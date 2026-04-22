# Digital Twin — Panasonic 18650PF Li-ion Battery

Physics-informed LSTM-based digital twin for SOC estimation of the Panasonic 18650PF (2.9 Ah) lithium-ion battery, using the dataset from [Kollmeyer, 2017](https://data.mendeley.com/datasets/wykht8y7tg/1).

The project compares four training regimes for the same LSTM architecture:

- **Vanilla LSTM** — trained with MSE loss and dropout regularization (p = 0.05)
- **Physics-Informed LSTM** — trained with MSE + physics consistency loss and dropout (p = 0.05)
- **Physics-Informed LSTM (no dropout)** — trained with MSE + physics consistency loss, no dropout (physics loss is the sole regularizer)
- **Vanilla LSTM (no dropout)** — trained with MSE loss, no dropout (unregularized baseline)

## Project Structure

```
dt_battery_lstm/
├─ data/
│  ├─ 25degC_prepared/          # Filtered, normalized CSV data with SOC
│  ├─ data_visualization/       # Generated plots (OCV, Nyquist, drive cycles)
│  ├─ load_battery_dataset.py   # Download & convert .mat → .csv
│  ├─ prepare_data.py           # Filter, normalize, and compute SOC
│  ├─ visualize_data.py         # Visualization utilities (OCV, drive cycle, Nyquist)
├─ curve_fitting/
│  ├─ ocv_soc_lookup.csv        # OCV vs SOC lookup table
│  ├─ r_in_soc_lookup.csv       # R_in vs SOC lookup table
│  ├─ ocv_lookup.py             # OCVModel class
│  ├─ rin_lookup.py             # InternalResistanceModel class
├─ config.py                    # Centralized hyperparameters & paths
├─ physics.py                   # Differentiable OCV & R_in (nn.Module), PhysicsLoss
├─ dataset.py                   # BatteryWindowDataset, create_dataloaders()
├─ model.py                     # VanillaLSTM (Hochreiter & Schmidhuber, 1997)
├─ engine.py                    # Training loop, evaluation, testing
├─ train.py                     # CLI entry point for training
├─ train_colab.ipynb            # Google Colab notebook for training
├─ inference_utils.py           # Model loading, prediction, metrics, MC uncertainty
├─ generate_results.py          # End-to-end evaluation: plots, comparison tables
├─ trained_models/              # Saved checkpoints (.pt) and training stats (.csv)
├─ results/                     # Generated plots and comparison JSON
├─ README.md
```

## Drive Cycles

| Cycle | Role | Standard | Duration | Key Characteristics | Battery Dynamics |
|-------|------|-----------|----------|--------------------|------------------|
| **LA92** | Train | US EPA LA92 | ~235 min | Aggressive urban driving; rapid accelerations/decelerations; high dynamic content | Large, frequent current swings; deep SOC transients; strong voltage hysteresis |
| **US06** | Validation | US EPA US06 | ~80 min | High-speed highway with aggressive accel; highest peak currents (~12 A) | Sustained high C-rates; significant ohmic drop (I·R_in); rapid SOC depletion and thermal stress |
| **Cycle 1** | Test | Custom | ~293 min | Mixed urban/highway with moderate dynamics | Moderate current variation; battery operates across wide SOC range; voltage follows OCV closely at low currents |
| **Cycle 2** | Test | Custom | ~241 min | Repetitive urban pattern with frequent start-stop | Pulsed discharge with short rest periods; SOC drops in stepwise manner; RC relaxation effects between pulses |
| **Cycle 3** | Test | Custom | ~293 min | Mixed profile with extended low-current cruise phases | Long periods of near-OCV operation; weak data-driven signal — model must rely on OCV(SOC) relationship; slow SOC drift |
| **Cycle 4** | Test | Custom | ~320 min | Includes deep discharge near end-of-life SOC (≤20%) | Operating in the nonlinear low-SOC region where OCV curve steepens and R_in increases; voltage dynamics become highly nonlinear |
| **HWFTa** | Test | US EPA HWFET (first half) | ~134 min | Steady-state highway cruising at ~100 km/h | Near-constant moderate discharge current; SOC decreases linearly; minimal transient dynamics — OCV dominates terminal voltage |
| **HWFTb** | Test | US EPA HWFET (second half) | ~135 min | Mirror of HWFTa but starting at lower SOC | Same steady dynamics as HWFTa but shifted to lower SOC band where R_in is slightly higher and OCV slope is steeper |
| **UDDS** | Test | US EPA UDDS (urban) | ~402 min | Standard urban driving schedule; low-speed, stop-and-go traffic | Frequent low-amplitude current pulses with short rests; diffusion and RC dynamics dominate; OCV recovery during pauses slowly adjusts SOC |
| **NN** | Test | Neural-network generated | ~186 min | Synthetically generated profile; non-standard dynamics | Irregular current patterns that don't follow typical drive cycle statistics; challenges generalization since the physics model wasn't fitted to these dynamics |

**Key insight for SOC estimation**: The difficulty of each cycle depends on which battery dynamics dominate. Cycles with high dynamic current (US06, LA92) produce strong data-driven signals but large ohmic drops. Cycles with sustained low currents (UDDS, HWFT) rely more on the OCV(SOC) relationship since the I·R_in term is small. Deep-discharge cycles (Cycle 4) push the model into the nonlinear OCV region, and synthetic profiles (NN) test pure generalization.

## Data Pipeline

### 1. Download & Convert (`data/load_battery_dataset.py`)

Downloads the 25 °C portion of the Mendeley dataset, converts all `.mat` files to `.csv`, drops the `TimeStamp` column, and stores the result in `data/25degC_csv/`.

```bash
python data/load_battery_dataset.py
```

### 2. Filter, Normalize & SOC Estimation (`data/prepare_data.py`)

Processes the raw CSV files (excluding EIS experiments):

- **Butterworth low-pass filter** (2nd order, 1 Hz cutoff) applied to `Current`, `Voltage`, `Battery_Temp_degC`, and `Chamber_Temp_degC`.
- **Z-normalization** (`sklearn.preprocessing.StandardScaler`) on the filtered Current, Voltage, and Battery_Temp_degC.
- **SOC via Coulomb counting**:

$$\frac{d\,SOC(t)}{dt} = \frac{\eta\, I(t)}{C_n}$$

with $C_n = 2.9\,\text{Ah}$ and $\eta = 0.98$. Discharge starts at SOC = 1, charge at SOC = 0.

```bash
python data/prepare_data.py
```

Each output CSV in `data/25degC_prepared/` contains: `Time`, `Current_filt`, `Voltage_filt`, `Temp_degC_filt`, `Chamber_Temp_degC_filt`, `Current_norm`, `Voltage_norm`, `Temp_degC_norm`, `SOC`.

## Curve Fitting

### OCV vs SOC (`curve_fitting/ocv_lookup.py`)

Extracts the open-circuit voltage curve from the C/20 discharge test. Only discharge-phase data is used, with a small IR correction ($R_{in} = 22\,\text{mOhm}$) to account for the residual voltage drop:

$$OCV \approx V_{measured} - I \cdot R_{in}$$

A 7th-degree polynomial is fitted (used by the physics-informed loss):

$$OCV(SOC) = \sum_{k=0}^{7} a_k \cdot SOC^k$$

**Fit quality**: RMSE = 0.011 V, Max |Error| = 0.107 V (at SOC ≈ 0 boundary).

```bash
python curve_fitting/ocv_lookup.py
```

The `OCVModel` class provides:
- `fit()` — fits the polynomial and returns coefficients
- `estimate(soc)` — returns OCV [V] for a given SOC value

### Internal Resistance vs SOC (`curve_fitting/rin_lookup.py`)

Extracts the battery's ohmic resistance from the EIS (electrochemical impedance spectroscopy) experiments. The method detects the sign change of the imaginary impedance `Zimg1` from positive to negative — at this crossover point, `Zreal1` represents the real part of the battery impedance, which corresponds to the internal resistance $R_{in}$:

$$R_{in} = Z_{real}\big|_{Z_{img} \text{ sign change } (+ \to -)}$$

SOC for each EIS spectrum is estimated from the accumulated amp-hours:

$$SOC = 1 - \frac{|AhAccu|}{C_n}$$

A 3rd-degree polynomial is fitted:

$$R_{in}(SOC) = b_3 \cdot SOC^3 + b_2 \cdot SOC^2 + b_1 \cdot SOC + b_0$$

| Coefficient | Value |
|---|---|
| $b_3$ | 1.3887 |
| $b_2$ | 0.2119 |
| $b_1$ | -3.5397 |
| $b_0$ | 23.0912 |

$R_{in}$ ranges from ~21 mOhm at high SOC to ~23 mOhm at low SOC.

```bash
python curve_fitting/rin_lookup.py
```

The `InternalResistanceModel` class provides:
- `fit()` — fits the polynomial and returns coefficients
- `estimate(soc)` — returns $R_{in}$ [mOhm] for a given SOC value

## SOC Estimation — LSTM Model

### Architecture

A vanilla LSTM as proposed by Hochreiter & Schmidhuber (1997):

```
Input (batch, seq_len, 2)       ← [Current_norm, Voltage_norm]
    │
    ▼
nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
    │
    ▼
lstm_out (batch, seq_len, 64)
    │
    ▼
nn.Dropout(p=0.05)  ← Turns ON/OFF depending on the model and dropout condition
    │
    ▼
nn.Linear(64, 1)
    │
    ▼
nn.Sigmoid()                    ← Constrains SOC to (0, 1)
    │
    ▼
Output (batch, seq_len)         ← SOC prediction at every timestep
```

**Key design choices**:

- **Sigmoid output**: SOC is bounded in (0, 1). The sigmoid activation enforces this constraint, which is critical for the physics-informed model — the OCV and R_in polynomials are only valid for SOC ∈ [0, 1], and predicting outside this range would cause numerical instability.
- **Sequence-to-sequence**: The model outputs SOC at every timestep (not just the last), providing richer gradient signal and enabling the physics loss to be applied at every step.
- **Input features**: `[Current_norm, Voltage_norm]` — the two signals directly linked by the circuit equation V = OCV(SOC) + I·R_in(SOC). Temperature is excluded because the physics model (OCV, R_in lookup tables) is temperature-independent (only 25 °C data available).
- **Dropout control**: The `apply_dropout` parameter in `forward()` controls whether dropout is applied:
  - `None` (default) — auto: dropout ON in training mode, OFF in eval mode
  - `True` — force dropout ON (for MC uncertainty estimation)
  - `False` — force dropout OFF (for physics-informed training without dropout)
- **17,217 trainable parameters** (LSTM: 17,152 + Linear: 64 + bias: 1).

### Train / Validation / Test Split

| Split | Cycle | Duration | Rows | Character |
|-------|-------|----------|------|-----------|
| **Train** | LA92 | 235 min | 140,874 | Aggressive urban, high dynamic content |
| **Validation** | US06 | 80 min | 48,061 | Aggressive highway, highest peak currents |
| **Test** | Cycle 1–4, HWFTa/b, UDDS, NN | ~1,408 min | ~838,000 | Diverse unseen profiles |

Training on a single cycle (LA92) and testing on 8 unseen cycles provides a strong generalization test — the model must learn real battery dynamics, not memorize a specific drive profile.

### Windowing

The time series is segmented into overlapping windows of 200 timesteps (20 seconds at 10 Hz sampling):

| Parameter | Train | Validation | Test |
|-----------|-------|------------|------|
| Window size | 200 steps (20 s) | 200 steps (20 s) | 200 steps (20 s) |
| Stride | 1 (overlapping) | 200 (non-overlapping) | 200 (non-overlapping) |

- **Training stride = 1** maximizes training samples (~140,675 windows from LA92) and acts as data augmentation.
- **Val/Test stride = 200** ensures non-overlapping, independent evaluation with no information leakage.

### Physics Model

The battery is modeled as an equivalent circuit with three elements in series:

```
                    I(t) ──>
            ┌───────── R_in(SOC) ───────── +
            │                              
            |                              
 OCV(SOC) ─────                           V(t)
           ───                            
            |                              
            └───────────────────────────── -
```

Where:
- $I(t)$ and $V(t)$ are the current and voltage of the battery, respectively.
- $OCV(SOC)$ — 7th-degree polynomial fitted to the C/20 discharge test
- $R_{in}(SOC)$ — 3rd-degree polynomial (mOhm → Ohm) fitted to EIS data

Both polynomials are implemented as differentiable `nn.Module` in `physics.py` using Horner's method, with coefficients stored as frozen `torch.Tensor` buffers. This ensures gradients flow from the physics loss through OCV(SOC) and R_in(SOC) back to the LSTM parameters — no detachment or NumPy conversion.

### Loss Functions

#### Model 1: Vanilla LSTM Loss (data-driven with dropout)

$$L = MSE(SOC_{pred},\, SOC_{true})$$

Dropout (p = 0.05) is active during training and validation, and provides uncertainty estimation at inference.

#### Model 2: Physics-Informed LSTM Loss (with dropout)

$$L = \alpha \cdot \underbrace{MSE(SOC_{pred},\, SOC_{true})}_{L_{data}} + \beta \cdot \underbrace{MSE(V_{measured},\, OCV(SOC_{pred}) + I \cdot R_{in}(SOC_{pred}))}_{L_{physics}}$$

With $\alpha = 1.0$ and $\beta = 0.1$ (default). Dropout (p = 0.05) is active during training and also during validation (to keep the physics loss signal consistent). At inference, dropout is OFF for deterministic predictions; activated via `apply_dropout=True` for MC uncertainty.

#### Model 3: Physics-Informed LSTM Loss (no dropout)

Same loss function as Model 2, but dropout is **completely disabled** (`apply_dropout=False`). The physics consistency loss acts as the sole regularizer. This configuration tests whether the physics loss alone provides sufficient regularization without dropout. MC uncertainty estimation is **not available** for this model.

#### Regularization trought dropout layer and physics-informed loss

**What the dropout layer does:** In the context of sequence modeling, dropout disrupts the LSTM's reliance on specific hidden state pathways, effectively training an ensemble of sub-networks within a single architecture — at inference time, when all neurons are active, the model benefits from this implicit averaging effect, which smooths predictions and can reduce overfitting, especially when training data is limited or contains noisy labels like Coulomb-counted SOC values.

**What the physics loss does:**  It measures how well the predicted SOC is consistent with Kirchhoff's voltage law. If the LSTM predicts an SOC that, when plugged into the circuit model, does not match the measured terminal voltage, the physics loss is high. This acts as a soft constraint / regularizer that steers predictions toward physically plausible values.

**Why this helps**:
1. **Regularization** — prevents the model from fitting noise in the SOC labels (which come from approximate Coulomb counting).
2. **Generalization** — the circuit model provides an inductive bias that does not depend on the training data distribution.
3. **Low-dynamics regimes** — when current excitation is small, the data-driven signal is weak, but the physics model still provides guidance via OCV(SOC).

**β weighting**: Too high forces the model to fit the approximate physics (which neglects RC dynamics), ignoring the data. Too low gives no physics benefit. β = 0.1 is a starting point where the physics loss contributes ~9% of the total loss.

**Key design principle**: All four models share the identical LSTM architecture. The only differences are the loss function (MSE vs. MSE+physics) and whether dropout is active. This ensures a fair comparison — any performance difference is attributable to the loss function and/or dropout configuration.

### Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 1e-3) |
| LR scheduler | ReduceLROnPlateau (factor = 0.5, patience = 5) |
| Gradient clipping | max_norm = 1.0 |
| Max epochs | 100 |
| Early stopping | patience = 15 (on validation loss) |
| Batch size | 64 |

### Four-Model Comparison

| | Model 1: Vanilla LSTM | Model 2: Physics LSTM | Model 3: Physics LSTM (no dropout) | Model 4: Vanilla LSTM (no dropout) |
|---|---|---|---|---|
| **Command** | `python train.py` | `python train.py --physics` | `python train.py --physics --no-dropout` | `python train.py --no-dropout` |
| **Loss function** | MSE | MSE + Physics | MSE + Physics | MSE |
| **Dropout during training** | ON (p = 0.05) | ON (p = 0.05) | OFF | OFF |
| **Dropout during validation** | OFF | ON (p = 0.05) | OFF | OFF |
| **Dropout during inference** | OFF | OFF | OFF | OFF |
| **MC uncertainty** | Available (`apply_dropout=True`) | Available (`apply_dropout=True`) | Not available | Not available |
| **Regularization** | Dropout only | Dropout + Physics | Physics only | None |
| **Checkpoint name** | `vanilla_lstm.pt` | `physics_lstm.pt` | `physics_lstm_no_dropout.pt` | `vanilla_lstm_no_dropout.pt` |

### Evaluation Metrics

For each test cycle:

- **RMSE** — Root Mean Square Error (primary metric, penalizes large errors)
- **MAE** — Mean Absolute Error (interpretable, robust to outliers)
- **Max Error** — Worst-case prediction error (important for safety-critical SOC estimation)

### Monte Carlo Uncertainty

The LSTM includes dropout (p = 0.05) which can be activated during inference for uncertainty estimation. Running multiple stochastic forward passes (with `apply_dropout=True`) produces a distribution of SOC predictions per timestep, from which the mean prediction and standard deviation are computed. The 95% confidence band (mean ± 2σ) is plotted alongside the true SOC for each test cycle via `generate_results.py`.

**Availability by model**:
- **Model 1** (Vanilla, dropout): MC uncertainty available
- **Model 2** (Physics, dropout): MC uncertainty available
- **Model 3** (Physics, no dropout): MC uncertainty **not available** — raises `ValueError` if attempted. Since dropout was never active during training, stochastic forward passes would produce meaningless uncertainty estimates.
- **Model 4** (Vanilla, no dropout): MC uncertainty **not available** — same reason as Model 3.

### Results Comparison

Run `python generate_results.py` to evaluate all trained models and produce:
- `results/vanilla_lstm_dp0%_vs_physics_lstm_dp0%.json` — comparison of no-dropout models
- `results/vanilla_lstm_dp5%_vs_physics_lstm_dp5%.json` — comparison of dropout models
- `results/mc_uncertainty_<cycle>.png` — MC uncertainty plots per cycle (Models 1 & 2 only)
- `results/training_curves.png` — train/val loss curves for all models
- `results/heatmap_rmse.png` — RMSE heatmap across models and cycles
- `results/heatmap_mae.png` — MAE heatmap across models and cycles

## Usage

Both models can also be trained on Google Colab using `train_colab.ipynb`.

### Train Model 1: Vanilla LSTM (with dropout)

```bash
python train.py
```

### Train Model 2: Physics-Informed LSTM (with dropout)

```bash
python train.py --physics
```

### Train Model 3: Physics-Informed LSTM (no dropout)

```bash
python train.py --physics --no-dropout
```

### Train Model 4: Vanilla LSTM (no dropout)

```bash
python train.py --no-dropout
```

You can optionally override the default beta for physics models:

```bash
python train.py --physics --beta 0.1
python train.py --physics --no-dropout --beta 0.1
```

### End-to-End Evaluation

```bash
python generate_results.py
```

Evaluates all trained models across all test cycles and generates:
- **MC uncertainty plots** per cycle with 95% confidence bands (Models 1 & 2 only; Models 3 & 4 have no MC uncertainty)
- **Training curves overlay** (train/val loss for all models)
- **RMSE and MAE heatmaps** across models and cycles
- **Comparison tables** with per-cycle RMSE, MAE, and Max Error (saved as JSON, one per dropout configuration)

Options:
- `--no-plots` — skip plot generation, only produce metrics tables
- `--device cpu` — force CPU inference
- `--mc-iterations 50` — number of MC dropout passes (default: 50)

### Utility Module (`inference_utils.py`)

Provides reusable functions for loading models and running predictions:
- `load_model(name, device)` — load a checkpoint from `trained_models/`
- `predict_cycle(model, dataset, device)` — deterministic inference
- `predict_cycle_with_uncertainty(model, dataset, device, n_iterations)` — MC dropout predictions returning (mean, std). Raises `ValueError` for Model 3 (no dropout).
- `compute_metrics(soc_true, soc_pred)` — compute RMSE, MAE, Max Error for arrays

Note: `predict_cycle_with_uncertainty()` raises a `ValueError` when called on a model trained with `--no-dropout`, since MC uncertainty estimation requires a trained dropout layer.

### Visualization (`data/visualize_data.py`)

```python
from data.visualize_data import plot_ocv_soc, plot_drive_cycle, plot_nyquist

# OCV(SOC) curve
plot_ocv_soc()

# Drive cycle data (Current, Voltage, SOC)
plot_drive_cycle("data/25degC_prepared/Drive cycles/<filename>.csv")

# Nyquist impedance plot across all SOC values
plot_nyquist()
```

## Hyperparameters Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SEQ_LEN` | 200 | Number of timesteps per input window (200 steps = 20 s at 10 Hz) |
| `TRAIN_STRIDE` | 1 | Stride between consecutive training windows (1 = maximum overlap / data augmentation) |
| `VAL_STRIDE` / `TEST_STRIDE` | 200 | Stride between consecutive validation/test windows (200 = non-overlapping) |
| `BATCH_SIZE` | 64 | Number of windows per mini-batch |
| `INPUT_SIZE` | 2 | Number of input features per timestep ([Current_norm, Voltage_norm]) |
| `HIDDEN_SIZE` | 64 | Number of hidden units in each LSTM layer |
| `NUM_LAYERS` | 1 | Number of stacked LSTM layers |
| `LEARNING_RATE` | 1e-3 | Initial learning rate for the Adam optimizer |
| `MAX_EPOCHS` | 100 | Maximum number of training epochs (with early stopping) |
| `PATIENCE` | 15 | Early stopping patience (epochs without val loss improvement) |
| `GRAD_CLIP` | 1.0 | Maximum gradient norm for gradient clipping |
| `ALPHA` | 1.0 | Weight for the data-driven MSE loss term (L_data) |
| `BETA` | 0.1 | Weight for the physics consistency loss term (L_physics) |
| `DROPOUT` | 0.05 | Dropout rate for MC uncertainty estimation and regularization |
| `DROPOUT_CONDITION` | True | Whether dropout is active during training (False → physics-only regularization via --no-dropout) |
| `LR_FACTOR` | 0.5 | Factor by which the learning rate is reduced on plateau |
| `LR_PATIENCE` | 5 | Number of epochs with no val loss improvement before reducing LR |

## Dataset Reference

Phillip Kollmeyer, "Panasonic 18650PF Li-ion Battery Data", Mendeley Data, V1, 2017. DOI: [10.17632/wykht8y7tg.1](https://data.mendeley.com/datasets/wykht8y7tg/1)

## Future Work

- **Add RC dynamics to physics model** — Include at least one RC pair (V = OCV + I·R_in + V_RC) to capture polarization/transient effects. This would make the physics loss significantly more meaningful.
- **Temperature-aware model** — Add temperature as an input feature and make OCV/R_in temperature-dependent (the dataset includes multiple temperatures).
- **Adaptive β scheduling** — Instead of fixed β=0.1, use gradient balancing (e.g., GradNorm or uncertainty-weighted multi-task loss) to dynamically weight data vs. physics loss.
- **Validate ground truth SOC** — Cross-validate Coulomb-counted SOC against HPPC-derived SOC or OCV-rest measurements to quantify label noise.
- **Add ablation studies** — Test β sensitivity (0.01, 0.05, 0.1, 0.5, 1.0), different polynomial degrees for OCV/R_in, and sequence length effects.
- **Add cross-validation** — Train on multiple cycles (not just LA92) and test on held-out ones to reduce single-cycle bias.

## Citation

If you use this software in your work, please cite:

Juan Aponte. (2026). Digital_Twin_Battery_with_LSTM(Version 1.0)