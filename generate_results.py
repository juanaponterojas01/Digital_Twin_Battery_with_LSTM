"""
End-to-end evaluation script for the battery SOC estimation project.

Evaluates the vanilla LSTM and physics-informed LSTM (beta=0.1) across
all test cycles, and generates:
  - SOC prediction plots per cycle (Vanilla vs Physics vs True)
  - Monte Carlo uncertainty plots per cycle (2 subplots with 95% CI)
  - Training curves overlay
  - Comparison table as JSON with per-cycle RMSE, MAE, and Max Error

Usage:
    python generate_results.py
    python generate_results.py --no-plots
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from dataset import BatteryWindowDataset
from inference_utils import (
    compute_metrics,
    load_model,
    predict_cycle,
    predict_cycle_with_uncertainty,
)

MODEL_NAMES = ["vanilla_lstm", "physics_lstm_beta_0.1"]
MODEL_LABELS = {"vanilla_lstm": "Vanilla LSTM", "physics_lstm_beta_0.1": "Physics LSTM (beta=0.1)"}
MODEL_COLORS = {"vanilla_lstm": "tab:blue", "physics_lstm_beta_0.1": "tab:orange"}

RESULTS_DIR = os.path.join(config.PROJECT_DIR, "results")
TRAINED_DIR = config.TRAINED_MODELS_DIR
MAX_PLOT_POINTS = 5000


def _load_cycle_dataset(cycle_name: str):
    """Load a drive cycle CSV and return its BatteryWindowDataset."""
    filename = config.DRIVE_CYCLES[cycle_name]
    csv_path = os.path.join(config.DATA_DIR, filename)
    df = pd.read_csv(csv_path)
    return BatteryWindowDataset(df, config.SEQ_LEN, config.TEST_STRIDE), df


def _decimate(arr, max_points):
    """Fast downsampling for plotting: keep every k-th point.

    Keeps all data when len(arr) <= max_points.
    """
    if max_points is None or len(arr) <= max_points:
        return arr, np.arange(len(arr))
    step = max(1, len(arr) // max_points)
    idx = np.arange(0, len(arr), step)
    return arr[idx], idx


def plot_training_curves_overlay(save_path):
    """Overlay training and validation loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, label, color in [
        ("vanilla_lstm", "Vanilla LSTM", "tab:blue"),
        ("physics_lstm_beta_0.1", "Physics LSTM (β=0.1)", "tab:orange"),
    ]:
        csv_path = os.path.join(TRAINED_DIR, f"training_stats_{name}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        epochs = df["epoch"].astype(int)
        train_loss = df["train_loss"].astype(float)
        val_loss = df["val_loss"].astype(float)

        axes[0].plot(epochs, train_loss, label=f"{label} (train)",
                     color=color, linewidth=1.5)
        axes[1].plot(epochs, val_loss, label=f"{label} (val)",
                     color=color, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def build_comparison_table(all_results):
    """Build a side-by-side comparison DataFrame."""
    cycles = config.TEST_CYCLES
    rows = []
    for c in cycles:
        v = all_results["vanilla_lstm"][c]
        p = all_results["physics_lstm_beta_0.1"][c]
        rmse_delta = ((p["RMSE"] - v["RMSE"]) / v["RMSE"]) * 100
        mae_delta = ((p["MAE"] - v["MAE"]) / v["MAE"]) * 100
        maxerr_delta = ((p["Max_Error"] - v["Max_Error"]) / v["Max_Error"]) * 100
        rows.append({
            "Cycle": c,
            "Vanilla RMSE": f"{v['RMSE']:.4f}",
            "Physics RMSE": f"{p['RMSE']:.4f}",
            "Impr%": f"{rmse_delta:+.2f}",
            "Vanilla MAE": f"{v['MAE']:.4f}",
            "Physics MAE": f"{p['MAE']:.4f}",
            "Impr%": f"{mae_delta:+.2f}",
            "Vanilla MaxErr": f"{v['Max_Error']:.4f}",
            "Physics MaxErr": f"{p['Max_Error']:.4f}",
            "MaxErr Delta%": f"{maxerr_delta:+.2f}",
        })
    return pd.DataFrame(rows)


def _extract_true_soc(dataset: BatteryWindowDataset) -> np.ndarray:
    """Extract ground truth SOC values from dataset windows."""
    all_soc = []
    for idx in range(len(dataset)):
        _, y, _, _ = dataset[idx]
        all_soc.append(y.numpy())
    return np.concatenate(all_soc).flatten()


def plot_soc_predictions(cycle_name, soc_true, soc_vanilla, soc_physics,
                          save_path, max_points=MAX_PLOT_POINTS):
    """One plot per cycle: truth vs Vanilla vs Physics SOC."""
    soc_true_d, t = _decimate(soc_true, max_points)
    soc_vanilla_d, _ = _decimate(soc_vanilla, max_points)
    soc_physics_d, _ = _decimate(soc_physics, max_points)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, soc_true_d, label="True SOC", color="tab:green", lw=1.5)
    ax.plot(t, soc_vanilla_d, label="Vanilla LSTM", color="tab:blue", lw=1)
    ax.plot(t, soc_physics_d, label="Physics LSTM", color="tab:orange", lw=1)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("SOC")
    ax.set_title(f"SOC Predictions — {cycle_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_mc_uncertainty(cycle_name, soc_true, mean_v, std_v, mean_p, std_p,
                        save_path, max_points=MAX_PLOT_POINTS):
    """Two subplots per cycle: MC uncertainty for Vanilla and Physics."""
    soc_true_d, t = _decimate(soc_true, max_points)
    mean_v_d, _ = _decimate(mean_v, max_points)
    std_v_d, _ = _decimate(std_v, max_points)
    mean_p_d, _ = _decimate(mean_p, max_points)
    std_p_d, _ = _decimate(std_p, max_points)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mean, std, label, color in [
        (axes[0], mean_v_d, std_v_d, "Vanilla LSTM", "tab:blue"),
        (axes[1], mean_p_d, std_p_d, "Physics LSTM", "tab:orange"),
    ]:
        ax.plot(t, soc_true_d, label="True SOC", color="tab:green", lw=1.5)
        ax.plot(t, mean, label=label, color=color, lw=1)
        ci_lo = np.clip(mean - 1.96 * std, 0, 1)
        ci_hi = np.clip(mean + 1.96 * std, 0, 1)
        ax.fill_between(t, ci_lo, ci_hi, color=color, alpha=0.2,
                        label="95% CI")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("SOC")
        ax.set_title(f"{label} — {cycle_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results for SOC estimation models")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (only compute metrics JSON)")
    parser.add_argument("--mc-iterations", type=int, default=50,
                        help="Number of MC dropout forward passes for uncertainty (default: 50)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading models...")
    model_vanilla = load_model("vanilla_lstm", device)
    model_physics = load_model("physics_lstm_beta_0.1", device)
    if model_vanilla is None:
        raise FileNotFoundError("Checkpoint vanilla_lstm.pt not found")
    if model_physics is None:
        raise FileNotFoundError("Checkpoint physics_lstm_beta_0.1.pt not found")

    all_results = {"vanilla_lstm": {}, "physics_lstm_beta_0.1": {}}

    for cycle_name in config.TEST_CYCLES:
        print(f"\n[{cycle_name}]")
        dataset, _ = _load_cycle_dataset(cycle_name)
        soc_true = _extract_true_soc(dataset)

        soc_vanilla = predict_cycle(model_vanilla, dataset, device)
        soc_physics = predict_cycle(model_physics, dataset, device)

        metrics_v = compute_metrics(soc_true, soc_vanilla)
        metrics_p = compute_metrics(soc_true, soc_physics)
        all_results["vanilla_lstm"][cycle_name] = metrics_v
        all_results["physics_lstm_beta_0.1"][cycle_name] = metrics_p

        print(f"  Vanilla — RMSE={metrics_v['RMSE']:.4f}  MAE={metrics_v['MAE']:.4f}")
        print(f"  Physics — RMSE={metrics_p['RMSE']:.4f}  MAE={metrics_p['MAE']:.4f}")

        if not args.no_plots:
            soc_path = os.path.join(RESULTS_DIR, f"soc_predictions_{cycle_name}.png")
            plot_soc_predictions(cycle_name, soc_true, soc_vanilla,
                                 soc_physics, soc_path)

            mean_v, std_v = predict_cycle_with_uncertainty(
                model_vanilla, dataset, device, n_iterations=args.mc_iterations)
            mean_p, std_p = predict_cycle_with_uncertainty(
                model_physics, dataset, device, n_iterations=args.mc_iterations)

            mc_path = os.path.join(RESULTS_DIR, f"mc_uncertainty_{cycle_name}.png")
            plot_mc_uncertainty(cycle_name, soc_true,
                                mean_v, std_v, mean_p, std_p, mc_path)

    if not args.no_plots:
        train_path = os.path.join(RESULTS_DIR, "training_curves.png")
        plot_training_curves_overlay(train_path)

    comparison_df = build_comparison_table(all_results)
    print("\n" + "=" * 60)
    print("Metrics Comparison:")
    print(comparison_df.to_string(index=False))

    json_path = os.path.join(RESULTS_DIR, "metrics_comparison.json")
    with open(json_path, "w") as f:
        json.dump(comparison_df.to_dict(orient="records"), f, indent=2)
    print(f"\nSaved: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()