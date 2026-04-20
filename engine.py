"""Core training engine for battery SOC estimation models.

Provides training, evaluation, testing, and comparison functions
that can be used from both train.py and train_colab.ipynb.
"""

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from test_model import compute_metrics
import config
from model import VanillaLSTM
from physics import PhysicsLoss, load_physics_coefficients


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    is_physics: bool,
    grad_clip: float = 1.0,
) -> tuple[float, float, float]:
    """Train the model for one epoch.

    Returns
    -------
    tuple[float, float, float]
        (avg_total_loss, avg_data_loss, avg_physics_loss)
    """
    model.train()
    running_loss = 0.0
    running_ml = 0.0
    running_phys = 0.0
    n_batches = 0

    for x, y, v, i in loader:
        x, y, v, i = x.to(device), y.to(device), v.to(device), i.to(device)
        soc_pred = model(x)

        if is_physics:
            loss, ml, phys = loss_fn(soc_pred, y, v, i)
        else:
            loss = loss_fn(soc_pred, y)
            ml = loss
            phys = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_ml += ml.item()
        running_phys += phys.item() if isinstance(phys, torch.Tensor) else phys
        n_batches += 1

    return running_loss / n_batches, running_ml / n_batches, running_phys / n_batches

@torch.no_grad()
def evaluate_with_metrics(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    is_physics: bool,
) -> dict[str, float]:
    """Evaluate the model on a dataset (losses + RMSE, MAE, Max Error).

    Returns
    -------
    dict[str, float]
        Keys: loss, ml_loss, phys_loss, rmse, mae, max_error
    """
    model.eval()
    total_loss = 0.0
    total_ml = 0.0
    total_phys = 0.0
    n_batches = 0
    all_pred = []
    all_true = []

    for x, y, v, i in loader:
        x, y, v, i = x.to(device), y.to(device), v.to(device), i.to(device)
        soc_pred = model(x)

        if is_physics:
            loss, ml, phys = loss_fn(soc_pred, y, v, i)
        else:
            loss = loss_fn(soc_pred, y)
            ml = loss
            phys = torch.tensor(0.0, device=device)

        total_loss += loss.item()
        total_ml += ml.item()
        total_phys += phys.item() if isinstance(phys, torch.Tensor) else phys
        n_batches += 1

        all_pred.append(soc_pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    pred = np.concatenate(all_pred).flatten()
    true = np.concatenate(all_true).flatten()
    metrics = compute_metrics(true, pred)

    return {
        "loss": total_loss / n_batches,
        "ml_loss": total_ml / n_batches,
        "phys_loss": total_phys / n_batches,
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "max_error": metrics["Max_Error"],
    }


@torch.no_grad()
def test_model(
    model: nn.Module,
    test_loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Test the model on all test cycles.

    Returns
    -------
    dict[str, dict[str, float]]
        Results mapping cycle name to dict with RMSE, MAE, Max_Error.
    """
    model.eval()
    results = {}

    for cycle_name, loader in test_loaders.items():
        soc_pred_all = []
        soc_true_all = []

        for x, y, v, i in loader:
            x = x.to(device)
            soc_pred = model(x)
            soc_pred_all.append(soc_pred.cpu().numpy())
            soc_true_all.append(y.numpy())

        pred = np.concatenate(soc_pred_all).flatten()
        true = np.concatenate(soc_true_all).flatten()

        results[cycle_name] = compute_metrics(true, pred)
        print(
            f"  {cycle_name:8s}  RMSE={results[cycle_name]['RMSE']:.4f}"
            f"  MAE={results[cycle_name]['MAE']:.4f}"
            f"  MaxErr={results[cycle_name]['Max_Error']:.4f}"
        )

    return results


CSV_FIELDS = [
    "epoch", "train_loss", "train_ml_loss", "train_phys_loss",
    "val_loss", "val_ml_loss", "val_phys_loss",
    "val_rmse", "val_mae", "val_max_error", "lr", "elapsed_s",
]


def train_model(
    is_physics: bool,
    beta: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loaders: dict[str, DataLoader],
) -> tuple[nn.Module, dict[str, dict[str, float]]]:
    """Train a battery SOC estimation model with early stopping.

    All hyperparameters (learning rate, epochs, patience, etc.) are read
    from ``config``.  Override them by modifying ``config`` before calling
    this function.

    Parameters
    ----------
    is_physics : bool
        Whether to use physics-informed loss.
    beta : float
        Weight for physics loss.
    train_loader : DataLoader
    val_loader : DataLoader
    test_loaders : dict[str, DataLoader]

    Returns
    -------
    tuple[nn.Module, dict[str, dict[str, float]]]
        (trained_model, test_results)
    """
    device = config.DEVICE
    alpha = config.ALPHA
    lr = config.LEARNING_RATE
    max_epochs = config.MAX_EPOCHS
    patience = config.PATIENCE
    grad_clip = config.GRAD_CLIP
    lr_factor = config.LR_FACTOR
    lr_patience = config.LR_PATIENCE
    results_dir = config.RESULTS_DIR
    trained_models_dir =  config.TRAINED_MODELS_DIR

    model_name = f"physics_lstm_beta_{beta}" if is_physics else "vanilla_lstm"

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"Device: {device}")
    print(f"Physics: {is_physics}  Beta: {beta}")
    print(f"{'=' * 60}\n")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)

    model = VanillaLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    if is_physics:
        ocv_coeffs, rin_coeffs = load_physics_coefficients()
        loss_fn = PhysicsLoss(ocv_coeffs, rin_coeffs, alpha=alpha, beta=beta).to(
            device
        )
        print(f"OCV polynomial degree: {len(ocv_coeffs) - 1}")
        print(f"R_in polynomial degree: {len(rin_coeffs) - 1}")
        print(f"Alpha={alpha}, Beta={beta}\n")
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )
    history_path = os.path.join(trained_models_dir, f"training_stats_{model_name}.csv")
    results_path = os.path.join(config.RESULTS_DIR, f"{model_name}_results.json")

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_state = None
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        train_loss, train_ml, train_phys = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, is_physics, grad_clip
        )
        val_metrics = evaluate_with_metrics(
            model, val_loader, loss_fn, device, is_physics
        )
        val_loss = val_metrics["loss"]
        val_ml = val_metrics["ml_loss"]
        val_phys = val_metrics["phys_loss"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_ml_loss": f"{train_ml:.6f}",
                "train_phys_loss": f"{train_phys:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_ml_loss": f"{val_ml:.6f}",
                "val_phys_loss": f"{val_phys:.6f}",
                "val_rmse": f"{val_metrics['rmse']:.6f}",
                "val_mae": f"{val_metrics['mae']:.6f}",
                "val_max_error": f"{val_metrics['max_error']:.6f}",
                "lr": f"{current_lr:.1e}",
                "elapsed_s": f"{elapsed:.1f}",
            }
        )

        if is_physics:
            print(
                f"Epoch {epoch:3d}/{max_epochs}  "
                f"train={train_loss:.6f} (ml={train_ml:.6f} phys={train_phys:.6f})  "
                f"val={val_loss:.6f} (ml={val_ml:.6f} phys={val_phys:.6f})  "
                f"val_RMSE={val_metrics['rmse']:.4f}  "
                f"lr={current_lr:.1e}  [{elapsed:.0f}s]"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{max_epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"val_RMSE={val_metrics['rmse']:.4f}  "
                f"lr={current_lr:.1e}  [{elapsed:.0f}s]"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {history_path}")

    model.load_state_dict(best_state)
    print(f"\nLoaded best model from epoch {best_epoch}")

    checkpoint_path = os.path.join(trained_models_dir, f"{model_name}.pt")
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "val_loss": best_val_loss,
            "is_physics": is_physics,
            "beta": beta,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "seq_len": config.SEQ_LEN,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTest Results:")
    test_results = test_model(model, test_loaders, device)

    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to {results_path}")

    return model, test_results
