"""Core training engine for battery SOC estimation models.

Provides training, evaluation, testing, and comparison functions
that can be used from both train.py and train_colab.ipynb.
"""
import csv
import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import VanillaLSTM
from physics import PhysicsLoss, load_physics_coefficients
from inference_utils import compute_metrics


logger = logging.getLogger(__name__)

effective_dropout = 0.0 if not config.DROPOUT_CONDITION else config.DROPOUT

@dataclass
class TrainingMetrics:
    """Container for training epoch metrics."""
    epoch: int
    train_loss: float
    train_ml_loss: float
    train_phys_loss: float
    val_loss: float
    val_ml_loss: float
    val_phys_loss: float
    val_rmse: float
    val_mae: float
    val_max_error: float
    lr: float
    elapsed_s: float


CSV_FIELDS = [
    "epoch", "train_loss", "train_ml_loss", "train_phys_loss",
    "val_loss", "val_ml_loss", "val_phys_loss",
    "val_rmse", "val_mae", "val_max_error", "lr", "elapsed_s",
]


def _compute_losses(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    is_physics: bool,
) -> tuple[float, float, float, list[torch.Tensor], list[torch.Tensor]]:
    """Compute losses on a dataset.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        DataLoader providing batches.
    loss_fn : nn.Module
        Loss function (MSELoss or PhysicsLoss).
    device : torch.device
        Device to run computation on.
    is_physics : bool
        Whether physics-informed loss is used.

    Returns
    -------
    tuple[float, float, float, list[torch.Tensor], list[torch.Tensor]]
        (avg_total_loss, avg_ml_loss, avg_phys_loss, all_preds, all_targets)
    """
    model.eval()
    running_loss = 0.0
    running_ml = 0.0
    running_phys = 0.0
    n_batches = 0
    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y, v, i in loader:
            x, y, v, i = x.to(device), y.to(device), v.to(device), i.to(device)
            # Model 1 (vanilla, dropout): apply_dropout=None → auto (OFF in eval mode)
            # Model 2 (physics, dropout): apply_dropout=True → forced ON for consistent validation
            # Model 3 (physics, no dropout): apply_dropout=False → forced OFF
            if not config.DROPOUT_CONDITION:
                apply_dropout_val = False
            elif is_physics:
                apply_dropout_val = True
            else:
                apply_dropout_val = None
            soc_pred = model(x, apply_dropout=apply_dropout_val)

            if is_physics:
                loss, ml, phys = loss_fn(soc_pred, y, v, i)
            else:
                loss = loss_fn(soc_pred, y)
                ml = loss
                phys = torch.tensor(0.0, device=device)

            running_loss += loss.item()
            running_ml += ml.item()
            running_phys += phys.item() if isinstance(phys, torch.Tensor) else phys
            n_batches += 1

            all_pred.append(soc_pred.cpu())
            all_true.append(y.cpu())

    return (
        running_loss / n_batches,
        running_ml / n_batches,
        running_phys / n_batches,
        all_pred,
        all_true,
    )


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

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer for gradient updates.
    loss_fn : nn.Module
        Loss function (MSELoss or PhysicsLoss).
    device : torch.device
        Device to run computation on.
    is_physics : bool
        Whether physics-informed loss is used.
    grad_clip : float
        Maximum gradient norm for clipping (default: 1.0).

    Returns
    -------
    tuple[float, float, float]
        (avg_total_loss, avg_ml_loss, avg_physics_loss)
    """
    model.train()
    running_loss = 0.0
    running_ml = 0.0
    running_phys = 0.0
    n_batches = 0

    for x, y, v, i in loader:
        x, y, v, i = x.to(device), y.to(device), v.to(device), i.to(device)
        # Model 3 (physics, no dropout): apply_dropout=False disables dropout during training
        # Models 1 & 2 (dropout active): apply_dropout=None → auto (ON in train mode)
        soc_pred = model(x, apply_dropout=None if config.DROPOUT_CONDITION else False)

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


def validate_with_metrics(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    is_physics: bool,
) -> dict[str, float]:
    """Evaluate model on a dataset and compute metrics.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        Validation/test data loader.
    loss_fn : nn.Module
        Loss function (MSELoss or PhysicsLoss).
    device : torch.device
        Device to run computation on.
    is_physics : bool
        Whether physics-informed loss is used.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: loss, ml_loss, phys_loss, rmse, mae, max_error
    """
    avg_loss, avg_ml, avg_phys, all_pred, all_true = _compute_losses(
        model, loader, loss_fn, device, is_physics
    )

    pred = np.concatenate([p.numpy() for p in all_pred]).flatten()
    true = np.concatenate([t.numpy() for t in all_true]).flatten()
    metrics = compute_metrics(true, pred)

    return {
        "loss": avg_loss,
        "ml_loss": avg_ml,
        "phys_loss": avg_phys,
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

    Parameters
    ----------
    model : nn.Module
        The model to evaluate (should be in eval mode).
    test_loaders : dict[str, DataLoader]
        Dictionary mapping cycle names to their data loaders.
    device : torch.device
        Device to run computation on.

    Returns
    -------
    dict[str, dict[str, float]]
        Results mapping cycle name to dict with RMSE, MAE, Max_Error.
    """
    model.eval()
    results: dict[str, dict[str, float]] = {}

    for cycle_name, loader in test_loaders.items():
        soc_pred_all: list[np.ndarray] = []
        soc_true_all: list[np.ndarray] = []

        for x, y, _, _ in loader:
            x = x.to(device)
            soc_pred = model(x)
            soc_pred_all.append(soc_pred.cpu().numpy())
            soc_true_all.append(y.numpy())

        predicts = np.concatenate(soc_pred_all).flatten()
        true_vals = np.concatenate(soc_true_all).flatten()

        results[cycle_name] = compute_metrics(true_vals, predicts)
        logger.info(
            f"  {cycle_name:8s}  RMSE={results[cycle_name]['RMSE']:.4f}"
            f"  MAE={results[cycle_name]['MAE']:.4f}"
            f"  MaxErr={results[cycle_name]['Max_Error']:.4f}"
        )

    return results


def _save_checkpoint(
    best_state: dict,
    best_epoch: int,
    best_val_loss: float,
    is_physics: bool,
    beta: float,
    trained_models_dir: str,
    model_name: str,
) -> None:
    """Save model checkpoint.

    Parameters
    ----------
    best_state : dict
        State dict of the best model.
    best_epoch : int
        Epoch number of best model.
    best_val_loss : float
        Validation loss of best model.
    is_physics : bool
        Whether physics-informed loss was used.
    beta : float
        Physics loss weight.
    trained_models_dir : str
        Directory to save checkpoint.
    model_name : str
        Name for the saved file.
    """
    checkpoint = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "val_loss": best_val_loss,
        "is_physics": is_physics,
        "beta": beta,
        "hidden_size": config.HIDDEN_SIZE,
        "num_layers": config.NUM_LAYERS,
        "dropout": effective_dropout,
        "dropout_condition": config.DROPOUT_CONDITION,
        "seq_len": config.SEQ_LEN,
    }
    checkpoint_path = os.path.join(trained_models_dir, f"{model_name}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Best model saved to {checkpoint_path}")


def _save_training_history(history: list[TrainingMetrics], trained_models_dir: str, model_name: str) -> None:
    """Save training history to CSV file.

    Parameters
    ----------
    history : list[TrainingMetrics]
        List of training metrics per epoch.
    trained_models_dir : str
        Directory to save the CSV file.
    model_name : str
        Name for the saved file.
    """
    history_path = os.path.join(trained_models_dir, f"training_stats_{model_name}.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for metrics in history:
            writer.writerow({
                "epoch": metrics.epoch,
                "train_loss": f"{metrics.train_loss:.6f}",
                "train_ml_loss": f"{metrics.train_ml_loss:.6f}",
                "train_phys_loss": f"{metrics.train_phys_loss:.6f}",
                "val_loss": f"{metrics.val_loss:.6f}",
                "val_ml_loss": f"{metrics.val_ml_loss:.6f}",
                "val_phys_loss": f"{metrics.val_phys_loss:.6f}",
                "val_rmse": f"{metrics.val_rmse:.6f}",
                "val_mae": f"{metrics.val_mae:.6f}",
                "val_max_error": f"{metrics.val_max_error:.6f}",
                "lr": f"{metrics.lr:.1e}",
                "elapsed_s": f"{metrics.elapsed_s:.1f}",
            })
    logger.info(f"Training history saved to {history_path}")


def _log_epoch(
    epoch: int,
    max_epochs: int,
    avg_train_loss: float,
    train_ml_loss: float,
    train_phys_loss: float,
    val_loss: float,
    val_ml: float,
    val_phys: float,
    val_rmse: float,
    current_lr: float,
    elapsed: float,
    is_physics: bool,
) -> None:
    """Log epoch progress to console.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    max_epochs : int
        Total number of epochs.
    avg_train_loss : float
        Average training loss.
    train_ml_loss : float
        ML component of training loss.
    train_phys_loss : float
        Physics component of training loss.
    val_loss : float
        Validation loss.
    val_ml : float
        ML component of validation loss.
    val_phys : float
        Physics component of validation loss.
    val_rmse : float
        Validation RMSE.
    current_lr : float
        Current learning rate.
    elapsed : float
        Elapsed time in seconds.
    is_physics : bool
        Whether physics-informed loss is used.
    """
    if is_physics:
        logger.info(
            f"Epoch {epoch:3d}/{max_epochs}  "
            f"train={avg_train_loss:.6f} (ml={train_ml_loss:.6f} phys={train_phys_loss:.6f})  "
            f"val={val_loss:.6f} (ml={val_ml:.6f} phys={val_phys:.6f})  "
            f"val_RMSE={val_rmse:.4f}  "
            f"lr={current_lr:.1e}  [{elapsed:.0f}s]"
        )
    else:
        logger.info(
            f"Epoch {epoch:3d}/{max_epochs}  "
            f"train={avg_train_loss:.6f}  val={val_loss:.6f}  "
            f"val_RMSE={val_rmse:.4f}  "
            f"lr={current_lr:.1e}  [{elapsed:.0f}s]"
        )


def train_model(
    is_physics: bool,
    beta: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loaders: dict[str, DataLoader],
) -> tuple[nn.Module, dict[str, dict[str, float]]]:
    """Train a battery SOC estimation model with early stopping.

    All hyperparameters (learning rate, epochs, patience, etc.) are read
    from ``config``. Override them by modifying ``config`` before calling
    this function.

    Parameters
    ----------
    is_physics : bool
        Whether to use physics-informed loss.
    beta : float
        Weight for physics loss.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    test_loaders : dict[str, DataLoader]
        Dictionary mapping cycle name to test data loader.

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
    trained_models_dir = config.TRAINED_MODELS_DIR

    model_name = "physics_lstm_no_dropout" if (is_physics and not config.DROPOUT_CONDITION) else (
        "physics_lstm" if is_physics else "vanilla_lstm"
    )

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"Device: {device}")
    print(f"Physics: {is_physics}  Beta: {beta}  Dropout: {effective_dropout}")
    print(f"{'=' * 60}\n")

    os.makedirs(trained_models_dir, exist_ok=True)

    # Initialize model
    # When DROPOUT_CONDITION=False, dropout is completely disabled (p=0.0)
    # since the model is regularized solely by the physics loss.
    
    model = VanillaLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=effective_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    if is_physics:
        ocv_coeffs, rin_coeffs = load_physics_coefficients()
        loss_fn = PhysicsLoss(ocv_coeffs, rin_coeffs, alpha=alpha, beta=beta).to(device)
        print(f"OCV polynomial degree: {len(ocv_coeffs) - 1}")
        print(f"R_in polynomial degree: {len(rin_coeffs) - 1}")
        print(f"Alpha={alpha}, Beta={beta}\n")
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_state: dict | None = None
    history: list[TrainingMetrics] = []
    t0 = time.time()

    for epoch in range(max_epochs):
        # Train Step
        avg_train_loss, train_ml_loss, train_phys_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, is_physics, grad_clip
        )

        # Validation Step
        val_metrics = validate_with_metrics(
            model, val_loader, loss_fn, device, is_physics
        )
        val_loss = val_metrics["loss"]
        val_ml = val_metrics["ml_loss"]
        val_phys = val_metrics["phys_loss"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append(
            TrainingMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                train_ml_loss=train_ml_loss,
                train_phys_loss=train_phys_loss,
                val_loss=val_loss,
                val_ml_loss=val_ml,
                val_phys_loss=val_phys,
                val_rmse=val_metrics["rmse"],
                val_mae=val_metrics["mae"],
                val_max_error=val_metrics["max_error"],
                lr=current_lr,
                elapsed_s=elapsed,
            )
        )

        _log_epoch(
            epoch, max_epochs,
            avg_train_loss, train_ml_loss, train_phys_loss,
            val_loss, val_ml, val_phys, val_metrics["rmse"],
            current_lr, elapsed, is_physics,
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

    _save_training_history(history, trained_models_dir, model_name)

    if best_state is None:
        raise RuntimeError("Training did not produce any valid model state")

    model.load_state_dict(best_state)
    print(f"\nLoaded best model from epoch {best_epoch}")

    _save_checkpoint(
        best_state, best_epoch, best_val_loss,
        is_physics, beta, trained_models_dir, model_name,
    )

    print("\nTest Results:")
    test_results = test_model(model, test_loaders, device)

    results_path = os.path.join(trained_models_dir, f"{model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to {results_path}")

    return model, test_results