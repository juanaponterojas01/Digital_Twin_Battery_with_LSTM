"""Utility functions for model inference, predictions, and metrics.

This module provides functions to:
- Load trained models from checkpoints
- Run inference on datasets (with/without MC dropout for uncertainty)
- Compute evaluation metrics (RMSE, MAE, Max Error)
"""

import os


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
from model import VanillaLSTM
from torch.utils.data import Dataset


def compute_metrics(soc_true: np.ndarray, soc_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and Max Error for SOC predictions.

    Parameters
    ----------
    soc_true : np.ndarray
        Ground truth SOC values, flattened.
    soc_pred : np.ndarray
        Predicted SOC values, flattened.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'RMSE', 'MAE', 'Max_Error'.
    """
    rmse = float(np.sqrt(mean_squared_error(soc_true, soc_pred)))
    mae = float(mean_absolute_error(soc_true, soc_pred))
    max_err = float(np.max(np.abs(soc_true - soc_pred)))
    return {"RMSE": rmse, "MAE": mae, "Max_Error": max_err}


def load_model(model_name: str, device: torch.device) -> nn.Module | None:
    """Load a trained model checkpoint from trained_models/.

    Parameters
    ----------
    model_name : str
        Base name of the checkpoint (without .pt extension).
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    nn.Module or None
        Loaded model in eval mode, or None if checkpoint not found.
    """
    path = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}.pt")
    if not os.path.isfile(path):
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VanillaLSTM(
        input_size=checkpoint.get("input_size", config.INPUT_SIZE),
        hidden_size=checkpoint.get("hidden_size", config.HIDDEN_SIZE),
        num_layers=checkpoint.get("num_layers", config.NUM_LAYERS),
        dropout=checkpoint.get("dropout", config.DROPOUT),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_cycle(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference on every window in a dataset.

    Stacks all input windows into a single batch tensor and performs one
    forward pass instead of looping window-by-window.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    dataset : Dataset
        Dataset to iterate over (must return x, y, v, i via __getitem__).
    device : torch.device
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Flattened array of SOC predictions.
    """
    all_x = [dataset[idx][0] for idx in range(len(dataset))]
    batch_x = torch.stack(all_x).to(device)
    with torch.no_grad():
        preds = model(batch_x).cpu().numpy()
    return preds.flatten()


def predict_cycle_with_uncertainty(
    model: VanillaLSTM,
    dataset: Dataset,
    device: torch.device,
    n_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo Dropout inference on a dataset.

    Not available for models trained with --no-dropout (DROPOUT_CONDITION=False),
    since dropout was never active during training and MC uncertainty would be
    meaningless.

    Parameters
    ----------
    model : VanillaLSTM
        Trained model (should be in eval mode).
    dataset : Dataset
        Dataset to iterate over (must return x, y, v, i via __getitem__).
    device : torch.device
        Device to run inference on.
    n_iterations : int
        Number of stochastic forward passes (default: 50).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mean_soc, std_soc) where:
        - mean_soc: (N,) SOC mean prediction for each timestep
        - std_soc: (N,) SOC uncertainty (standard deviation)

    Raises
    ------
    ValueError
        If the model was trained with --no-dropout (MC uncertainty is not available).
    """
    if model.dropout.p == 0.0:
        raise ValueError(
            "MC uncertainty estimation is not available for models trained with "
            "--no-dropout. This model was regularized solely by the physics loss "
            "and its dropout layer has probability 0."
        )

    all_x = [dataset[idx][0] for idx in range(len(dataset))]
    batch_x = torch.stack(all_x).to(device)
    mean_soc, std_soc = model.predict_mc_uncertainty(
        batch_x, n_iterations=n_iterations
    )
    return mean_soc.flatten(), std_soc.flatten()