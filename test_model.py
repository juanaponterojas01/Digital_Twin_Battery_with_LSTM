"""
Utility functions to load trained models, make predictions,
evaluate metrics, and estimate Monte Carlo uncertainty.
"""

import os
import torch
from torch import nn
from typing import Dict
import numpy as np
import config
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import VanillaLSTM


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


def load_model(model_name: str, device: torch.device) -> nn.Module:
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
        input_size=checkpoint.get("hidden_size", config.INPUT_SIZE)
        if "input_size" not in checkpoint
        else checkpoint.get("input_size", config.INPUT_SIZE),
        hidden_size=checkpoint.get("hidden_size", config.HIDDEN_SIZE),
        num_layers=checkpoint.get("num_layers", config.NUM_LAYERS),
        dropout=checkpoint.get("dropout", config.DROPOUT),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_cycle(model, dataset, device) -> np.ndarray:
    """Run inference on every window in a dataset and concatenate predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    dataset : BatteryWindowDataset
        Dataset to iterate over.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Flattened array of SOC predictions.
    """
    all_pred = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, y, v, i = dataset[idx]
            x = x.unsqueeze(0).to(device)
            pred = model(x).squeeze(0).squeeze(-1).cpu().numpy()
            all_pred.append(pred)
    return np.concatenate(all_pred).flatten()


def predict_cycle_with_uncertainty(
    model, dataset, device, n_iterations: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo Dropout inference on a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataset : BatteryWindowDataset
        Dataset to iterate over.
    device : torch.device
        Device to run inference on.
    n_iterations : int
        Number of stochastic forward passes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mean_pred, std_pred) — SOC mean and uncertainty per timestep.
    """
    all_means = []
    all_stds = []

    for idx in range(len(dataset)):
        x, y, v, i = dataset[idx]
        x = x.unsqueeze(0).to(device)
        mean_pred, std_pred = model.predict_mc_uncertainty(x, n_iterations=n_iterations)
        all_means.append(mean_pred)
        all_stds.append(std_pred)

    mean_pred = np.concatenate(all_means).flatten()
    std_pred = np.concatenate(all_stds).flatten()
    return mean_pred, std_pred


@torch.no_grad()
def evaluate_cycle_metrics(model, test_loaders, device) -> Dict:
    """Run inference on all test cycles and compute per-cycle metrics.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    test_loaders : dict[str, DataLoader]
        Mapping of cycle name to its DataLoader.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict mapping cycle name to RMSE, MAE, and Max_Error.
    """
    results = {}
    for cycle_name, loader in test_loaders.items():
        preds = []
        true_vals = []
        for x, y, v, i in loader:
            x = x.to(device)
            soc_pred = model(x).cpu().numpy()
            preds.append(soc_pred.flatten())
            true_vals.append(y.numpy().flatten())

        preds = np.concatenate(preds)
        true_vals = np.concatenate(true_vals)
        metrics = compute_metrics(true_vals, preds)
        results[cycle_name] = {
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Max_Error": metrics["Max_Error"],
        }
    return results