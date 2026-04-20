"""
End-to-end evaluation script for the battery SOC estimation project.

Evaluates the vanilla LSTM and physics-informed LSTM (beta=0.1) across
all test cycles, and generates:
  - SOC prediction plots with Monte Carlo uncertainty bands
  - Training curves overlay
  - Comparison table with per-cycle RMSE, MAE, and Max Error (saved as CSV/JSON)

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
from dataset import BatteryWindowDataset, create_dataloaders
from test_model import (
    compute_metrics,
    load_model,
    predict_cycle,
    predict_cycle_with_uncertainty,
    evaluate_cycle_metrics,
)

MODEL_NAMES = ["vanilla_lstm", "physics_lstm_beta_0.1"]
MODEL_LABELS = {"vanilla_lstm": "Vanilla LSTM", "physics_lstm_beta_0.1": "Physics LSTM (β=0.1)"}
MODEL_COLORS = {"vanilla_lstm": "tab:blue", "physics_lstm_beta_0.1": "tab:orange"}

MAX_PLOT_POINTS = 3000
RESULTS_DIR = os.path.join(config.PROJECT_DIR, "results")
TRAINED_DIR = config.TRAINED_MODELS_DIR


def _load_cycle_dataset(cycle_name: str):
    """Load a drive cycle CSV and return its BatteryWindowDataset."""
    filename = config.DRIVE_CYCLES[cycle_name]
    csv_path = os.path.join(config.DATA_DIR, filename)
    df = pd.read_csv(csv_path)
    return BatteryWindowDataset(df, config.SEQ_LEN, config.TEST_STRIDE), df


def plot_soc_comparison(cycle_name, df, time_min, true_soc,
                        vanilla_pred, vanilla_std,
                        physics_pred, physics_std,
                        save_path):
    """Plot SOC predictions with 95% confidence bands for both models."""
    stride = max(1, len(time_min) // MAX_PLOT_POINTS)
    t = time_min[::stride]
    y_true = true_soc[::stride]
    y_van = vanilla_pred[::stride]
    s_van = vanilla_std[::stride]
    y_phy = physics_pred[::stride]
    s_phy = physics_std[::stride]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(t, y_van - 2 * s_van, y_van + 2 * s_van,
                    color=MODEL_COLORS["vanilla_lstm"], alpha=0.2,
                    label="Vanilla LSTM 95% CI")
    ax.plot(t, y_van, color=MODEL_COLORS["vanilla_lstm"], linewidth=1.2,
            label="Vanilla LSTM")

    ax.fill_between(t, y_phy - 2 * s_phy, y_phy + 2 * s_phy,
                    color=MODEL_COLORS["physics_lstm_beta_0.1"], alpha=0.2,
                    label="Physics LSTM 95% CI")
    ax.plot(t, y_phy, color=MODEL_COLORS["physics_lstm_beta_0.1"], linewidth=1.2,
            label="Physics LSTM (β=0.1)")

    ax.plot(t, y_true, "k--", linewidth=1.0, label="True SOC", alpha=0.7)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SOC")
    ax.set_title(f"SOC Estimation — {cycle_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


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


def plot_val_rmse_comparison(save_path):
    """Overlay validation RMSE over epochs for both models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, label, color in [
        ("vanilla_lstm", "Vanilla LSTM", "tab:blue"),
        ("physics_lstm_beta_0.1", "Physics LSTM (β=0.1)", "tab:orange"),
    ]:
        csv_path = os.path.join(TRAINED_DIR, f"training_stats_{name}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        epochs = df["epoch"].astype(int)
        val_rmse = df["val_rmse"].astype(float)
        ax.plot(epochs, val_rmse, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

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
            "RMSE Δ%": f"{rmse_delta:+.2f}",
            "Vanilla MAE": f"{v['MAE']:.4f}",
            "Physics MAE": f"{p['MAE']:.4f}",
            "MAE Δ%": f"{mae_delta:+.2f}",
            "Vanilla MaxErr": f"{v['Max_Error']:.4f}",
            "Physics MaxErr": f"{p['Max_Error']:.4f}",
            "MaxErr Δ%": f"{maxerr_delta:+.2f}",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate results and plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--device", default=None, help="Force device (cpu/cuda).")
    parser.add_argument("--mc-iterations", type=int, default=50,
                        help="Number of MC dropout passes (default: 50).")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else config.DEVICE)
    print(f"Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, _, test_loaders = create_dataloaders()

    all_results = {}
    for name in MODEL_NAMES:
        model = load_model(name, device)
        if model is None:
            print(f"WARNING: {name}.pt not found. Train it first.")
            continue
        print(f"\nEvaluating {MODEL_LABELS[name]}...")
        all_results[name] = evaluate_cycle_metrics(model, test_loaders, device)
        for c in config.TEST_CYCLES:
            m = all_results[name][c]
            print(f"  {c}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  MaxErr={m['Max_Error']:.4f}")

    if len(all_results) < 2:
        print("\nNot all models found. Exiting.")
        return

    comparison = build_comparison_table(all_results)
    print("\n" + "=" * 80)
    print(comparison.to_string(index=False))
    print("=" * 80)

    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to {csv_path}")

    json_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results JSON saved to {json_path}")

    if args.no_plots:
        print("\nSkipping plots (--no-plots).")
        return

    print("\nGenerating plots...")

    plot_training_curves_overlay(os.path.join(RESULTS_DIR, "training_curves_overlay.png"))
    plot_val_rmse_comparison(os.path.join(RESULTS_DIR, "val_rmse_comparison.png"))

    for cycle_name in config.TEST_CYCLES:
        dataset, df = _load_cycle_dataset(cycle_name)
        time_min = (df["Time"].values / 60.0).astype(np.float32)
        true_soc = df["SOC"].values.astype(np.float32)[:len(dataset) * config.SEQ_LEN]

        print(f"  MC predictions for {cycle_name}...")
        v_mean, v_std = predict_cycle_with_uncertainty(
            load_model("vanilla_lstm", device), dataset, device,
            n_iterations=args.mc_iterations,
        )
        p_mean, p_std = predict_cycle_with_uncertainty(
            load_model("physics_lstm_beta_0.1", device), dataset, device,
            n_iterations=args.mc_iterations,
        )

        plot_soc_comparison(
            cycle_name, df, time_min, true_soc,
            v_mean, v_std, p_mean, p_std,
            save_path=os.path.join(RESULTS_DIR, f"soc_{cycle_name.lower()}.png"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
