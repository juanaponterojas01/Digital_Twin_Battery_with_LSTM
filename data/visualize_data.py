"""Visualization utilities for the battery dataset.

Provides functions to plot:
- OCV(SOC) curve from the lookup table,
- Current, Voltage, and SOC (filtered and normalized) for a drive-cycle file,
- Nyquist impedance plot across all SOC values from the EIS experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This package must be run from the project root so that curve_fitting is importable.
from curve_fitting.ocv_lookup import OCVModel

import config

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "25degC_prepared")
PLOTS_DIR = os.path.join(PROJECT_DIR, "data", "data_visualization")


def plot_ocv_soc(ocv_csv: str = None, save: bool = True) -> None:
    """Plot the OCV–SOC curve from the lookup table.

    Parameters
    ----------
    ocv_csv : str, optional
        Path to the OCV lookup CSV. Defaults to
        ``curve_fitting/ocv_soc_lookup.csv``.
    save : bool
        If True, save the figure to ``data/data_visualization/``.
    """
    if ocv_csv is None:
        ocv_csv = os.path.join(PROJECT_DIR, "curve_fitting", "ocv_soc_lookup.csv")

    df = pd.read_csv(ocv_csv)
    soc = df["SOC [-]"].values
    ocv = df["OCV [V]"].values

    model = OCVModel(df)
    model.fit()

    soc_fine = np.linspace(0, 1, 500)
    ocv_fine = np.array([model.estimate(s) for s in soc_fine])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(soc, ocv, s=5, alpha=0.3, label="C/20 data")
    ax.plot(soc_fine, ocv_fine, "r-", label=f"Poly-{model.degree} fit", linewidth=1.5)
    ax.set_xlabel("SOC [-]")
    ax.set_ylabel("OCV [V]")
    ax.set_title("Open-Circuit Voltage vs SOC (25 °C)")
    ax.legend()
    ax.grid(True)

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, "ocv_soc_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_drive_cycle(csv_path: str, save: bool = True) -> None:
    """Plot filtered and normalized Current, Voltage, and SOC for a drive-cycle file.

    Parameters
    ----------
    csv_path : str
        Path to a prepared drive-cycle CSV file.
    save : bool
        If True, save the figure to ``data/data_visualization/``.
    """
    df = pd.read_csv(csv_path)
    time_min = df["Time"].values / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(time_min, df["Current_filt"], color="tab:blue")
    axes[0].set_ylabel("Current_filt [A]")
    axes[0].grid(True)

    axes[1].plot(time_min, df["Voltage_filt"], color="tab:orange")
    axes[1].set_ylabel("Voltage_filt [V]")
    axes[1].grid(True)

    axes[2].plot(time_min, df["SOC"], color="tab:green")
    axes[2].set_ylabel("SOC [-]")
    axes[2].set_xlabel("Time [min]")
    axes[2].grid(True)

    title = os.path.basename(csv_path).replace(".csv", "")
    fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, f"drive_cycle_{title[:20]}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_nyquist(eis_csv_dir: str = None, save: bool = True) -> None:
    """Plot Nyquist impedance diagrams at all SOC values from the EIS experiments.

    Reads the 14 EIS CSV files (skipping the TS summary), estimates SOC
    from ``SOC = 1 - |AhAccu|/C_n``, and plots Re(Z) vs -Im(Z).

    Parameters
    ----------
    eis_csv_dir : str, optional
        Path to the directory with EIS CSV files. Defaults to
        ``data/25degC_prepared/EIS``.
    save : bool
        If True, save the figure to ``data/data_visualization/``.
    """
    if eis_csv_dir is None:
        eis_csv_dir = os.path.join(PROJECT_DIR, "data", "25degC_prepared", "EIS")

    fig, ax = plt.subplots(figsize=(10, 7))

    line_styles = [".", "s", "d", "X", "v", "+", "*"]

    for i, filename in enumerate(sorted(os.listdir(eis_csv_dir))):
        if not filename.lower().endswith(".csv") or "TS" in filename:
            continue

        filepath = os.path.join(eis_csv_dir, filename)
        df = pd.read_csv(filepath, sep=";", skiprows=29)
        df = df[pd.to_numeric(df["AhAccu"], errors="coerce").notna()].reset_index(
            drop=True
        )

        if "Zreal1" not in df.columns or "Zimg1" not in df.columns:
            continue

        zreal = pd.to_numeric(df["Zreal1"], errors="coerce").values
        zimg = pd.to_numeric(df["Zimg1"], errors="coerce").values
        ah_acc = pd.to_numeric(df["AhAccu"], errors="coerce").values

        valid = np.isfinite(zreal) & np.isfinite(zimg)
        zreal = zreal[valid]
        zimg = zimg[valid]

        soc = 1.0 - abs(ah_acc[0]) / config.C_NOMINAL

        marker = line_styles[i % len(line_styles)]

        ax.scatter(
            zreal,
            -1 * zimg,
            s=20,
            label=f"SOC = {soc:.2f}",
            linewidth=0.8,
            marker=marker,
        )
        ax.plot(
            zreal,
            -1 * zimg,
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_xlim(18, 35)
    ax.set_ylim(-9, 10)
    ax.set_xlabel("Re(Z) [mOhm]")
    ax.set_ylabel("-Im(Z) [mOhm]")
    ax.set_title("Nyquist Plot — EIS at 25 °C")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, "nyquist_eis_25degC.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)