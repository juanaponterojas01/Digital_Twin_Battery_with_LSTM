"""Filter, normalize, and compute SOC for the battery CSV dataset.

Reads raw CSV files from ``data/25degC_csv/`` (skipping the EIS folder),
applies a 2nd-order Butterworth low-pass filter and Z-normalization to
Current, Voltage, and Battery_Temp_degC, estimates SOC via Coulomb
counting, and writes the prepared data to ``data/25degC_prepared/``.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

C_NOMINAL = config.C_NOMINAL
ETA = 0.98
CUTOFF_HZ = 1.0
FILTER_ORDER = 2

SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "25degC_csv")
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "25degC_prepared")


def butterworth_lowpass_filter(data: np.ndarray, cutoff_hz: float,
                               fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Apply a Butterworth low-pass filter to *data*."""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1.0:
        return data.copy()
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, data)


def z_normalize(data: np.ndarray) -> np.ndarray:
    """Z-normalize a 1-D array using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()


def compute_soc(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Estimate SOC via Coulomb counting.

    Uses dSOC/dt = eta * I(t) / C_n with C_n = 2.9 Ah and eta = 0.98.
    Discharge starts at SOC = 1.0, charge starts at SOC = 0.0.
    """
    dt = np.zeros(len(time_s))
    dt[1:] = np.diff(time_s)

    delta_soc = ETA * current_a * dt / (C_NOMINAL * 3600.0)
    soc = np.cumsum(delta_soc)

    first_nonzero_idx = 0
    for i in range(len(current_a)):
        if abs(current_a[i]) > 1e-6:
            first_nonzero_idx = i
            break

    if current_a[first_nonzero_idx] < 0:
        soc = 1.0 + soc
    else:
        soc = 0.0 + soc

    return soc


def process_file(df: pd.DataFrame, cutoff_hz: float = CUTOFF_HZ) -> pd.DataFrame:
    """Filter, normalize, and compute SOC for a single DataFrame."""
    time_s = df["Time"].values.astype(np.float64)

    dt = np.diff(time_s)
    valid_dt = dt[dt > 0]
    fs = 1.0 / np.median(valid_dt) if len(valid_dt) > 0 else 1.0

    current = df["Current"].values.astype(np.float64)
    voltage = df["Voltage"].values.astype(np.float64)
    temp = df["Battery_Temp_degC"].values.astype(np.float64)
    chamber_temp = df["Chamber_Temp_degC"].values.astype(np.float64)

    if fs > 2 * cutoff_hz:
        current_filt = butterworth_lowpass_filter(current, cutoff_hz, fs)
        voltage_filt = butterworth_lowpass_filter(voltage, cutoff_hz, fs)
        temp_filt = butterworth_lowpass_filter(temp, cutoff_hz, fs)
        chamber_temp_filt = butterworth_lowpass_filter(chamber_temp, cutoff_hz, fs)
    else:
        current_filt = current.copy()
        voltage_filt = voltage.copy()
        temp_filt = temp.copy()
        chamber_temp_filt = chamber_temp.copy()

    current_norm = z_normalize(current_filt)
    voltage_norm = z_normalize(voltage_filt)
    temp_norm = z_normalize(temp_filt)

    soc = compute_soc(time_s, current_filt)

    return pd.DataFrame({
        "Time": time_s,
        "Current_filt": current_filt,
        "Voltage_filt": voltage_filt,
        "Temp_degC_filt": temp_filt,
        "Chamber_Temp_degC_filt": chamber_temp_filt,
        "Current_norm": current_norm,
        "Voltage_norm": voltage_norm,
        "Temp_degC_norm": temp_norm,
        "SOC": soc,
    })


def process_directory(source_dir: str, dest_dir: str,
                      cutoff_hz: float = CUTOFF_HZ) -> None:
    """Process all CSV files in *source_dir* and write to *dest_dir*.

    EIS CSV files are copied as-is (they have a different format and are
    not filtered/normalized). All other CSV files are processed with the
    Butterworth filter, Z-normalization, and SOC estimation.
    """
    import shutil

    os.makedirs(dest_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        dest_subdir = os.path.join(dest_dir, rel_path) if rel_path != "." else dest_dir
        os.makedirs(dest_subdir, exist_ok=True)

        is_eis = "EIS" in root

        for filename in files:
            if not filename.lower().endswith(".csv"):
                continue

            src_path = os.path.join(root, filename)
            dest_path = os.path.join(dest_subdir, filename)

            if is_eis:
                shutil.copy2(src_path, dest_path)
                continue

            print(f"  Processing: {filename}")
            try:
                df = pd.read_csv(src_path)
                required = {"Time", "Current", "Voltage", "Battery_Temp_degC", "Chamber_Temp_degC"}
                if not required.issubset(df.columns):
                    missing = required - set(df.columns)
                    print(f"    Skipping (missing columns: {missing}).")
                    continue
                result = process_file(df, cutoff_hz)
                result.to_csv(dest_path, index=False)
            except Exception as e:
                print(f"    ERROR processing {filename}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter, normalize, and compute SOC for battery CSV data."
    )
    parser.add_argument(
        "--source", type=str, default=SOURCE_DIR,
        help="Source directory with raw CSV files.",
    )
    parser.add_argument(
        "--dest", type=str, default=DEST_DIR,
        help="Destination directory for prepared CSV files.",
    )
    parser.add_argument(
        "--cutoff", type=float, default=CUTOFF_HZ,
        help=f"Butterworth cutoff frequency in Hz (default: {CUTOFF_HZ}).",
    )
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    dest = os.path.abspath(args.dest)

    if not os.path.isdir(source):
        print(f"Error: source directory '{source}' does not exist.")
        return

    print(f"Source: {source}")
    print(f"Destination: {dest}")
    print(f"Cutoff frequency: {args.cutoff} Hz")
    process_directory(source, dest, args.cutoff)
    print("Done.")


if __name__ == "__main__":
    main()
