"""OCV–SOC lookup table and polynomial model.

Extracts OCV(SOC) from the C/20 discharge test, exports a CSV lookup
table, and provides the :class:`OCVModel` class with a 5th-degree
polynomial fit.
"""

import os
import argparse
import numpy as np
import pandas as pd

import config

C_NOMINAL = config.C_NOMINAL
R_IN_OHM = config.R_IN_OCV_CORRECTION

DEFAULT_C20_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "25degC_prepared",
    "C20 OCV and 1C discharge tests_start_of_tests",
    "05-08-17_13.26 C20 OCV Test_C20_25dC.csv",
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ocv_soc_lookup.csv"
)


def extract_ocv_soc(c20_csv_path: str) -> pd.DataFrame:
    """Extract OCV vs SOC from a C/20 discharge CSV.

    Uses only the discharge phase (I < 0) and applies a small IR
    correction using the nominal internal resistance to approximate the
    true open-circuit voltage.

    Parameters
    ----------
    c20_csv_path : str
        Path to the C/20 test CSV file (prepared data with
        ``Voltage_filt`` and ``SOC`` columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``SOC [-]`` and ``OCV [V]``.
    """
    df = pd.read_csv(c20_csv_path)
    df = df.dropna(subset=["Voltage_filt", "SOC"]).reset_index(drop=True)

    voltage = df["Voltage_filt"].values.astype(np.float64)
    soc = df["SOC"].values.astype(np.float64)
    current = df["Current_filt"].values.astype(np.float64)

    discharge_mask = current < -0.001
    soc_d = soc[discharge_mask]
    v_d = voltage[discharge_mask]

    valid = (soc_d >= 0.0) & (soc_d <= 1.0)
    soc_d = soc_d[valid]
    v_d = v_d[valid]

    current_discharge = np.mean(current[discharge_mask][valid])
    ocv_d = v_d - current_discharge * R_IN_OHM

    unique_soc = np.unique(np.round(soc_d, 6))
    ocv_values = []
    soc_values = []

    for s in unique_soc:
        mask = np.isclose(soc_d, s, atol=1e-5)
        ocv_values.append(np.mean(ocv_d[mask]))
        soc_values.append(s)

    return pd.DataFrame({"SOC [-]": soc_values, "OCV [V]": ocv_values})


class OCVModel:
    """5th-degree polynomial model for OCV as a function of SOC.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``SOC [-]`` and ``OCV [V]`` columns.
    degree : int, optional
        Polynomial degree (default 5).
    """

    def __init__(self, df: pd.DataFrame, degree: int = 7):
        self.degree = degree
        self.soc = df["SOC [-]"].values
        self.ocv = df["OCV [V]"].values
        self.coeffs = None

    def fit(self) -> np.ndarray:
        """Fit a polynomial of degree ``self.degree`` to the OCV data.

        Returns
        -------
        np.ndarray
            Polynomial coefficients (highest power first).
        """
        self.coeffs = np.polyfit(self.soc, self.ocv, self.degree)
        return self.coeffs

    def estimate(self, soc: float) -> float:
        """Estimate OCV for a given SOC value.

        Parameters
        ----------
        soc : float
            State of charge (0–1).

        Returns
        -------
        float
            Estimated open-circuit voltage in volts.
        """
        if self.coeffs is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return float(np.polyval(self.coeffs, soc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OCV(SOC) from C20 data and create a lookup table."
    )
    parser.add_argument(
        "c20_csv", type=str, nargs="?", default=DEFAULT_C20_PATH,
        help="Path to the C20 OCV test CSV file.",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output CSV path for the lookup table.",
    )
    args = parser.parse_args()

    c20_path = os.path.abspath(args.c20_csv)
    if not os.path.isfile(c20_path):
        print(f"Error: file '{c20_path}' does not exist.")
        return

    print(f"Extracting OCV(SOC) from: {c20_path}")
    df = extract_ocv_soc(c20_path)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Lookup table saved to: {output_path}")
    print(f"  {len(df)} unique SOC points, "
          f"OCV range: [{df['OCV [V]'].min():.4f}, {df['OCV [V]'].max():.4f}] V\n")

    model = OCVModel(df)
    coeffs = model.fit()
    print(f"Polynomial coefficients (degree {model.degree}): {coeffs}")

    print("\nSample estimations:")
    for soc_val in np.arange(0, 1.1, 0.1):
        print(f"  SOC={soc_val:.1f} -> OCV={model.estimate(soc_val):.4f} V")

    ocv_pred = np.array([model.estimate(s) for s in df["SOC [-]"].values])
    rmse = np.sqrt(np.mean((ocv_pred - df["OCV [V]"].values) ** 2))
    max_err = np.max(np.abs(ocv_pred - df["OCV [V]"].values))
    print(f"\nFitting quality: RMSE = {rmse:.6f} V, Max |Error| = {max_err:.6f} V")


if __name__ == "__main__":
    main()
