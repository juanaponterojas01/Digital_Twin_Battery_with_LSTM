"""Internal-resistance–SOC lookup table and polynomial model.

Extracts R_in from EIS experiment data by detecting the sign change of
the imaginary impedance (Zimg1) from positive to negative and taking the
corresponding real impedance (Zreal1) as the ohmic resistance. Exports a
CSV lookup table and provides the :class:`InternalResistanceModel` class
with a 3rd-degree polynomial fit.
"""

import os
import argparse
import numpy as np
import pandas as pd

import config

C_NOMINAL = config.C_NOMINAL

DEFAULT_EIS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "25degC_prepared", "EIS",
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "r_in_soc_lookup.csv"
)


def extract_rin_from_eis(eis_dir: str) -> pd.DataFrame:
    """Extract R_in and SOC from EIS experiment CSV files.

    For each EIS file, finds the first index where Zimg1 transitions from
    positive to negative and records the corresponding Zreal1 value as
    the internal resistance. SOC is computed from the accumulated
    amp-hours: ``SOC = 1 - |AhAccu| / C_n``.

    Parameters
    ----------
    eis_dir : str
        Path to the directory containing EIS CSV files.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``SOC [-]`` and ``R_in [mOhm]``.
    """
    records = []

    for filename in sorted(os.listdir(eis_dir)):
        if not filename.lower().endswith(".csv"):
            continue
        if "TS" in filename:
            continue

        filepath = os.path.join(eis_dir, filename)
        df = pd.read_csv(filepath, sep=";", skiprows=29)
        df = df[pd.to_numeric(df["AhAccu"], errors="coerce").notna()].reset_index(drop=True)

        if "Zreal1" not in df.columns or "Zimg1" not in df.columns or "AhAccu" not in df.columns:
            print(f"  Skipping {filename}: missing required columns.")
            continue

        zreal = pd.to_numeric(df["Zreal1"], errors="coerce").values
        zimg = pd.to_numeric(df["Zimg1"], errors="coerce").values
        ah_acc = pd.to_numeric(df["AhAccu"], errors="coerce").values

        sign_change_idx = None
        for i in range(1, len(zimg)):
            if zimg[i - 1] >= 0 and zimg[i] < 0:
                sign_change_idx = i
                break

        if sign_change_idx is None:
            print(f"  Skipping {filename}: no positive-to-negative sign change in Zimg1.")
            continue

        r_in = zreal[sign_change_idx]
        soc = 1.0 - abs(ah_acc[0]) / C_NOMINAL

        records.append({"SOC [-]": soc, "R_in [mOhm]": r_in})
        print(f"  {filename}: SOC={soc:.4f}, R_in={r_in:.4f} mOhm")

    return pd.DataFrame(records)


class InternalResistanceModel:
    """3rd-degree polynomial model for internal resistance vs SOC.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``SOC [-]`` and ``R_in [mOhm]`` columns.
    degree : int, optional
        Polynomial degree (default 3).
    """

    def __init__(self, df: pd.DataFrame, degree: int = 3):
        self.degree = degree
        self.soc = df["SOC [-]"].values
        self.r_in = df["R_in [mOhm]"].values
        self.coeffs = None

    def fit(self) -> np.ndarray:
        """Fit a polynomial of degree ``self.degree`` to the R_in data.

        Returns
        -------
        np.ndarray
            Polynomial coefficients (highest power first).
        """
        self.coeffs = np.polyfit(self.soc, self.r_in, self.degree)
        return self.coeffs

    def estimate(self, soc: float) -> float:
        """Estimate R_in for a given SOC value.

        Parameters
        ----------
        soc : float
            State of charge (0–1).

        Returns
        -------
        float
            Estimated internal resistance in mOhm.
        """
        if self.coeffs is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return float(np.polyval(self.coeffs, soc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract R_in vs SOC from EIS data and create a lookup table."
    )
    parser.add_argument(
        "eis_dir", type=str, nargs="?", default=DEFAULT_EIS_DIR,
        help="Path to the EIS directory containing CSV files.",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output CSV path for the lookup table.",
    )
    args = parser.parse_args()

    eis_dir = os.path.abspath(args.eis_dir)
    if not os.path.isdir(eis_dir):
        print(f"Error: directory '{eis_dir}' does not exist.")
        return

    print(f"Extracting R_in from EIS data in: {eis_dir}")
    df = extract_rin_from_eis(eis_dir)

    if df.empty:
        print("No data extracted. Exiting.")
        return

    df = df.sort_values("SOC [-]").reset_index(drop=True)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nLookup table saved to: {output_path}")
    print(df.to_string(index=False))

    model = InternalResistanceModel(df)
    coeffs = model.fit()
    print(f"\nPolynomial coefficients (degree {model.degree}): {coeffs}")

    print("\nSample estimations:")
    for soc_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"  SOC={soc_val:.1f} -> R_in={model.estimate(soc_val):.4f} mOhm")


if __name__ == "__main__":
    main()
