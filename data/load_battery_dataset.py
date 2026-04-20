"""Convert the Panasonic 18650PF battery dataset.

Extracts the 25degC folder from the local Mendeley dataset zip, converts all
.mat files to .csv (dropping the TimeStamp column), and stores the result in
``data/25degC_csv/``.

The raw zip file must be placed at ``data/wykht8y7tg-1.zip``.
Download it from https://data.mendeley.com/datasets/wykht8y7tg/1
"""

import os
import shutil
import argparse
import tempfile
import zipfile
import scipy.io
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "25degC_csv")
RAW_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wykht8y7tg-1.zip")


def mat_to_dataframe(mat_path: str) -> pd.DataFrame:
    """Convert a single .mat file into a pandas DataFrame."""
    mat_data = scipy.io.loadmat(mat_path)
    meas = mat_data["meas"]
    field_names = meas.dtype.names

    data = {}
    for field in field_names:
        values = meas[field][0, 0]
        if values.dtype == np.uint8:
            values = values.astype(np.float64)
        if values.dtype == object:
            values = np.array([str(v.flatten()[0]) if v.size > 0 else "" for v in values])
        else:
            values = values.flatten()
        data[field] = values

    return pd.DataFrame(data)


def convert_directory(source_dir: str, dest_dir: str) -> None:
    """Walk *source_dir*, convert every .mat to .csv and copy the rest."""
    os.makedirs(dest_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        dest_subdir = os.path.join(dest_dir, rel_path) if rel_path != "." else dest_dir
        os.makedirs(dest_subdir, exist_ok=True)

        for filename in files:
            src_file = os.path.join(root, filename)

            if filename.lower().endswith(".mat"):
                csv_filename = os.path.splitext(filename)[0] + ".csv"
                dest_file = os.path.join(dest_subdir, csv_filename)
                print(f"  Converting: {filename} -> {csv_filename}")
                try:
                    df = mat_to_dataframe(src_file)
                    if "TimeStamp" in df.columns:
                        df.drop(columns=["TimeStamp"], inplace=True)
                    df.to_csv(dest_file, index=False)
                except Exception as e:
                    print(f"  ERROR converting {filename}: {e}")
            else:
                dest_file = os.path.join(dest_subdir, filename)
                shutil.copy2(src_file, dest_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the Panasonic 18650PF 25degC dataset from a local zip."
    )
    parser.add_argument(
        "--zip", type=str, default=RAW_ZIP,
        help="Path to the local dataset zip file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.zip):
        print(f"ERROR: Zip file not found at {args.zip}")
        print("Please download the dataset from https://data.mendeley.com/datasets/wykht8y7tg/1")
        print("and place it at the path above.")
        return

    if os.path.isdir(DATA_DIR) and any(
        f.endswith(".csv") for _, _, files in os.walk(DATA_DIR) for f in files
    ):
        print(f"CSV data already exists in {DATA_DIR}. Skipping conversion.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = os.path.join(tmp, "raw")

        print(f"Extracting dataset from {args.zip} ...")
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(extract_dir)

        source_25deg = None
        for root, dirs, files in os.walk(extract_dir):
            if os.path.basename(root) == "25degC":
                source_25deg = root
                break

        if source_25deg is None:
            print("ERROR: Could not find '25degC' folder in the archive.")
            return

        print(f"Converting .mat files from {source_25deg} ...")
        convert_directory(source_25deg, DATA_DIR)

    print(f"Done. CSV files saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
