"""Dataset and DataLoader utilities for battery SOC estimation.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class BatteryWindowDataset(Dataset):
    """Sliding-window dataset for battery SOC estimation.

    Each sample contains:
        x       : (seq_len, 2) — [Current_norm, Voltage_norm]
        y       : (seq_len,)   — SOC ground truth
        voltage : (seq_len,)   — Voltage_filt (for physics loss)
        current : (seq_len,)   — Current_filt (for physics loss)

    Parameters
    ----------
    df : pd.DataFrame
        Prepared battery data with columns: Current_norm, Voltage_norm, SOC,
        Voltage_filt, Current_filt.
    seq_len : int
        Window length in timesteps.
    stride : int
        Step size between consecutive windows.
    """

    _REQUIRED_COLUMNS = ["Current_norm", "Voltage_norm", "SOC", "Voltage_filt", "Current_filt"]

    def __init__(self, df: pd.DataFrame, seq_len: int, stride: int):
        missing = [c for c in self._REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"BatteryWindowDataset: missing columns {missing}. "
                f"Expected: {self._REQUIRED_COLUMNS}"
            )
        self.seq_len = seq_len
        self.stride = stride

        self.current_norm = df["Current_norm"].values.astype(np.float32)
        self.voltage_norm = df["Voltage_norm"].values.astype(np.float32)
        self.soc = df["SOC"].values.astype(np.float32)
        self.voltage_filt = df["Voltage_filt"].values.astype(np.float32)
        self.current_filt = df["Current_filt"].values.astype(np.float32)

        self.indices = list(range(0, len(self.soc) - seq_len + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end = start + self.seq_len

        x = np.stack([self.current_norm[start:end],
                       self.voltage_norm[start:end]], axis=1)
        y = self.soc[start:end]
        v = self.voltage_filt[start:end]
        i = self.current_filt[start:end]

        return (torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(v),
                torch.from_numpy(i))


def _load_cycle(cycle_name: str) -> pd.DataFrame:
    """Load a prepared drive cycle CSV by name."""
    filename = config.DRIVE_CYCLES[cycle_name]
    path = os.path.join(config.DATA_DIR, filename)
    return pd.read_csv(path)


def create_dataloaders():
    """Create train, validation, and test DataLoaders.

    Parameters
    ----------
    config module

    Returns
    -------
    tuple[DataLoader, DataLoader, dict[str, DataLoader]]
        (train_loader, val_loader, test_loaders_dict)

        test_loaders_dict maps cycle name to its DataLoader.
    """
    batch_size = config.BATCH_SIZE
    seq_len =  config.SEQ_LEN
    train_stride = config.TRAIN_STRIDE
    val_stride = config.VAL_STRIDE
    test_stride = config.TEST_STRIDE

    train_df = _load_cycle(config.TRAIN_CYCLE)
    train_dataset = BatteryWindowDataset(train_df, seq_len, train_stride)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)

    val_df = _load_cycle(config.VAL_CYCLE)
    val_dataset = BatteryWindowDataset(val_df, seq_len, val_stride)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    test_loaders = {}
    for cycle_name in config.TEST_CYCLES:
        df = _load_cycle(cycle_name)
        dataset = BatteryWindowDataset(df, seq_len, test_stride)
        test_loaders[cycle_name] = DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0,
                                              pin_memory=True)

    print(f"Train: {config.TRAIN_CYCLE} — {len(train_dataset):,} windows")
    print(f"Val:   {config.VAL_CYCLE} — {len(val_dataset):,} windows")
    for name, loader in test_loaders.items():
        print(f"Test:  {name} — {len(loader.dataset):,} windows")

    return train_loader, val_loader, test_loaders
