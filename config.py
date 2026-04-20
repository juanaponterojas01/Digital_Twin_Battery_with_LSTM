"""Centralized configuration for the battery SOC estimation project."""

import os
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "25degC_prepared", "Drive cycles")
TRAINED_MODELS_DIR = os.path.join(PROJECT_DIR, "trained_models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


DRIVE_CYCLES = {
    "Cycle1": "03-18-17_02.17 25degC_Cycle_1_Pan18650PF.csv",
    "Cycle2": "03-19-17_03.25 25degC_Cycle_2_Pan18650PF.csv",
    "Cycle3": "03-19-17_09.07 25degC_Cycle_3_Pan18650PF.csv",
    "Cycle4": "03-19-17_14.31 25degC_Cycle_4_Pan18650PF.csv",
    "US06": "03-20-17_01.43 25degC_US06_Pan18650PF.csv",
    "HWFTa": "03-20-17_05.56 25degC_HWFTa_Pan18650PF.csv",
    "HWFTb": "03-20-17_19.27 25degC_HWFTb_Pan18650PF.csv",
    "UDDS": "03-21-17_00.29 25degC_UDDS_Pan18650PF.csv",
    "LA92": "03-21-17_09.38 25degC_LA92_Pan18650PF.csv",
    "NN": "03-21-17_16.27 25degC_NN_Pan18650PF.csv",
}

TRAIN_CYCLE = "LA92"  # Drive cycle used for training (key in DRIVE_CYCLES)
VAL_CYCLE = "US06"  # Drive cycle used for validation (key in DRIVE_CYCLES)
TEST_CYCLES = ["Cycle1", "Cycle2", "Cycle3", "Cycle4", "HWFTa", "HWFTb", "UDDS", "NN"]  # Drive cycles used for testing

SEQ_LEN = 200  # 20 s at 10 Hz
TRAIN_STRIDE = 1  # Overlapping for augmentation
VAL_STRIDE = 200  # Non-overlapping
TEST_STRIDE = 200  # Non-overlapping
BATCH_SIZE = 64

INPUT_SIZE = 2  # [Current_norm, Voltage_norm]
HIDDEN_SIZE = 64
NUM_LAYERS = 1  # Vanilla LSTM
DROPOUT = 0.2  # MC Dropout rate for uncertainty estimation

LEARNING_RATE = 1e-3  # Adam
MAX_EPOCHS = 100  # With early stopping
PATIENCE = 15  # Epochs without improvement
GRAD_CLIP = 1.0  # Max norm

ALPHA = 1.0  # Data loss weight
BETA = 0.1  # Physics loss weight (tunable via --beta)

LR_FACTOR = 0.5  # ReduceLROnPlateau factor
LR_PATIENCE = 5  # ReduceLROnPlateau patience

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device ("cuda" or "cpu")

C_NOMINAL = 2.9  # Nominal battery capacity in Ah (Panasonic 18650PF)
ETA_CHARGE = 0.98  # Coulombic efficiency during charging
ETA_DISCHARGE = 1.0  # Coulombic efficiency during discharging
R_IN_OCV_CORRECTION = 0.022  # Ohmic resistance correction for C/20 OCV extraction (Ohm)
MILLIOHM_TO_OHM = 1e-3  # Conversion factor from mOhm to Ohm
