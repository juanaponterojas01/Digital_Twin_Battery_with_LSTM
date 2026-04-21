"""Training script for battery SOC estimation models.

Usage:
    # Train vanilla LSTM (with dropout)
    python train.py

    # Train physics-informed LSTM (with dropout)
    python train.py --physics

    # Train physics-informed LSTM (without dropout — physics-only regularization)
    python train.py --physics --no-dropout
"""

import argparse

import torch

import config
from dataset import create_dataloaders
from engine import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate battery SOC estimation models."
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Use physics-informed loss (MSE + physics consistency).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=config.BETA,
        help=f"Weight for physics loss (default: {config.BETA}).",
    )
    parser.add_argument(
        "--no-dropout",
        action="store_true",
        help="Disable dropout entirely (physics-only regularization). "
             "MC uncertainty estimation will not be available.",
    )
    args = parser.parse_args()

    if args.no_dropout:
        config.DROPOUT_CONDITION = False

    train_loader, val_loader, test_loaders = create_dataloaders()

    train_model(
        is_physics=args.physics,
        beta=args.beta,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loaders=test_loaders,
    )


if __name__ == "__main__":
    main()