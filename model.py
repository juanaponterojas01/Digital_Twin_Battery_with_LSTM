"""LSTM model for battery SOC estimation.

Implements the vanilla LSTM (Hochreiter & Schmidhuber, 1997) with a
sigmoid output to constrain SOC predictions to (0, 1), and a dropout
layer for Monte Carlo uncertainty estimation.
"""

import numpy as np
import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    """Single-layer LSTM for SOC estimation with sigmoid output and MC dropout.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep (default: 2 for Current + Voltage).
    hidden_size : int
        Number of hidden units in the LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied after the LSTM output .
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, apply_dropout: bool | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_size).
        apply_dropout : bool | None
            Controls dropout behavior:
            - None  → auto (dropout ON in train mode, OFF in eval mode)
            - True  → force dropout ON (for MC uncertainty estimation)
            - False → force dropout OFF (for physics-informed training without dropout)

        Returns
        -------
        torch.Tensor
            Predicted SOC of shape (batch, seq_len), values in (0, 1).
        """
        lstm_out, _ = self.lstm(x)
        if apply_dropout is True or (apply_dropout is None and self.training):
            lstm_out = self.dropout(lstm_out)
        soc = self.sigmoid(self.fc(lstm_out))
        return soc.squeeze(-1)

    def predict_mc_uncertainty(self, x: torch.Tensor, n_iterations: int = 50):
        """Monte Carlo Dropout for uncertainty estimation.

        Performs multiple stochastic forward passes with dropout activated
        to estimate prediction uncertainty. Not available for models trained
        with --no-dropout (DROPOUT_CONDITION=False).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, 2).
        n_iterations : int
            Number of forward passes (default: 50).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_pred, std_pred) where:
            - mean_pred: (batch, seq_len) — SOC estimate
            - std_pred: (batch, seq_len) — Uncertainty (standard deviation)

        Raises
        ------
        ValueError
            If dropout probability is 0 (MC uncertainty requires p > 0).
        """
        if self.dropout.p == 0.0:
            raise ValueError(
                "MC dropout uncertainty estimation requires dropout probability > 0. "
                "This model was trained with --no-dropout and cannot produce uncertainty estimates."
            )

        predictions = []
        with torch.no_grad():
            for _ in range(n_iterations):
                pred = self(x, apply_dropout=True)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        return mean_pred, std_pred
