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
        Dropout probability applied after the LSTM output (default: 0.2).
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Predicted SOC of shape (batch, seq_len), values in (0, 1).
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        soc = self.sigmoid(self.fc(lstm_out))
        return soc.squeeze(-1)

    def predict_mc_uncertainty(self, x: torch.Tensor, n_iterations: int = 50):
        """Monte Carlo Dropout for uncertainty estimation.

        Works with already trained models. Sets the model to train mode
        to activate dropout, performs multiple stochastic forward passes,
        then returns the model to eval mode.

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
        """
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_iterations):
                pred = self(x)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        self.eval()

        return mean_pred, std_pred
