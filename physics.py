"""Differentiable physics models for battery SOC estimation.

Provides PyTorch nn.Module implementations of:
- OCV(SOC): open-circuit voltage as a polynomial function of SOC
- R_in(SOC): internal resistance as a polynomial function of SOC

Also provides the PhysicsLoss class that combines MSE data loss with
a physics consistency loss based on Kirchhoff's voltage law:
    V = OCV(SOC) + I * R_in(SOC)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from curve_fitting.ocv_lookup import OCVModel
from curve_fitting.rin_lookup import InternalResistanceModel

import config

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate polynomial using Horner's method.

    Parameters
    ----------
    coeffs : torch.Tensor
        Polynomial coefficients in decreasing power order [a_n, ..., a_1, a_0].
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Polynomial evaluated at x.
    """
    result = coeffs[0] * torch.ones_like(x)
    for i in range(1, len(coeffs)):
        result = result * x + coeffs[i]
    return result


def load_physics_coefficients():
    """Load OCV and R_in polynomial coefficients from the existing curve fitting models.

    Returns
    -------
    tuple[list[float], list[float]]
        (ocv_coeffs, rin_coeffs) where ocv_coeffs are the 7th-degree OCV polynomial
        coefficients and rin_coeffs are the 3rd-degree R_in polynomial coefficients,
        both in decreasing power order.
    """

    ocv_csv = os.path.join(PROJECT_DIR, "curve_fitting", "ocv_soc_lookup.csv")
    ocv_df = pd.read_csv(ocv_csv)
    ocv_model = OCVModel(ocv_df)
    ocv_coeffs = ocv_model.fit().tolist()

    rin_csv = os.path.join(PROJECT_DIR, "curve_fitting", "r_in_soc_lookup.csv")
    rin_df = pd.read_csv(rin_csv)
    rin_model = InternalResistanceModel(rin_df)
    rin_coeffs = rin_model.fit().tolist()

    return ocv_coeffs, rin_coeffs


class OCVModelTorch(nn.Module):
    """Differentiable OCV(SOC) polynomial model in PyTorch.

    Parameters
    ----------
    coeffs : list[float]
        Polynomial coefficients in decreasing power order.
    """

    def __init__(self, coeffs: list[float]):
        super().__init__()
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, soc: torch.Tensor) -> torch.Tensor:
        """Estimate OCV [V] for given SOC values.

        Parameters
        ----------
        soc : torch.Tensor
            State of charge values, typically in (0, 1).

        Returns
        -------
        torch.Tensor
            Open-circuit voltage in volts.
        """
        return _polyval(self.coeffs, soc)


class RinModelTorch(nn.Module):
    """Differentiable R_in(SOC) polynomial model in PyTorch.

    Output is in Ohm (converts from mOhm by dividing by 1000).

    Parameters
    ----------
    coeffs : list[float]
        Polynomial coefficients in decreasing power order (mOhm units).
    """

    def __init__(self, coeffs: list[float]):
        super().__init__()
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, soc: torch.Tensor) -> torch.Tensor:
        """Estimate R_in [Ohm] for given SOC values.

        Parameters
        ----------
        soc : torch.Tensor
            State of charge values, typically in (0, 1).

        Returns
        -------
        torch.Tensor
            Internal resistance in Ohm.
        """
        return _polyval(self.coeffs, soc) * config.MILLIOHM_TO_OHM


class PhysicsLoss(nn.Module):
    """Combined data loss and physics consistency loss.

    L_total = alpha * MSE(SOC_pred, SOC_true) + beta * MSE(V_true, OCV(SOC_pred) + I * R_in(SOC_pred))

    Parameters
    ----------
    ocv_coeffs : list[float]
        OCV polynomial coefficients.
    rin_coeffs : list[float]
        R_in polynomial coefficients.
    alpha : float
        Weight for the data (MSE) loss.
    beta : float
        Weight for the physics consistency loss.
    """

    def __init__(self, ocv_coeffs: list[float], rin_coeffs: list[float],
                 alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.ocv_model = OCVModelTorch(ocv_coeffs)
        self.rin_model = RinModelTorch(rin_coeffs)
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, soc_pred: torch.Tensor, soc_true: torch.Tensor,
                voltage_true: torch.Tensor, current_true: torch.Tensor):
        """Compute the combined physics-informed loss.

        Parameters
        ----------
        soc_pred : torch.Tensor
            Predicted SOC, shape (batch, seq_len).
        soc_true : torch.Tensor
            Ground truth SOC, shape (batch, seq_len).
        voltage_true : torch.Tensor
            Measured terminal voltage [V], shape (batch, seq_len).
        current_true : torch.Tensor
            Measured current [A], shape (batch, seq_len).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (total_loss, data_loss, physics_loss)
        """
        L_ml = self.mse(soc_pred, soc_true)

        v_predicted = self.ocv_model(soc_pred) + current_true * self.rin_model(soc_pred)
        L_phys = self.mse(voltage_true, v_predicted)

        total = self.alpha * L_ml + self.beta * L_phys
        return total, L_ml, L_phys

if __name__ == "__main__":
    ocv_coeffs, rin_coeffs = load_physics_coefficients()
    physics_loss = PhysicsLoss(ocv_coeffs, rin_coeffs)
    print(physics_loss)