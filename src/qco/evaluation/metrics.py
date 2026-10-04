"""Regression evaluation metrics for QCO — Phase 14."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


def compute_regression_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

    Args:
        y_true: Ground truth target values (tensor or array).
        y_pred: Predicted values (tensor or array).

    Returns:
        Dict with keys: 'mae', 'rmse', 'r2'.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # Total sum of squares for R2
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))

    if ss_tot > 1e-8:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        # Constant target edge case
        r2 = 1.0 if ss_res < 1e-8 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
