# sensitivity.py
# Sensitivity analysis utilities for Probabilistic Fracture Mechanics
# Author: Ikechukwu Onyegiri + Copilot

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dataclasses import dataclass


@dataclass
class SensitivityResult:
    method: str
    # correlations indexed by variable name
    correlations: pd.DataFrame  # columns: ["rho", "pvalue"], index: variable names
    notes: str = ""


# ---------------------------------------------------------------------
# Utility: rank transform
# ---------------------------------------------------------------------

def _rank_transform(x: np.ndarray) -> np.ndarray:
    """Return ranks (1..n) for vector x with average ranks for ties."""
    temp = x.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(x) + 1, dtype=float)
    # Average ranks for ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    avg = sums / counts
    return avg[inv]


# ---------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------

def spearman_sensitivity(samples: Dict[str, np.ndarray],
                         outputs: Dict[str, np.ndarray],
                         target: str = "fail") -> SensitivityResult:
    """
    Compute Spearman rank correlation between each input and target output.
    target:
        - "fail": failure indicator (1=fail, 0=pass)
        - any key in outputs dict, e.g., "a_EOL_mm", "Kr", "Lr"
    """
    y = outputs["fail"] if target == "fail" else outputs[target]
    rows = []
    for name, x in samples.items():
        # skip NaN-only vectors
        if np.all(np.isnan(x)):
            continue
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 3:
            continue
        rho, p = spearmanr(x[m], y[m])
        rows.append((name, rho, p))
    df = pd.DataFrame(rows, columns=["var", "rho", "pvalue"]).set_index("var").sort_values("rho", ascending=False)
    return SensitivityResult(method="Spearman", correlations=df, notes="Rank-based monotonic sensitivity.")


# ---------------------------------------------------------------------
# PRCC: Partial Rank Correlation Coefficient
# ---------------------------------------------------------------------

def prcc_sensitivity(samples: Dict[str, np.ndarray],
                     outputs: Dict[str, np.ndarray],
                     target: str = "fail") -> SensitivityResult:
    """
    Compute PRCC of each input vs target output while controlling for others.
    Steps:
      - rank-transform all variables and output
      - regress each variable and output on the others
      - PRCC = Spearman correlation between residuals
    """
    # Build data matrix
    keys = [k for k in samples.keys() if not np.all(np.isnan(samples[k]))]
    X = np.column_stack([samples[k] for k in keys]).astype(float)
    y = outputs["fail"] if target == "fail" else outputs[target]
    # Remove rows with NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask, :]
    y = y[mask]

    if X.shape[0] < max(10, X.shape[1] + 3):
        # Not enough samples for PRCC
        return SensitivityResult(method="PRCC", correlations=pd.DataFrame(columns=["rho", "pvalue"]), notes="Insufficient samples.")

    # Rank-transform
    Xr = np.apply_along_axis(_rank_transform, 0, X)
    yr = _rank_transform(y)

    def regress_residuals(a: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Residual from linear regression a ~ B (least squares)
        BtB = B.T @ B
        try:
            coef = np.linalg.solve(BtB, B.T @ a)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(BtB) @ (B.T @ a)
        return a - B @ coef

    rows = []
    p = Xr.shape[1]
    for j, name in enumerate(keys):
        # Other vars
        idx = [i for i in range(p) if i != j]
        Bj = Xr[:, idx]
        xj_res = regress_residuals(Xr[:, j], Bj)
        y_res = regress_residuals(yr, Bj)
        rho, pval = spearmanr(xj_res, y_res)
        rows.append((name, rho, pval))

    df = pd.DataFrame(rows, columns=["var", "rho", "pvalue"]).set_index("var").sort_values("rho", ascending=False)
    return SensitivityResult(method="PRCC", correlations=df, notes="Partial rank correlation controlling for other inputs.")


# ---------------------------------------------------------------------
# One-at-a-time (OAT) sweep
# ---------------------------------------------------------------------

def oat_sweep(var_name: str,
              baseline: Dict[str, float],
              sweep_values: np.ndarray,
              model_func: Callable[[Dict[str, float]], float]) -> pd.DataFrame:
    """
    Perform OAT sweep for a single variable.
    baseline: mapping of variable -> baseline value
    sweep_values: values to assign to var_name
    model_func: function that takes a dict of variables and returns scalar response (e.g., PoF proxy or EOL a)
    Returns DataFrame with ['value','response'].
    """
    rows = []
    for v in sweep_values:
        ctx = baseline.copy()
        ctx[var_name] = float(v)
        resp = model_func(ctx)
        rows.append((float(v), float(resp)))
    return pd.DataFrame(rows, columns=["value", "response"])


# ---------------------------------------------------------------------
# Tornado data builder
# ---------------------------------------------------------------------

def tornado_from_spearman(sens: SensitivityResult, top: int = 10, absolute: bool = True) -> pd.DataFrame:
    """
    Prepare tornado-plot-friendly DataFrame from a Spearman/PRCC result.
    Returns DataFrame with columns ['var','rho'] sorted by |rho| descending.
    """
    df = sens.correlations.copy()
    if df.empty:
        return df
    df = df.copy()
    df["rho_abs"] = df["rho"].abs() if absolute else df["rho"]
    df = df.sort_values("rho_abs", ascending=False).head(top)
    return df[["rho", "pvalue"]].rename_axis("var").reset_index()
