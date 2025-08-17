# fatigue.py
# Fatigue crack growth engine for BS 7910 ECA
# Author: Ikechukwu Onyegiri + Copilot

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from models import PipeGeometry, FlawGeometry, FCGRParams

# ----------------------------
# ΔK Calculation
# ----------------------------

def delta_K_surface(pipe: PipeGeometry, flaw: FlawGeometry, delta_sigma: float) -> float:
    """
    Compute ΔK for a surface flaw under stress range Δσ.
    Simplified BS 7910 solution.
    """
    a = flaw.a / 1000.0  # mm -> m
    Q = 1.0 + 1.464 * (flaw.a / flaw.c)**1.65
    beta = 1.12
    return beta * delta_sigma * np.sqrt(np.pi * a / 1000.0) * np.sqrt(Q)


def delta_K_embedded(pipe: PipeGeometry, flaw: FlawGeometry, delta_sigma: float) -> float:
    """
    Compute ΔK for an embedded flaw under stress range Δσ.
    """
    a = flaw.a / 1000.0
    beta = 1.0
    return beta * delta_sigma * np.sqrt(np.pi * a / 1000.0)


# ----------------------------
# Paris Law Integration
# ----------------------------

def integrate_paris(flaw: FlawGeometry, pipe: PipeGeometry, fcgr: FCGRParams,
                    histogram: pd.DataFrame, flaw_type: str = "surface") -> Tuple[FlawGeometry, float]:
    """
    Integrate crack growth using Paris law over a histogram of stress ranges.
    histogram: DataFrame with columns ['delta_sigma','cycles']
    Returns updated flaw and total Δa [mm].
    """
    da_total = 0.0
    new_flaw = flaw.copy()
    for _, row in histogram.iterrows():
        delta_sigma = row['delta_sigma']
        cycles = row['cycles']
        if flaw_type == "surface":
            deltaK = delta_K_surface(pipe, new_flaw, delta_sigma)
        else:
            deltaK = delta_K_embedded(pipe, new_flaw, delta_sigma)
        if deltaK < fcgr.deltaK_th:
            continue
        deltaK_eff = min(deltaK, fcgr.deltaK_cap)
        da_per_cycle = fcgr.C * (deltaK_eff**fcgr.m) * fcgr.env_factor
        da = da_per_cycle * cycles * 1000.0  # convert m to mm
        new_flaw.a += da
        da_total += da
    return new_flaw, da_total


# ----------------------------
# Orcaflex Histogram Parser (Placeholder)
# ----------------------------
def parse_orcaflex_histogram(file_path: str) -> pd.DataFrame:
    """
    Placeholder for Orcaflex histogram parsing.
    Expected format: CSV with columns ['Range','Cycles'] or similar.
    Returns DataFrame with ['delta_sigma','cycles'].
    """
    df = pd.read_csv(file_path)
    # Map Orcaflex columns to expected format
    if 'Range' in df.columns:
        df.rename(columns={'Range': 'delta_sigma', 'Cycles': 'cycles'}, inplace=True)
    return df[['delta_sigma', 'cycles']]
