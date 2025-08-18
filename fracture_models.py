# fracture_models.py
# Core fracture mechanics functions for BS 7910 ECA
# Author: Ikechukwu Onyegiri + Copilot

from __future__ import annotations
import numpy as np
from typing import Tuple, List
from models import (
 PipeGeometry, FlawGeometry, StressStrainCurve, JRCurve,
 FADOption, DeterministicInputs, FlawType
)

# ----------------------------
# FAD Construction
# ----------------------------

def build_fad(option: FADOption, mat_curve: StressStrainCurve, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Failure Assessment Diagram (FAD).
    Option 1: Kr = sqrt(1 - 0.14*Lr^2) for Lr <= 1.0, else Kr = 0
    Option 2: Derived from stress-strain curve (BS 7910 Annex)
    """
    Lr = np.linspace(0, 1.2, n_points)
    if option == FADOption.OPTION_1:
        Kr = np.sqrt(1 - 0.14 * Lr**2)
        Kr[Lr > 1.0] = 0.0
    else:
        # Option 2: Use stress-strain curve to compute Kr vs Lr
        eps, sig = mat_curve.as_arrays()
        YS = mat_curve.yield_strength()
        Kr = []
        for lr in Lr:
            ref_stress = lr * YS
            # Find strain at ref_stress
            if ref_stress <= sig[0]:
                Kr.append(1.0)
            else:
                idx = np.searchsorted(sig, ref_stress)
                if idx >= len(sig):
                    Kr.append(0.0)
                else:
                    # Simplified: Kr = sqrt(1 - (strain/strain_u))
                    strain_u = eps[-1]
                    strain_ref = eps[idx]
                    Kr.append(max(0.0, np.sqrt(1 - strain_ref / strain_u)))
        Kr = np.array(Kr)
    return Lr, Kr


# ----------------------------
# Stress Intensity Factor (K)
# ----------------------------

def calc_K_surface(pipe: PipeGeometry, flaw: FlawGeometry, sigma: float) -> float:
    """
    Simplified BS 7910 solution for semi-elliptical surface flaw in pipe.
    sigma: applied membrane+bending stress [MPa]
    Returns K [MPa*sqrt(m)]
    """
    a = flaw.a / 1000.0  # mm -> m
    Q = 1.0 + 1.464 * (flaw.a / flaw.c)**1.65
    beta = 1.12  # shape factor approx
    return beta * sigma * np.sqrt(np.pi * a) * np.sqrt(Q)
    #return beta * sigma * np.sqrt(np.pi * a / 1000.0) * np.sqrt(Q)


def calc_K_embedded(pipe: PipeGeometry, flaw: FlawGeometry, sigma: float) -> float:
    """
    Simplified embedded flaw K solution.
    """
    a = flaw.a / 1000.0
    beta = 1.0
    return beta * sigma * np.sqrt(np.pi * a)
    #return beta * sigma * np.sqrt(np.pi * a / 1000.0)


# ----------------------------
# J-integral estimation
# ----------------------------
def calc_J(K: float, E: float = 207000.0, nu: float = 0.3) -> float:
    """
    Convert K to J using J = K^2 / E' (plane strain)
    E' = E / (1 - nu^2)
    """
    E_prime = E / (1 - nu**2)
    return (K**2) / E_prime


# ----------------------------
# Ductile tearing engine
# ----------------------------
def ductile_tearing(flaw: FlawGeometry, jr_curve: JRCurve, J_applied_func, tearing_limit: float) -> Tuple[FlawGeometry, float]:
    """
    Compute stable tearing for one reeling event.
    J_applied_func: function of da -> J_applied
    tearing_limit: max allowed tearing [mm]
    Returns (updated flaw, actual tearing)
    """
    da_candidates = np.linspace(0, tearing_limit, 200)
    J_applied = np.array([J_applied_func(da) for da in da_candidates])
    J_resist = np.array([jr_curve.J_at_da(da) for da in da_candidates])
    idx = np.argmax(J_applied < J_resist)  # last stable point
    if idx == 0:
        da = 0.0
    else:
        da = da_candidates[idx]
    new_flaw = flaw.copy()
    new_flaw.a += da
    return new_flaw, da


# ----------------------------
# Embedded → Surface re-characterization
# ----------------------------

def check_recharacterization(flaw: FlawGeometry, rule_ratio: float) -> FlawGeometry:
    """
    If ligament < rule_ratio * a, convert embedded flaw to surface flaw.
    """
    if flaw.ligament is not None and flaw.ligament < rule_ratio * flaw.a:
        new_flaw = flaw.copy()
        new_flaw.flaw_type = FlawType.SURFACE_OD
        new_flaw.ligament = None
        return new_flaw
    return flaw

