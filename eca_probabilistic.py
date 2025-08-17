# eca_probabilistic.py
# Probabilistic Fracture Mechanics (PFM) Monte Carlo engine
# Author: Ikechukwu Onyegiri + Copilot

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import copy

from models import (
    ProbabilisticInputs, DeterministicInputs, PipeGeometry, FlawGeometry,
    StressStrainCurve, StressStrainPoint, FCGRParams, Loading, Distribution
)
from eca_deterministic import run_deterministic_eca


# ----------------------------
# Helpers
# ----------------------------

def _scale_stress_strain_curve(curve: StressStrainCurve, new_ys: float, new_uts: float) -> StressStrainCurve:
    """
    Scale an existing stress–strain curve to hit sampled YS/UTS while preserving shape.
    This is a pragmatic approach for PFM studies when you only vary YS/UTS.
    """
    eps, sig = curve.as_arrays()
    if len(sig) < 2:
        return curve

    # Current properties
    try:
        ys_old = curve.yield_strength()
    except Exception:
        ys_old = sig[0]
    uts_old = sig.max()

    # Piecewise linear scaling: below YS scale to new_ys/ys_old; above scale gradually to new_uts/uts_old
    scale_low = new_ys / max(ys_old, 1e-6)
    scale_high = new_uts / max(uts_old, 1e-6)

    # Blend factor vs strain (0 near origin → 1 at ultimate)
    w = (sig - ys_old) / max(uts_old - ys_old, 1e-6)
    w = np.clip(w, 0.0, 1.0)

    sig_new = sig * ((1 - w) * scale_low + w * scale_high)

    pts = [StressStrainPoint(strain=float(eps[i]), stress=float(sig_new[i])) for i in range(len(eps))]
    return StressStrainCurve(points=pts, is_true=curve.is_true)


def _maybe_update_fcgr(base: DeterministicInputs, C: Optional[float], m: Optional[float], which: str):
    """Update FCGR C,m on install/operation if random variables are supplied."""
    if which == "install" and base.fcgr_install:
        C_new = base.fcgr_install.C if C is None else C
        m_new = base.fcgr_install.m if m is None else m
        base.fcgr_install = FCGRParams(
            C=C_new, m=m_new,
            deltaK_th=base.fcgr_install.deltaK_th,
            deltaK_cap=base.fcgr_install.deltaK_cap,
            env_factor=base.fcgr_install.env_factor
        )
    if which == "operation" and base.fcgr_operation:
        C_new = base.fcgr_operation.C if C is None else C
        m_new = base.fcgr_operation.m if m is None else m
        base.fcgr_operation = FCGRParams(
            C=C_new, m=m_new,
            deltaK_th=base.fcgr_operation.deltaK_th,
            deltaK_cap=base.fcgr_operation.deltaK_cap,
            env_factor=base.fcgr_operation.env_factor
        )


# ----------------------------
# Main PFM runner
# ----------------------------

def run_pfm(prob: ProbabilisticInputs) -> Dict[str, Any]:
    """
    Execute Monte Carlo PFM:
      - Sample input RVs
      - Run deterministic ECA per sample (ductile tearing → FCG → EOL FAD check)
      - Compute PoF (fraction failing EOL)
      - Return traces for sensitivity

    Returns:
      {
        "PoF": float,
        "results": {
           "pass": [bool]*n,
           "Kr": [float]*n,
           "Lr": [float]*n,
           "a_EOL_mm": [float]*n
        },
        "samples": { "a": [...], "c": [...], "ligament": [...], "YS": [...], ... }
      }
    """
    n = prob.n_samples
    rng = np.random.default_rng(prob.seed)

    # Sample inputs
    samples = {}
    def S(rv_name: str, rv) -> np.ndarray:
        if rv is None:
            return np.full(n, np.nan)
        arr = rv.sample(n, rng)
        samples[rv_name] = arr
        return arr

    a0      = S("a", prob.rv_flaw_a)
    c0      = S("c", prob.rv_flaw_c)
    lig0    = S("ligament", prob.rv_ligament) if prob.rv_ligament else np.array([np.nan]*n)
    ys      = S("YS", prob.rv_Y)
    uts     = S("UTS", prob.rv_UTS)
    t       = S("t", prob.rv_t)
    OD      = S("OD", prob.rv_OD)
    sig_i   = S("sigma_install", prob.rv_sigma_install)
    sig_o   = S("sigma_oper", prob.rv_sigma_oper)
    C_i     = S("C_install", prob.rv_fcgr_C) if prob.rv_fcgr_C else np.array([np.nan]*n)
    m_i     = S("m_install", prob.rv_fcgr_m) if prob.rv_fcgr_m else np.array([np.nan]*n)
    # For operation we reuse same C,m unless specific RVs are added in future

    # Containers
    ok = np.zeros(n, dtype=bool)
    Kr = np.zeros(n)
    Lr = np.zeros(n)
    a_EOL = np.zeros(n)

    # Base deterministic context to clone
    base0: DeterministicInputs = prob.base

    for i in range(n):
        base = copy.deepcopy(base0)

        # Geometry
        base.pipe = PipeGeometry(t=float(t[i]), OD=float(OD[i]))

        # Flaw
        base.flaw0 = FlawGeometry(
            a=float(a0[i]),
            c=float(c0[i]),
            ligament=None if np.isnan(lig0[i]) else float(lig0[i]),
            flaw_type=base.flaw0.flaw_type  # keep selected type from base UI
        )

        # Material curve scaled to sampled YS/UTS
        base.mat_curve = _scale_stress_strain_curve(base.mat_curve, float(ys[i]), float(uts[i]))

        # Loads (membrane only for simplicity; UI can add bending later)
        if not np.isnan(sig_i[i]):
            base.install_loading = Loading(sigma_m=float(sig_i[i]), sigma_b=0.0, pressure=0.0)
        if not np.isnan(sig_o[i]):
            base.operation_loading = Loading(sigma_m=float(sig_o[i]), sigma_b=0.0, pressure=0.0)

        # FCGR parameters (optional)
        Ci = None if np.isnan(C_i[i]) else float(C_i[i])
        mi = None if np.isnan(m_i[i]) else float(m_i[i])
        _maybe_update_fcgr(base, Ci, mi, which="install")
        _maybe_update_fcgr(base, Ci, mi, which="operation")

        # Run deterministic ECA for this sample
        out = run_deterministic_eca(base)
        eol = out["EOL_check"]
        ok[i] = bool(eol["pass"])
        Kr[i] = float(eol["Kr"])
        Lr[i] = float(eol["Lr"])
        # final flaw depth = last trace value if present; else initial
        if out["flaw_evolution"]:
            a_EOL[i] = float(out["flaw_evolution"][-1][1])
        else:
            a_EOL[i] = base.flaw0.a

    pof = float(np.mean(~ok))

    return {
        "PoF": pof,
        "results": {
            "pass": ok.tolist(),
            "Kr": Kr.tolist(),
            "Lr": Lr.tolist(),
            "a_EOL_mm": a_EOL.tolist()
        },
        "samples": samples
    }
