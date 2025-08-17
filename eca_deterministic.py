# eca_deterministic.py
# Deterministic ECA engine per BS 7910
# Author: Ikechukwu Onyegiri + Copilot

from __future__ import annotations
from typing import Dict, List
import numpy as np
from models import DeterministicInputs, Phase
from fracture_models import (
    build_fad, calc_K_surface, calc_K_embedded, calc_J,
    ductile_tearing, check_recharacterization
)
from fatigue import integrate_paris

def run_deterministic_eca(inputs: DeterministicInputs) -> Dict:
    """
    Perform deterministic ECA:
    1. Ductile tearing (reeling events)
    2. Installation fatigue crack growth
    3. Operation fatigue crack growth
    4. EOL fracture check on FAD
    Returns dict with:
        - 'flaw_evolution': list of (phase, a_mm)
        - 'EOL_check': (Kr, Lr, pass/fail)
        - 'FAD': (Lr[], Kr[])
    """
    trace = []
    flaw = inputs.flaw0.copy()
    pipe = inputs.pipe

    # Build FAD
    Lr_curve, Kr_curve = build_fad(inputs.fad_option, inputs.mat_curve)

    # ----------------------------
    # Phase 1: Reeling (ductile tearing)
    # ----------------------------
    if inputs.reeling_events and inputs.jr_curve_reeling:
        for event in inputs.reeling_events:
            # Approximate J_applied as linear with Δa for now
            def J_applied_func(da):
                # Simplified: J = alpha * (strain)^2 * (a + da)
                alpha = 1e6  # placeholder scaling
                return alpha * (event.axial_strain**2) * (flaw.a + da)
            flaw, da = ductile_tearing(
                flaw,
                inputs.jr_curve_reeling,
                J_applied_func,
                tearing_limit=inputs.tearing_limit_fraction_t * pipe.t
            )
            trace.append((Phase.REELING.value, flaw.a))
            # Check re-characterization
            flaw = check_recharacterization(flaw, inputs.rechar_rule_ligament_over_a)

    # ----------------------------
    # Phase 2: Installation FCG
    # ----------------------------
    if inputs.install_histogram is not None and inputs.fcgr_install:
        flaw, da = integrate_paris(flaw, pipe, inputs.fcgr_install, inputs.install_histogram,
                                   flaw_type="surface" if flaw.is_surface() else "embedded")
        trace.append((Phase.INSTALLATION.value, flaw.a))
        flaw = check_recharacterization(flaw, inputs.rechar_rule_ligament_over_a)

    # ----------------------------
    # Phase 3: Operation FCG
    # ----------------------------
    if inputs.operation_histogram is not None and inputs.fcgr_operation:
        flaw, da = integrate_paris(flaw, pipe, inputs.fcgr_operation, inputs.operation_histogram,
                                   flaw_type="surface" if flaw.is_surface() else "embedded")
        trace.append((Phase.OPERATION.value, flaw.a))
        flaw = check_recharacterization(flaw, inputs.rechar_rule_ligament_over_a)

    # ----------------------------
    # EOL Fracture Check
    # ----------------------------
    sigma = 0.0
    if inputs.operation_loading:
        sigma = inputs.operation_loading.combined()
    elif inputs.install_loading:
        sigma = inputs.install_loading.combined()

    if flaw.is_surface():
        K_applied = calc_K_surface(pipe, flaw, sigma)
    else:
        K_applied = calc_K_embedded(pipe, flaw, sigma)

    J_applied = calc_J(K_applied, inputs.youngs_modulus, inputs.poisson)
    YS = inputs.mat_curve.yield_strength()
    flow = inputs.mat_curve.flow_strength()
    Lr = sigma / flow if flow > 0 else 0.0
    Kr = K_applied / (np.sqrt(J_applied * inputs.youngs_modulus)) if J_applied > 0 else 0.0

    # Check against FAD
    fad_Kr_allow = np.interp(Lr, Lr_curve, Kr_curve)
    passes = Kr <= fad_Kr_allow

    return {
        "flaw_evolution": trace,
        "EOL_check": {"Kr": Kr, "Lr": Lr, "allowable_Kr": fad_Kr_allow, "pass": passes},
        "FAD": {"Lr": Lr_curve.tolist(), "Kr": Kr_curve.tolist()}
    }
