# pages/deterministic.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st
from models import (
    PipeGeometry, FlawGeometry, FlawType, DeterministicInputs, FADOption,
    ReelingStrainEvent, Loading
)
from eca_deterministic import run_deterministic_eca
from plotting import plot_fad, plot_flaw_evolution
from utils_ui import (
    upload_or_build_stress_strain, upload_jr_curve, upload_histogram,
    flaw_type_selector, reeling_events_editor, fcgr_inputs,
)


def _build_inputs() -> DeterministicInputs:
    st.header("Deterministic ECA")
    g1, g2, g3 = st.columns(3)
    with g1:
        OD = st.number_input("Pipe OD (mm)", min_value=20.0, value=273.0, step=1.0)
    with g2:
        t = st.number_input("Wall thickness, t (mm)", min_value=2.0, value=20.0, step=0.5)
    with g3:
        nu = st.number_input("Poisson's ratio", min_value=0.0, max_value=0.5, value=0.3, step=0.01)

    ftype = flaw_type_selector()
    c1, c2, c3 = st.columns(3)
    with c1:
        a0 = st.number_input("Initial flaw depth, a0 (mm)", min_value=0.01, value=2.0, step=0.1)
    with c2:
        c0 = st.number_input("Half-length along weld, c0 (mm)", min_value=0.1, value=10.0, step=0.5)
    with c3:
        ligament = st.number_input("Ligament to nearest surface (mm) – embedded only", min_value=0.0, value=5.0, step=0.5)
        ligament = None if ftype in (FlawType.SURFACE_OD, FlawType.SURFACE_ID) else float(ligament)

    mat_curve = upload_or_build_stress_strain(key_prefix="det_")

    st.subheader("Global loading for EOL check")
    l1, l2, l3 = st.columns(3)
    with l1:
        sig_m_inst = st.number_input("Install membrane σ (MPa)", value=0.0, step=5.0)
    with l2:
        sig_m_oper = st.number_input("Operation membrane σ (MPa)", value=0.0, step=5.0)
    with l3:
        sigma_b = st.number_input("Bending σ (MPa) – not used in K yet", value=0.0, step=5.0)

    fad_opt = st.selectbox("FAD option", ["Option 1", "Option 2 (from curve)"])
    fad_opt = FADOption.OPTION_1 if fad_opt.startswith("Option 1") else FADOption.OPTION_2

    jr = upload_jr_curve(key_prefix="det_")
    events = reeling_events_editor(key_prefix="det_")
    events_models = [ReelingStrainEvent(name=e["name"], axial_strain=e["axial_strain"], cycles=e["cycles"]) for e in events]

    # Histograms
    hist_i = upload_histogram("Installation histogram (optional)", key_prefix="det_")
    hist_o = upload_histogram("Operation histogram (optional)", key_prefix="det_")

    fcgr_i = fcgr_inputs("Installation", key_prefix="det_")
    fcgr_o = fcgr_inputs("Operation", key_prefix="det_")

    # Tearing & re-characterization rules
    s1, s2 = st.columns(2)
    with s1:
        tear_frac_t = st.slider("Max ductile tearing per event (fraction of t)", min_value=0.02, max_value=0.5, value=0.10, step=0.01)
    with s2:
        rechar_rule = st.slider("Embedded→Surface rule: ligament < r * a", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

    base = DeterministicInputs(
        pipe=PipeGeometry(t=float(t), OD=float(OD)),
        flaw0=FlawGeometry(a=float(a0), c=float(c0), ligament=ligament, flaw_type=ftype),
        mat_curve=mat_curve,
        fad_option=fad_opt,
        jr_curve_reeling=jr,
        fcgr_install=fcgr_i,
        fcgr_operation=fcgr_o,
        reeling_events=events_models,
        install_histogram=hist_i,
        operation_histogram=hist_o,
        install_loading=Loading(sigma_m=float(sig_m_inst), sigma_b=float(sigma_b)),
        operation_loading=Loading(sigma_m=float(sig_m_oper), sigma_b=float(sigma_b)),
        poisson=float(nu),
        tearing_limit_fraction_t=float(tear_frac_t),
        rechar_rule_ligament_over_a=float(rechar_rule),
    )
    return base


def run():
    base = _build_inputs()
    run_it = st.button("Run deterministic ECA", type="primary")
    if not run_it:
        st.info("Configure inputs then click **Run deterministic ECA**.")
        return

    with st.spinner("Running deterministic ECA..."):
        try:
            out = run_deterministic_eca(base)
        except Exception as e:
            st.exception(e)
            return

    st.success("Completed.")
    eol = out["EOL_check"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lr (ref. stress ratio)", f"{eol['Lr']:.3f}")
    with c2:
        st.metric("Kr (fracture ratio)", f"{eol['Kr']:.3f}")
    with c3:
        st.metric("Allowable Kr on FAD", f"{eol['allowable_Kr']:.3f}")
    with c4:
        st.metric("EOL status", "PASS" if eol['pass'] else "FAIL")

    # Plots
    fad_fig = plot_fad(out["FAD"], eol_point=(eol["Lr"], eol["Kr"]))
    st.plotly_chart(fad_fig, use_container_width=True)

    st.subheader("Flaw evolution")
    trace = out.get("flaw_evolution", [])
    evo_fig = plot_flaw_evolution(trace)
    st.plotly_chart(evo_fig, use_container_width=True)

    st.download_button(
        "Download results JSON",
        data=json.dumps(out, indent=2).encode("utf-8"),
        file_name="eca_deterministic_result.json",
        mime="application/json",
    )
