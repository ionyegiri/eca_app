# pages/probabilistic.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st
from models import (
    PipeGeometry, FlawGeometry, FlawType, DeterministicInputs, FADOption,
    StressStrainCurve, StressStrainPoint,
    ProbabilisticInputs, RandomVar, Distribution, Loading, FCGRParams,
)
from eca_probabilistic import run_pfm
from plotting import plot_distribution, plot_tornado
from sensitivity import spearman_sensitivity, tornado_from_spearman
from utils_ui import upload_or_build_stress_strain, flaw_type_selector


# -------------- Helpers --------------

def _rv_editor(name: str, default_dist: Distribution, defaults: dict, key_prefix: str = "") -> RandomVar:
    with st.expander(f"Random variable – {name}", expanded=False):
        dist_name = st.selectbox(
            "Distribution",
            [d.name for d in Distribution],
            index=[d for d in Distribution].index(default_dist),
            key=f"{key_prefix}{name}dist",
        )
        dist = Distribution[dist_name]
        params = {}
        if dist == Distribution.DETERM:
            params["value"] = st.number_input("Value", value=float(defaults.get("value", 0.0)), key=f"{key_prefix}{name}val")
        elif dist == Distribution.NORMAL:
            c1, c2 = st.columns(2)
            with c1:
                params["mean"] = st.number_input("Mean", value=float(defaults.get("mean", 0.0)), key=f"{key_prefix}{name}mean")
            with c2:
                params["std"] = st.number_input("Std dev", value=float(defaults.get("std", 1.0)), key=f"{key_prefix}{name}std")
        elif dist == Distribution.LOGNORMAL:
            c1, c2 = st.columns(2)
            with c1:
                params["mu"] = st.number_input("mu (log space)", value=float(defaults.get("mu", 0.0)), key=f"{key_prefix}{name}mu")
            with c2:
                params["sigma"] = st.number_input("sigma (log space)", value=float(defaults.get("sigma", 0.25)), key=f"{key_prefix}{name}sig")
        elif dist == Distribution.UNIFORM:
            c1, c2 = st.columns(2)
            with c1:
                params["low"] = st.number_input("Low", value=float(defaults.get("low", 0.0)), key=f"{key_prefix}{name}low")
            with c2:
                params["high"] = st.number_input("High", value=float(defaults.get("high", 1.0)), key=f"{key_prefix}{name}high")
        elif dist == Distribution.TRIANGULAR:
            c1, c2, c3 = st.columns(3)
            with c1:
                params["left"] = st.number_input("Left", value=float(defaults.get("left", 0.0)), key=f"{key_prefix}{name}left")
            with c2:
                params["mode"] = st.number_input("Mode", value=float(defaults.get("mode", 0.0)), key=f"{key_prefix}{name}mode")
            with c3:
                params["right"] = st.number_input("Right", value=float(defaults.get("right", 1.0)), key=f"{key_prefix}{name}right")
        return RandomVar(name=name, dist=dist, params=params)


def _base_deterministic() -> DeterministicInputs:
    st.header("Probabilistic ECA (Monte Carlo)")
    st.caption("Define the base deterministic context. Random variables will override these per sample.")
    c1, c2, c3 = st.columns(3)
    with c1:
        OD = st.number_input("Pipe OD (mm)", min_value=20.0, value=273.0, step=1.0)
    with c2:
        t = st.number_input("Wall thickness, t (mm)", min_value=2.0, value=20.0, step=0.5)
    with c3:
        nu = st.number_input("Poisson's ratio", min_value=0.0, max_value=0.5, value=0.3, step=0.01)

    ftype = flaw_type_selector(key_prefix="pfm_")
    c1, c2, c3 = st.columns(3)
    with c1:
        a0 = st.number_input("Initial a0 (mm)", min_value=0.01, value=2.0, step=0.1)
    with c2:
        c0 = st.number_input("Half-length c0 (mm)", min_value=0.1, value=10.0, step=0.5)
    with c3:
        ligament = st.number_input("Ligament (mm) – embedded only", min_value=0.0, value=5.0, step=0.5)
        ligament = None if ftype in (FlawType.SURFACE_OD, FlawType.SURFACE_ID) else float(ligament)

    mat_curve = upload_or_build_stress_strain(key_prefix="pfm_")

    st.subheader("Global loading (membrane σ)")
    l1, l2 = st.columns(2)
    with l1:
        sig_m_inst = st.number_input("Install σ (MPa)", value=0.0, step=5.0)
    with l2:
        sig_m_oper = st.number_input("Operation σ (MPa)", value=0.0, step=5.0)

    fad_opt = st.selectbox("FAD option", ["Option 1", "Option 2 (from curve)"])
    fad_opt = FADOption.OPTION_1 if fad_opt.startswith("Option 1") else FADOption.OPTION_2

    # Optional FCGR baseline (C,m) reused on both phases unless RVs override
    st.subheader("Baseline FCGR (applied if RVs not set)")
    c1, c2, c3 = st.columns(3)
    with c1:
        fcgr_C = st.number_input("C (baseline)", value=1e-12, format="%e")
    with c2:
        fcgr_m = st.number_input("m (baseline)", value=3.0, step=0.1)
    with c3:
        env = st.number_input("Env. factor", value=1.0, step=0.1)

    base = DeterministicInputs(
        pipe=PipeGeometry(t=float(t), OD=float(OD)),
        flaw0=FlawGeometry(a=float(a0), c=float(c0), ligament=ligament, flaw_type=ftype),
        mat_curve=mat_curve,
        fad_option=fad_opt,
        install_loading=Loading(sigma_m=float(sig_m_inst)),
        operation_loading=Loading(sigma_m=float(sig_m_oper)),
        fcgr_install=FCGRParams(C=float(fcgr_C), m=float(fcgr_m), env_factor=float(env)),
        fcgr_operation=FCGRParams(C=float(fcgr_C), m=float(fcgr_m), env_factor=float(env)),
        # leave histograms and JR-curve empty in PFM for now
    )
    return base


def run():
    base = _base_deterministic()

    st.subheader("Sampling setup")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("Samples (n)", min_value=10, max_value=200000, value=5000, step=100)
    with c2:
        seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42)

    # Random variables
    st.divider()
    st.subheader("Random variables")
    rv_a = _rv_editor("a", Distribution.NORMAL, {"mean": base.flaw0.a, "std": 0.5}, key_prefix="pfm_")
    rv_c = _rv_editor("c", Distribution.NORMAL, {"mean": base.flaw0.c, "std": 1.0}, key_prefix="pfm_")
    rv_lig = None
    if base.flaw0.ligament is not None:
        rv_lig = _rv_editor("ligament", Distribution.DETERM, {"value": base.flaw0.ligament}, key_prefix="pfm_")
    rv_Y = _rv_editor("YS", Distribution.NORMAL, {"mean": base.mat_curve.yield_strength(), "std": 30.0}, key_prefix="pfm_")
    rv_UTS = _rv_editor("UTS", Distribution.NORMAL, {"mean": base.mat_curve.ultimate_strength(), "std": 30.0}, key_prefix="pfm_")
    rv_t = _rv_editor("t", Distribution.NORMAL, {"mean": base.pipe.t, "std": 0.5}, key_prefix="pfm_")
    rv_OD = _rv_editor("OD", Distribution.NORMAL, {"mean": base.pipe.OD, "std": 1.0}, key_prefix="pfm_")
    rv_sig_i = _rv_editor("sigma_install", Distribution.DETERM, {"value": base.install_loading.sigma_m if base.install_loading else 0.0}, key_prefix="pfm_")
    rv_sig_o = _rv_editor("sigma_oper", Distribution.DETERM, {"value": base.operation_loading.sigma_m if base.operation_loading else 0.0}, key_prefix="pfm_")
    # Optional FCGR randomness
    rv_C = _rv_editor("fcgr_C", Distribution.DETERM, {"value": base.fcgr_install.C if base.fcgr_install else 1e-12}, key_prefix="pfm_")
    rv_m = _rv_editor("fcgr_m", Distribution.DETERM, {"value": base.fcgr_install.m if base.fcgr_install else 3.0}, key_prefix="pfm_")

    prob = ProbabilisticInputs(
        n_samples=int(n), seed=int(seed),
        rv_flaw_a=rv_a, rv_flaw_c=rv_c, rv_ligament=rv_lig,
        rv_Y=rv_Y, rv_UTS=rv_UTS, rv_t=rv_t, rv_OD=rv_OD,
        rv_sigma_install=rv_sig_i, rv_sigma_oper=rv_sig_o,
        rv_fcgr_C=rv_C, rv_fcgr_m=rv_m,
        base=base,
    )

    run_it = st.button("Run probabilistic ECA", type="primary")
    if not run_it:
        st.info("Set up sampling and RVs, then click **Run probabilistic ECA**.")
        return

    with st.spinner("Sampling and running Monte Carlo..."):
        try:
            out = run_pfm(prob)
        except Exception as e:
            st.exception(e)
            return

    st.success("Completed.")
    st.metric("Probability of Failure (PoF)", f"{out['PoF']*100:.2f}%")

    # Distributions
    res = out["results"]
    cols = st.columns(3)
    figs = [
        plot_distribution(np.array(res["Lr"]), "Lr distribution", x_label="Lr"),
        plot_distribution(np.array(res["Kr"]), "Kr distribution", x_label="Kr"),
        plot_distribution(np.array(res["a_EOL_mm"]), "a at EOL (mm)", x_label="a_EOL", unit="mm"),
    ]
    for col, fig in zip(cols, figs):
        with col:
            st.plotly_chart(fig, use_container_width=True)

    # Sensitivity (Spearman)
    st.subheader("Sensitivity (Spearman)")
    samples = out["samples"]
    outputs = {
        "fail": np.array([0 if p else 1 for p in res["pass"]], dtype=float),
        "a_EOL_mm": np.array(res["a_EOL_mm"], dtype=float),
        "Kr": np.array(res["Kr"], dtype=float),
        "Lr": np.array(res["Lr"], dtype=float),
    }
    sens = spearman_sensitivity(samples, outputs, target="fail")
    df_tornado = tornado_from_spearman(sens, top=10)
    fig_tor = plot_tornado(df_tornado)
    st.plotly_chart(fig_tor, use_container_width=True)

    st.download_button(
        "Download PFM results JSON",
        data=json.dumps(out, indent=2).encode("utf-8"),
        file_name="eca_probabilistic_result.json",
        mime="application/json",
    )
