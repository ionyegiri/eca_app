# utils_ui.py
from __future__ import annotations
from typing import List, Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np
from models import (
    StressStrainCurve, StressStrainPoint, JRCurve, JRCurvePoint,
    PipeGeometry, FlawGeometry, FlawType, FCGRParams,
)

SS_CURVE_HELP = (
    "Upload CSV with columns: 'strain','stress' (strain as fraction, e.g., 0.002). "
    "Alternatively use the quick builder with YS/UTS inputs."
)

JR_CURVE_HELP = "Upload CSV with columns: 'da','J' (da in mm, J in N/mm)."

HIST_HELP = (
    "Upload CSV with columns: 'delta_sigma','cycles'. 'delta_sigma' in MPa, 'cycles' count."
)


def upload_or_build_stress_strain(key_prefix: str = "") -> StressStrainCurve:
    st.subheader("Material stress–strain curve")
    src = st.radio(
        "Provide curve via:", ["Upload CSV", "Quick builder (YS/UTS)"], key=f"{key_prefix}ss_src"
    )
    if src == "Upload CSV":
        f = st.file_uploader("CSV file", type=["csv"], help=SS_CURVE_HELP, key=f"{key_prefix}ss_csv")
        if f is not None:
            df = pd.read_csv(f)
            if not set(["strain", "stress"]).issubset(df.columns):
                st.error("CSV must have 'strain' and 'stress' columns.")
                return StressStrainCurve()
            pts = [StressStrainPoint(float(r["strain"]), float(r["stress"])) for _, r in df.iterrows()]
            return StressStrainCurve(points=pts, is_true=False)
        return StressStrainCurve()
    # Quick builder
    col1, col2, col3 = st.columns(3)
    with col1:
        ys = st.number_input("Yield strength, YS (MPa)", min_value=50.0, max_value=2000.0, value=450.0, key=f"{key_prefix}ys")
    with col2:
        uts = st.number_input("Ultimate strength, UTS (MPa)", min_value=ys, max_value=3000.0, value=max(550.0, ys), key=f"{key_prefix}uts")
    with col3:
        eps_u = st.number_input("Ultimate strain (engineering)", min_value=0.05, max_value=1.0, value=0.15, step=0.01, key=f"{key_prefix}epsu")
    # Simple piecewise linear curve: (0,0) -> (0.002, YS) -> (eps_u, UTS)
    pts = [
        StressStrainPoint(0.0, 0.0),
        StressStrainPoint(0.002, float(ys)),
        StressStrainPoint(float(eps_u), float(uts)),
    ]
    return StressStrainCurve(points=pts, is_true=False)


def upload_jr_curve(key_prefix: str = "") -> Optional[JRCurve]:
    st.subheader("Reeling JR-curve (optional)")
    f = st.file_uploader("CSV file with da,J", type=["csv"], help=JR_CURVE_HELP, key=f"{key_prefix}jr")
    if f is None:
        return None
    df = pd.read_csv(f)
    if not set(["da", "J"]).issubset(df.columns):
        st.error("CSV must have 'da' and 'J' columns.")
        return None
    pts = [JRCurvePoint(float(r["da"]), float(r["J"])) for _, r in df.iterrows()]
    return JRCurve(points=pts)


def upload_histogram(label: str, key_prefix: str = "") -> Optional[pd.DataFrame]:
    st.subheader(label)
    f = st.file_uploader("CSV file with delta_sigma,cycles", type=["csv"], help=HIST_HELP, key=f"{key_prefix}{label}")
    if f is None:
        return None
    try:
        df = pd.read_csv(f)
        if set(["Range", "Cycles"]).issubset(df.columns):
            df = df.rename(columns={"Range": "delta_sigma", "Cycles": "cycles"})
        if not set(["delta_sigma", "cycles"]).issubset(df.columns):
            st.error("CSV must have 'delta_sigma' and 'cycles' columns, or 'Range','Cycles'.")
            return None
        df = df[["delta_sigma", "cycles"]].copy()
        df["delta_sigma"] = pd.to_numeric(df["delta_sigma"], errors="coerce")
        df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce").astype(int)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Failed to read histogram: {e}")
        return None


def flaw_type_selector(key_prefix: str = "") -> FlawType:
    mapping = {
        "Surface – OD": FlawType.SURFACE_OD,
        "Surface – ID": FlawType.SURFACE_ID,
        "Embedded – near OD": FlawType.EMBEDDED_NEAR_OD,
        "Embedded – midwall": FlawType.EMBEDDED_MIDWALL,
        "Embedded – near ID": FlawType.EMBEDDED_NEAR_ID,
    }
    label = st.selectbox("Flaw type", list(mapping.keys()), key=f"{key_prefix}flawtype")
    return mapping[label]


def reeling_events_editor(key_prefix: str = "") -> list:
    st.subheader("Reeling events (optional)")

    # Show guidance in a caption instead of using the 'help=' kwarg
    st.caption("Axial strain in percent. Add rows as needed.")

    df0 = pd.DataFrame([
        {"name": "Reel-on", "axial_strain_%": 1.5, "cycles": 1},
        {"name": "Aligner", "axial_strain_%": 1.0, "cycles": 1},
    ])

    # ❗ Removed: help="..."  (older Streamlit versions don’t support this on data_editor)
    df = st.data_editor(
        df0,
        num_rows="dynamic",
        key=f"{key_prefix}reel_events",
    )

    events = []
    try:
        for _, r in df.iterrows():
            events.append({
                "name": str(r["name"]),
                "axial_strain": float(r["axial_strain_%"]) / 100.0,
                "cycles": int(r["cycles"]),
            })
    except Exception:
        pass

    return events


def fcgr_inputs(label: str, key_prefix: str = "") -> Optional[FCGRParams]:
    st.subheader(f"Paris law parameters – {label}")
    enabled = st.checkbox(f"Include {label} fatigue crack growth", value=False, key=f"{key_prefix}{label}en")
    if not enabled:
        return None
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        C = st.number_input("C", value=1e-12, format="%e", key=f"{key_prefix}{label}C")
    with c2:
        m = st.number_input("m", value=3.0, step=0.1, key=f"{key_prefix}{label}m")
    with c3:
        dKth = st.number_input("ΔK_th (MPa√m)", value=0.0, step=0.1, key=f"{key_prefix}{label}th")
    with c4:
        dKcap = st.number_input("ΔK cap (MPa√m)", value=100.0, step=1.0, key=f"{key_prefix}{label}cap")
    with c5:
        env = st.number_input("Env. factor", value=1.0, step=0.1, key=f"{key_prefix}{label}env")
    return FCGRParams(C=float(C), m=float(m), deltaK_th=float(dKth), deltaK_cap=float(dKcap), env_factor=float(env))
