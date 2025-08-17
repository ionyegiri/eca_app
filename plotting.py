# plotting.py
# Plotting utilities for ECA Streamlit app (Plotly-based)
# Author: Ikechukwu Onyegiri + Copilot

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import Phase

# -------------------------------------------------------
# FAD plot: Option 1/2 curve, EOL point, and (optional) path markers
# -------------------------------------------------------

def plot_fad(fad_data: Dict, eol_point: Optional[Tuple[float, float]] = None,
             path_points: Optional[List[Tuple[float, float]]] = None,
             title: str = "Failure Assessment Diagram (FAD)") -> go.Figure:
    """
    fad_data: {"Lr": [..], "Kr": [..]}
    eol_point: (Lr, Kr) at End-of-Life
    path_points: optional list of (Lr, Kr) along the evolution (if available)
    """
    Lr = np.array(fad_data["Lr"], dtype=float)
    Kr = np.array(fad_data["Kr"], dtype=float)

    fig = go.Figure()

    # FAD envelope
    fig.add_trace(go.Scatter(
        x=Lr, y=Kr,
        mode="lines",
        name="FAD envelope",
        line=dict(color="#2a9d8f", width=3)
    ))

    # Optional crack path (e.g., reeling -> install -> operation evolution in FAD space)
    if path_points and len(path_points) > 1:
        xp = [p[0] for p in path_points]
        yp = [p[1] for p in path_points]
        fig.add_trace(go.Scatter(
            x=xp, y=yp,
            mode="lines+markers",
            name="Evolution path",
            line=dict(color="#264653", dash="dash"),
            marker=dict(color="#264653", size=7, symbol="diamond")
        ))

    # EOL point
    if eol_point is not None:
        fig.add_trace(go.Scatter(
            x=[eol_point[0]], y=[eol_point[1]],
            mode="markers",
            name="EOL point",
            marker=dict(color="#e76f51", size=10, symbol="x")
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Lr (Reference stress ratio)",
        yaxis_title="Kr (Fracture ratio)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(range=[0, max(1.2, float(Lr.max()) if len(Lr) else 1.2)])
    fig.update_yaxes(range=[0, 1.2])
    return fig


# -------------------------------------------------------
# Flaw evolution plot (phase-wise)
# -------------------------------------------------------

def plot_flaw_evolution(trace: List[Tuple[str, float]],
                        title: str = "Flaw Evolution per Phase") -> go.Figure:
    """
    trace: list of (phase_name, a_mm) in chronological order
    """
    if not trace:
        fig = go.Figure()
        fig.update_layout(title="No flaw evolution data available.", template="plotly_white")
        return fig

    phases = [t[0] for t in trace]
    a_vals = [t[1] for t in trace]

    # Collate by phase (if multiple entries, show step progression)
    x = list(range(1, len(a_vals) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=a_vals, mode="lines+markers",
        name="a (depth, mm)",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8)
    ))

    # annotate phases across x positions
    annotations = []
    for i, (p, a) in enumerate(trace, start=1):
        short = p.title()
        annotations.append(dict(x=i, y=a, text=short, showarrow=True, arrowhead=1, yshift=12))
    fig.update_layout(
        title=title,
        xaxis_title="Step (chronological)",
        yaxis_title="Flaw depth a [mm]",
        template="plotly_white",
        annotations=annotations
    )
    return fig


# -------------------------------------------------------
# Distribution plots (histogram + ECDF)
# -------------------------------------------------------

def plot_distribution(data: np.ndarray, title: str,
                      bins: int = 30, show_ecdf: bool = True,
                      x_label: str = "", unit: str = "") -> go.Figure:
    clean = data[~np.isnan(data)]
    fig = make_subplots(
        rows=1, cols=2 if show_ecdf else 1,
        subplot_titles=("Histogram", "ECDF" if show_ecdf else None)
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=clean,
        nbinsx=bins,
        name="Histogram",
        marker=dict(color="#457b9d")
    ), row=1, col=1)

    # ECDF
    if show_ecdf:
        xs = np.sort(clean)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name="ECDF",
            line=dict(color="#e76f51")
        ), row=1, col=2)

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False
    )
    fig.update_xaxes(title_text=f"{x_label}{' ['+unit+']' if unit else ''}", row=1, col=1)
    if show_ecdf:
        fig.update_xaxes(title_text=f"{x_label}{' ['+unit+']' if unit else ''}", row=1, col=2)
        fig.update_yaxes(title_text="F(x)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    return fig


# -------------------------------------------------------
# Tornado (sensitivity) plot
# -------------------------------------------------------

def plot_tornado(df: pd.DataFrame, title: str = "Sensitivity (Tornado)") -> go.Figure:
    """
    df columns expected: ['var','rho','pvalue'] from sensitivity.tornado_from_spearman(...)
    Shows absolute |rho| as bar length with sign coloring.
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No sensitivity data available.", template="plotly_white")
        return fig

    # order by absolute influence descending
    df = df.copy()
    df["rho_abs"] = df["rho"].abs()
    df.sort_values("rho_abs", ascending=True, inplace=True)  # horizontal bars, smallest at bottom
    colors = ["#2a9d8f" if r >= 0 else "#e76f51" for r in df["rho"]]

    fig = go.Figure(go.Bar(
        x=df["rho_abs"],
        y=df["var"],
        orientation="h",
        marker_color=colors,
        hovertemplate="|rho|=%{x:.3f}<br>rho=%{customdata[0]:.3f}<br>p=%{customdata[1]:.3g}",
        customdata=df[["rho", "pvalue"]].values
    ))
    fig.update_layout(
        title=title,
        xaxis_title="|rho| (absolute correlation strength)",
        yaxis_title="Input variable",
        template="plotly_white",
        margin=dict(l=120, r=40, t=60, b=40)
    )
    return fig


# -------------------------------------------------------
# Helper: FAD path from evolution trace (rough mapping)
# -------------------------------------------------------

def fad_path_from_trace(trace: List[Tuple[str, float]],
                        sigma_path: Optional[List[float]] = None,
                        E: float = 207000.0,
                        flow: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Construct a (very) rough FAD-space path from phase trace and a path of applied stresses:
      - Lr ~ sigma / flow
      - Kr ~ placeholder (noting we generally need K and J per point)
    This is provided only for qualitative visualization unless full per-step K/J are supplied.
    """
    if not trace or sigma_path is None or flow is None or flow <= 0:
        return []
    path = []
    for (_, _a), sig in zip(trace, sigma_path):
        Lr = sig / flow
        # Qualitative: bring Kr as Lr-proportional placeholder (bounded)
        Kr = max(0.0, 1.0 - 0.14 * Lr**2) ** 0.5
        path.append((Lr, Kr))
    return path
