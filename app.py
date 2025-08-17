import streamlit as st
from models import DeterministicInputs, ProbabilisticInputs, PipeGeometry, FlawGeometry, StressStrainCurve, StressStrainPoint
from eca_deterministic import run_deterministic_eca
from eca_probabilistic import run_pfm
from plotting import plot_fad, plot_flaw_evolution, plot_distribution, plot_tornado
from sensitivity import spearman_sensitivity, tornado_from_spearman
import pandas as pd
import numpy as np

st.set_page_config(page_title="ECA Analysis Tool", layout="wide")

st.title("Engineering Critical Assessment (ECA) App")

# Sidebar navigation
analysis_type = st.sidebar.radio("Select Analysis Type", ["Deterministic", "Probabilistic", "Sensitivity"])

if analysis_type == "Deterministic":
    st.header("Deterministic ECA")
    # Inputs
    t = st.number_input("Wall Thickness (mm)", value=20.0)
    OD = st.number_input("Outer Diameter (mm)", value=273.0)
    a = st.number_input("Flaw Depth a (mm)", value=2.0)
    c = st.number_input("Flaw Half-Length c (mm)", value=10.0)

    if st.button("Run Deterministic ECA"):
        pipe = PipeGeometry(t=t, OD=OD)
        flaw = FlawGeometry(a=a, c=c)
        # Placeholder material curve
        mat_curve = StressStrainCurve(points=[StressStrainPoint(0.002, 450), StressStrainPoint(0.1, 550)])
        inputs = DeterministicInputs(pipe=pipe, flaw0=flaw, mat_curve=mat_curve)
        result = run_deterministic_eca(inputs)

        st.subheader("Results")
        st.json(result["EOL_check"])
        st.plotly_chart(plot_fad(result["FAD"], eol_point=(result["EOL_check"]["Lr"], result["EOL_check"]["Kr"])))
        st.plotly_chart(plot_flaw_evolution(result["flaw_evolution"]))

elif analysis_type == "Probabilistic":
    st.header("Probabilistic ECA")
    n_samples = st.number_input("Number of Samples", value=1000)
    if st.button("Run Probabilistic Analysis"):
        # Placeholder: build ProbabilisticInputs
        st.write("Running Monte Carlo simulation...")
        # Call run_pfm(prob_inputs)
        # Display PoF, histograms, etc.

elif analysis_type == "Sensitivity":
    st.header("Sensitivity Analysis")
    st.write("Upload results from probabilistic run or compute on the fly.")
