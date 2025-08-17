# app.py
import streamlit as st

st.set_page_config(
    page_title="ECA – Deterministic & Probabilistic",
    layout="wide",
    page_icon="📈",
)

st.sidebar.title("ECA Navigation")
page = st.sidebar.radio(
    "Choose analysis type",
    ["Deterministic ECA", "Probabilistic ECA"],
    index=0,
)

if page == "Deterministic ECA":
    from pages import deterministic
    deterministic.run()
else:
    from pages import probabilistic
    probabilistic.run()
