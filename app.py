import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px

st.set_page_config(page_title="Catalyst Copilot", layout="wide")

st.title("🧠 Catalyst Copilot — Event-Driven Portfolio Optimizer")

st.sidebar.header("Upload Catalyst Data")
uploaded_file = st.sidebar.file_uploader("Upload your catalyst CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Catalyst Registry Preview")
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("Upload a catalyst file to begin (CSV format).")

st.sidebar.subheader("Portfolio Inputs")
cash = st.sidebar.number_input("Current Cash (€)", value=2500.00, step=100.00)
nlv = st.sidebar.number_input("Net Liquidation Value (€)", value=3500.00, step=100.00)

st.sidebar.markdown("---")
st.sidebar.caption("CMP + DRM + PRB Engine (v4.6.1)")

st.write("### System Modules Overview")
modules = {
    "CMP": "Catalyst Memory Persistence — retains catalyst registry across sessions.",
    "DRM": "Dynamic Reinforcement Model — adjusts reinvestment rate based on Sharpe.",
    "PRB": "Portfolio Rebalancer — optimizes allocation using adaptive Sharpe weighting.",
    "NEV-L": "Neural EV Lite — Bayesian analog probability refinement.",
    "PBT": "Prompt Backtester — Simulates past catalyst cycles to validate model accuracy.",
    "EVP": "Empirical Validation Protocol — live broker calibration."
}
st.table(pd.DataFrame.from_dict(modules, orient="index", columns=["Description"]))

st.write("### Performance Tracker")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("NLV", f"€{nlv:,.2f}")
with col2:
    st.metric("Cash", f"€{cash:,.2f}")
with col3:
    st.metric("Reinvestable Capital", f"€{nlv * 0.15:,.2f}")

st.markdown("---")
st.caption("Educational use only — system operates on public and verifiable data sources.")
