# app.py
import streamlit as st

st.set_page_config(page_title="Catalyst Copilot", layout="wide")
st.title("Catalyst Copilot — sanity check ✅")

st.write(
    "If you can see this, the app is deployed correctly. "
    "We’ll add real features once the base is live."
)

with st.expander("Environment"):
    st.write({
        "streamlit_version": st.__version__,
    })
