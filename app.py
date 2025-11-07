# app.py
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Catalyst Portfolio Optimizer", layout="wide")

# ---------- Helpers ----------
def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df

def _safe_to_float_array(series: pd.Series):
    return pd.to_numeric(series, errors="coerce").to_numpy()

def _download_bytes(name: str, df: pd.DataFrame):
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(f"Download template CSV: {name}", buff.getvalue(), file_name=name, mime="text/csv")

def _fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{x*100:.1f}%"

@st.cache_data(show_spinner=False)
def _template_positions():
    return pd.DataFrame({
        "ticker": ["ALT","MIST","KURA","IOVA"],
        "shares": [0,0,0,0],
        "avg": [0.00,0.00,0.00,0.00],
        "price": [0.00,0.00,0.00,0.00],
    })

@st.cache_data(show_spinner=False)
def _template_watchlist():
    # p_* sum to 1.0 row-wise
    return pd.DataFrame({
        "ticker": ["ALT","MIST","KURA","IOVA"],
        "event": ["SABCS abstract","FDA meeting window","Earnings Q3","BLA update"],
        "date": ["2025-12-10","2025-11-20","2025-11-07","2025-11-15"],
        "p_bull": [0.35,0.30,0.25,0.40],
        "p_base": [0.45,0.50,0.55,0.45],
        "p_bear": [0.20,0.20,0.20,0.15],
        "target_bull": [8.50,6.20,17.50,16.00],
        "target_base": [6.00,4.80,14.00,13.00],
        "target_bear": [3.80,3.60,10.50,10.00],
        "confidence": [0.60,0.55,0.50,0.65]
    })

# ---------- Standardizers ----------
def _std_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize broker CSV to: ticker, shares, avg, price"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker","shares","avg","price"])

    rename_map = {
        "Ticker":"ticker","SYMBOL":"ticker","Symbol":"ticker","symbol":"ticker",
        "Shares":"shares","Qty":"shares","QTY":"shares","Quantity":"shares","quantity":"shares","Position":"shares",
        "Avg":"avg","avg_cost":"avg","average_cost":"avg","Average Price":"avg","Avg Price":"avg","Cost Basis":"avg",
        "Price":"price","price":"price","last":"price","mark":"price","current_price":"price","Last Price":"price"
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Ensure all needed columns exist
    for col in ["ticker","shares","avg","price"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = _coerce_numeric(df, ["shares","avg","price"])

    # Fix invalid prices (use avg if price missing/<=0)
    price_arr = _safe_to_float_array(df["price"])
    avg_arr   = _safe_to_float_array(df["avg"])
    invalid = ~np.isfinite(price_arr) | (price_arr <= 0)
    price_arr[invalid] = avg_arr[invalid]
    df["price"] = price_arr

    df["shares"] = df["shares"].fillna(0).astype(int)
    df["avg"]    = df["avg"].fillna(0.0)
    df["price"]  = df["price"].fillna(0.0)

    out = df[["ticker","shares","avg","price"]].dropna(subset=["ticker"]).reset_index(drop=True)
    return out

def _std_watchlist_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize catalyst CSV to required EV fields."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "ticker","event","date",
            "p_bull","p_base","p_bear",
            "target_bull","target_base","target_bear",
            "confidence"
        ])

    rename_map = {
        "Ticker":"ticker","SYMBOL":"ticker","symbol":"ticker",
        "Event":"event","Catalyst":"event",
        "Date":"date","EventDate":"date",
        "pBull":"p_bull","pBase":"p_base","pBear":"p_bear",
        "TargetBull":"target_bull","TargetBase":"target_base","TargetBear":"target_bear",
        "Conf":"confidence","Confidence":"confidence"
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Require columns
    needed = ["ticker","event","date","p_bull","p_base","p_bear","target_bull","target_base","target_bear","confidence"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["event"] = df["event"].astype(str).str.strip()
    # Dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")

    # Numerics
    df = _coerce_numeric(df, ["p_bull","p_base","p_bear","target_bull","target_base","target_bear","confidence"])

    # Probability discipline
    # If probs expressed in %, convert if needed
    for col in ["p_bull","p_base","p_bear","confidence"]:
        if df[col].dropna().max() > 1.0:
            df[col] = df[col] / 100.0

    # Normalize probabilities where all present
    probs = df[["p_bull","p_base","p_bear"]].to_numpy(dtype=float)
    rows, cols = probs.shape
    for i in range(rows):
        if np.all(np.isfinite(probs[i])):
            s = probs[i].sum()
            if s > 0:
                probs[i] = probs[i] / s
    df["p_bull"], df["p_base"], df["p_bear"] = probs[:,0], probs[:,1], probs[:,2]

    # Confidence clamp
    df["confidence"] = df["confidence"].fillna(0.6).clip(0.0, 1.0)

    # Targets sanitize
    for col in ["target_bull","target_base","target_bear"]:
        df[col] = df[col].fillna(0.0).clip(lower=0.0)

    out = df[needed].dropna(subset=["ticker"]).reset_index(drop=True)
    return out

# ---------- EV Math ----------
def compute_ev(wl: pd.DataFrame, prices: pd.Series | dict | None = None) -> pd.DataFrame:
    df = wl.copy()
    if prices is None:
        df["price"] = np.nan
    else:
        if isinstance(prices, dict):
            df["price"] = df["ticker"].map(lambda t: prices.get(t, np.nan))
        else:
            # assume series indexed by ticker
            m = prices
            df["price"] = df["ticker"].map(lambda t: m.get(t, np.nan))

    df["EV_sh"] = (
        df["p_bull"] * df["target_bull"] +
        df["p_base"] * df["target_base"] +
        df["p_bear"] * df["target_bear"]
    )

    df["EV_spread"] = df["EV_sh"] - df["price"]
    df["EV_pct"] = np.where(
        pd.notna(df["price"]) & (df["price"] > 0),
        (df["EV_sh"] / df["price"] - 1.0),
        np.nan
    )

    # Sanity columns
    df["Prob_sum"] = (df["p_bull"] + df["p_base"] + df["p_bear"])
    df["Prob_ok"] = np.isclose(df["Prob_sum"], 1.0, atol=1e-3)

    return df

def suggest_sizes(evdf: pd.DataFrame, budget: float, max_per_ticker: float = 0.15):
    """Simple position sizing suggestion: weight by (EV_pct * confidence), cap per ticker."""
    x = evdf.copy()
    # Replace NaNs
    x["EV_pct"] = x["EV_pct"].fillna(0.0).clip(lower=-1.0, upper=10.0)
    x["confidence"] = x["confidence"].fillna(0.6).clip(0.0, 1.0)

    score = (x["EV_pct"].clip(lower=0) * (0.5 + 0.5 * x["confidence"])).to_numpy()
    score = np.where(np.isfinite(score) & (score > 0), score, 0.0)
    if score.sum() <= 0:
        x["weight"] = 0.0
        x["alloc"] = 0.0
        x["shares_suggested"] = 0
        return x

    weights = score / score.sum()
    weights = np.minimum(weights, max_per_ticker)
    if weights.sum() == 0:
        x["weight"] = 0.0
    else:
        weights = weights / weights.sum()
        x["weight"] = weights

    x["alloc"] = x["weight"] * float(budget)
    # shares rounding uses current price (fallback to target_base when price missing)
    px = x["price"].fillna(x["target_base"]).replace(0, np.nan)
    x["shares_suggested"] = np.floor(x["alloc"] / px).fillna(0).astype(int)
    return x

# ---------- UI ----------
st.title("Catalyst Portfolio Optimizer (v4.6.1 runtime)")

with st.sidebar:
    st.header("Data Inputs")
    st.caption("Upload BOTH CSVs. You can download templates below.")

    up_pos = st.file_uploader("Positions CSV (ticker, shares, avg, price)", type=["csv"], key="pos_up")
    up_wl  = st.file_uploader("Catalyst Watchlist CSV (bull/base/bear + targets)", type=["csv"], key="wl_up")

    st.divider()
    _download_bytes("positions_template.csv", _template_positions())
    _download_bytes("watchlist_template.csv", _template_watchlist())

    st.divider()
    budget = st.number_input("Available cash budget (€)", min_value=0.0, value=1000.0, step=100.0, help="Used for sizing suggestions.")
    max_cap = st.slider("Max weight per ticker", 0.05, 0.5, 0.15, 0.05)

# Load state every rerun so uploaders remain visible and data persists
if "positions" not in st.session_state:
    st.session_state.positions = _template_positions()
if "watchlist" not in st.session_state:
    st.session_state.watchlist = _template_watchlist()

# Parse uploads
pos_err = None
wl_err = None
if up_pos:
    try:
        raw = pd.read_csv(up_pos)
        st.session_state.positions = _std_positions_df(raw)
    except Exception as e:
        pos_err = f"Positions CSV error: {e}"

if up_wl:
    try:
        raw = pd.read_csv(up_wl)
        st.session_state.watchlist = _std_watchlist_df(raw)
    except Exception as e:
        wl_err = f"Watchlist CSV error: {e}"

if pos_err:
    st.error(pos_err)
if wl_err:
    st.error(wl_err)

colA, colB = st.columns([1,2], gap="large")

with colA:
    st.subheader("Portfolio (positions)")
    st.dataframe(st.session_state.positions, use_container_width=True)
    # Portfolio MV and cash (optional: let user input cash)
    mv = (st.session_state.positions["shares"] * st.session_state.positions["price"]).sum()
    avg_cost_val = (st.session_state.positions["shares"] * st.session_state.positions["avg"]).sum()
    st.metric("Market Value", f"€{mv:,.2f}")
    st.metric("Cost Basis (sum shares*avg)", f"€{avg_cost_val:,.2f}")

with colB:
    st.subheader("Catalyst Watchlist (with EV math)")
    wl = st.session_state.watchlist.copy()
    prices_map = dict(zip(st.session_state.positions["ticker"], st.session_state.positions["price"]))
    evdf = compute_ev(wl, prices_map)

    # Display small quality flags
    bad_prob = evdf.loc[~evdf["Prob_ok"], ["ticker","p_bull","p_base","p_bear","Prob_sum"]]
    if len(bad_prob):
        st.warning("Probability rows not summing to 1.0 were normalized. Check these:", icon="⚠️")
        st.dataframe(bad_prob, use_container_width=True)

    st.dataframe(
        evdf[["ticker","event","date","price","p_bull","p_base","p_bear",
              "target_bull","target_base","target_bear","EV_sh","EV_spread","EV_pct","confidence"]],
        use_container_width=True
    )

st.divider()

# Join with positions for a unified view
st.subheader("Unified View: Positions × Watchlist EV")
pos = st.session_state.positions.copy()
uni = pd.merge(evdf, pos, on="ticker", how="left", suffixes=("","_pos"))
uni["pos_value"] = (uni["shares"] * uni["price_pos"]).fillna(0.0)
uni["upside_%"] = np.where(
    pd.notna(uni["price_pos"]) & (uni["price_pos"] > 0),
    (uni["EV_sh"] / uni["price_pos"] - 1.0),
    np.nan
)
st.dataframe(
    uni[["ticker","event","date","shares","price_pos","avg","EV_sh","EV_pct","upside_%","confidence"]],
    use_container_width=True
)

# Sizing suggestions based on EV%
st.subheader("Sizing Suggestions (uses budget slider)")
sug = suggest_sizes(evdf, budget=budget, max_per_ticker=max_cap)
sug_view = sug[["ticker","event","price","EV_sh","EV_pct","confidence","weight","alloc","shares_suggested"]].copy()
sug_view["EV_pct"] = sug_view["EV_pct"].apply(_fmt_pct)
sug_view["weight"] = sug_view["weight"].apply(_fmt_pct)
st.dataframe(sug_view, use_container_width=True)

# Export suggested orders
csv_buf = io.StringIO()
export_df = sug[["ticker","shares_suggested","price","EV_sh","EV_pct","confidence","alloc","weight"]].copy()
export_df.to_csv(csv_buf, index=False)
st.download_button("Export suggested orders (CSV)", csv_buf.getvalue(), "suggested_orders.csv", "text/csv")

st.divider()
st.caption(f"Run timestamp (ET-normalized not applied here): {datetime.utcnow().isoformat()}Z")
st.caption("Tip: keep probabilities consistent (sum=1.0). EV/sh = p_bull*target_bull + p_base*target_base + p_bear*target_bear; EV% = EV/sh ÷ price − 1.")
