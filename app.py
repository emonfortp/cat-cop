import os
import io
import math
from datetime import datetime, timezone
from dateutil import parser as dtparser

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Cat-Cop: Catalyst Copilot", layout="wide")
st.title("🧪 Cat-Cop — Catalyst Portfolio Copilot")

# -----------------------------
# UTILITIES
# -----------------------------
NUMERIC_KW = dict(errors="coerce", downcast="float")

def _now_utc():
    return datetime.now(timezone.utc)

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], **NUMERIC_KW)
    return df

def _std_cols(df, mapping):
    """
    Rename columns according to a priority mapping list.
    mapping: dict canonical_name -> list of possible names (lowercased match)
    """
    lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for canon, candidates in mapping.items():
        for cand in candidates:
            if cand in lower:
                rename_map[lower[cand]] = canon
                break
    df = df.rename(columns=rename_map)
    return df

def _parse_date(s):
    if pd.isna(s):
        return pd.NaT
    try:
        return dtparser.parse(str(s)).astimezone(timezone.utc).date()
    except Exception:
        return pd.NaT

@st.cache_data(ttl=300, show_spinner=False)
def fetch_quotes_yf(tickers):
    """Fetch last price via yfinance (cached). Returns dict {ticker: price}."""
    import yfinance as yf
    prices = {}
    if not tickers:
        return prices
    # yfinance prefers batched Tickers but single calls are most reliable
    for t in tickers:
        try:
            y = yf.Ticker(t)
            info = y.fast_info  # fast path
            p = None
            if info is not None:
                p = info.get("last_price", None)
                if p is None:
                    p = info.get("last_close", None)
            if p is None:
                hist = y.history(period="5d", auto_adjust=False)
                if not hist.empty:
                    p = float(hist["Close"].iloc[-1])
            if p is not None and np.isfinite(p) and p > 0:
                prices[t] = float(p)
        except Exception:
            # Keep going; missing quotes should not kill the app
            continue
    return prices

# -----------------------------
# INPUTS — SIDEBAR
# -----------------------------
st.sidebar.header("Inputs")

st.sidebar.markdown("**1) Positions CSV (optional)**")
pos_file = st.sidebar.file_uploader(
    "Upload positions CSV (columns like: ticker/symbol, qty, avg, price)",
    type=["csv"], key="pos_csv")

st.sidebar.markdown("**2) Catalyst CSV (optional)**")
cat_file = st.sidebar.file_uploader(
    "Upload catalyst CSV (columns like: ticker, date, p_bull/p_base/p_bear, t_bull/t_base/t_bear, confidence)",
    type=["csv"], key="cat_csv")

use_live_quotes = st.sidebar.toggle("Enable live quotes (yfinance, cached 5 min)", value=False)
ev_threshold = st.sidebar.number_input("Short-Fuse EV% minimum", value=15.0, step=1.0)
conf_threshold = st.sidebar.number_input("Short-Fuse Confidence minimum", value=0.70, step=0.05, min_value=0.0, max_value=1.0)

budget = st.sidebar.number_input("Budget for new ideas (€)", value=1000.0, step=50.0, min_value=0.0)
max_positions = st.sidebar.number_input("Max suggestions", value=5, step=1, min_value=1)

# -----------------------------
# LOAD / STANDARDIZE POSITIONS
# -----------------------------
def load_positions():
    if pos_file is None:
        # Safe default empty structure
        return pd.DataFrame(columns=["ticker", "qty", "avg", "price"])
    try:
        df = pd.read_csv(pos_file)
    except Exception as e:
        st.sidebar.error(f"Positions CSV error: {e}")
        return pd.DataFrame(columns=["ticker", "qty", "avg", "price"])

    df = _std_cols(df, {
        "ticker": ["ticker", "symbol", "sym"],
        "qty":    ["qty", "quantity", "shares", "units"],
        "avg":    ["avg", "avgcost", "average_cost", "avg_cost", "cost_basis"],
        "price":  ["price", "last", "mark"]
    })
    # Ensure canonical columns exist
    for c in ["ticker", "qty", "avg", "price"]:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numerics
    df = _coerce_numeric(df, ["qty", "avg", "price"])

    # Clean ticker
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Fill improper price with avg if needed
    df["price"] = np.where(~np.isfinite(df["price"]) | (df["price"] <= 0), df["avg"], df["price"])
    df["avg"]   = np.where(~np.isfinite(df["avg"]) | (df["avg"] <= 0), df["price"], df["avg"])

    # Drop empties
    df = df[df["ticker"].str.len() > 0]
    return df

positions = load_positions()

# -----------------------------
# LOAD / STANDARDIZE CATALYSTS
# -----------------------------
def load_catalysts():
    if cat_file is None:
        return pd.DataFrame(columns=[
            "ticker","event","date","p_bull","p_base","p_bear","t_bull","t_base","t_bear","confidence"
        ])
    try:
        df = pd.read_csv(cat_file)
    except Exception as e:
        st.sidebar.error(f"Catalyst CSV error: {e}")
        return pd.DataFrame(columns=[
            "ticker","event","date","p_bull","p_base","p_bear","t_bull","t_base","t_bear","confidence"
        ])

    df = _std_cols(df, {
        "ticker":     ["ticker", "symbol", "sym"],
        "event":      ["event", "catalyst", "name"],
        "date":       ["date", "event_date", "when"],
        "p_bull":     ["p_bull", "prob_bull", "bull_prob"],
        "p_base":     ["p_base", "prob_base", "base_prob"],
        "p_bear":     ["p_bear", "prob_bear", "bear_prob"],
        "t_bull":     ["t_bull", "target_bull", "bull_target"],
        "t_base":     ["t_base", "target_base", "base_target"],
        "t_bear":     ["t_bear", "target_bear", "bear_target"],
        "confidence": ["confidence", "conf", "cl"]
    })

    # Missing columns → create
    for c in ["event","confidence"]:
        if c not in df.columns: df[c] = np.nan

    # Types
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["p_bull","p_base","p_bear","t_bull","t_base","t_bear","confidence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], **NUMERIC_KW)

    # Date & DTE
    df["date"] = df["date"].apply(_parse_date)
    today = _now_utc().date()
    df["DTE"] = (pd.to_datetime(df["date"]) - pd.to_datetime(today)).dt.days

    # Keep only meaningful rows
    df = df[df["ticker"].str.len() > 0]
    return df

catalysts = load_catalysts()

# -----------------------------
# QUOTES MERGE
# -----------------------------
all_tickers = sorted(set(positions["ticker"].dropna().tolist() + catalysts["ticker"].dropna().tolist()))
quote_map = {}
if use_live_quotes and all_tickers:
    with st.spinner("Fetching quotes..."):
        quote_map = fetch_quotes_yf(all_tickers)

def _effective_price(row):
    # If we have a live quote, use it; else price; else avg
    t = str(row.get("ticker", "")).upper()
    live = quote_map.get(t, np.nan)
    if np.isfinite(live) and live > 0:
        return float(live)
    p = row.get("price", np.nan)
    a = row.get("avg", np.nan)
    if np.isfinite(p) and p > 0:
        return float(p)
    if np.isfinite(a) and a > 0:
        return float(a)
    return np.nan

# -----------------------------
# EV MATH
# -----------------------------
def compute_ev_table(positions_df, catalysts_df):
    if catalysts_df.empty:
        return pd.DataFrame(columns=[
            "ticker","event","date","DTE","p_bull","p_base","p_bear",
            "t_bull","t_base","t_bear","confidence","price","EV_sh","EV_pct"
        ])
    df = catalysts_df.copy()

    # Attach price (from positions or quotes)
    pos_simple = positions_df[["ticker","price","avg"]].drop_duplicates()
    df = df.merge(pos_simple, on="ticker", how="left")
    df["price"] = df.apply(_effective_price, axis=1)
    df = _coerce_numeric(df, ["price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","confidence"])

    # Probability discipline & normalization (if user CSV doesn't sum to 1)
    probs = df[["p_bull","p_base","p_bear"]].fillna(0.0).values
    sums = probs.sum(axis=1)
    # If any sum is 0, fall back to priors
    priors = np.array([0.35, 0.45, 0.20])
    need_prior = (sums <= 0) | ~np.isfinite(sums)
    probs[need_prior, :] = priors
    # Normalize
    sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(sums == 0, 1.0, sums)
    df[["p_bull","p_base","p_bear"]] = probs

    # EV/sh = sum(prob * target)
    df["EV_sh"] = df["p_bull"]*df["t_bull"] + df["p_base"]*df["t_base"] + df["p_bear"]*df["t_bear"]

    # EV% relative to current effective price
    df["EV_pct"] = np.where(
        np.isfinite(df["price"]) & (df["price"] > 0),
        (df["EV_sh"] - df["price"]) / df["price"] * 100.0,
        np.nan
    )

    cols = ["ticker","event","date","DTE","p_bull","p_base","p_bear",
            "t_bull","t_base","t_bear","confidence","price","EV_sh","EV_pct"]
    return df[cols].sort_values(["DTE","EV_pct"], ascending=[True, False])

ev_table = compute_ev_table(positions, catalysts)

# -----------------------------
# DISPLAY — PORTFOLIO
# -----------------------------
c1, c2 = st.columns([1, 2], gap="large")

with c1:
    st.subheader("Positions")
    if positions.empty:
        st.info("No positions uploaded yet.")
    else:
        # Recompute effective display price
        pos = positions.copy()
        pos["price"] = pos.apply(_effective_price, axis=1)
        pos["mv"] = pos["qty"] * pos["price"]
        pos = _coerce_numeric(pos, ["qty","avg","price","mv"])
        st.dataframe(pos, use_container_width=True, hide_index=True)

with c2:
    st.subheader("Catalysts (EV computed)")
    if ev_table.empty:
        st.info("No catalyst data uploaded yet.")
    else:
        st.dataframe(ev_table, use_container_width=True, hide_index=True)

# -----------------------------
# SHORT-FUSE PANEL
# -----------------------------
st.markdown("---")
st.subheader("⚡ Short-Fuse (DTE ≤ 15 & (EV% ≥ threshold or Confidence ≥ threshold))")
if ev_table.empty:
    st.info("Upload catalyst CSV to view Short-Fuse ideas.")
else:
    sf = ev_table.copy()
    sf = sf[(sf["DTE"].fillna(9999) <= 15) & (
            (sf["EV_pct"].fillna(-1e9) >= ev_threshold) |
            (sf["confidence"].fillna(0.0) >= conf_threshold)
        )]
    if sf.empty:
        st.info("No items meet Short-Fuse criteria at the moment.")
    else:
        st.dataframe(sf, use_container_width=True, hide_index=True)

# -----------------------------
# SUGGESTED ALLOCATION (simple EV% rank within budget)
# -----------------------------
st.markdown("---")
st.subheader("Suggested Allocation (EV%-weighted, budget constrained)")

def suggest_alloc(ev_df, budget_eur, max_n=5):
    df = ev_df.copy()
    df = df[np.isfinite(df["EV_pct"]) & np.isfinite(df["price"]) & (df["price"] > 0)]
    if df.empty or budget_eur <= 0:
        return pd.DataFrame(columns=["ticker","price","EV_pct","suggest_shares","alloc_eur"])

    # Rank by EV% then by sooner DTE
    df = df.sort_values(["EV_pct","DTE"], ascending=[False, True]).head(max_n)

    # Normalize weights by EV% (positive only)
    df["w_raw"] = df["EV_pct"].clip(lower=0.0)
    if df["w_raw"].sum() <= 0:
        df["w"] = 1.0 / len(df)
    else:
        df["w"] = df["w_raw"] / df["w_raw"].sum()

    df["alloc_eur"] = df["w"] * budget_eur
    df["suggest_shares"] = np.floor(df["alloc_eur"] / df["price"]).astype(int)
    df["alloc_eur"] = df["suggest_shares"] * df["price"]

    out = df[["ticker","price","EV_pct","suggest_shares","alloc_eur"]]
    # Remove zeros
    out = out[out["suggest_shares"] > 0]
    return out

alloc = suggest_alloc(ev_table, budget, int(max_positions))
if alloc.empty:
    st.info("No allocation suggestion (check EV% & prices).")
else:
    st.dataframe(alloc, use_container_width=True, hide_index=True)
    st.caption("Tip: Use limit or bracket orders; tighten stops on leverage per your LCM rules.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
**Notes**
- EV/sh = p_bull·t_bull + p_base·t_base + p_bear·t_bear. EV% is relative to current effective price (live quote if enabled, else CSV price/avg).
- CSVs are flexible — the app auto-maps columns like `symbol→ticker`, `average_cost→avg`.  
- Live quotes are cached 5 minutes to keep things snappy. Disable them if first load is slow.
""")
