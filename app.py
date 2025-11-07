# app.py
# Cat-Cop: Catalyst Portfolio Optimizer (Streamlit)
# - Robust CSV ingestion (positions + catalysts)
# - Cached quotes
# - EV math from bull/base/bear
# - Alerts / Short-Fuse panel (DTE ≤ 15 & EV% threshold)
# - Greedy allocator with numeric-safety fixes
# - Fully self-contained (no fancy env vars required)

import os
import io
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional quotes; we guard imports & failures gracefully
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


# --------------------------- Streamlit Page ---------------------------

st.set_page_config(page_title="Cat-Cop", page_icon="🧠", layout="wide")
st.title("🧠 Catalyst Portfolio Optimizer")

st.caption(
    "v4.6.1 • Positions + Catalyst EV • Cached Quotes • Alerts/Short-Fuse (≤15d) • "
    "Greedy Allocator (numeric-safe) • This app never executes trades."
)


# --------------------------- Helpers ---------------------------

@st.cache_data(show_spinner=False, ttl=900)
def _yf_price_one(ticker: str) -> Optional[float]:
    """Fetch latest price via yfinance; return None on failure."""
    if not YF_OK or not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        px = t.fast_info.last_price if hasattr(t, "fast_info") else None
        if px is None or not np.isfinite(px) or px <= 0:
            hist = t.history(period="1d")
            if hist is not None and len(hist) > 0:
                px = float(hist["Close"].iloc[-1])
        if px is None or not np.isfinite(px) or px <= 0:
            return None
        return float(px)
    except Exception:
        return None


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _to_num(series, default=None):
    out = pd.to_numeric(series, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


# --------------------------- Positions Ingest ---------------------------

def _std_positions_df(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Standardize positions CSV to:
      ['ticker','qty','avg','price'] where price falls back to avg if missing/invalid.
    Accepts a wide range of broker exports:
      - columns we try: ticker/symbol, quantity/qty/shares, avg/avgcost/cost, price/last/close
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ticker", "qty", "avg", "price"])

    df = _norm_cols(raw)

    # Map likely headers
    col_map = {}
    # ticker
    for c in ["ticker", "symbol", "underlying", "security"]:
        if c in df.columns:
            col_map["ticker"] = c
            break
    # qty
    for c in ["qty", "quantity", "shares", "position"]:
        if c in df.columns:
            col_map["qty"] = c
            break
    # avg cost
    for c in ["avg", "avgcost", "averagecost", "cost_basis", "cost"]:
        if c in df.columns:
            col_map["avg"] = c
            break
    # price
    for c in ["price", "last", "close", "mark"]:
        if c in df.columns:
            col_map["price"] = c
            break

    # Create missing columns
    for need in ["ticker", "qty", "avg", "price"]:
        if need not in col_map:
            df[need] = np.nan

    out = pd.DataFrame({
        "ticker": df[col_map.get("ticker", "ticker")] if col_map.get("ticker") else df["ticker"],
        "qty": _to_num(df[col_map.get("qty", "qty")] if col_map.get("qty") else df["qty"], default=0),
        "avg": _to_num(df[col_map.get("avg", "avg")] if col_map.get("avg") else df["avg"]),
        "price": _to_num(df[col_map.get("price", "price")] if col_map.get("price") else df["price"]),
    })

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    # If price invalid → fall back to avg
    out["price"] = np.where(~np.isfinite(out["price"]) | (out["price"] <= 0), out["avg"], out["price"])
    # Prune empties
    out = out[(out["ticker"] != "") & out["qty"].fillna(0).astype(float).gt(0)]
    out.reset_index(drop=True, inplace=True)
    return out


# --------------------------- Catalyst / Watchlist Ingest ---------------------------

def _std_watchlist_df(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Standardize catalyst watchlist to the fields we need:
      ['ticker','event_date','bull_target','base_target','bear_target','p_bull','p_base','p_bear','confidence']
    If probs missing, default priors {0.35, 0.45, 0.20} and normalize to 1.
    """
    cols_needed = [
        "ticker", "event_date",
        "bull_target", "base_target", "bear_target",
        "p_bull", "p_base", "p_bear",
        "confidence"
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols_needed)

    df = _norm_cols(raw)

    # Ensure columns exist
    for c in cols_needed:
        if c not in df.columns:
            df[c] = np.nan

    # Clean
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    # Dates
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date

    # Numerics
    for c in ["bull_target", "base_target", "bear_target", "p_bull", "p_base", "p_bear", "confidence"]:
        df[c] = _to_num(df[c])

    # Defaults for probabilities if missing
    df["p_bull"] = df["p_bull"].fillna(0.35)
    df["p_base"] = df["p_base"].fillna(0.45)
    df["p_bear"] = df["p_bear"].fillna(0.20)
    # Normalize to 1
    s = df["p_bull"] + df["p_base"] + df["p_bear"]
    s = s.replace(0, np.nan)
    df["p_bull"] = df["p_bull"] / s
    df["p_base"] = df["p_base"] / s
    df["p_bear"] = df["p_bear"] / s

    # Confidence default 0.60
    df["confidence"] = df["confidence"].fillna(0.60)

    # Drop rows without ticker or targets
    must = ["bull_target", "base_target", "bear_target"]
    ok_mask = (df["ticker"] != "")
    for c in must:
        ok_mask &= pd.notna(df[c])
    df = df.loc[ok_mask].copy()

    # DTE
    today = date.today()
    df["DTE"] = (pd.to_datetime(df["event_date"]) - pd.to_datetime(today)).dt.days

    # Keep essentials
    order = ["ticker", "event_date", "DTE",
             "bull_target", "base_target", "bear_target",
             "p_bull", "p_base", "p_bear", "confidence"]
    df = df[order].reset_index(drop=True)
    return df


# --------------------------- EV Math ---------------------------

def _ensure_prices(df: pd.DataFrame, use_quotes: bool = True) -> pd.DataFrame:
    """
    Guarantee there is a 'price' column for EV calc:
      - if 'price' exists and valid, keep it
      - else fetch via yfinance (cached), or leave NaN if not available
    """
    df = df.copy()
    if "price" not in df.columns:
        df["price"] = np.nan

    need_price = ~pd.notna(df["price"]) | (df["price"] <= 0)
    if use_quotes:
        for i in df[need_price].index:
            tk = str(df.at[i, "ticker"]).upper().strip()
            px = _yf_price_one(tk)
            if px is not None:
                df.at[i, "price"] = float(px)
    return df


def _calc_ev_table(pos_df: pd.DataFrame, wl_df: pd.DataFrame, use_quotes: bool = True) -> pd.DataFrame:
    """
    Join positions with watchlist by ticker (outer), fetch prices if needed, then compute:
      EV_price = p_bull*Bull + p_base*Base + p_bear*Bear
      EV_%     = (EV_price - price) / price * 100
    Returns EV table per ticker.
    """
    # Unique tickers from either side
    tickers = pd.unique(pd.concat([
        pos_df["ticker"] if "ticker" in pos_df.columns else pd.Series(dtype=str),
        wl_df["ticker"] if "ticker" in wl_df.columns else pd.Series(dtype=str)
    ], ignore_index=True)).tolist()

    # Build base frame
    base = pd.DataFrame({"ticker": tickers})
    # Attach any current price from positions (if there)
    if "price" in pos_df.columns:
        px_map = pos_df.dropna(subset=["ticker", "price"]).drop_duplicates("ticker").set_index("ticker")["price"]
        base = base.join(px_map, on="ticker")
    else:
        base["price"] = np.nan

    # Attach scenario & probs from watchlist
    merge_cols = ["ticker", "event_date", "DTE",
                  "bull_target", "base_target", "bear_target",
                  "p_bull", "p_base", "p_bear", "confidence"]
    base = base.merge(wl_df[merge_cols], on="ticker", how="left")

    # Ensure prices
    base = _ensure_prices(base, use_quotes=use_quotes)

    # EV math (only where price valid)
    base["EV_price"] = np.nan
    valid_mask = pd.notna(base["price"]) & (base["price"] > 0)
    base.loc[valid_mask, "EV_price"] = (
        base.loc[valid_mask, "p_bull"] * base.loc[valid_mask, "bull_target"] +
        base.loc[valid_mask, "p_base"] * base.loc[valid_mask, "base_target"] +
        base.loc[valid_mask, "p_bear"] * base.loc[valid_mask, "bear_target"]
    )
    base["EV_pct"] = (base["EV_price"] - base["price"]) / base["price"] * 100.0
    # Niceties
    base["DTE"] = base["DTE"].fillna(999999).astype(int)
    # Order
    cols = ["ticker", "price", "EV_price", "EV_pct", "event_date", "DTE",
            "bull_target", "base_target", "bear_target",
            "p_bull", "p_base", "p_bear", "confidence"]
    base = base[cols].sort_values(["DTE", "EV_pct"], ascending=[True, False]).reset_index(drop=True)
    return base


# --------------------------- EV Table Standardizer (allocator uses this) ---------------------------

def _std_ev_df(ev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize EV table for allocator:
      - ticker, price, EV_pct
      - coerce numerics safely
    """
    if ev_df is None or ev_df.empty:
        return pd.DataFrame(columns=["ticker", "price", "EV_pct"])

    df = ev_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for need in ["ticker", "price", "ev_pct"]:
        if need not in df.columns:
            df[need] = np.nan

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["ev_pct"] = pd.to_numeric(df["ev_pct"], errors="coerce")

    df = df[["ticker", "price", "ev_pct"]]
    return df


# --------------------------- Allocator (numeric-safe) ---------------------------

def suggest_alloc(ev_df: pd.DataFrame, budget_eur: float, max_n: int = 5) -> pd.DataFrame:
    """
    Greedy equal-euro allocator for the best EV_pct tickers.
    Returns: ticker | price | EV_pct | suggest_shares | alloc_eur
    """
    if ev_df is None or ev_df.empty or budget_eur is None or budget_eur <= 0:
        return pd.DataFrame(columns=["ticker", "price", "EV_pct", "suggest_shares", "alloc_eur"])

    df = _std_ev_df(ev_df)

    valid_mask = (
        pd.notna(df["price"]) & (df["price"] > 0) &
        pd.notna(df["ev_pct"])
    )
    df = df.loc[valid_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "price", "EV_pct", "suggest_shares", "alloc_eur"])

    df.sort_values("ev_pct", ascending=False, inplace=True)
    df = df.head(int(max_n)).copy()

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["ticker", "price", "EV_pct", "suggest_shares", "alloc_eur"])

    per_name_eur = float(budget_eur) / n
    df["suggest_shares"] = np.floor(per_name_eur / df["price"]).astype(int)
    df["alloc_eur"] = df["suggest_shares"] * df["price"]
    df = df[df["suggest_shares"] > 0].copy()

    df.rename(columns={"ev_pct": "EV_pct"}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------- Sidebar: Inputs ---------------------------

st.sidebar.header("⚙️ Inputs")

with st.sidebar.expander("💼 Upload Positions CSV", expanded=True):
    st.write("Expected fields (flexible): **ticker/symbol**, **qty/quantity**, **avg/avgcost**, optional **price**.")
    f_pos = st.file_uploader("Positions CSV", type=["csv"], key="pos_up")
    if f_pos is not None:
        try:
            pos_raw = pd.read_csv(f_pos)
            st.session_state.positions = _std_positions_df(pos_raw)
            st.success(f"Loaded {len(st.session_state.positions)} positions.")
        except Exception as e:
            st.error(f"Positions CSV error: {e}")

with st.sidebar.expander("🗓️ Upload Catalyst Watchlist CSV", expanded=True):
    st.write("Columns: **ticker,event_date,bull_target,base_target,bear_target,p_bull,p_base,p_bear,confidence**.")
    f_wl = st.file_uploader("Catalyst Watchlist CSV", type=["csv"], key="wl_up")
    if f_wl is not None:
        try:
            wl_raw = pd.read_csv(f_wl)
            st.session_state.watchlist = _std_watchlist_df(wl_raw)
            st.success(f"Loaded {len(st.session_state.watchlist)} catalyst rows.")
        except Exception as e:
            st.error(f"Watchlist CSV error: {e}")

with st.sidebar.expander("💶 Budget & Filters", expanded=True):
    budget = st.number_input("Available budget (EUR)", min_value=0.0, value=1000.0, step=100.0)
    max_positions = st.number_input("Max names to allocate", min_value=1, value=5, step=1)
    use_quotes = st.toggle("Fetch quotes (cache 15 min)", value=True)
    ev_alert_threshold = st.slider("Alerts: EV% ≥", min_value=-50.0, max_value=300.0, value=20.0, step=5.0)
    dte_limit = st.slider("Alerts: DTE ≤ days", min_value=1, max_value=60, value=15, step=1)

# Initialize session
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=["ticker", "qty", "avg", "price"])
if "watchlist" not in st.session_state:
    st.session_state.watchlist = pd.DataFrame(columns=[
        "ticker", "event_date", "DTE",
        "bull_target", "base_target", "bear_target",
        "p_bull", "p_base", "p_bear", "confidence"
    ])


# --------------------------- Main Panels ---------------------------

left, right = st.columns([1.2, 1])

with left:
    st.subheader("📊 Positions (standardized)")
    st.dataframe(st.session_state.positions, use_container_width=True)

    st.subheader("🧭 Catalyst Watchlist (standardized)")
    st.dataframe(st.session_state.watchlist, use_container_width=True)

    st.subheader("🧮 EV Table")
    try:
        ev_table = _calc_ev_table(
            st.session_state.positions.copy(),
            st.session_state.watchlist.copy(),
            use_quotes=use_quotes
        )
        st.dataframe(ev_table, use_container_width=True)
        st.caption(f"EV dtypes: {dict(ev_table.dtypes)}")
    except Exception as e:
        st.error(f"EV calc error: {e}")
        ev_table = pd.DataFrame(columns=["ticker", "price", "EV_price", "EV_pct", "event_date", "DTE"])

with right:
    st.subheader("⏰ Alerts / Short-Fuse (DTE ≤ threshold)")
    try:
        alerts = ev_table.copy()
        alerts = alerts[
            pd.notna(alerts["EV_pct"]) &
            pd.notna(alerts["DTE"]) &
            (alerts["DTE"] <= int(dte_limit)) &
            (alerts["EV_pct"] >= float(ev_alert_threshold))
        ].sort_values(["DTE", "EV_pct"], ascending=[True, False])
        if alerts.empty:
            st.info("No alert meets current thresholds.")
        else:
            st.dataframe(alerts[["ticker","event_date","DTE","price","EV_price","EV_pct","confidence"]],
                         use_container_width=True)
    except Exception as e:
        st.error(f"Alerts error: {e}")

    st.subheader("🧩 Suggested Allocation")
    try:
        # Normalize EV table before allocation (prevents dtype issues)
        ev_table_std = _std_ev_df(ev_table)
        alloc = suggest_alloc(ev_table_std, budget, int(max_positions))
        if alloc.empty:
            st.info("No allocation suggestion (check EV% & prices).")
        else:
            st.dataframe(alloc, use_container_width=True)
            tot_alloc = float(alloc["alloc_eur"].sum())
            st.metric("Allocated (EUR)", f"{tot_alloc:,.2f}")
    except Exception as e:
        st.error(f"Allocator error: {e}")


# --------------------------- Sample CSV Downloaders ---------------------------

st.divider()
st.subheader("📥 Sample CSV Templates")

sample_pos = pd.DataFrame({
    "ticker": ["KURA", "ALT", "IOVA"],
    "qty": [50, 65, 30],
    "avg": [12.10, 4.30, 42.00],
    "price": [np.nan, np.nan, np.nan]
})

sample_wl = pd.DataFrame({
    "ticker": ["ALT", "KURA", "IOVA"],
    "event_date": [(date.today()+timedelta(days=7)).isoformat(),
                   (date.today()+timedelta(days=2)).isoformat(),
                   (date.today()+timedelta(days=14)).isoformat()],
    "bull_target": [8.50, 15.00, 58.00],
    "base_target": [6.00, 13.00, 48.00],
    "bear_target": [3.80, 10.50, 38.00],
    "p_bull": [0.30, 0.35, 0.30],
    "p_base": [0.50, 0.50, 0.50],
    "p_bear": [0.20, 0.15, 0.20],
    "confidence": [0.65, 0.60, 0.60]
})

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download positions_template.csv",
        data=sample_pos.to_csv(index=False).encode("utf-8"),
        file_name="positions_template.csv",
        mime="text/csv"
    )
with c2:
    st.download_button(
        "⬇️ Download catalyst_watchlist_template.csv",
        data=sample_wl.to_csv(index=False).encode("utf-8"),
        file_name="catalyst_watchlist_template.csv",
        mime="text/csv"
    )

st.caption("Tip: if your broker export has different headers, upload it anyway — the normalizer maps common variants automatically.")


# --------------------------- Footer ---------------------------

st.divider()
st.caption(
    "Educational only. Not investment advice. Quotes via yfinance (cached 15 min) when enabled; "
    "otherwise EV uses provided/positions prices."
)
