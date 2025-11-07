# app.py — Catalyst Copilot (Quotes + Short-Fuse + Save/Load + IBKR Export)
# ------------------------------------------------------------------------
# What’s included
# - Upload: positions.csv, watchlist.csv, optional prices.csv
# - Live quotes via yfinance (5-min cache), toggleable
# - EV math (bull/base/bear), prob auto-normalization
# - Short-Fuse alerts (DTE ≤ 15) with EV% & confidence filters
# - Save/Load Watchlist (download/upload)
# - IBKR Order CSV export (Basket-style)
# - Robust guards + template CSVs

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
import io

# Optional live quotes
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

st.set_page_config(page_title="Catalyst Copilot", layout="wide")
TZ_ET = timezone(timedelta(hours=-5))
TODAY_ET = datetime.now(TZ_ET).date()

# -----------------------
# Helpers
# -----------------------
def _to_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (datetime, date)):
        return pd.to_datetime(x).date()
    try:
        return pd.to_datetime(str(x)).date()
    except Exception:
        return pd.NaT

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _normalize_probs(row):
    p_cols = ["p_bull", "p_base", "p_bear"]
    vals = [ _safe_float(row.get(c, np.nan), 0.0) for c in p_cols ]
    vals = [0.0 if not np.isfinite(v) else max(0.0, v) for v in vals]
    s = sum(vals)
    if s <= 0:
        vals = [0.35, 0.45, 0.20]; s = 1.0
    norm = [v/s for v in vals]
    return dict(zip(p_cols, norm)), (abs(sum(norm) - 1.0) > 1e-6)

def _compute_ev_price(row):
    cp = _safe_float(row.get("price"), np.nan)
    tb = _safe_float(row.get("target_bull"), cp)
    tm = _safe_float(row.get("target_base"), cp)
    tr = _safe_float(row.get("target_bear"), cp)
    p  = row["p_bull"], row["p_base"], row["p_bear"]
    return p[0]*tb + p[1]*tm + p[2]*tr

def _compute_ev_pct(row):
    cp = _safe_float(row.get("price"), np.nan)
    if not np.isfinite(cp) or cp <= 0:
        return np.nan
    ev_price = _compute_ev_price(row)
    return (ev_price / cp - 1.0) * 100.0

def _compute_dte(row):
    d = row.get("date")
    if pd.isna(d):
        return np.nan
    dd = _to_date(d)
    if dd is pd.NaT:
        return np.nan
    return (dd - TODAY_ET).days

# -----------------------
# CSV templates / loaders
# -----------------------
TEMPLATE_WATCHLIST = pd.DataFrame(
    [
        {
            "ticker":"ALT","event":"SABCS late-breaker window","date":"2025-12-10",
            "p_bull":0.40,"p_base":0.45,"p_bear":0.15,
            "target_bull":8.50,"target_base":6.00,"target_bear":3.80,
            "confidence":0.70
        },
        {
            "ticker":"MIST","event":"FDA label update window","date":"2025-12-05",
            "p_bull":0.38,"p_base":0.47,"p_bear":0.15,
            "target_bull":3.60,"target_base":2.90,"target_bear":2.10,
            "confidence":0.65
        },
    ]
)

TEMPLATE_POSITIONS = pd.DataFrame(
    [
        {"ticker":"ALT","qty":0,"avg":5.20,"price":np.nan},
        {"ticker":"MIST","qty":0,"avg":2.45,"price":np.nan},
    ]
)

def _download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def _read_csv(uploaded, name_hint=""):
    try:
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"{name_hint} CSV read error: {e}")
        return pd.DataFrame()

# -----------------------
# Quote fetcher (cache 5m)
# -----------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(tickers: list) -> pd.DataFrame:
    if not YF_AVAILABLE or len(tickers) == 0:
        return pd.DataFrame(columns=["ticker","price"])
    uniq = sorted(set([t.strip() for t in tickers if isinstance(t, str) and t.strip()]))
    data = []
    try:
        info = yf.download(tickers=uniq, period="1d", interval="1m",
                           group_by="ticker", threads=True, progress=False)
    except Exception:
        info = None
    if info is not None and not isinstance(info.columns, pd.MultiIndex):
        last = pd.to_numeric(info["Close"]).dropna()
        px = float(last.iloc[-1]) if len(last) else np.nan
        data.append({"ticker": uniq[0], "price": px})
    elif info is not None:
        for t in uniq:
            try:
                close = pd.to_numeric(info[(t, "Close")]).dropna()
                px = float(close.iloc[-1]) if len(close) else np.nan
            except Exception:
                px = np.nan
            data.append({"ticker": t, "price": px})
    else:
        for t in uniq:
            try:
                q = yf.Ticker(t).history(period="1d", interval="1m")
                px = float(q["Close"].dropna().iloc[-1]) if len(q) else np.nan
            except Exception:
                px = np.nan
            data.append({"ticker": t, "price": px})
    out = pd.DataFrame(data).drop_duplicates(subset=["ticker"])
    return out

# -----------------------
# Standardizers
# -----------------------
def _std_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker","qty","avg","price","mv"])
    df.columns = [c.lower() for c in df.columns]
    for c in ["ticker","qty","avg"]:
        if c not in df.columns:
            df[c] = np.nan
    if "price" not in df.columns:
        df["price"] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["qty"]    = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["avg"]    = pd.to_numeric(df["avg"], errors="coerce")
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")
    df["price"]  = np.where(~np.isfinite(df["price"]) | (df["price"]<=0), df["avg"], df["price"])
    df["mv"]     = df["qty"] * df["price"]
    return df[["ticker","qty","avg","price","mv"]]

def _std_watchlist_df(df: pd.DataFrame):
    cols = [
        "ticker","event","date",
        "p_bull","p_base","p_bear",
        "target_bull","target_base","target_bear",
        "confidence","price","dte","ev_pct"
    ]
    if df.empty:
        return pd.DataFrame(columns=cols), []
    df.columns = [c.lower().strip() for c in df.columns]
    needed = ["ticker","event","date","p_bull","p_base","p_bear",
              "target_bull","target_base","target_bear","confidence"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["event"]  = df["event"].astype(str).fillna("").str.strip()
    df["date"]   = df["date"].apply(_to_date)
    for c in ["p_bull","p_base","p_bear","target_bull","target_base","target_bear","confidence","price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # normalize probs
    norm_ps = df.apply(_normalize_probs, axis=1, result_type="expand")
    probs = norm_ps.iloc[:,0].tolist()
    flags = norm_ps.iloc[:,1].tolist()
    df["p_bull"] = [p["p_bull"] for p in probs]
    df["p_base"] = [p["p_base"] for p in probs]
    df["p_bear"] = [p["p_bear"] for p in probs]
    # DTE
    df["dte"] = df.apply(_compute_dte, axis=1)
    return df, flags

def _merge_prices(base_df: pd.DataFrame, live_df: pd.DataFrame, uploaded_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    if not uploaded_df.empty:
        up = uploaded_df.copy()
        up.columns = [c.lower().strip() for c in up.columns]
        if "ticker" in up.columns and "price" in up.columns:
            up["ticker"] = up["ticker"].astype(str).str.upper().str.strip()
            up["price"]  = pd.to_numeric(up["price"], errors="coerce")
            df = df.merge(up[["ticker","price"]].rename(columns={"price":"price_uploaded"}), on="ticker", how="left")
        else:
            df["price_uploaded"] = np.nan
    else:
        df["price_uploaded"] = np.nan
    if not live_df.empty:
        lv = live_df.copy()
        lv["ticker"] = lv["ticker"].astype(str).str.upper().str.strip()
        lv["price"]  = pd.to_numeric(lv["price"], errors="coerce")
        df = df.merge(lv.rename(columns={"price":"price_live"}), on="ticker", how="left")
    else:
        df["price_live"] = np.nan
    cur = []
    for _, r in df.iterrows():
        for key in ("price_uploaded","price_live","price"):
            v = r.get(key, np.nan)
            if np.isfinite(v) and v > 0:
                cur.append(v); break
        else:
            cur.append(np.nan)
    df["price"] = pd.to_numeric(cur, errors="coerce")
    return df

# -----------------------
# Sidebar: inputs
# -----------------------
st.sidebar.title("⚙️ Inputs & Settings")

st.sidebar.markdown("**1) Upload CSVs**")
pos_up = st.sidebar.file_uploader("Positions CSV (ticker, qty, avg, [price])", type=["csv"], key="pos_csv")
wl_up  = st.sidebar.file_uploader("Watchlist CSV (see template)", type=["csv"], key="wl_csv")
px_up  = st.sidebar.file_uploader("Optional Prices CSV (ticker, price)", type=["csv"], key="px_csv")

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    _download_button(TEMPLATE_POSITIONS, "positions_template.csv", "⬇️ Positions template")
with col_t2:
    _download_button(TEMPLATE_WATCHLIST, "watchlist_template.csv", "⬇️ Watchlist template")

st.sidebar.markdown("---")
use_live = st.sidebar.checkbox("Enable live quotes (yfinance, 5-min cache)", value=True and YF_AVAILABLE, disabled=not YF_AVAILABLE)
if not YF_AVAILABLE:
    st.sidebar.info("yfinance not available — install it to enable live quotes.")

ev_floor = st.sidebar.slider("EV% display floor", min_value=-50, max_value=200, value=-10, step=5)
conf_floor = st.sidebar.slider("Confidence floor", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Budget (for sizing suggestions)**")
budget = st.sidebar.number_input("Cash available", value=1000.0, min_value=0.0, step=100.0)

# -----------------------
# Load & standardize
# -----------------------
pos_df_raw = _read_csv(pos_up, "Positions") if pos_up else pd.DataFrame()
wl_df_raw  = _read_csv(wl_up, "Watchlist") if wl_up else pd.DataFrame()
px_df_raw  = _read_csv(px_up, "Prices") if px_up else pd.DataFrame()

positions = _std_positions_df(pos_df_raw)
watchlist, prob_flags = _std_watchlist_df(wl_df_raw)

tickers_for_quotes = sorted({t for t in positions["ticker"].tolist() + watchlist["ticker"].tolist() if isinstance(t, str) and t})
live_prices = fetch_live_prices(tickers_for_quotes) if use_live and tickers_for_quotes else pd.DataFrame()

if not positions.empty:
    positions = _merge_prices(positions, live_prices, px_df_raw)
if not watchlist.empty:
    watchlist = _merge_prices(watchlist, live_prices, px_df_raw)
    watchlist["dte"] = watchlist.apply(_compute_dte, axis=1)
    watchlist["ev_pct"] = watchlist.apply(_compute_ev_pct, axis=1)
    if any(prob_flags):
        st.warning("Some watchlist rows had probabilities normalized to sum 1.000.")

# -----------------------
# Main layout
# -----------------------
st.title("🚀 Catalyst Copilot")

top1, top2, top3 = st.columns([2,2,1])
with top1:
    st.subheader("Positions")
    if positions.empty:
        st.info("Upload positions to see MV and live prices.")
    else:
        disp_pos = positions.copy()
        disp_pos["price"] = disp_pos["price"].round(4)
        disp_pos["mv"] = disp_pos["mv"].round(2)
        st.dataframe(disp_pos, use_container_width=True, height=240)

with top2:
    st.subheader("Watchlist (EV math)")
    if watchlist.empty:
        st.info("Upload watchlist to compute EV%.")
    else:
        disp_wl = watchlist.copy()
        m = (disp_wl["ev_pct"] >= ev_floor) & (disp_wl["confidence"].fillna(0) >= conf_floor)
        disp_wl = disp_wl[m].copy()
        for c in ["p_bull","p_base","p_bear"]:
            disp_wl[c] = pd.to_numeric(disp_wl[c], errors="coerce").round(3)
        for c in ["target_bull","target_base","target_bear","price","ev_pct","confidence"]:
            disp_wl[c] = pd.to_numeric(disp_wl[c], errors="coerce")
        disp_wl["ev_pct"] = disp_wl["ev_pct"].round(2)
        disp_wl["price"] = disp_wl["price"].round(4)
        disp_wl["dte"] = disp_wl["dte"].astype("Int64")
        st.dataframe(disp_wl, use_container_width=True, height=240)

with top3:
    st.subheader("Today (ET)")
    st.metric("Date", str(TODAY_ET))
    st.caption("Quotes cache: 5 min.")

st.markdown("---")

# -----------------------
# Save / Load Watchlist
# -----------------------
st.subheader("💾 Save / Load Watchlist (post-calc)")
if watchlist.empty:
    st.info("Upload a watchlist to enable save/export.")
else:
    col_s1, col_s2, col_s3 = st.columns([1,1,2])
    with col_s1:
        _download_button(watchlist, "watchlist_current.csv", "⬇️ Download current watchlist")
    with col_s2:
        saved_up = st.file_uploader("Load saved watchlist CSV", type=["csv"], key="saved_wl")
        if saved_up:
            wl_loaded = _read_csv(saved_up, "Saved Watchlist")
            if not wl_loaded.empty:
                # Re-run standardize + EV compute so it stays consistent
                wl_loaded, flags2 = _std_watchlist_df(wl_loaded)
                wl_loaded = _merge_prices(wl_loaded, live_prices, px_df_raw)
                wl_loaded["dte"] = wl_loaded.apply(_compute_dte, axis=1)
                wl_loaded["ev_pct"] = wl_loaded.apply(_compute_ev_pct, axis=1)
                watchlist = wl_loaded.copy()
                st.success("Saved watchlist loaded and recalculated.")

st.markdown("---")

# -----------------------
# Short-Fuse Alerts
# -----------------------
st.subheader("⏱️ Short-Fuse Alerts (DTE ≤ 15)")
col_sf1, col_sf2, col_sf3 = st.columns([1,1,4])
with col_sf1:
    sf_ev = st.number_input("Min EV% (Short-Fuse)", value=10.0, step=5.0)
with col_sf2:
    sf_conf = st.number_input("Min confidence", value=0.60, step=0.05, min_value=0.0, max_value=1.0)

if watchlist.empty:
    st.info("Upload a watchlist to populate Short-Fuse.")
    sf = pd.DataFrame()
else:
    sf = watchlist.copy()
    sf = sf[(sf["dte"].apply(lambda x: np.isfinite(x) and x<=15)) &
            ((sf["ev_pct"]>=sf_ev) | (sf["confidence"]>=sf_conf))].copy()
    sf["rank_score"] = (sf["ev_pct"].fillna(0.0)) * (sf["confidence"].fillna(0.0))
    sf = sf.sort_values(by=["rank_score","dte"], ascending=[False, True])
    for c in ["ev_pct","confidence","price","target_bull","target_base","target_bear"]:
        sf[c] = pd.to_numeric(sf[c], errors="coerce")
    sf["ev_pct"] = sf["ev_pct"].round(2)
    sf["confidence"] = sf["confidence"].round(2)
    sf["price"] = sf["price"].round(4)
    sf["dte"] = sf["dte"].astype("Int64")
    st.dataframe(sf[["ticker","event","date","dte","price","ev_pct","confidence",
                     "p_bull","p_base","p_bear","target_bull","target_base","target_bear"]],
                 use_container_width=True, height=280)

    if not sf.empty:
        _download_button(sf, "short_fuse_alerts.csv", "⬇️ Download Short-Fuse CSV")

# -----------------------
# Sizing suggestions
# -----------------------
st.markdown("---")
st.subheader("📐 Sizing Suggestions (EV% × Confidence)")
if watchlist.empty or budget <= 0:
    st.info("Upload a watchlist and set a positive budget to see suggestions.")
    sug = pd.DataFrame()
else:
    sug = watchlist.copy()
    sug["score"] = sug["ev_pct"].clip(lower=0).fillna(0) * sug["confidence"].fillna(0)
    if sug["score"].sum() <= 0:
        st.info("No positive EV opportunities at current thresholds.")
    else:
        sug["w"] = sug["score"] / sug["score"].sum()
        sug["alloc"] = sug["w"] * budget
        sug["px"] = pd.to_numeric(sug["price"], errors="coerce")
        sug["qty_suggested"] = np.floor(np.where(sug["px"]>0, sug["alloc"]/sug["px"], 0)).astype(int)
        out = sug[["ticker","event","date","dte","price","ev_pct","confidence","w","alloc","qty_suggested"]].copy()
        out["w"] = (out["w"]*100).round(2)
        out["alloc"] = out["alloc"].round(2)
        out["price"] = out["price"].round(4)
        st.dataframe(out.sort_values("w", ascending=False), use_container_width=True, height=280)
        _download_button(out, "sizing_suggestions.csv", "⬇️ Export suggested orders")

# -----------------------
# IBKR Order CSV Export
# -----------------------
st.markdown("---")
st.subheader("🧺 IBKR Order CSV (Basket-style)")
st.caption("Creates a broker-ready CSV with one row per ticker. You can import it into IBKR BasketTrader / TWS via 'File → Import Basket'.")
with st.expander("Order parameters"):
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        order_action = st.selectbox("Action", ["BUY","SELL"], index=0)
        order_type = st.selectbox("OrderType", ["LMT","MKT","STP","STP LMT"], index=0)
        tif = st.selectbox("TIF", ["DAY","GTC"], index=0)
    with col_o2:
        qty_source = st.selectbox("Quantity source", ["qty_suggested (from sizing)", "manual fixed qty"], index=0)
        fixed_qty = st.number_input("If manual, quantity per line", value=100, min_value=1, step=1)
    with col_o3:
        limit_method = st.selectbox("Limit price method (for LMT/STP LMT)", [
            "current price",
            "target_base",
            "mid(target_base, target_bull) = 0.5*(tb+tm)",
            "price * (1 + 0.5*EV%)"
        ], index=0)
        slip_bps = st.number_input("Add slippage (bps, e.g. 25 = 0.25%)", value=0, min_value=0, step=5)

def _calc_limit_price(row, method):
    cp = _safe_float(row.get("price"), np.nan)
    tb = _safe_float(row.get("target_bull"), np.nan)
    tm = _safe_float(row.get("target_base"), np.nan)
    evp = _safe_float(row.get("ev_pct"), np.nan)
    if method == "current price":
        base = cp
    elif method == "target_base":
        base = tm if np.isfinite(tm) else cp
    elif method == "mid(target_base, target_bull) = 0.5*(tb+tm)":
        if np.isfinite(tb) and np.isfinite(tm):
            base = 0.5*(tb+tm)
        else:
            base = cp
    else:  # price * (1 + 0.5*EV%)
        if np.isfinite(cp) and np.isfinite(evp):
            base = cp * (1.0 + 0.5*evp/100.0)
        else:
            base = cp
    return base

if not watchlist.empty:
    # Base table for export comes from sizing or watchlist with 1-share default
    export_df = watchlist.copy()
    # attach sizing if available
    if not sug.empty and "qty_suggested" in sug.columns:
        export_df = export_df.merge(sug[["ticker","qty_suggested"]], on="ticker", how="left")
    export_df["qty_suggested"] = export_df["qty_suggested"].fillna(0).astype(int)
    # decide qty
    if qty_source.startswith("qty_suggested"):
        qvec = export_df["qty_suggested"].copy()
        # guard: if all zeros, default to 100
        if (qvec<=0).all():
            qvec = pd.Series([100]*len(export_df), index=export_df.index)
    else:
        qvec = pd.Series([int(fixed_qty)]*len(export_df), index=export_df.index)
    export_df["Quantity"] = qvec.clip(lower=1)

    # compute limit
    lim = export_df.apply(lambda r: _calc_limit_price(r, limit_method), axis=1)
    lim = pd.to_numeric(lim, errors="coerce")
    # apply slippage in bps (positive increases limit price for buys)
    if order_action == "BUY":
        lim = lim * (1.0 + (float(slip_bps)/10000.0))
    else:
        lim = lim * (1.0 - (float(slip_bps)/10000.0))
    export_df["LmtPrice"] = np.where(np.isfinite(lim), lim, np.nan)

    # Assemble IBKR basket fields (minimal viable set)
    ibkr = pd.DataFrame({
        "Symbol": export_df["ticker"].astype(str),
        "Action": order_action,
        "Quantity": export_df["Quantity"].astype(int),
        "OrderType": order_type,
        "LmtPrice": np.where(order_type in ("LMT","STP LMT"), export_df["LmtPrice"].round(4), ""),
        "TIF": tif,
        # Optional columns you can add later: "Exchange","Currency","AuxPrice"(for STP), etc.
    })

    st.dataframe(ibkr, use_container_width=True, height=260)
    _download_button(ibkr, "ibkr_orders.csv", "⬇️ Download IBKR Order CSV")
else:
    st.info("Upload a watchlist to enable IBKR export.")

# -----------------------
# Footer notes
# -----------------------
with st.expander("Notes / Columns"):
    st.markdown("""
**positions.csv** → `ticker, qty, avg, [price]`  
**watchlist.csv** → `ticker, event, date (YYYY-MM-DD), p_bull, p_base, p_bear, target_bull, target_base, target_bear, confidence`  
**prices.csv (optional)** → `ticker, price`

- Probabilities are auto-normalized to **1.000** if your row doesn’t.
- **EV% = ((p_bull*target_bull + p_base*target_base + p_bear*target_bear) / price − 1) × 100**.
- **DTE** = (event date − today ET). Live quotes cache is **5 minutes**.
- **IBKR CSV**: minimal Basket fields → Symbol, Action, Quantity, OrderType, LmtPrice (if LMT), TIF.
""")
