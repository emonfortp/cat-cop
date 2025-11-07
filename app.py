# app.py — Catalyst Copilot (robust boot + EV math + quotes optional + short-fuse + save/load + IBKR CSV)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
import io
import time

# --- Optional quotes ---
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

st.set_page_config(page_title="Catalyst Copilot", layout="wide")

# Safe "today in ET" (no timezone libs beyond stdlib)
try:
    TZ_ET = timezone(timedelta(hours=-5))
    TODAY_ET = datetime.now(TZ_ET).date()
except Exception:
    TODAY_ET = datetime.utcnow().date()

# ------------- helpers -------------
def _to_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (datetime, date)):
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return pd.NaT
    try:
        return pd.to_datetime(str(x)).date()
    except Exception:
        return pd.NaT

def _to_num(s, default=np.nan):
    try:
        v = float(s)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _normalize_probs_row(row):
    p_cols = ["p_bull", "p_base", "p_bear"]
    vals = []
    for c in p_cols:
        v = _to_num(row.get(c, np.nan), 0.0)
        if not np.isfinite(v) or v < 0:
            v = 0.0
        vals.append(v)
    s = sum(vals)
    if s <= 0:
        vals = [0.35, 0.45, 0.20]; s = 1.0
    vals = [v / s for v in vals]
    return dict(zip(p_cols, vals)), (abs(sum(vals) - 1.0) > 1e-6)

def _ev_price(row):
    price_now = _to_num(row.get("price"))
    tb = _to_num(row.get("target_bull"), price_now)
    tm = _to_num(row.get("target_base"), price_now)
    tr = _to_num(row.get("target_bear"), price_now)
    p = (row.get("p_bull", 0.35), row.get("p_base", 0.45), row.get("p_bear", 0.20))
    try:
        return float(p[0])*tb + float(p[1])*tm + float(p[2])*tr
    except Exception:
        return np.nan

def _ev_pct(row):
    cp = _to_num(row.get("price"))
    if not np.isfinite(cp) or cp <= 0:
        return np.nan
    evp = _ev_price(row)
    if not np.isfinite(evp):
        return np.nan
    return (evp / cp - 1.0) * 100.0

def _dte(row):
    d = row.get("date")
    if pd.isna(d):
        return np.nan
    dd = _to_date(d)
    if dd is pd.NaT:
        return np.nan
    return (dd - TODAY_ET).days

# ------------- templates -------------
TEMPLATE_WATCHLIST = pd.DataFrame([{
    "ticker":"ALT","event":"SABCS late-breaker window","date":"2025-12-10",
    "p_bull":0.40,"p_base":0.45,"p_bear":0.15,
    "target_bull":8.50,"target_base":6.00,"target_bear":3.80,
    "confidence":0.70
},{
    "ticker":"MIST","event":"FDA label update window","date":"2025-12-05",
    "p_bull":0.38,"p_base":0.47,"p_bear":0.15,
    "target_bull":3.60,"target_base":2.90,"target_bear":2.10,
    "confidence":0.65
}])

TEMPLATE_POSITIONS = pd.DataFrame([
    {"ticker":"ALT","qty":0,"avg":5.20,"price":np.nan},
    {"ticker":"MIST","qty":0,"avg":2.45,"price":np.nan},
])

def _download(df, fname, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=fname, mime="text/csv")

def _read_csv(up, name="CSV"):
    try:
        return pd.read_csv(up)
    except Exception as e:
        st.error(f"{name} read error: {e}")
        return pd.DataFrame()

# ------------- quotes (robust; per-ticker; cached 5m) -------------
@st.cache_data(ttl=300)
def fetch_quote_one(ticker: str) -> float:
    if not YF_AVAILABLE or not ticker:
        return np.nan
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not hist.empty and "Close" in hist:
            return float(pd.to_numeric(hist["Close"]).dropna().iloc[-1])
        # Fallback daily
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty and "Close" in hist:
            return float(pd.to_numeric(hist["Close"]).dropna().iloc[-1])
    except Exception:
        return np.nan
    return np.nan

def fetch_quotes_list(tickers):
    out = []
    for t in tickers:
        px = fetch_quote_one(t)
        out.append({"ticker": t, "price": px})
        # prevent API hammering on first boot
        time.sleep(0.05)
    return pd.DataFrame(out)

# ------------- standardizers -------------
def _std_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker","qty","avg","price","mv"])
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    for c in ["ticker","qty","avg"]:
        if c not in df.columns:
            df[c] = np.nan
    if "price" not in df.columns:
        df["price"] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["avg"] = pd.to_numeric(df["avg"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # If price is bad, fall back to avg
    bad = (~np.isfinite(df["price"])) | (df["price"] <= 0)
    df.loc[bad, "price"] = df.loc[bad, "avg"]
    df["mv"] = df["qty"] * df["price"]
    return df[["ticker","qty","avg","price","mv"]]

def _std_watchlist(df: pd.DataFrame):
    cols = ["ticker","event","date","p_bull","p_base","p_bear",
            "target_bull","target_base","target_bear","confidence",
            "price","dte","ev_pct"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols), []
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    for c in ["ticker","event","date","p_bull","p_base","p_bear",
              "target_bull","target_base","target_bear","confidence"]:
        if c not in df.columns:
            df[c] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["event"] = df["event"].astype(str).fillna("").str.strip()
    df["date"] = df["date"].apply(_to_date)
    for c in ["p_bull","p_base","p_bear","target_bull","target_base","target_bear","confidence","price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # normalize probs
    norm = df.apply(_normalize_probs_row, axis=1, result_type="expand")
    plist = norm.iloc[:,0].tolist()
    flags = norm.iloc[:,1].tolist()
    df["p_bull"] = [p.get("p_bull",0.35) for p in plist]
    df["p_base"] = [p.get("p_base",0.45) for p in plist]
    df["p_bear"] = [p.get("p_bear",0.20) for p in plist]
    # DTE
    df["dte"] = df.apply(_dte, axis=1)
    return df, flags

def _merge_prices(base: pd.DataFrame, live: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    # uploaded prices
    if uploaded is not None and not uploaded.empty:
        up = uploaded.copy()
        up.columns = [str(c).lower().strip() for c in up.columns]
        if "ticker" in up.columns and "price" in up.columns:
            up["ticker"] = up["ticker"].astype(str).str.upper().str.strip()
            up["price"]  = pd.to_numeric(up["price"], errors="coerce")
            df = df.merge(up[["ticker","price"]].rename(columns={"price":"price_uploaded"}), on="ticker", how="left")
        else:
            df["price_uploaded"] = np.nan
    else:
        df["price_uploaded"] = np.nan
    # live prices
    if live is not None and not live.empty and "ticker" in live.columns and "price" in live.columns:
        lv = live.copy()
        lv["ticker"] = lv["ticker"].astype(str).str.upper().str.strip()
        lv["price"]  = pd.to_numeric(lv["price"], errors="coerce")
        df = df.merge(lv.rename(columns={"price":"price_live"}), on="ticker", how="left")
    else:
        df["price_live"] = np.nan
    # choose best price
    chosen = []
    for _, r in df.iterrows():
        for k in ("price_uploaded","price_live","price","avg"):
            v = _to_num(r.get(k))
            if np.isfinite(v) and v > 0:
                chosen.append(v); break
        else:
            chosen.append(np.nan)
    df["price"] = pd.to_numeric(chosen, errors="coerce")
    return df

# ------------- sidebar -------------
st.sidebar.title("⚙️ Inputs & Settings")
st.sidebar.markdown("**Upload CSVs**")

pos_up = st.sidebar.file_uploader("Positions CSV (ticker, qty, avg, [price])", type=["csv"], key="pos_csv")
wl_up  = st.sidebar.file_uploader("Watchlist CSV (see template)", type=["csv"], key="wl_csv")
px_up  = st.sidebar.file_uploader("Optional Prices CSV (ticker, price)", type=["csv"], key="px_csv")

c1, c2 = st.sidebar.columns(2)
with c1: _download(TEMPLATE_POSITIONS, "positions_template.csv", "⬇️ Positions template")
with c2: _download(TEMPLATE_WATCHLIST, "watchlist_template.csv", "⬇️ Watchlist template")

st.sidebar.markdown("---")
live_default = False  # start disabled to ensure first boot always works
use_live = st.sidebar.checkbox("Enable live quotes (cache 5m)", value=live_default, disabled=not YF_AVAILABLE)
if not YF_AVAILABLE:
    st.sidebar.info("yfinance not available; quotes disabled.")

ev_floor = st.sidebar.slider("EV% display floor", min_value=-50, max_value=200, value=-10, step=5)
conf_floor = st.sidebar.slider("Confidence floor", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sizing Budget**")
budget = st.sidebar.number_input("Cash available", value=1000.0, min_value=0.0, step=100.0)

# ------------- load data -------------
pos_raw = _read_csv(pos_up, "Positions") if pos_up else pd.DataFrame()
wl_raw  = _read_csv(wl_up, "Watchlist") if wl_up else pd.DataFrame()
px_raw  = _read_csv(px_up, "Prices") if px_up else pd.DataFrame()

positions = _std_positions(pos_raw)
watchlist, flags = _std_watchlist(wl_raw)

# quotes
tickers = sorted({t for t in positions.get("ticker", pd.Series(dtype=str)).tolist() +
                       watchlist.get("ticker", pd.Series(dtype=str)).tolist()
                  if isinstance(t, str) and t})

live_prices = pd.DataFrame()
if use_live and len(tickers) > 0:
    live_prices = fetch_quotes_list(tickers)

# price merge
if not positions.empty:
    positions = _merge_prices(positions, live_prices, px_raw)
if not watchlist.empty:
    watchlist = _merge_prices(watchlist, live_prices, px_raw)
    watchlist["dte"] = watchlist.apply(_dte, axis=1)
    watchlist["ev_pct"] = watchlist.apply(_ev_pct, axis=1)
    if any(flags):
        st.warning("Some rows had probabilities normalized to 1.000.")

# ------------- UI -------------
st.title("🚀 Catalyst Copilot")

a, b, c = st.columns([2,2,1])
with a:
    st.subheader("Positions")
    if positions.empty:
        st.info("Upload positions to see MV and prices.")
    else:
        dfp = positions.copy()
        dfp["price"] = pd.to_numeric(dfp["price"], errors="coerce").round(4)
        dfp["mv"] = pd.to_numeric(dfp["mv"], errors="coerce").round(2)
        st.dataframe(dfp, use_container_width=True, height=240)

with b:
    st.subheader("Watchlist (EV math)")
    if watchlist.empty:
        st.info("Upload watchlist to compute EV%.")
    else:
        dfw = watchlist.copy()
        # filter floors
        dfw = dfw[
            (pd.to_numeric(dfw["ev_pct"], errors="coerce").fillna(-1e9) >= ev_floor) &
            (pd.to_numeric(dfw["confidence"], errors="coerce").fillna(0) >= conf_floor)
        ].copy()
        # pretty
        for ccol in ["p_bull","p_base","p_bear"]:
            dfw[ccol] = pd.to_numeric(dfw[ccol], errors="coerce").round(3)
        for ccol in ["target_bull","target_base","target_bear","price","ev_pct","confidence"]:
            dfw[ccol] = pd.to_numeric(dfw[ccol], errors="coerce")
        dfw["price"] = dfw["price"].round(4)
        dfw["ev_pct"] = dfw["ev_pct"].round(2)
        try:
            dfw["dte"] = dfw["dte"].astype("Int64")
        except Exception:
            pass
        st.dataframe(dfw, use_container_width=True, height=240)

with c:
    st.subheader("Today (ET)")
    st.metric("Date", str(TODAY_ET))
    st.caption("Quotes cache: 5 min (when enabled)")

st.markdown("---")

# Save/Load watchlist
st.subheader("💾 Save / Load Watchlist")
if watchlist.empty:
    st.info("Upload a watchlist to enable save/export.")
else:
    c11, c12 = st.columns(2)
    with c11:
        _download(watchlist, "watchlist_current.csv", "⬇️ Download current watchlist")
    with c12:
        saved = st.file_uploader("Load saved watchlist CSV", type=["csv"], key="saved_wl")
        if saved:
            wl_loaded = _read_csv(saved, "Saved Watchlist")
            if not wl_loaded.empty:
                wl_loaded, f2 = _std_watchlist(wl_loaded)
                wl_loaded = _merge_prices(wl_loaded, live_prices, px_raw)
                wl_loaded["dte"] = wl_loaded.apply(_dte, axis=1)
                wl_loaded["ev_pct"] = wl_loaded.apply(_ev_pct, axis=1)
                watchlist = wl_loaded.copy()
                st.success("Saved watchlist loaded & recalculated.")

st.markdown("---")

# Short-Fuse
st.subheader("⏱️ Short-Fuse (DTE ≤ 15)")
c21, c22 = st.columns(2)
with c21:
    sf_ev = st.number_input("Min EV% (Short-Fuse)", value=10.0, step=5.0)
with c22:
    sf_conf = st.number_input("Min confidence", value=0.60, step=0.05, min_value=0.0, max_value=1.0)

if watchlist.empty:
    st.info("Upload a watchlist to populate Short-Fuse.")
    sf = pd.DataFrame()
else:
    sf = watchlist.copy()
    sf["dte"] = pd.to_numeric(sf["dte"], errors="coerce")
    sf = sf[(sf["dte"].apply(lambda x: np.isfinite(x) and x <= 15)) &
            ((pd.to_numeric(sf["ev_pct"], errors="coerce") >= sf_ev) |
             (pd.to_numeric(sf["confidence"], errors="coerce") >= sf_conf))].copy()
    sf["rank_score"] = pd.to_numeric(sf["ev_pct"], errors="coerce").fillna(0) * \
                       pd.to_numeric(sf["confidence"], errors="coerce").fillna(0)
    sf = sf.sort_values(by=["rank_score","dte"], ascending=[False, True])
    for cc in ["ev_pct","confidence","price","target_bull","target_base","target_bear"]:
        sf[cc] = pd.to_numeric(sf[cc], errors="coerce")
    sf["ev_pct"] = sf["ev_pct"].round(2)
    sf["confidence"] = sf["confidence"].round(2)
    sf["price"] = sf["price"].round(4)
    try:
        sf["dte"] = sf["dte"].astype("Int64")
    except Exception:
        pass
    st.dataframe(
        sf[["ticker","event","date","dte","price","ev_pct","confidence",
            "p_bull","p_base","p_bear","target_bull","target_base","target_bear"]],
        use_container_width=True, height=280
    )
    if not sf.empty:
        _download(sf, "short_fuse_alerts.csv", "⬇️ Download Short-Fuse CSV")

st.markdown("---")

# Sizing
st.subheader("📐 Sizing Suggestions (EV% × Confidence)")
if watchlist.empty or budget <= 0:
    st.info("Upload a watchlist and set Budget > 0 to see suggestions.")
    sug = pd.DataFrame()
else:
    sug = watchlist.copy()
    evv = pd.to_numeric(sug["ev_pct"], errors="coerce").fillna(0).clip(lower=0)
    conf = pd.to_numeric(sug["confidence"], errors="coerce").fillna(0)
    sug["score"] = evv * conf
    if float(sug["score"].sum()) <= 0:
        st.info("No positive EV opportunities at current thresholds.")
    else:
        total = float(sug["score"].sum())
        sug["w"] = sug["score"] / total if total > 0 else 0.0
        sug["alloc"] = sug["w"] * float(budget)
        sug["px"] = pd.to_numeric(sug["price"], errors="coerce")
        sug["qty_suggested"] = np.floor(np.where(sug["px"] > 0, sug["alloc"]/sug["px"], 0)).astype(int)
        out = sug[["ticker","event","date","dte","price","ev_pct","confidence","w","alloc","qty_suggested"]].copy()
        out["w"] = (pd.to_numeric(out["w"], errors="coerce")*100).round(2)
        out["alloc"] = pd.to_numeric(out["alloc"], errors="coerce").round(2)
        out["price"] = pd.to_numeric(out["price"], errors="coerce").round(4)
        st.dataframe(out.sort_values("w", ascending=False), use_container_width=True, height=280)
        _download(out, "sizing_suggestions.csv", "⬇️ Export suggested orders")

st.markdown("---")

# IBKR export
st.subheader("🧺 IBKR Order CSV (Basket)")
with st.expander("Order parameters"):
    o1, o2, o3 = st.columns(3)
    with o1:
        order_action = st.selectbox("Action", ["BUY","SELL"], index=0)
        order_type = st.selectbox("OrderType", ["LMT","MKT","STP","STP LMT"], index=0)
        tif = st.selectbox("TIF", ["DAY","GTC"], index=0)
    with o2:
        qty_mode = st.selectbox("Quantity source", ["qty_suggested (from sizing)", "manual fixed qty"], index=0)
        fixed_qty = st.number_input("If manual, quantity per line", value=100, min_value=1, step=1)
    with o3:
        limit_method = st.selectbox("Limit method (for LMT/STP LMT)", [
            "current price",
            "target_base",
            "mid(tb, tm)",
            "price * (1 + 0.5*EV%)"
        ], index=0)
        slip_bps = st.number_input("Slippage (bps)", value=0, min_value=0, step=5)

def _limit_from_row(row, method):
    cp = _to_num(row.get("price"))
    tb = _to_num(row.get("target_bull"))
    tm = _to_num(row.get("target_base"))
    evp = _to_num(row.get("ev_pct"))
    if method == "current price":
        base = cp
    elif method == "target_base":
        base = tm if np.isfinite(tm) else cp
    elif method == "mid(tb, tm)":
        base = 0.5*(tb+tm) if np.isfinite(tb) and np.isfinite(tm) else cp
    else:
        base = cp * (1.0 + 0.5*evp/100.0) if np.isfinite(cp) and np.isfinite(evp) else cp
    return base

if watchlist.empty:
    st.info("Upload a watchlist to enable IBKR export.")
else:
    base = watchlist.copy()
    # attach sizing if available
    if "qty_suggested" in locals() and not sug.empty and "qty_suggested" in sug.columns:
        try:
            base = base.merge(sug[["ticker","qty_suggested"]], on="ticker", how="left")
        except Exception:
            base["qty_suggested"] = 0
    else:
        base["qty_suggested"] = 0

    if qty_mode.startswith("qty_suggested"):
        q = base["qty_suggested"].fillna(0).astype(int)
        if (q<=0).all():
            q = pd.Series([100]*len(base), index=base.index)
    else:
        q = pd.Series([int(fixed_qty)]*len(base), index=base.index)

    lim = base.apply(lambda r: _limit_from_row(r, limit_method), axis=1)
    lim = pd.to_numeric(lim, errors="coerce")
    if order_action == "BUY":
        lim = lim * (1.0 + float(slip_bps)/10000.0)
    else:
        lim = lim * (1.0 - float(slip_bps)/10000.0)

    ibkr = pd.DataFrame({
        "Symbol": base["ticker"].astype(str),
        "Action": order_action,
        "Quantity": q.clip(lower=1).astype(int),
        "OrderType": order_type,
        "LmtPrice": np.where(order_type in ("LMT","STP LMT"), pd.to_numeric(lim, errors="coerce").round(4), ""),
        "TIF": tif
    })
    st.dataframe(ibkr, use_container_width=True, height=260)
    _download(ibkr, "ibkr_orders.csv", "⬇️ Download IBKR Order CSV")

# footer
with st.expander("CSV Schemas / EV math"):
    st.markdown("""
**positions.csv** → `ticker, qty, avg, [price]`  
**watchlist.csv** → `ticker, event, date (YYYY-MM-DD), p_bull, p_base, p_bear, target_bull, target_base, target_bear, confidence`  
**prices.csv (optional)** → `ticker, price`  

EV% = ((p_bull·target_bull + p_base·target_base + p_bear·target_bear) / price − 1) × 100  
DTE = (event date − today ET). Quotes cache: 5 min (when enabled).
""")
