# app.py — Catalyst Copilot v0.5 (unified, hardened, EV-complete)
# Cards → Watchlist (ranked) → EV math → Portfolio EV → Reinvest planner
# Uploads: positions.csv and watchlist.csv. All states guarded.

from __future__ import annotations
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

st.set_page_config(page_title="Catalyst Copilot v0.5", layout="wide")

# =============== Utilities ===============

EXPECTED_CARD_COLS = [
    "ticker","type","date","price",
    "t_bull","t_base","t_bear",
    "p_bull","p_base","p_bear",
    "news","flow","conf"
]

def _to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        s = str(x).strip().replace("€", "").replace(",", "")
        if s == "":
            return default
        return float(s)
    except:
        return default

def _clip01(x, default=0.0):
    try:
        return max(0.0, min(1.0, float(x)))
    except:
        return default

def _parse_date(s):
    if isinstance(s, (datetime, date)):
        return pd.to_datetime(s).date()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except:
            pass
    return None

def _norm_probs_row(r):
    # Use provided probs if valid; otherwise default priors and renormalize.
    pb = _to_float(r.get("p_bull"))
    pm = _to_float(r.get("p_base"))
    pr = _to_float(r.get("p_bear"))
    if any(map(lambda v: v is np.nan or v is None, [pb, pm, pr])):
        pb, pm, pr = 0.35, 0.45, 0.20
    s = pb + pm + pr
    if s <= 0 or not np.isfinite(s):
        pb, pm, pr, s = 0.35, 0.45, 0.20, 1.0
    if abs(s - 1.0) > 1e-9:
        pb, pm, pr = pb/s, pm/s, pr/s
    r["p_bull"], r["p_base"], r["p_bear"] = pb, pm, pr
    return r

def _ev_from_row(r):
    price = _to_float(r.get("price"))
    t_bull = _to_float(r.get("t_bull"))
    t_base = _to_float(r.get("t_base"))
    t_bear = _to_float(r.get("t_bear"))
    if not all(np.isfinite(v) for v in [price, t_bull, t_base, t_bear]):
        return np.nan, np.nan
    ev_sh = (r["p_bull"] * (t_bull - price)
           + r["p_base"] * (t_base - price)
           + r["p_bear"] * (t_bear - price))
    ev_pct = (ev_sh / price) * 100 if price else np.nan
    return ev_sh, ev_pct

def _odds_row(r):
    price = _to_float(r.get("price"))
    t_bull = _to_float(r.get("t_bull"))
    t_bear = _to_float(r.get("t_bear"))
    p_bull = _to_float(r.get("p_bull"), 0.35)
    p_bear = _to_float(r.get("p_bear"), 0.20)
    if not all(np.isfinite(v) for v in [price, t_bull, t_bear, p_bull, p_bear]):
        return 0.0
    up = max(t_bull - price, 0.0) * p_bull
    down = max(price - t_bear, 0.0) * p_bear
    if down <= 0:
        return 99.0
    return float(np.clip(up / (down + 1e-9), 0, 99))

def _download(df: pd.DataFrame, label: str, fname: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=fname, mime="text/csv")

def _std_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    # Accepts many header variants and returns standard columns.
    rename_map = {
        "Ticker":"ticker", "symbol":"ticker", "SYMBOL":"ticker",
        "Shares":"shares","qty":"shares","quantity":"shares",
        "Avg":"avg","avg_cost":"avg","average_cost":"avg",
        "Price":"price","last":"price","mark":"price","current_price":"price",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    for col in ["ticker","shares","avg","price"]:
        if col not in df.columns:
            df[col] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for col in ["shares","avg","price"]:
        df[col] = df[col].apply(_to_float)
    df = df[["ticker","shares","avg","price"]].copy()
    df = df.dropna(subset=["ticker"])
    df["shares"] = df["shares"].fillna(0).astype(int)
    df["price"] = np.where(~np.isfinite(df["price"]) | (df["price"]<=0), df["avg"], df["price"])
    df["avg"] = df["avg"].fillna(0.0)
    df["price"] = df["price"].fillna(0.0)
    return df

# =============== Session state ===============

if "cards" not in st.session_state:
    st.session_state.cards: list[dict] = []
if "cash" not in st.session_state:
    st.session_state.cash = 0.0
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=["ticker","shares","avg","price"])

# =============== Sidebar (Portfolio I/O) ===============

with st.sidebar:
    st.header("Portfolio")
    cash_in = st.number_input("Cash (€)", value=float(st.session_state.cash), step=50.0)
    if st.button("Save Cash"):
        st.session_state.cash = float(cash_in)
        st.success("Cash saved")

    st.markdown("---")
    st.subheader("Add / Update Position")
    p_ticker = st.text_input("Ticker").strip().upper()
    p_avg = st.number_input("Avg Cost", value=0.0, step=0.01, format="%.2f")
    p_shares = st.number_input("Shares", value=0, step=1)
    p_price = st.number_input("Current Price", value=0.0, step=0.01, format="%.2f")
    if st.button("Add/Update Position"):
        if p_ticker:
            df = st.session_state.positions.copy()
            df = _std_positions_df(df)
            mask = df["ticker"] == p_ticker
            if mask.any():
                df.loc[mask, ["shares","avg","price"]] = [int(p_shares), float(p_avg), float(p_price)]
            else:
                df = pd.concat([df, pd.DataFrame([{
                    "ticker": p_ticker, "shares": int(p_shares),
                    "avg": float(p_avg), "price": float(p_price)
                }])], ignore_index=True)
            st.session_state.positions = _std_positions_df(df)
            st.success(f"Saved {p_ticker}")
        else:
            st.warning("Enter a ticker.")

    if st.button("Clear Positions"):
        st.session_state.positions = pd.DataFrame(columns=["ticker","shares","avg","price"])
        st.info("Positions cleared.")

    st.markdown("**Upload positions CSV** (flexible headers)")
    up_pos = st.file_uploader("Upload positions.csv", type=["csv"], key="pos_csv")
    if up_pos is not None:
        try:
            pdf = pd.read_csv(up_pos)
        except Exception:
            up_pos.seek(0)
            pdf = pd.read_csv(io.StringIO(up_pos.getvalue().decode("utf-8", errors="ignore")))
        st.session_state.positions = _std_positions_df(pdf)
        st.success("Positions loaded.")
    if not st.session_state.positions.empty:
        _download(st.session_state.positions, "Download positions.csv", "positions.csv")

# =============== Header ===============

st.title("🧠 Catalyst Copilot — v0.5")
st.caption("Cards → Watchlist (ranked) → EV math → Portfolio EV → Reinvest planner | Uploads supported")

# =============== Cards (entry) ===============

st.header("Catalyst Cards — add & rank")

c1, c2, c3, c4 = st.columns(4)
with c1:
    sel_type = st.selectbox("Type", ["PDUFA","Readout","CHMP","Earnings","Policy","Conf"], index=0)
with c2:
    t_bull = st.number_input("Target Bull*", value=8.50, step=0.10, format="%.2f")
with c3:
    t_base = st.number_input("Target Base*", value=5.50, step=0.10, format="%.2f")
with c4:
    t_bear = st.number_input("Target Bear*", value=3.20, step=0.10, format="%.2f")

d1, d2, d3, d4 = st.columns(4)
with d1:
    price_in = st.number_input("Current Price (optional)", value=0.00, step=0.01, format="%.2f")
with d2:
    p_bear = st.number_input("p_bear*", value=0.20, step=0.05, min_value=0.0, max_value=1.0)
with d3:
    news = st.number_input("News score (0–1)", value=0.0, step=0.05, min_value=0.0, max_value=1.0)
with d4:
    flow = st.number_input("Flow score (0–1)", value=0.0, step=0.05, min_value=0.0, max_value=1.0)

e1, e2 = st.columns([2,2])
with e1:
    ticker_in = st.text_input("Ticker*", value="ALT").strip().upper()
with e2:
    event_date = st.text_input("Event Date*", value=date.today().strftime("%Y/%m/%d"))

conf = st.slider("Confidence (0–1)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)

add_col1, add_col2 = st.columns([1,3])
with add_col1:
    if st.button("Add Card", type="primary"):
        d = _parse_date(event_date)
        if not ticker_in or not d:
            st.warning("Enter a valid ticker and date.")
        else:
            # Derive p_base from p_bear and prior p_bull=0.35 (kept consistent with prompt).
            p_base = max(0.0, 1.0 - float(p_bear) - 0.35)
            # Fill price from positions if not provided
            fallback = np.nan
            if price_in and price_in > 0:
                fallback = price_in
            else:
                # Try map from positions
                pos = _std_positions_df(st.session_state.positions.copy())
                match = pos[pos["ticker"] == ticker_in]
                if not match.empty:
                    fallback = _to_float(match.iloc[0]["price"])
            card = dict(
                ticker=ticker_in, type=sel_type, date=str(d),
                price=_to_float(fallback),
                t_bull=_to_float(t_bull), t_base=_to_float(t_base), t_bear=_to_float(t_bear),
                p_bull=0.35, p_base=p_base, p_bear=_clip01(p_bear),
                news=_clip01(news), flow=_clip01(flow), conf=_clip01(conf)
            )
            st.session_state.cards.append(card)
            st.success(f"Card added for {ticker_in}")

with add_col2:
    st.markdown("**Upload watchlist.csv (optional)** — columns accepted:")
    st.code(", ".join(EXPECTED_CARD_COLS))
    up_wl = st.file_uploader("Upload watchlist.csv", type=["csv"], key="wl_csv")
    if up_wl is not None:
        try:
            wdf = pd.read_csv(up_wl)
        except Exception:
            up_wl.seek(0)
            wdf = pd.read_csv(io.StringIO(up_wl.getvalue().decode("utf-8", errors="ignore")))
        # Standardize & append
        wdf = wdf.rename(columns=str.lower)
        # Ensure all expected cols exist:
        for c in EXPECTED_CARD_COLS:
            if c not in wdf.columns:
                wdf[c] = np.nan
        # Clean types
        wdf["ticker"] = wdf["ticker"].astype(str).str.upper()
        for c in ["price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","news","flow","conf"]:
            wdf[c] = wdf[c].apply(_to_float if c not in ["news","flow","conf"] else _clip01)
        wdf["date"] = wdf["date"].astype(str)
        # Extend session cards
        st.session_state.cards.extend(wdf[EXPECTED_CARD_COLS].to_dict(orient="records"))
        st.success("Watchlist appended.")

# =============== Build Watchlist + EV ===============

wl = pd.DataFrame(st.session_state.cards)
for c in EXPECTED_CARD_COLS:
    if c not in wl.columns:
        wl[c] = np.nan

if not wl.empty:
    wl["ticker"] = wl["ticker"].astype(str).str.upper()
    wl = wl.apply(_norm_probs_row, axis=1)

    # DTE
    wl["dte"] = wl["date"].apply(_parse_date).apply(
        lambda d: (pd.to_datetime(d) - pd.Timestamp.today().normalize()).days if d else np.nan
    )

    # EV & odds
    ev_sh, ev_pct, odds = [], [], []
    for _, r in wl.iterrows():
        es, ep = _ev_from_row(r)
        ev_sh.append(es); ev_pct.append(ep); odds.append(_odds_row(r))
    wl["ev_sh"], wl["ev_pct"], wl["odds"] = ev_sh, ev_pct, odds

    # Rank: EV%, near-term preference, confidence/news/flow light boost
    wl["rank_score"]  = wl["ev_pct"].fillna(-1e9)
    wl["rank_score"] += (100 - wl["dte"].fillna(9999))/1000.0
    wl["rank_score"] += (wl["conf"].fillna(0)*0.5 + wl["news"].fillna(0)*0.25 + wl["flow"].fillna(0)*0.25)

    wl = wl.sort_values(["rank_score"], ascending=False).reset_index(drop=True)

st.subheader("Catalyst Watchlist (ranked)")
show_cols = ["ticker","type","date","dte","price","t_bull","t_base","t_bear",
             "p_bull","p_base","p_bear","conf","news","flow","ev_sh","ev_pct","odds"]
if wl.empty:
    st.info("No cards yet. Add a card or upload a watchlist.")
    st.dataframe(pd.DataFrame(columns=show_cols), use_container_width=True, height=220)
else:
    st.dataframe(wl[show_cols], use_container_width=True, height=280)
    _download(wl[show_cols], "Download watchlist_with_ev.csv", "watchlist_ev.csv")

st.markdown("---")

# =============== Portfolio EV view ===============

st.header("📊 Portfolio — EV & Reinvest Plan")

pos = _std_positions_df(st.session_state.positions.copy())
if pos.empty:
    st.info("No positions yet. Add in the sidebar or upload a CSV.")
else:
    # Map best EV per ticker to positions
    ev_map = {}
    if not wl.empty:
        best = wl.sort_values(["ticker","rank_score"], ascending=[True, False]).drop_duplicates("ticker")
        ev_map = {r["ticker"]: r["ev_pct"] for _, r in best.iterrows()}
    pos["EV_pct"] = pos["ticker"].map(ev_map).fillna(0.0)
    pos["MktVal_EUR"] = pos["shares"] * pos["price"]
    pos["EV_EUR"] = pos["MktVal_EUR"] * pos["EV_pct"] / 100.0

    st.dataframe(pos[["ticker","shares","avg","price","MktVal_EUR","EV_pct","EV_EUR"]],
                 use_container_width=True, height=240)

    mv = pos["MktVal_EUR"].sum()
    sum_ev = pos["EV_EUR"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Value (€)", f"{mv:,.2f}")
    c2.metric("Sum EV (€)", f"{sum_ev:,.2f}")
    c3.metric("Cash (€)", f"{st.session_state.cash:,.2f}")

# =============== Reinvest Planner ===============

st.subheader("🔁 Reinvest Planner")
reinvr = st.slider("Reinvest rate r (0–100%)", 0, 100, value=60, step=5)
default_budget = round(st.session_state.cash * reinvr / 100.0, 2)
budget = st.number_input("Budget (€)", value=float(default_budget), step=50.0)

st.caption("Sizing ∝ max(0, odds × EV%) under your budget. Uses watchlist EV signals.")

alloc_tbl = pd.DataFrame(columns=["ticker","price","ev_pct","odds","alloc_EUR","buy_shares"])
if budget > 0 and not wl.empty:
    cands = wl.copy()
    cands = cands[cands["ev_pct"].fillna(-1) > 0]
    if not cands.empty:
        score = (cands["ev_pct"].clip(lower=0.0) * cands["odds"].clip(lower=0.0))
        if score.sum() > 0:
            w = score / score.sum()
            alloc = w * budget
            pr = cands["price"].apply(_to_float).replace(0, np.nan)
            buy = np.floor(alloc / pr).fillna(0).astype(int)
            alloc_tbl = pd.DataFrame({
                "ticker": cands["ticker"].values,
                "price": cands["price"].values,
                "ev_pct": cands["ev_pct"].values,
                "odds": cands["odds"].values,
                "alloc_EUR": alloc.values,
                "buy_shares": buy.values
            }).sort_values("alloc_EUR", ascending=False).reset_index(drop=True)

st.dataframe(alloc_tbl, use_container_width=True, height=260)
if not alloc_tbl.empty:
    _download(alloc_tbl, "Download reinvest_plan.csv", "reinvest_plan.csv")

st.markdown("---")
st.caption("v0.5 — unified & hardened. Empty-state safe; EV math complete; CSV upload/download for positions & watchlist; reinvest planner ready.")
