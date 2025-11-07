# app.py — Catalyst Copilot v0.3 (CSV ingest + per-position EV + reinvest plan)
# Minimal deps: streamlit, pandas, numpy

import math
import io
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Catalyst Copilot", layout="wide")
st.title("⚗️ Catalyst Copilot — v0.3")

# -----------------------------
# Helpers
# -----------------------------
def clip(x, lo, hi):
    return max(lo, min(hi, x))

def dte(date_str: str) -> int:
    try:
        d = dt.datetime.fromisoformat(date_str.replace("/", "-"))
        return max((d.date() - dt.date.today()).days, 0)
    except Exception:
        return 0

def money(x): 
    try: return f"{x:,.2f}"
    except: return "-"

def pct(x):
    try: return f"{100*x:,.1f}%"
    except: return "-"

def normalize_ticker(t):
    return (t or "").strip().upper()

# -----------------------------
# Session state init
# -----------------------------
if "cash" not in st.session_state: st.session_state.cash = 0.0
if "positions" not in st.session_state: 
    # positions keyed by ticker
    st.session_state.positions: Dict[str, Dict] = {}
if "cards" not in st.session_state:
    st.session_state.cards: List[Dict] = []

# -----------------------------
# Left: Portfolio + CSV ingest
# -----------------------------
with st.sidebar:
    st.subheader("Portfolio")
    cash = st.number_input("Cash (€)", min_value=0.0, value=float(st.session_state.cash), step=50.0)
    if st.button("Save Cash"): 
        st.session_state.cash = cash
        st.success("Cash saved")

    st.markdown("---")
    st.subheader("Add / Update Position")
    tkr = st.text_input("Ticker", value="")
    avg = st.number_input("Avg Cost", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    sh  = st.number_input("Shares", min_value=0, value=0, step=1)
    px  = st.number_input("Current Price", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    if st.button("Add/Update Position"):
        T = normalize_ticker(tkr)
        if T and sh>0:
            st.session_state.positions[T] = {"ticker":T, "avg":avg, "shares":int(sh), "price":px}
            st.success(f"Position saved: {T}")
        else:
            st.warning("Please enter ticker and shares > 0")

    if st.button("Clear Positions"):
        st.session_state.positions.clear()
        st.info("Positions cleared")

    st.markdown("---")
    st.subheader("📥 Import from IBKR / broker CSV")
    """
    Accepted columns (any of):  
    - **Ticker/Symbol**, **Quantity/Shares/Position**, **Average Cost/Avg Price**, **Price/Mark Price/Close**  
    Export from **IBKR > PortfolioAnalyst > Positions (CSV)**.
    """
    up = st.file_uploader("Upload CSV", type=["csv"])

    def auto_map_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower().strip(): c for c in df.columns}
        # Try mappings
        sym = next((cols[k] for k in cols if k in ["symbol","ticker","underlying"]), None)
        qty = next((cols[k] for k in cols if k in ["quantity","shares","position"]), None)
        avgc= next((cols[k] for k in cols if k in ["average cost","avg price","costbasis","cost basis","avg cost"]), None)
        prc = next((cols[k] for k in cols if k in ["mark price","price","close","last price","market price"]), None)

        missing = [n for n,v in [("Ticker",sym),("Shares",qty),("Avg",avgc),("Price",prc)] if v is None]
        if missing: 
            st.warning(f"Missing columns in CSV: {', '.join(missing)}. Showing detected columns instead.")
        out = pd.DataFrame()
        if sym is not None:  out["ticker"] = df[sym].astype(str).map(normalize_ticker)
        if qty is not None:  out["shares"] = pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int)
        if avgc is not None: out["avg"]    = pd.to_numeric(df[avgc], errors="coerce").fillna(0.0)
        if prc is not None:  out["price"]  = pd.to_numeric(df[prc], errors="coerce").fillna(0.0)
        return out.dropna(subset=["ticker"]).query("ticker != ''")

    if up is not None:
        try:
            raw = pd.read_csv(up)
            mapped = auto_map_cols(raw)
            for _, r in mapped.iterrows():
                if r.get("shares",0) > 0:
                    st.session_state.positions[r["ticker"]] = {
                        "ticker": r["ticker"],
                        "avg": float(r.get("avg",0.0)),
                        "shares": int(r.get("shares",0)),
                        "price": float(r.get("price",0.0)),
                    }
            st.success(f"Imported {len(mapped)} positions.")
        except Exception as e:
            st.error(f"CSV parse error: {e}")

# -----------------------------
# Center: Catalyst Cards
# -----------------------------
st.markdown("### Catalyst Cards — add & rank")

col0, col1, col2, col3, col4, col5, col6 = st.columns([1.2,1,1,1,1,1,1])

with col0:
    cat_type = st.selectbox("Type", ["PDUFA","CHMP","Readout","Earnings","Policy","Conf","APD","Other"])
with col1:
    target_bull = st.number_input("Target Bull*", min_value=0.0, value=0.0, step=0.1)
with col2:
    target_base = st.number_input("Target Base*", min_value=0.0, value=0.0, step=0.1)
with col3:
    target_bear = st.number_input("Target Bear*", min_value=0.0, value=0.0, step=0.1)
with col4:
    p_bear = st.number_input("p_bear*", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
with col5:
    news = st.number_input("News score (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
with col6:
    flow = st.number_input("Flow score (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

c1, c2, c3 = st.columns(3)
with c1:
    tkr_in = st.text_input("Ticker*", value="")
with c2:
    date_in = st.date_input("Event Date*", value=dt.date.today() + dt.timedelta(days=10))
with c3:
    conf = st.slider("Confidence (0–1)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

p_bull = clip(1.0 - p_bear - 0.45, 0.05, 0.9)   # simple prior split
p_base = clip(1.0 - p_bear - p_bull, 0.05, 0.9)

if st.button("Add Card"):
    T = normalize_ticker(tkr_in)
    if not T:
        st.warning("Please enter ticker.")
    else:
        card = {
            "ticker": T,
            "type": cat_type,
            "date": str(date_in),
            "dte": dte(str(date_in)),
            "t_bull": float(target_bull),
            "t_base": float(target_base),
            "t_bear": float(target_bear),
            "p_bull": float(p_bull),
            "p_base": float(p_base),
            "p_bear": float(p_bear),
            "conf": float(conf),
            "news": float(news), 
            "flow": float(flow),
        }
        st.session_state.cards.append(card)
        st.success(f"Card added for {T}")

# -----------------------------
# Rank catalysts
# -----------------------------
cards_df = pd.DataFrame(st.session_state.cards)
if not cards_df.empty:
    # MathOdds (simple): weighted probs + recency & soft signal
    recency = np.exp(-cards_df["dte"]/30.0)
    odds = (0.5*cards_df["p_bull"] + 0.35*cards_df["p_base"] + 0.15*(1-cards_df["p_bear"])) \
           * (0.6*cards_df["conf"] + 0.2*cards_df["news"] + 0.2*cards_df["flow"]) \
           * (0.5 + 0.5*recency)
    # EV%
    # price may come from positions if we have it, else assume t_base for display
    prices = []
    for _, r in cards_df.iterrows():
        P = st.session_state.positions.get(r["ticker"], {})
        px = float(P.get("price", r["t_base"] or 0.0))
        prices.append(px if px>0 else 0.0)
    prices = np.array(prices)
    ev_sh = (cards_df["p_bull"]*cards_df["t_bull"] + cards_df["p_base"]*cards_df["t_base"] + cards_df["p_bear"]*cards_df["t_bear"]) - prices
    ev_pct = np.where(prices>0, ev_sh/prices, 0.0)

    cards_df = cards_df.assign(price=prices, ev_pct=ev_pct, odds=odds)
    cards_df = cards_df.sort_values(["odds","ev_pct"], ascending=False, ignore_index=True)

    st.markdown("#### Catalyst Watchlist (ranked)")
    st.dataframe(
        cards_df[["ticker","type","date","dte","price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","ev_pct","odds"]],
        use_container_width=True, height=320
    )

    csv_buf = io.StringIO()
    cards_df.to_csv(csv_buf, index=False)
    st.download_button("Download watchlist as CSV", data=csv_buf.getvalue(), file_name="watchlist.csv", mime="text/csv")
else:
    st.info("Add a catalyst card to see rankings.")

# -----------------------------
# Right: Portfolio EV + Reinvest plan
# -----------------------------
st.markdown("---")
st.markdown("### 📈 Portfolio — EV & Reinvest Plan")

# Build positions table with EV per position using best matching card per ticker
pos_df = pd.DataFrame(list(st.session_state.positions.values()))
if not pos_df.empty:
    # map card by ticker (best ranked)
    best = {}
    if not cards_df.empty:
        for t in cards_df["ticker"].unique():
            best[t] = cards_df[cards_df["ticker"]==t].iloc[0].to_dict()

    ev_perc_list, ev_abs_list = [], []
    for _, r in pos_df.iterrows():
        px = float(r.get("price",0.0))
        t = r["ticker"]
        if t in best and px>0:
            b = best[t]
            ev_sh = (b["p_bull"]*b["t_bull"] + b["p_base"]*b["t_base"] + b["p_bear"]*b["t_bear"]) - px
            ev_pct = ev_sh/px
        else:
            ev_pct = 0.0
            ev_sh = 0.0
        ev_perc_list.append(ev_pct)
        ev_abs_list.append(ev_sh * float(r.get("shares",0)))
    pos_df["EV_%"] = ev_perc_list
    pos_df["EV_€"] = ev_abs_list
    pos_df["MktVal€"] = pos_df["price"] * pos_df["shares"]

    left, right = st.columns([1.2,1])
    with left:
        st.dataframe(
            pos_df[["ticker","shares","avg","price","MktVal€","EV_%","EV_€"]],
            use_container_width=True, height=260
        )
    with right:
        total_mv = pos_df["MktVal€"].sum()
        total_ev = pos_df["EV_€"].sum()
        st.metric("Market Value (€)", money(total_mv))
        st.metric("Sum EV (€)", money(total_ev))
        st.metric("Cash (€)", money(st.session_state.cash))

    # Reinvest plan — allocate a fraction of cash to top catalysts not already held
    st.markdown("#### 🔁 Reinvest Planner")
    r = st.slider("Reinvest rate r (0–100%)", 0, 100, 60, step=5) / 100.0
    budget = st.session_state.cash * r
    st.write(f"Budget: **€{money(budget)}**")

    if not cards_df.empty and budget>0:
        # candidates = top 5 not held
        held = set(pos_df["ticker"].tolist())
        cands = cards_df[~cards_df["ticker"].isin(held)].head(5).copy()
        # naive sizing: proportional to odds*ev_pct, price-aware
        w = (cands["odds"].clip(1e-6) * (cands["ev_pct"].clip(lower=0)+1e-6))
        w = w / w.sum()
        target_euros = w * budget
        buy_shares = (target_euros / cands["price"]).fillna(0).astype(int).clip(lower=0)
        plan = cands[["ticker","price","ev_pct","odds"]].copy()
        plan["alloc€"] = target_euros.round(2)
        plan["buy_shares"] = buy_shares
        st.dataframe(plan, use_container_width=True)
        st.caption("Sizing = proportional to (odds × EV%) under cash budget; refine later with PRB.")
    else:
        st.info("Need budget > 0 and at least one catalyst not already held.")

else:
    st.info("No positions yet. Add manually or upload a broker CSV.")

st.caption("v0.3 — CSV ingest, per-position EV, reinvest planner. Next: order tickets & historical backtest.")
