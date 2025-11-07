# app.py — Catalyst Copilot v0.4  (Momentum Engine Upgrade)
# ---------------------------------------------------------
# Features
# - Portfolio entry + positions table (CSV-ready)
# - Catalyst Cards -> EV% math + ranking
# - Momentum Engine: ΔEV% vs last scan (per ticker)
# - Reinvest Planner: allocates cash by (odds × EV%) and returns buy sizes
# - CSV export for ranked watchlist
#
# Notes
# - No external APIs; everything local & prompt-spec compliant
# - Safe probability handling and confidence gating
# - Session-persistent EV memory (clears when app restarts)

import math
from datetime import datetime, date
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------ #
# App config
# ------------------------------ #
st.set_page_config(
    page_title="Catalyst Copilot v0.4",
    page_icon="🧠",
    layout="wide",
)

# ------------------------------ #
# Session boot
# ------------------------------ #
def _boot_state():
    if "cash" not in st.session_state: st.session_state.cash = 0.0
    if "positions" not in st.session_state:
        st.session_state.positions = pd.DataFrame(
            columns=["ticker", "shares", "avg", "price"]
        )
    if "cards" not in st.session_state:
        st.session_state.cards = pd.DataFrame(
            columns=[
                "ticker", "type", "date",
                "t_bull", "t_base", "t_bear",
                "p_bull", "p_base", "p_bear",
                "news", "flow", "conf"
            ]
        )
    # Momentum memory: last EV% per ticker
    if "ev_last" not in st.session_state:
        st.session_state.ev_last: Dict[str, float] = {}

_boot_state()

# ------------------------------ #
# Helpers
# ------------------------------ #
CAT_TYPES = ["PDUFA", "CHMP", "Readout", "Earnings", "Policy", "Conf", "Other"]

def clamp(x, a, b): return max(a, min(b, x))

def prob_triplet(p_bull: float, p_bear: float):
    # Make sure numbers are sane and sum to ~1
    p_bull = clamp(p_bull, 0.0, 0.95)
    p_bear = clamp(p_bear, 0.0, 0.95)
    p_base = 1.0 - p_bull - p_bear
    if p_base < 0:  # renormalize evenly if overshoot
        total = p_bull + p_bear
        p_bull /= total
        p_bear /= total
        p_base = 0.0
    return round(p_bull, 4), round(p_base, 4), round(p_bear, 4)

def w_conf(conf, news, flow):
    # Simple confidence gate that respects prompt spec (0..1)
    # Conf gets 0.7 weight; news/flow 0.15 each
    return clamp(0.7*conf + 0.15*news + 0.15*flow, 0.0, 1.0)

def expected_return_pct(price, t_bull, t_base, t_bear, p_bull, p_base, p_bear, conf_w):
    # EV/sh = Σ p_i * (target/price - 1), then scale by confidence blend
    if price <= 0:
        return 0.0
    r_bull = (t_bull / price) - 1.0
    r_base = (t_base / price) - 1.0
    r_bear = (t_bear / price) - 1.0
    ev = p_bull * r_bull + p_base * r_base + p_bear * r_bear
    return float(ev * conf_w)

def calc_odds(p_bull, p_base, p_bear):
    # Light "MathOdds" proxy: bulls plus half the base vs bear
    return float(clamp(p_bull + 0.5 * p_base - p_bear, 0.0, 1.0))

def dte_from_str(s: str):
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
    except:
        try:
            d = datetime.strptime(s, "%Y/%m/%d").date()
        except:
            return None
    return (d - date.today()).days

def momentum_tag(ticker: str, ev_pct_now: float):
    prev = st.session_state.ev_last.get(ticker)
    st.session_state.ev_last[ticker] = ev_pct_now
    if prev is None:
        return "—", 0.0
    delta = ev_pct_now - prev
    if delta > 0.01:  # >1% EV improvement
        return "Up", delta
    if delta < -0.01:
        return "Down", delta
    return "Flat", delta

def fmt_pct(x):
    try: return f"{100*x:.2f}%"
    except: return "-"

def safe_float(x):
    try: return float(x)
    except: return 0.0

# ------------------------------ #
# Sidebar — Portfolio
# ------------------------------ #
with st.sidebar:
    st.subheader("Portfolio")
    cash_in = st.number_input("Cash (€)", value=float(st.session_state.cash), step=50.0, format="%.2f")
    if st.button("Save Cash", use_container_width=True):
        st.session_state.cash = float(cash_in)
        st.success("Cash saved")

    st.markdown("---")
    st.subheader("Add / Update Position")
    tkr = st.text_input("Ticker", placeholder="e.g., ALT").upper().strip()
    avg = st.number_input("Avg Cost", value=0.0, step=0.01, format="%.2f")
    shs = st.number_input("Shares", value=0, step=10)
    px  = st.number_input("Current Price", value=0.0, step=0.01, format="%.2f")
    if st.button("Add/Update Position", use_container_width=True):
        if tkr:
            df = st.session_state.positions.copy()
            if (df["ticker"] == tkr).any():
                df.loc[df["ticker"] == tkr, ["shares", "avg", "price"]] = [shs, avg, px]
            else:
                df = pd.concat([df, pd.DataFrame([{"ticker": tkr, "shares": shs, "avg": avg, "price": px}])], ignore_index=True)
            st.session_state.positions = df
            st.success(f"Saved position for {tkr}")
        else:
            st.warning("Please enter a ticker.")

    if st.button("Clear Positions", use_container_width=True):
        st.session_state.positions = st.session_state.positions.iloc[0:0]
        st.info("Positions cleared")

# ------------------------------ #
# Main — Header
# ------------------------------ #
st.title("🧠 Catalyst Copilot — v0.4")

# ------------------------------ #
# Catalyst Cards (input & add)
# ------------------------------ #
st.subheader("Catalyst Cards — add & rank")

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
with c1:
    cat_type = st.selectbox("Type", CAT_TYPES, index=0)
with c2:
    ticker_in = st.text_input("Ticker*", value="", placeholder="e.g., ALT").upper().strip()
with c3:
    event_date = st.text_input("Event Date*", value=date.today().strftime("%Y/%m/%d"))
with c4:
    conf = st.slider("Confidence (0–1)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)

c5, c6, c7, c8 = st.columns(4)
with c5:
    t_bull = st.number_input("Target Bull*", value=8.50, step=0.10, format="%.2f")
with c6:
    t_base = st.number_input("Target Base*", value=5.50, step=0.10, format="%.2f")
with c7:
    t_bear = st.number_input("Target Bear*", value=3.20, step=0.10, format="%.2f")
with c8:
    p_bear_raw = st.number_input("p_bear*", value=0.20, step=0.01, min_value=0.0, max_value=1.0)

c9, c10, c11 = st.columns(3)
with c9:
    news = st.number_input("News score (0–1)", value=0.00, step=0.05, min_value=0.0, max_value=1.0)
with c10:
    flow = st.number_input("Flow score (0–1)", value=0.00, step=0.05, min_value=0.0, max_value=1.0)
with c11:
    p_bull_raw = st.number_input("p_bull*", value=0.35, step=0.01, min_value=0.0, max_value=1.0)

# Derived probability triplet (sum to 1)
p_bull, p_base, p_bear = prob_triplet(p_bull_raw, p_bear_raw)
st.caption(f"Probabilities → bull **{p_bull:.2f}**, base **{p_base:.2f}**, bear **{p_bear:.2f}** (auto-normalized)")

if st.button("Add Card", type="primary"):
    if not ticker_in:
        st.warning("Please enter a ticker.")
    else:
        row = {
            "ticker": ticker_in,
            "type": cat_type,
            "date": event_date.replace(".", "/").replace("-", "/"),
            "t_bull": float(t_bull),
            "t_base": float(t_base),
            "t_bear": float(t_bear),
            "p_bull": float(p_bull),
            "p_base": float(p_base),
            "p_bear": float(p_bear),
            "news": float(news),
            "flow": float(flow),
            "conf": float(conf),
        }
        st.session_state.cards = pd.concat([st.session_state.cards, pd.DataFrame([row])], ignore_index=True)
        st.success(f"Added catalyst card for {ticker_in}")

# ------------------------------ #
# Compute EV / Ranking (with Momentum)
# ------------------------------ #
st.subheader("Catalyst Watchlist (ranked)")

def current_price_for(tkr: str) -> float:
    df = st.session_state.positions
    if df is not None and not df.empty:
        m = df[df["ticker"].str.upper() == tkr.upper()]
        if not m.empty:
            return float(m.iloc[0]["price"])
    # fallback if not in portfolio: require manual price in targets or set 1.0 to avoid zero-div
    return 1.0

def build_ranked_watchlist(cards: pd.DataFrame) -> pd.DataFrame:
    if cards.empty:
        return pd.DataFrame(columns=[
            "ticker","type","date","dte","price","t_bull","t_base","t_bear",
            "p_bull","p_base","p_bear","ev_pct","odds","momentum","Δev"
        ])

    rows = []
    for _, r in cards.iterrows():
        price = current_price_for(r["ticker"])
        conf_w = w_conf(r["conf"], r["news"], r["flow"])
        ev_pct = expected_return_pct(
            price, r["t_bull"], r["t_base"], r["t_bear"],
            r["p_bull"], r["p_base"], r["p_bear"], conf_w
        )
        odds = calc_odds(r["p_bull"], r["p_base"], r["p_bear"])
        dte = dte_from_str(str(r["date"]))
        mom_tag, delta = momentum_tag(r["ticker"], ev_pct)

        rows.append({
            "ticker": r["ticker"],
            "type": r["type"],
            "date": pd.to_datetime(str(r["date"]).replace("/", "-"), errors="coerce").date(),
            "dte": dte if dte is not None else np.nan,
            "price": price,
            "t_bull": r["t_bull"],
            "t_base": r["t_base"],
            "t_bear": r["t_bear"],
            "p_bull": r["p_bull"],
            "p_base": r["p_base"],
            "p_bear": r["p_bear"],
            "ev_pct": ev_pct,
            "odds": odds,
            "momentum": mom_tag,
            "Δev": delta
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["ev_pct", "dte"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return df

watch = build_ranked_watchlist(st.session_state.cards)

# Pretty table
if watch.empty:
    st.info("No catalyst cards yet.")
else:
    show = watch.copy()
    show["ev_pct"] = show["ev_pct"].map(fmt_pct)
    show["Δev"] = show["Δev"].apply(lambda x: f"{100*x:+.2f}%" if isinstance(x, (float,int)) else "—")
    st.dataframe(
        show[["ticker","type","date","dte","price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","ev_pct","odds","momentum","Δev"]],
        use_container_width=True,
        hide_index=True
    )

    # CSV export
    csv = watch.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download watchlist as CSV",
        data=csv,
        file_name=f"catalyst_watchlist_{date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# ------------------------------ #
# Portfolio EV & Reinvest Plan
# ------------------------------ #
st.subheader("📈 Portfolio — EV & Reinvest Plan")

pos = st.session_state.positions.copy()
if pos.empty:
    st.info("Add positions in the sidebar to see portfolio analytics.")
else:
    # Merge EV% by latest watchlist (if multiple cards per ticker, take max EV%)
    ev_map = watch.groupby("ticker")["ev_pct"].max().to_dict()
    pos["MktVal€"] = pos["shares"] * pos["price"]
    pos["EV_%"] = pos["ticker"].map(ev_map).fillna(0.0)
    pos["EV_€"] = pos["MktVal€"] * pos["EV_%"]

    mv = float(pos["MktVal€"].sum())
    ev_sum = float(pos["EV_€"].sum())
    cash = float(st.session_state.cash)

    cA, cB = st.columns([1.6, 1])
    with cA:
        st.dataframe(
            pos[["ticker","shares","avg","price","MktVal€","EV_%","EV_€"]],
            use_container_width=True
        )
    with cB:
        st.metric("Market Value (€)", f"{mv:,.2f}")
        st.metric("Sum EV (€)", f"{ev_sum:,.2f}")
        st.metric("Cash (€)", f"{cash:,.2f}")

# ------------------------------ #
# Reinvest Planner (lite auto-rebalance)
# ------------------------------ #
st.subheader("🧮 Reinvest Planner")

col1, col2 = st.columns([1, 3])
with col1:
    r_slider = st.slider("Reinvest rate r (0–100%)", min_value=0, max_value=100, value=60, step=5)
    budget = (r_slider/100.0) * float(st.session_state.cash)
    st.caption(f"Budget: **€{budget:,.2f}**")

with col2:
    # Candidate pool = top catalysts with positive EV and known price
    if watch.empty:
        st.info("Add catalyst cards to generate a reinvest plan.")
        alloc_df = pd.DataFrame(columns=["ticker","price","ev_pct","odds","alloc€","buy_shares"])
    else:
        cand = watch[watch["ev_pct"] > 0].copy()
        cand["score"] = cand["odds"] * cand["ev_pct"].clip(lower=0)
        if cand["score"].sum() <= 0 or budget <= 0:
            alloc_df = pd.DataFrame(columns=["ticker","price","ev_pct","odds","alloc€","buy_shares"])
        else:
            cand["w"] = cand["score"] / cand["score"].sum()
            cand["alloc€"] = cand["w"] * budget
            cand["buy_shares"] = np.floor(cand["alloc€"] / cand["price"])
            alloc_df = cand[["ticker","price","ev_pct","odds","alloc€","buy_shares"]].reset_index(drop=True)

    if alloc_df.empty:
        st.info("No positive-EV ideas or zero budget — nothing to allocate.")
    else:
        show_alloc = alloc_df.copy()
        show_alloc["ev_pct"] = show_alloc["ev_pct"].map(fmt_pct)
        st.dataframe(show_alloc, use_container_width=True, hide_index=True)

st.caption("v0.4 — Momentum Engine active. Next: order tickets + historical PBT tab.")
