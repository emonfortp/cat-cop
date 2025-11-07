# app.py — Catalyst Copilot v0.4.1 (KeyError-hardened, ASCII-safe)
# Cards → Watchlist (ranked) → EV math → Portfolio EV → Reinvest planner

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

st.set_page_config(page_title="Catalyst Copilot v0.4.1", layout="wide")

# ---------- Helpers ----------
def _to_float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(str(x).replace(",", "").replace("€", ""))
    except:
        return default

def _clip01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except:
        return 0.0

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
    p = pd.Series(
        [
            _to_float(r.get("p_bull"), np.nan),
            _to_float(r.get("p_base"), np.nan),
            _to_float(r.get("p_bear"), np.nan),
        ],
        index=["p_bull", "p_base", "p_bear"],
    )
    if p.isna().any():
        p = pd.Series([0.35, 0.45, 0.20], index=["p_bull", "p_base", "p_bear"])
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        p = pd.Series([0.35, 0.45, 0.20], index=["p_bull", "p_base", "p_bear"])
        s = 1.0
    if abs(s - 1.0) > 1e-6:
        p = p / s
    r["p_bull"], r["p_base"], r["p_bear"] = p.values
    return r

def _ev_from_row(r):
    price = _to_float(r.get("price"))
    t_bull = _to_float(r.get("t_bull"))
    t_base = _to_float(r.get("t_base"))
    t_bear = _to_float(r.get("t_bear"))
    if any(map(lambda x: (x is None) or (not np.isfinite(x)), [price, t_bull, t_base, t_bear])):
        return np.nan, np.nan
    ev_sh = (r["p_bull"] * (t_bull - price)
           + r["p_base"] * (t_base - price)
           + r["p_bear"] * (t_bear - price))
    ev_pct = (ev_sh / price) * 100 if price and np.isfinite(price) else np.nan
    return ev_sh, ev_pct

def _odds_row(r):
    up = max(_to_float(r.get("t_bull")) - _to_float(r.get("price")), 0.0) * r.get("p_bull", 0.35)
    down = max(_to_float(r.get("price")) - _to_float(r.get("t_bear")), 0.0) * r.get("p_bear", 0.20)
    if down <= 0:
        return 99.0
    return float(np.clip(up / (down + 1e-9), 0, 99))

def _download_csv_button(df, label, fname):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=fname, mime="text/csv")

# ---------- Session State ----------
if "cards" not in st.session_state:
    st.session_state.cards = []  # list of dicts
if "cash" not in st.session_state:
    st.session_state.cash = 0.0
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=["ticker", "shares", "avg", "price"])

# Ensure cards dataframe has expected columns, even when empty
EXPECTED_CARD_COLS = ["ticker","type","date","price",
                      "t_bull","t_base","t_bear",
                      "p_bull","p_base","p_bear",
                      "news","flow","conf"]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Portfolio")
    cash = st.number_input("Cash (€)", value=float(st.session_state.cash), step=50.0)
    if st.button("Save Cash"):
        st.session_state.cash = float(cash)
        st.success("Cash saved")

    st.markdown("---")
    st.subheader("Add / Update Position")
    pt = st.text_input("Ticker").strip().upper()
    pav = st.number_input("Avg Cost", value=0.0, step=0.01, format="%.2f")
    psh = st.number_input("Shares", value=0, step=1)
    pcp = st.number_input("Current Price", value=0.0, step=0.01, format="%.2f")
    if st.button("Add/Update Position"):
        if pt and psh >= 0:
            df = st.session_state.positions.copy()
            for col in ["ticker","shares","avg","price"]:
                if col not in df.columns:
                    df[col] = pd.Series(dtype=float if col != "ticker" else str)
            mask = df["ticker"] == pt
            if mask.any():
                df.loc[mask, ["shares", "avg", "price"]] = [psh, pav, pcp]
            else:
                df = pd.concat(
                    [df, pd.DataFrame([{"ticker": pt, "shares": psh, "avg": pav, "price": pcp}])],
                    ignore_index=True,
                )
            st.session_state.positions = df
            st.success(f"Saved {pt}")
        else:
            st.warning("Enter valid ticker and shares.")

    if st.button("Clear Positions"):
        st.session_state.positions = pd.DataFrame(columns=["ticker", "shares", "avg", "price"])
        st.info("Positions cleared.")

# ---------- Header ----------
st.title("🧠 Catalyst Copilot — v0.4.1")
st.caption("Cards → Watchlist (ranked) → EV math → Portfolio EV → Reinvest planner (hardened for empty states).")

# ---------- Cards ----------
st.header("Catalyst Cards — add & rank")

c1, c2, c3, c4 = st.columns(4)
with c1:
    sel_type = st.selectbox("Type", ["PDUFA", "Readout", "CHMP", "Earnings", "Policy", "Conf"], index=0)
with c2:
    t_bull = st.number_input("Target Bull*", value=8.50, step=0.10, format="%.2f")
with c3:
    t_base = st.number_input("Target Base*", value=5.50, step=0.10, format="%.2f")
with c4:
    t_bear = st.number_input("Target Bear*", value=3.20, step=0.10, format="%.2f")

c5, c6, c7, c8 = st.columns(4)
with c5:
    p_bear = st.number_input("p_bear*", value=0.20, step=0.05, min_value=0.0, max_value=1.0)
with c6:
    news = st.number_input("News score (0–1)", value=0.0, step=0.05, min_value=0.0, max_value=1.0)
with c7:
    flow = st.number_input("Flow score (0–1)", value=0.0, step=0.05, min_value=0.0, max_value=1.0)
with c8:
    conf = st.slider("Confidence (0–1)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)

c9, c10 = st.columns([2, 2])
with c9:
    ticker = st.text_input("Ticker*", value="ALT").strip().upper()
with c10:
    ev_date = st.text_input("Event Date*", value=date.today().strftime("%Y/%m/%d"))

if st.button("Add Card", type="primary"):
    d = _parse_date(ev_date)
    if not ticker or not d:
        st.warning("Please enter a valid Ticker and Date.")
    else:
        p_base = max(0.0, 1.0 - float(p_bear) - 0.35)
        card = dict(
            ticker=ticker, type=sel_type, date=str(d),
            price=np.nan,
            t_bull=_to_float(t_bull), t_base=_to_float(t_base), t_bear=_to_float(t_bear),
            p_bull=0.35, p_base=p_base, p_bear=_clip01(p_bear),
            news=_clip01(news), flow=_clip01(flow), conf=_clip01(conf)
        )
        st.session_state.cards.append(card)
        st.success(f"Card added for {ticker}")

# Build watchlist safely
wl = pd.DataFrame(st.session_state.cards)
for col in EXPECTED_CARD_COLS:
    if col not in wl.columns:
        wl[col] = np.nan

if not wl.empty:
    wl = wl.apply(_norm_probs_row, axis=1)
    wl["dte"] = wl["date"].apply(_parse_date).apply(
        lambda d: (pd.to_datetime(d) - pd.Timestamp.today().normalize()).days if d else np.nan
    )
    ev_sh, ev_pct, odds = [], [], []
    for _, r in wl.iterrows():
        es, ep = _ev_from_row(r)
        ev_sh.append(es); ev_pct.append(ep); odds.append(_odds_row(r))
    wl["ev_sh"], wl["ev_pct"], wl["odds"] = ev_sh, ev_pct, odds
    wl["rank_score"] = wl["ev_pct"].fillna(-1e9) + (100 - wl["dte"].fillna(9999))/1000.0
    wl = wl.sort_values("rank_score", ascending=False).reset_index(drop=True)

st.subheader("Catalyst Watchlist (ranked)")
show_cols = ["ticker","type","date","dte","price","t_bull","t_base","t_bear",
             "p_bull","p_base","p_bear","ev_sh","ev_pct","odds"]
if wl.empty:
    st.info("No cards yet. Add a card to see rankings.")
    st.dataframe(pd.DataFrame(columns=show_cols), use_container_width=True)
else:
    st.dataframe(wl[show_cols], use_container_width=True)
    _download_csv_button(wl[show_cols], "Download Watchlist (with EV)", "watchlist_ev.csv")

st.markdown("---")

# ---------- Portfolio EV ----------
st.header("📊 Portfolio — EV & Reinvest Plan")

pos = st.session_state.positions.copy()
for col in ["ticker","shares","avg","price"]:
    if col not in pos.columns:
        pos[col] = pd.Series(dtype=float if col != "ticker" else str)

for c in ["shares", "avg", "price"]:
    pos[c] = pos[c].apply(_to_float)

if not pos.empty:
    pos["ticker"] = pos["ticker"].astype(str).str.upper()
    pos["price"] = np.where(pos["price"].isna() | (pos["price"] <= 0), pos["avg"], pos["price"])

    ev_map = {}
    if not wl.empty:
        best = wl.sort_values(["ticker", "rank_score"], ascending=[True, False]).drop_duplicates(subset=["ticker"])
        ev_map = {r["ticker"]: r["ev_pct"] for _, r in best.iterrows()}

    pos["EV_%"] = pos["ticker"].map(ev_map).fillna(0.0)
    pos["MktVal_EUR"] = pos["shares"] * pos["price"]
    pos["EV_EUR"] = pos["MktVal_EUR"] * pos["EV_%"] / 100.0

    st.dataframe(pos[["ticker","shares","avg","price","MktVal_EUR","EV_%","EV_EUR"]],
                 use_container_width=True)

    mv = pos["MktVal_EUR"].sum()
    sum_ev = pos["EV_EUR"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Value (€)", f"{mv:,.2f}")
    c2.metric("Sum EV (€)", f"{sum_ev:,.2f}")
    c3.metric("Cash (€)", f"{st.session_state.cash:,.2f}")
else:
    st.info("No positions yet. Add or upload a CSV.")

# ---------- Reinvest Planner ----------
st.subheader("🔁 Reinvest Planner")
r = st.slider("Reinvest rate r (0–100%)", min_value=0, max_value=100, value=60, step=5)
default_budget = round(st.session_state.cash * r / 100.0, 2)
budget = st.number_input("Budget (€)", value=float(default_budget), step=50.0)
st.caption("Sizing ∝ max(0, odds × EV%) under your budget.")

alloc_tbl = pd.DataFrame(columns=["ticker","price","ev_pct","odds","alloc_EUR","buy_shares"])
if not wl.empty and budget > 0:
    cands = wl.copy()
    cands = cands[cands["ev_pct"].fillna(-1) > 0]
    if not cands.empty:
        score = (cands["ev_pct"].clip(lower=0.0) * cands["odds"].clip(lower=0.0))
        if score.sum() > 0:
            w = score / score.sum()
            alloc_EUR = w * budget
            pr = cands["price"].apply(_to_float).replace(0, np.nan)
            buy = np.floor(alloc_EUR / pr).fillna(0).astype(int)
            alloc_tbl = pd.DataFrame({
                "ticker": cands["ticker"].values,
                "price": cands["price"].values,
                "ev_pct": cands["ev_pct"].values,
                "odds": cands["odds"].values,
                "alloc_EUR": alloc_EUR,
                "buy_shares": buy
            }).sort_values("alloc_EUR", ascending=False).reset_index(drop=True)

st.dataframe(alloc_tbl, use_container_width=True)
_download_csv_button(alloc_tbl, "Download Reinvest Plan", "reinvest_plan.csv")

st.markdown("---")
st.caption("v0.4.1 — hardened: no KeyErrors on empty tables; EV math & planner intact.")
