# app.py — Catalyst Copilot v0.4 (unified)
# - Cards -> Watchlist (ranked)
# - EV math (per-card & per-position)
# - Portfolio uploader & EV aggregation
# - Reinvest planner
# - CSV import/export + templates

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

st.set_page_config(page_title="Catalyst Copilot v0.4", layout="wide")

# ---------- Helpers ----------
def _to_float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""): return default
        return float(str(x).replace(",", "").replace("€",""))
    except:
        return default

def _clip01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except:
        return 0.0

def _parse_date(s):
    if isinstance(s, (datetime, date)): return pd.to_datetime(s).date()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try: return datetime.strptime(str(s), fmt).date()
        except: pass
    return None

def _norm_probs_row(r):
    p = pd.Series([_to_float(r.get("p_bull"), np.nan),
                   _to_float(r.get("p_base"), np.nan),
                   _to_float(r.get("p_bear"), np.nan)],
                   index=["p_bull","p_base","p_bear"])
    if p.isna().any():
        # fallback to priors if any missing
        p = pd.Series([0.35,0.45,0.20], index=["p_bull","p_base","p_bear"])
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        p = pd.Series([0.35,0.45,0.20], index=["p_bull","p_base","p_bear"]); s=1.0
    if abs(s-1.0) > 1e-6: p = p/s
    r["p_bull"], r["p_base"], r["p_bear"] = p.values
    return r

def _ev_from_row(r):
    price  = _to_float(r.get("price"))
    t_bull = _to_float(r.get("t_bull"))
    t_base = _to_float(r.get("t_base"))
    t_bear = _to_float(r.get("t_bear"))
    if any(map(lambda x: (x is None) or (not np.isfinite(x)), [price,t_bull,t_base,t_bear])):
        return np.nan, np.nan
    ev_sh = (r["p_bull"]*(t_bull-price)
           + r["p_base"]*(t_base-price)
           + r["p_bear"]*(t_bear-price))
    ev_pct = (ev_sh/price)*100 if price and np.isfinite(price) else np.nan
    return ev_sh, ev_pct

def _odds_row(r):
    up = max(_to_float(r.get("t_bull")) - _to_float(r.get("price")), 0.0) * r["p_bull"]
    down = max(_to_float(r.get("price")) - _to_float(r.get("t_bear")), 0.0) * r["p_bear"]
    if down <= 0: return 99.0
    return float(np.clip(up/(down+1e-9), 0, 99))

def _download_csv_button(df, label, fname):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=fname, mime="text/csv")

# ---------- Session State ----------
if "cards" not in st.session_state:
    st.session_state.cards = []  # list of dicts
if "cash" not in st.session_state:
    st.session_state.cash = 0.0
if "positions" not in st.session_state:
    # ticker, shares, avg, price (optional current price)
    st.session_state.positions = pd.DataFrame(columns=["ticker","shares","avg","price"])

# ---------- Sidebar: Portfolio ----------
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
            mask = (df["ticker"] == pt)
            if mask.any():
                df.loc[mask, ["shares","avg","price"]] = [psh, pav, pcp]
            else:
                df = pd.concat([df, pd.DataFrame([{"ticker":pt,"shares":psh,"avg":pav,"price":pcp}])], ignore_index=True)
            st.session_state.positions = df
            st.success(f"Saved {pt}")
        else:
            st.warning("Enter a valid ticker and shares.")

    if st.button("Clear Positions"):
        st.session_state.positions = pd.DataFrame(columns=["ticker","shares","avg","price"])
        st.info("Positions cleared.")

    st.markdown("---")
    st.subheader("Positions CSV")
    st.caption("Upload a CSV with columns: `ticker,shares,avg,price` (price optional).")
    up_pos = st.file_uploader("Upload positions.csv", type=["csv"], key="pos_csv")
    if up_pos is not None:
        try:
            dfp = pd.read_csv(up_pos)
            need = {"ticker","shares","avg"}
            if not need.issubset(set(map(str.lower, dfp.columns))):
                raise ValueError("Missing required columns.")
            # normalize columns
            cols = {c.lower():c for c in dfp.columns}
            dfp.rename(columns={cols.get("ticker","ticker"):"ticker",
                                cols.get("shares","shares"):"shares",
                                cols.get("avg","avg"):"avg",
                                cols.get("price","price"):"price"}, inplace=True)
            dfp["ticker"] = dfp["ticker"].astype(str).str.upper()
            for c in ["shares","avg","price"]:
                if c in dfp.columns: dfp[c] = dfp[c].apply(_to_float)
            st.session_state.positions = dfp[["ticker","shares","avg","price"] if "price" in dfp.columns else ["ticker","shares","avg"]]
            st.success(f"Loaded {len(dfp)} positions.")
        except Exception as e:
            st.error(f"Positions CSV error: {e}")

    # template
    tmpl = pd.DataFrame({
        "ticker":["ALT","MIST","KURA"],
        "shares":[130,656,50],
        "avg":[3.90,2.04,9.52],
        "price":[3.92,1.86,9.95],
    })
    _download_csv_button(tmpl, "Download positions template", "positions_template.csv")

# ---------- Main: Header ----------
st.title("🧠 Catalyst Copilot — v0.4 (Unified)")
st.caption("Cards → Watchlist (ranked) → EV math → Portfolio EV → Reinvest planner. All in one.")

# ---------- Section A: Cards & Watchlist ----------
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

c5, c6, c7, c8 = st.columns(4)
with c5:
    p_bear = st.number_input("p_bear*", value=0.20, step=0.05, min_value=0.0, max_value=1.0)
with c6:
    news = st.number_input("News score (0–1)", value=0.0, step=0.05, min_value=0.0, max_value=1.0)
with c7:
    flow = st.number_input("Flow score (0–1)", value=0.0, step=0.05, min_value=0.0, max_value=1.0)
with c8:
    conf = st.slider("Confidence (0–1)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)

c9, c10 = st.columns([2,2])
with c9:
    ticker = st.text_input("Ticker*", value="ALT").strip().upper()
with c10:
    ev_date = st.text_input("Event Date*", value=date.today().strftime("%Y/%m/%d"))

if st.button("Add Card", type="primary"):
    d = _parse_date(ev_date)
    if not ticker or not d:
        st.warning("Please enter a valid Ticker and Event Date.")
    else:
        # priors and normalization
        p_base = max(0.0, 1.0 - float(p_bear) - 0.35)  # quick seed; will normalize later anyway
        card = dict(
            ticker=ticker, type=sel_type,
            date=str(d),
            price=np.nan,              # fill later from positions or manual
            t_bull=_to_float(t_bull), t_base=_to_float(t_base), t_bear=_to_float(t_bear),
            p_bull=0.35, p_base=p_base, p_bear=_clip01(p_bear),
            news=_clip01(news), flow=_clip01(flow), conf=_clip01(conf)
        )
        st.session_state.cards.append(card)
        st.success(f"Card added for {ticker}")

# Watchlist dataframe
wl = pd.DataFrame(st.session_state.cards) if st.session_state.cards else pd.DataFrame(
    columns=["ticker","type","date","price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","news","flow","conf"]
)

# Allow CSV import for watchlist too
with st.expander("📥 Import Watchlist CSV (optional)"):
    st.caption("Columns accepted: ticker,type,date,price,t_bull,t_base,t_bear,p_bull,p_base,p_bear,news,flow,conf")
    up_wl = st.file_uploader("Upload watchlist.csv", type=["csv"], key="wl_csv")
    if up_wl is not None:
        try:
            add = pd.read_csv(up_wl)
            add.columns = [c.strip().lower() for c in add.columns]
            # normalize column names
            rename = {
                "ticker":"ticker","type":"type","date":"date","price":"price",
                "t_bull":"t_bull","t_base":"t_base","t_bear":"t_bear",
                "p_bull":"p_bull","p_base":"p_base","p_bear":"p_bear",
                "news":"news","flow":"flow","conf":"conf"
            }
            add = add.rename(columns={c:rename[c] for c in add.columns if c in rename})
            # types
            if "ticker" in add.columns: add["ticker"] = add["ticker"].astype(str).str.upper()
            for c in ["price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","news","flow","conf"]:
                if c in add.columns: add[c] = add[c].apply(_to_float)
            # append
            wl = pd.concat([wl, add], ignore_index=True)
            st.success(f"Loaded {len(add)} watchlist rows.")
        except Exception as e:
            st.error(f"Watchlist CSV error: {e}")

# fill missing price from positions table when possible
if not wl.empty and "price" in wl.columns:
    pos_map = {r["ticker"]: r["price"] for _, r in st.session_state.positions.fillna(0).iterrows()}
    wl["price"] = wl.apply(lambda r: _to_float(r["price"], pos_map.get(r.get("ticker"), np.nan)), axis=1)

# compute DTE, normalize probs, EV, odds
if not wl.empty:
    # DTE
    wl["dte"] = wl["date"].apply(_parse_date).apply(lambda d: (pd.to_datetime(d) - pd.Timestamp.today().normalize()).days if d else np.nan)

    # normalize probabilities
    wl = wl.apply(_norm_probs_row, axis=1)

    # EV math
    ev_sh, ev_pct, odds = [], [], []
    for _, r in wl.iterrows():
        es, ep = _ev_from_row(r)
        ev_sh.append(es); ev_pct.append(ep); odds.append(_odds_row(r))
    wl["ev_sh"] = ev_sh
    wl["ev_pct"] = ev_pct
    wl["odds"] = odds

    # rank (highest EV% first, then nearest DTE)
    wl["rank_score"] = wl["ev_pct"].fillna(-1e9) + (100 - wl["dte"].fillna(9999))/1000.0
    wl = wl.sort_values(by=["rank_score"], ascending=False).reset_index(drop=True)

st.subheader("Catalyst Watchlist (ranked)")
show_cols = ["ticker","type","date","dte","price","t_bull","t_base","t_bear",
             "p_bull","p_base","p_bear","ev_sh","ev_pct","odds"]
st.dataframe(wl[ [c for c in show_cols if c in wl.columns] ], use_container_width=True)

_download_csv_button(wl, "Download watchlist (with EV)", "watchlist_ev.csv")

st.markdown("---")

# ---------- Section B: Portfolio EV & Reinvest ----------
st.header("📊 Portfolio — EV & Reinvest Plan")

# merge positions with best watchlist EV per ticker (if any)
pos = st.session_state.positions.copy()
for c in ["shares","avg","price"]:
    if c in pos.columns: pos[c] = pos[c].apply(_to_float)

if not pos.empty:
    # price fallback to avg if missing
    pos["price"] = np.where(pos["price"].isna() | (pos["price"]<=0), pos["avg"], pos["price"])
    # map EV% by ticker (take the top-ranked row per ticker)
    if not wl.empty:
        best = wl.sort_values(["ticker","rank_score"], ascending=[True, False]).drop_duplicates(subset=["ticker"])
        ev_map = {r["ticker"]: r["ev_pct"] for _, r in best.iterrows()}
    else:
        ev_map = {}
    pos["EV_%"] = pos["ticker"].map(ev_map).fillna(0.0)

    pos["MktVal€"] = pos["shares"] * pos["price"]
    pos["EV_€"]    = pos["MktVal€"] * pos["EV_%"] / 100.0

    st.dataframe(pos[["ticker","shares","avg","price","MktVal€","EV_%","EV_€"]], use_container_width=True)

    mv = pos["MktVal€"].sum()
    sum_ev = pos["EV_€"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Value (€)", f"{mv:,.2f}")
    c2.metric("Sum EV (€)", f"{sum_ev:,.2f}")
    c3.metric("Cash (€)", f"{st.session_state.cash:,.2f}")
else:
    st.info("No positions yet. Add some on the left, or upload a CSV.")

st.subheader("🔁 Reinvest Planner")
r = st.slider("Reinvest rate r (0–100%)", min_value=0, max_value=100, value=60, step=5)
budget = st.number_input("Budget (€) (uses cash × r)", value=float(round(st.session_state.cash * r/100.0, 2)), step=50.0)
st.caption("Sizing ∝ max(0, odds × EV%) under your budget. You can refine this later.")

alloc_tbl = pd.DataFrame(columns=["ticker","price","ev_pct","odds","alloc€","buy_shares"])
if not wl.empty and budget > 0:
    # positive EV rows only
    cands = wl.copy()
    cands = cands[cands["ev_pct"].fillna(-1) > 0]
    if not cands.empty:
        score = (cands["ev_pct"].clip(lower=0.0) * cands["odds"].clip(lower=0.0))
        if score.sum() <= 0:
            st.info("All candidates have zero score (no positive EV).")
        else:
            w = score / score.sum()
            alloc€ = w * budget
            pr = cands["price"].apply(_to_float).fillna(0.0).replace(0, np.nan)
            buy = np.floor(alloc€ / pr).fillna(0).astype(int)
            alloc_tbl = pd.DataFrame({
                "ticker": cands["ticker"].values,
                "price": cands["price"].values,
                "ev_pct": cands["ev_pct"].values,
                "odds": cands["odds"].values,
                "alloc€": alloc€,
                "buy_shares": buy
            }).sort_values("alloc€", ascending=False).reset_index(drop=True)

st.dataframe(alloc_tbl, use_container_width=True)
_download_csv_button(alloc_tbl, "Download reinvest plan", "reinvest_plan.csv")

st.markdown("---")
st.caption("v0.4 — unified & hardened: CSV ingest, per-card & per-position EV, reinvest planner, templates, and safe math. Next: order tickets & backtests.")
