# app.py
import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ——— App Basics ———
st.set_page_config(page_title="Catalyst Copilot v0.2", layout="wide")
st.title("Catalyst Copilot v0.2")
st.caption("Minimal, fast-deploy build: portfolio input → catalyst cards → EV math → ranked watchlist.")

# ——— Helpers ———
TZ = ZoneInfo("Europe/Berlin")  # your timezone per project
TODAY = datetime.now(TZ).date()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def compute_dte(event_date: date) -> int:
    return (event_date - TODAY).days

def clip(x, lo, hi):  # simple clip
    return max(lo, min(hi, x))

def compute_ev_per_share(p_bull, p_base, p_bear, t_bull, t_base, t_bear, px_now):
    # EV per share relative to current price
    # EV% = Σ(p_i * (target_i/px_now - 1))
    if px_now <= 0:
        return 0.0, 0.0
    ev_pct = p_bull*(t_bull/px_now - 1) + p_base*(t_base/px_now - 1) + p_bear*(t_bear/px_now - 1)
    ev_abs = ev_pct * px_now
    return ev_pct, ev_abs

def compute_csi(ivz, volz, news=0.0, flow=0.0):
    # CSI = 0.4 IVz + 0.3 Volz + 0.2 News + 0.1 Flow
    return 0.4*ivz + 0.3*volz + 0.2*news + 0.1*flow

def validate_probs(pb, pm, pr):
    s = pb+pm+pr
    return abs(s-1.0) < 1e-6

# ——— Session state ———
if "positions" not in st.session_state:
    st.session_state.positions = []  # list of dicts
if "cards" not in st.session_state:
    st.session_state.cards = []      # list of dicts
if "cash" not in st.session_state:
    st.session_state.cash = 0.0

# ——— Sidebar: Portfolio Input ———
with st.sidebar:
    st.header("Portfolio")
    cash_in = st.text_input("Cash (€)", value=str(st.session_state.cash or "0"))
    if st.button("Save Cash"):
        st.session_state.cash = safe_float(cash_in, 0.0)
        st.success(f"Cash set to €{st.session_state.cash:,.2f}")

    st.subheader("Add / Update Position")
    colp1, colp2 = st.columns(2)
    with colp1:
        px_ticker = st.text_input("Ticker", value="")
        px_shares = st.text_input("Shares", value="0")
    with colp2:
        px_avg = st.text_input("Avg Cost", value="0")
        px_price = st.text_input("Current Price", value="0")

    if st.button("Add/Update Position"):
        t = px_ticker.strip().upper()
        if not t:
            st.error("Ticker is required.")
        else:
            # upsert
            found = False
            for p in st.session_state.positions:
                if p["ticker"] == t:
                    p.update({
                        "shares": safe_float(px_shares, 0.0),
                        "avg_cost": safe_float(px_avg, 0.0),
                        "price": safe_float(px_price, 0.0),
                    })
                    found = True
                    break
            if not found:
                st.session_state.positions.append({
                    "ticker": t,
                    "shares": safe_float(px_shares, 0.0),
                    "avg_cost": safe_float(px_avg, 0.0),
                    "price": safe_float(px_price, 0.0),
                })
            st.success(f"Saved position: {t}")

    if st.button("Clear Positions"):
        st.session_state.positions = []
        st.info("Positions cleared.")

# ——— Main: Positions Table ———
st.subheader("Positions")
if st.session_state.positions:
    dfp = pd.DataFrame(st.session_state.positions)
    dfp["MV"] = dfp["shares"] * dfp["price"]
    st.dataframe(dfp, use_container_width=True)
    st.markdown(f"**Cash:** €{st.session_state.cash:,.2f} | **MV:** €{dfp['MV'].sum():,.2f} | **NLV (approx):** €{(dfp['MV'].sum()+st.session_state.cash):,.2f}")
else:
    st.info("Add positions in the sidebar to see MV / NLV.")

st.divider()

# ——— Catalyst Card Form ———
st.subheader("Add Catalyst Card")
with st.form("card_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        c_ticker = st.text_input("Ticker*", "")
        c_type = st.selectbox("Type*", ["PDUFA","CHMP","Readout","Earnings","Policy","APD","Conf","Other"])
        c_date = st.date_input("Event Date*", value=TODAY)
    with c2:
        c_price = st.text_input("Current Price*", "0")
        c_target_bull = st.text_input("Target Bull*", "0")
        c_target_base = st.text_input("Target Base*", "0")
        c_target_bear = st.text_input("Target Bear*", "0")
    with c3:
        c_p_bull = st.text_input("p_bull*", "0.35")
        c_p_base = st.text_input("p_base*", "0.45")
        c_p_bear = st.text_input("p_bear*", "0.20")
        c_conf = st.slider("Confidence (0–1)", 0.0, 1.0, 0.60, 0.01)
    with c4:
        c_ivz = st.text_input("IV z", "0")
        c_volz = st.text_input("Vol z", "0")
        c_news = st.text_input("News score (0–1)", "0")
        c_flow = st.text_input("Flow score (0–1)", "0")

    submitted = st.form_submit_button("Add Card")
    if submitted:
        try:
            p_bull = safe_float(c_p_bull, 0.0)
            p_base = safe_float(c_p_base, 0.0)
            p_bear = safe_float(c_p_bear, 0.0)
            # normalize if slightly off
            s = p_bull + p_base + p_bear
            if s <= 0:
                raise ValueError("Probabilities must sum to > 0.")
            p_bull /= s; p_base /= s; p_bear /= s
            if abs(p_bull + p_base + p_bear - 1.0) > 1e-6:
                raise ValueError("Probability normalization failed.")

            price_now = safe_float(c_price, 0.0)
            t_bull = safe_float(c_target_bull, 0.0)
            t_base = safe_float(c_target_base, 0.0)
            t_bear = safe_float(c_target_bear, 0.0)

            ivz = safe_float(c_ivz, 0.0)
            volz = safe_float(c_volz, 0.0)
            news = safe_float(c_news, 0.0)
            flow = safe_float(c_flow, 0.0)

            dte = compute_dte(c_date)
            ev_pct, ev_abs = compute_ev_per_share(p_bull, p_base, p_bear, t_bull, t_base, t_bear, price_now)
            csi = compute_csi(ivz, volz, news, flow)

            st.session_state.cards.append({
                "ticker": c_ticker.strip().upper(),
                "type": c_type,
                "date": c_date.isoformat(),
                "dte": dte,
                "price": price_now,
                "t_bull": t_bull,
                "t_base": t_base,
                "t_bear": t_bear,
                "p_bull": round(p_bull, 3),
                "p_base": round(p_base, 3),
                "p_bear": round(p_bear, 3),
                "ev_pct": ev_pct,           # relative to current px
                "ev_abs": ev_abs,           # € per share notionally
                "conf": c_conf,
                "ivz": ivz,
                "volz": volz,
                "news": news,
                "flow": flow,
                "csi": csi,
            })
            st.success("Catalyst card added.")
        except Exception as e:
            st.error(f"Error: {e}")

# ——— Watchlist / Ranking ———
st.subheader("Catalyst Watchlist (ranked)")
if st.session_state.cards:
    df = pd.DataFrame(st.session_state.cards)
    # Rank by short-fuse-ish priority: higher CSI, sooner DTE (but not negative), higher EV%
    # Score = (CSI z-lite) * time decay * (1 + EV%)
    # time decay simple: 1 / (1 + max(dte,0))
    df["time_decay"] = 1.0 / (1.0 + df["dte"].clip(lower=0))
    df["score"] = (df["csi"]) * df["time_decay"] * (1.0 + df["ev_pct"])
    df = df.sort_values(["score"], ascending=False)

    st.dataframe(
        df[[
            "ticker","type","date","dte","price",
            "t_bull","t_base","t_bear",
            "p_bull","p_base","p_bear",
            "ev_pct","csi","conf","score"
        ]],
        use_container_width=True
    )

    # Download
    st.download_button(
        "Download watchlist as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"catalyst_watchlist_{TODAY.isoformat()}.csv",
        mime="text/csv"
    )
else:
    st.info("Add a catalyst card to see the ranked watchlist.")

st.divider()
st.caption("v0.2 — next: broker CSV ingest, EV-by-position, reinvest plan, and order ticket helpers.")
