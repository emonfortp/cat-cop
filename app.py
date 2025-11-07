# app.py — Catalyst Copilot v0.2.1 (hardened)
# - Adds zoneinfo fallback
# - Stricter input sanitization
# - Defensive probability normalization
# - Clear error messages instead of silent crashes

from datetime import datetime, date
import math
import os

import pandas as pd
import streamlit as st

# ---------- Timezone safe import ----------
# Some runtimes don’t ship zoneinfo. Fall back to naive local time.
try:
    from zoneinfo import ZoneInfo  # py>=3.9
    TZ = ZoneInfo("Europe/Berlin")
    def today_local():
        return datetime.now(TZ).date()
except Exception:
    TZ = None
    def today_local():
        # Fallback: naive local date
        return datetime.now().date()

TODAY = today_local()

# ---------- Streamlit page ----------
st.set_page_config(page_title="Catalyst Copilot v0.2.1", layout="wide")
st.title("Catalyst Copilot v0.2.1")
st.caption("Portfolio → Catalyst cards → EV math → Ranked watchlist (hardened build).")

# ---------- Helpers ----------
def safe_float(x, default=0.0):
    """Cast to float safely; handles '', None, commas, spaces."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def compute_dte(event_date: date) -> int:
    try:
        return (event_date - TODAY).days
    except Exception:
        return 0

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def normalize_probs(p_bull, p_base, p_bear):
    ps = [safe_float(p_bull, 0.0), safe_float(p_base, 0.0), safe_float(p_bear, 0.0)]
    s = sum(ps)
    if s <= 0:
        # default priors
        return 0.35, 0.45, 0.20
    pn = [max(0.0, p) / s for p in ps]
    # guard for floating drift
    total = sum(pn)
    if total <= 0:
        return 0.35, 0.45, 0.20
    # rescale to exactly 1.0
    pn = [p / total for p in pn]
    return round(pn[0], 3), round(pn[1], 3), round(pn[2], 3)

def compute_ev_per_share(p_bull, p_base, p_bear, t_bull, t_base, t_bear, px_now):
    px = safe_float(px_now, 0.0)
    if px <= 0:
        return 0.0, 0.0
    # EV% relative to current price
    ev_pct = (
        safe_float(p_bull) * (safe_float(t_bull)/px - 1.0) +
        safe_float(p_base) * (safe_float(t_base)/px - 1.0) +
        safe_float(p_bear) * (safe_float(t_bear)/px - 1.0)
    )
    ev_abs = ev_pct * px  # absolute €/share notionally
    return ev_pct, ev_abs

def compute_csi(ivz, volz, news=0.0, flow=0.0):
    return (
        0.4 * safe_float(ivz) +
        0.3 * safe_float(volz) +
        0.2 * clip(safe_float(news), 0.0, 1.0) +
        0.1 * clip(safe_float(flow), 0.0, 1.0)
    )

# ---------- Session state ----------
if "positions" not in st.session_state:
    st.session_state.positions = []  # list of dicts
if "cards" not in st.session_state:
    st.session_state.cards = []      # list of dicts
if "cash" not in st.session_state:
    st.session_state.cash = 0.0

# ---------- Sidebar: Portfolio ----------
with st.sidebar:
    st.header("Portfolio")

    cash_in = st.text_input("Cash (€)", value=str(st.session_state.cash or "0"))
    if st.button("Save Cash", use_container_width=True):
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

    if st.button("Add/Update Position", type="primary", use_container_width=True):
        t = (px_ticker or "").strip().upper()
        if not t:
            st.error("Ticker is required.")
        else:
            upsert = {
                "ticker": t,
                "shares": safe_float(px_shares, 0.0),
                "avg_cost": safe_float(px_avg, 0.0),
                "price": safe_float(px_price, 0.0),
            }
            found = False
            for p in st.session_state.positions:
                if p["ticker"] == t:
                    p.update(upsert)
                    found = True
                    break
            if not found:
                st.session_state.positions.append(upsert)
            st.success(f"Saved position: {t}")

    if st.button("Clear Positions", use_container_width=True):
        st.session_state.positions = []
        st.info("Positions cleared.")

# ---------- Positions ----------
st.subheader("Positions")
if st.session_state.positions:
    dfp = pd.DataFrame(st.session_state.positions)
    # Fill missing numeric fields safely
    for col in ["shares", "avg_cost", "price"]:
        dfp[col] = dfp[col].apply(lambda x: safe_float(x, 0.0))
    dfp["MV"] = dfp["shares"] * dfp["price"]
    st.dataframe(dfp, use_container_width=True)
    st.markdown(
        f"**Cash:** €{st.session_state.cash:,.2f} | "
        f"**MV:** €{dfp['MV'].sum():,.2f} | "
        f"**NLV (approx):** €{(dfp['MV'].sum() + st.session_state.cash):,.2f}"
    )
else:
    st.info("Add positions in the sidebar to see MV / NLV.")

st.divider()

# ---------- Catalyst Card Form ----------
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
            # Normalize probabilities robustly
            p_bull, p_base, p_bear = normalize_probs(c_p_bull, c_p_base, c_p_bear)

            price_now = safe_float(c_price, 0.0)
            t_bull = safe_float(c_target_bull, 0.0)
            t_base = safe_float(c_target_base, 0.0)
            t_bear = safe_float(c_target_bear, 0.0)

            ivz = safe_float(c_ivz, 0.0)
            volz = safe_float(c_volz, 0.0)
            news = clip(safe_float(c_news, 0.0), 0.0, 1.0)
            flow = clip(safe_float(c_flow, 0.0), 0.0, 1.0)

            dte = compute_dte(c_date if isinstance(c_date, date) else TODAY)
            ev_pct, ev_abs = compute_ev_per_share(p_bull, p_base, p_bear, t_bull, t_base, t_bear, price_now)
            csi = compute_csi(ivz, volz, news, flow)

            t_sym = (c_ticker or "").strip().upper()
            if not t_sym:
                raise ValueError("Ticker is required.")

            st.session_state.cards.append({
                "ticker": t_sym,
                "type": c_type,
                "date": c_date.isoformat() if isinstance(c_date, date) else str(c_date),
                "dte": dte,
                "price": price_now,
                "t_bull": t_bull,
                "t_base": t_base,
                "t_bear": t_bear,
                "p_bull": p_bull,
                "p_base": p_base,
                "p_bear": p_bear,
                "ev_pct": ev_pct,
                "ev_abs": ev_abs,
                "conf": round(safe_float(c_conf, 0.6), 2),
                "ivz": ivz,
                "volz": volz,
                "news": news,
                "flow": flow,
                "csi": csi,
            })
            st.success("Catalyst card added.")
        except Exception as e:
            st.error(f"Error adding card: {e}")

# ---------- Watchlist / Ranking ----------
st.subheader("Catalyst Watchlist (ranked)")
if st.session_state.cards:
    df = pd.DataFrame(st.session_state.cards).copy()

    # Defensive typing
    for col in ["dte","price","t_bull","t_base","t_bear","p_bull","p_base","p_bear","ev_pct","csi","conf"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # time decay → nearer events rank higher (no negatives)
    df["time_decay"] = 1.0 / (1.0 + df["dte"].clip(lower=0))
    # final score
    df["score"] = df["csi"] * df["time_decay"] * (1.0 + df["ev_pct"])

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

    st.download_button(
        "Download watchlist as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"catalyst_watchlist_{TODAY.isoformat()}.csv",
        mime="text/csv"
    )
else:
    st.info("Add a catalyst card to see the ranked watchlist.")

st.divider()
st.caption("v0.2.1 — hardened. Next: broker CSV ingest, per-position EV, reinvest plan, order ticket helpers.")
