# app.py — Catalyst Copilot v0.4.2 (IBKR PortfolioAnalyst Import + QC)
# --------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime

st.set_page_config(page_title="Catalyst Copilot", layout="wide")

# --- State ---
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=["ticker","shares","avg","price"])
if "cards" not in st.session_state:
    st.session_state.cards = pd.DataFrame(columns=[
        "ticker","type","date","t_bull","t_base","t_bear","p_bull","p_base","p_bear","news","flow","conf"
    ])
if "cash" not in st.session_state: st.session_state.cash = 0.0
if "ev_last" not in st.session_state: st.session_state.ev_last = {}

# --- Helpers ---
def clamp(x, lo, hi):
    try: x = float(x)
    except: x = 0.0
    return max(lo, min(hi, x))

def prob_triplet(p_bull, p_bear):
    p_bull = clamp(p_bull, 0.0, 0.99)
    p_bear = clamp(p_bear, 0.0, 0.99)
    p_base = 1.0 - (p_bull + p_bear)
    if p_base < 0:
        total = max(p_bull + p_bear, 1e-12)
        p_bull /= total; p_bear /= total; p_base = 0.0
    s = p_bull + p_base + p_bear
    if s <= 0: return (0.0, 1.0, 0.0)
    p_bull, p_base, p_bear = (p_bull/s, p_base/s, p_bear/s)
    return (round(p_bull,4), round(p_base,4), round(p_bear,4))

def dte_from_str(s):
    if not isinstance(s, str): return np.nan
    s = s.strip().replace(".", "/").replace("-", "/")
    for fmt in ("%Y/%m/%d","%Y/%m/%d %H:%M","%Y/%m/%d %H:%M:%S"):
        try: return (datetime.strptime(s, fmt).date() - date.today()).days
        except: pass
    return np.nan

def calc_odds(p_bull, p_base, p_bear):
    up, dn = p_bull, p_bear
    return 0.0 if (up+dn)<=0 else round(up/(up+dn),4)

def w_conf(conf, news, flow):
    return round(0.5*clamp(conf,0,1) + 0.3*clamp(news,0,1) + 0.2*clamp(flow,0,1), 4)

def expected_return_pct(price, t_bull, t_base, t_bear, p_bull, p_base, p_bear, confw):
    price = float(price or 0.0)
    if price <= 0: return 0.0
    r_bull = (t_bull - price)/price
    r_base = (t_base - price)/price
    r_bear = (t_bear - price)/price
    ev = p_bull*r_bull + p_base*r_base + p_bear*r_bear
    ev *= (0.6 + 0.4*confw)
    return round(float(ev),4)

def current_price_for(ticker):
    df = st.session_state.positions
    if df.empty: return np.nan
    row = df[df["ticker"]==ticker]
    if row.empty: return np.nan
    return float(row.iloc[0]["price"]) if pd.notna(row.iloc[0]["price"]) else np.nan

def momentum_tag(key, ev_now):
    prev = st.session_state.ev_last.get(key)
    st.session_state.ev_last[key] = ev_now
    if prev is None: return "—", 0.0
    delta = ev_now - prev
    if delta > 0.01: return "Up", round(delta,4)
    if delta < -0.01: return "Down", round(delta,4)
    return "Flat", round(delta,4)

# --- NEW: IBKR / Generic PortfolioAnalyst parser ---
def parse_portfolio_analyst(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Maps common IBKR PortfolioAnalyst / generic exports to positions schema:
    Output columns: ticker, shares, avg, price
    Tries multiple possible column names.
    """
    cols = {c.lower(): c for c in df_raw.columns}
    def pick(*opts):
        for o in opts:
            if o in cols: return cols[o]
        return None

    c_ticker = pick("symbol","ticker","conid","underlying symbol")
    c_shares = pick("quantity","position","shares","qty")
    c_avg    = pick("cost basis","average cost","avg cost","avg_cost","avg")
    c_price  = pick("mark price","price","last price","market price","mark")

    if c_ticker is None or c_shares is None:
        raise ValueError("Required columns not found (need Symbol/Ticker and Quantity/Position).")

    out = pd.DataFrame()
    out["ticker"] = (df_raw[c_ticker].astype(str)
                     .str.upper().str.extract(r"([A-Z\.\-]+)")[0].fillna("").str.strip())
    out["shares"] = pd.to_numeric(df_raw[c_shares], errors="coerce").fillna(0).astype(int)
    out["avg"]    = pd.to_numeric(df_raw[c_avg],    errors="coerce").fillna(0.0) if c_avg else 0.0
    out["price"]  = pd.to_numeric(df_raw[c_price],  errors="coerce").fillna(np.nan) if c_price else np.nan

    # Filter valid rows
    out = out[(out["ticker"]!="") & (out["shares"]>=0)].copy()
    # Aggregate duplicates by ticker
    agg = out.groupby("ticker", as_index=False).agg({
        "shares":"sum",
        "avg":"mean",
        "price":"mean"
    })
    # Ensure float types
    for c in ["avg","price"]:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")
    return agg.reset_index(drop=True)

# --- Watchlist builder ---
def build_ranked_watchlist(cards: pd.DataFrame) -> pd.DataFrame:
    if cards.empty:
        return pd.DataFrame(columns=[
            "ticker","type","date","dte","price","t_bull","t_base","t_bear",
            "p_bull","p_base","p_bear","ev_pct","odds","momentum","Δev"
        ])
    rows=[]
    for _, r in cards.iterrows():
        tkr = str(r["ticker"]).upper().strip()
        ctyp = str(r["type"]).strip()
        raw_date = str(r["date"])
        dte = dte_from_str(raw_date)

        tb = float(pd.to_numeric(r["t_bull"], errors="coerce") or 0.0)
        ts = float(pd.to_numeric(r["t_base"], errors="coerce") or 0.0)
        td = float(pd.to_numeric(r["t_bear"], errors="coerce") or 0.0)
        pb = float(pd.to_numeric(r["p_bull"], errors="coerce") or 0.0)
        ps = float(pd.to_numeric(r["p_base"], errors="coerce") or 0.0)
        pn = float(pd.to_numeric(r["p_bear"], errors="coerce") or 0.0)
        news = float(pd.to_numeric(r.get("news",0.0), errors="coerce") or 0.0)
        flow = float(pd.to_numeric(r.get("flow",0.0), errors="coerce") or 0.0)
        conf = float(pd.to_numeric(r.get("conf",0.0), errors="coerce") or 0.0)

        price = current_price_for(tkr)
        confw = w_conf(conf, news, flow)
        ev = expected_return_pct(price, tb, ts, td, pb, ps, pn, confw)
        odds = calc_odds(pb, ps, pn)
        tag, delta = momentum_tag((tkr, ctyp), ev)

        try:
            date_disp = pd.to_datetime(raw_date.replace("/", "-"), errors="coerce").date()
        except: date_disp = None

        rows.append({
            "ticker": tkr, "type": ctyp, "date": date_disp, "dte": dte,
            "price": price, "t_bull": tb, "t_base": ts, "t_bear": td,
            "p_bull": pb, "p_base": ps, "p_bear": pn,
            "ev_pct": ev, "odds": odds, "momentum": tag, "Δev": delta
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["ev_pct","dte","odds"], ascending=[False, True, False],
                        na_position="last").reset_index(drop=True)
    return df

# --- Layout ---
left, right = st.columns([0.28, 0.72])

with left:
    st.markdown("## 🧾 Portfolio")
    cash_val = st.number_input("Cash (€)", min_value=0.0, value=float(st.session_state.cash), step=50.0)
    if st.button("Save Cash", use_container_width=True):
        st.session_state.cash = float(cash_val); st.success("Cash saved")

    st.markdown("---")
    st.markdown("### ➕ Add / Update Position")
    tkr_in = st.text_input("Ticker", value="")
    avg_in = st.number_input("Avg Cost", min_value=0.0, value=0.0, step=0.01)
    shr_in = st.number_input("Shares", min_value=0, value=0, step=1)
    prc_in = st.number_input("Current Price", min_value=0.0, value=0.0, step=0.01)

    def upsert_position():
        if not tkr_in.strip():
            st.warning("Enter a ticker."); return
        t = tkr_in.upper().strip()
        row = pd.DataFrame([{"ticker":t,"shares":int(shr_in),"avg":float(avg_in),"price":float(prc_in)}])
        df = st.session_state.positions.copy()
        df = df[df["ticker"] != t]
        st.session_state.positions = pd.concat([df, row], ignore_index=True)
        st.success(f"Position saved for {t}")
    st.button("Add/Update Position", on_click=upsert_position, use_container_width=True)

    st.markdown("#### Upload Positions CSV (simple)")
    pos_csv = st.file_uploader("Columns: ticker, shares, avg, price", type=["csv"], key="pos_csv_simple")
    if pos_csv is not None:
        try:
            df = pd.read_csv(pos_csv)
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
            for c in ["avg","price"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            df = df[(df["ticker"]!="") & (df["shares"]>=0)]
            st.session_state.positions = df.reset_index(drop=True)
            st.success("Positions CSV loaded.")
        except Exception as e:
            st.error(f"Positions CSV error: {e}")

    st.download_button(
        "Download current Positions CSV",
        data=st.session_state.positions.to_csv(index=False).encode("utf-8"),
        file_name="positions_current.csv",
        mime="text/csv",
        use_container_width=True
    )

with right:
    st.markdown("## 🤖 Catalyst Copilot — v0.4.2")

    # --- NEW: IBKR / PortfolioAnalyst Import (ALWAYS VISIBLE) ---
    st.markdown("### 📥 Import — Portfolio Analyst (IBKR/Generic)")
    ibkr_file = st.file_uploader("Upload IBKR PortfolioAnalyst or generic broker CSV", type=["csv"], key="ibkr_pa")
    if ibkr_file is not None:
        try:
            raw = pd.read_csv(ibkr_file)
            parsed = parse_portfolio_analyst(raw)
            if parsed.empty:
                st.warning("No valid rows found after parsing.")
            else:
                # Merge into positions (ticker-level overwrite)
                base = st.session_state.positions.copy()
                base = base[~base["ticker"].isin(parsed["ticker"])]
                st.session_state.positions = pd.concat([base, parsed], ignore_index=True)
                st.success(f"Imported {len(parsed)} tickers from PortfolioAnalyst CSV.")
                st.dataframe(parsed, use_container_width=True)
        except Exception as e:
            st.error(f"PortfolioAnalyst import error: {e}")

    with st.expander("Catalyst Cards — add & rank", expanded=True):
        c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
        typ = c1.selectbox("Type", ["PDUFA","Readout","CHMP","Earnings","Policy","Conf"], index=0)
        t_bull = c2.number_input("Target Bull*", value=0.0, step=0.10, format="%.2f")
        t_base = c3.number_input("Target Base*", value=0.0, step=0.10, format="%.2f")
        t_bear = c4.number_input("Target Bear*", value=0.0, step=0.10, format="%.2f")

        d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
        p_bear_in = d1.number_input("p_bear*", value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        news = d2.number_input("News score (0–1)", value=0.0, min_value=0.0, max_value=1.0, step=0.05)
        flow = d3.number_input("Flow score (0–1)", value=0.0, min_value=0.0, max_value=1.0, step=0.05)
        conf = d4.slider("Confidence (0–1)", 0.0, 1.0, 0.60)

        e1, e2 = st.columns([1.2, 2])
        ticker_card = e1.text_input("Ticker*", value="", placeholder="e.g., ALT").upper().strip()
        event_date = e2.text_input("Event Date*", value=date.today().strftime("%Y/%m/%d"))

        p_bull_in = clamp(1.0 - p_bear_in, 0.0, 1.0) * 0.6
        p_bull, p_base, p_bear = prob_triplet(p_bull_in, p_bear_in)

        if st.button("Add Card", use_container_width=True):
            if not ticker_card: st.warning("Ticker required for card.")
            else:
                row = {
                    "ticker": ticker_card, "type": typ, "date": event_date,
                    "t_bull": t_bull, "t_base": t_base, "t_bear": t_bear,
                    "p_bull": p_bull, "p_base": p_base, "p_bear": p_bear,
                    "news": news, "flow": flow, "conf": conf
                }
                st.session_state.cards = pd.concat([st.session_state.cards, pd.DataFrame([row])], ignore_index=True)
                st.success(f"Card added: {ticker_card} / {typ}")

        st.markdown("#### Upload Catalyst Cards CSV")
        cards_csv = st.file_uploader(
            "Columns: ticker,type,date,t_bull,t_base,t_bear,p_bull,p_base,p_bear,news,flow,conf",
            type=["csv"], key="cards_csv")
        if cards_csv is not None:
            try:
                df = pd.read_csv(cards_csv)
                need = {"ticker","type","date","t_bull","t_base","t_bear","p_bull","p_base","p_bear"}
                if not need.issubset(df.columns):
                    miss = sorted(list(need - set(df.columns)))
                    raise ValueError(f"Missing columns: {miss}")
                df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
                st.session_state.cards = df.reset_index(drop=True)
                st.success("Catalyst cards CSV loaded.")
            except Exception as e:
                st.error(f"Cards CSV error: {e}")

        st.download_button(
            "Download current Catalyst Cards CSV",
            data=st.session_state.cards.to_csv(index=False).encode("utf-8"),
            file_name="catalyst_cards_current.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("### 🧮 Catalyst Watchlist (ranked)")
    watch = build_ranked_watchlist(st.session_state.cards)
    st.dataframe(watch, use_container_width=True)

    st.markdown("### 📈 Portfolio — EV & Reinvest Plan")
    pos = st.session_state.positions.copy()
    if pos.empty:
        st.info("No positions yet. Add a position on the left or import a CSV.")
    else:
        pos["MktVal€"] = pos["shares"] * pos["price"]
        pos["EV_%"] = 0.0
        if not watch.empty:
            ev_map = watch.groupby("ticker")["ev_pct"].max().to_dict()
            pos["EV_%"] = pos["ticker"].map(ev_map).fillna(0.0)
        pos["EV_€"] = pos["MktVal€"] * pos["EV_%"]
        mv = round(float(pos["MktVal€"].sum()), 2)
        sum_ev = round(float(pos["EV_€"].sum()), 2)
        c1, c2, c3 = st.columns([2.6, 1, 1])
        with c1: st.dataframe(pos[["ticker","shares","avg","price","MktVal€","EV_%","EV_€"]], use_container_width=True)
        with c2: st.metric("Market Value (€)", f"{mv:,.2f}"); st.metric("Sum EV (€)", f"{sum_ev:,.2f}")
        with c3: st.metric("Cash (€)", f"{st.session_state.cash:,.2f}")

        st.markdown("#### 🔁 Reinvest Planner")
        r = st.slider("Reinvest rate r (0–100%)", 0, 100, 60)
        budget = round(st.session_state.cash * (r/100.0), 2)
        st.caption(f"Budget: €{budget:,.2f}")
        cand = watch[(watch["ev_pct"]>0)].copy()
        cand = cand[pd.to_numeric(cand["price"], errors="coerce")>0]
        cand["score"] = cand["odds"] * cand["ev_pct"].clip(lower=0)
        if cand.empty or cand["score"].sum()<=0 or budget<=0:
            alloc_df = pd.DataFrame(columns=["ticker","price","ev_pct","odds","alloc€","buy_shares"])
            st.info("No positive-EV candidates with valid prices, or budget = 0.")
        else:
            cand["w"] = cand["score"]/cand["score"].sum()
            cand["alloc€"] = cand["w"]*budget
            cand["buy_shares"] = np.floor(cand["alloc€"]/cand["price"]).astype(int)
            alloc_df = cand[["ticker","price","ev_pct","odds","alloc€","buy_shares"]].reset_index(drop=True)
        st.dataframe(alloc_df, use_container_width=True)
        st.caption("Sizing ∝ (odds × EV%) under cash budget. (PRB risk caps coming next).")

# --- Math Audit ---
st.markdown("---")
st.caption("🔎 Math Audit")
alerts = []
if not st.session_state.cards.empty:
    cards = st.session_state.cards
    # Check probabilities
    sdiff = np.abs((cards[["p_bull","p_base","p_bear"]].sum(axis=1) - 1.0)) > 1e-6
    if sdiff.any(): alerts.append(f"{int(sdiff.sum())} card(s) with probabilities not summing to 1.000.")
if not st.session_state.positions.empty and st.session_state.positions["price"].le(0).any():
    alerts.append("Some position prices ≤ 0 (update from left panel or CSV import).")
if alerts:
    for a in alerts: st.error(a)
else:
    st.success("All core math checks passed.")
st.caption("v0.4.2 — adds IBKR PortfolioAnalyst importer + robust column mapping.")
