# app.py
# Catalyst EV Calculator — unified single-file Streamlit app
# Paste into Streamlit, then run: streamlit run app.py

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Catalyst EV Calculator", layout="wide")

# -------------------------
# Helpers (parsing & mapping)
# -------------------------
def _num_series(s):
    return pd.to_numeric(s, errors="coerce")

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _map_positions_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a variety of headers and map them into:
    ticker, shares, avg, price
    """
    df = _clean_cols(df_raw)
    # Candidate headers for each target
    cand = {
        "ticker": ["ticker","symbol","sym","asset","security","instrument"],
        "shares": ["shares","qty","quantity","position","units"],
        "avg":    ["avg","avgcost","avg_cost","averagecost","average_cost","cost_basis","avg price","average price"],
        "price":  ["price","last","mktprice","marketprice","close","px","current price"]
    }

    out = pd.DataFrame()
    # TICKER required
    for k in cand["ticker"]:
        if k in df.columns:
            out["ticker"] = df[k]
            break
    if "ticker" not in out.columns:
        raise ValueError("Positions CSV error: missing 'ticker' (aka symbol).")

    # SHARES required
    for k in cand["shares"]:
        if k in df.columns:
            out["shares"] = _num_series(df[k])
            break
    if "shares" not in out.columns:
        raise ValueError("Positions CSV error: missing 'shares' (aka qty).")

    # AVG optional
    for k in cand["avg"]:
        if k in df.columns:
            out["avg"] = _num_series(df[k])
            break
    if "avg" not in out.columns:
        out["avg"] = np.nan

    # PRICE optional (we can use avg as fallback later)
    for k in cand["price"]:
        if k in df.columns:
            out["price"] = _num_series(df[k])
            break
    if "price" not in out.columns:
        out["price"] = np.nan

    # normalize tickers
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    # drop rows with no shares or no ticker
    out = out.dropna(subset=["ticker","shares"])
    out["shares"] = out["shares"].astype(float)
    return out

def _map_cards_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a variety of headers and map them into the minimal schema:
    ticker, t_bull, t_base, t_bear, p_bull, p_base, p_bear, last(optional)
    """
    df = _clean_cols(df_raw)
    def pick(df, names, required=True, default=np.nan):
        for n in names:
            if n in df.columns:
                return _num_series(df[n]) if df[n].dtype != object else df[n]
        if required:
            raise ValueError(f"Catalyst CSV error: missing one of {names}")
        return pd.Series(default, index=df.index, dtype="float64")

    # Ticker
    tick = None
    for n in ["ticker","symbol","sym","asset"]:
        if n in df.columns:
            tick = df[n].astype(str).str.upper().str.strip()
            break
    if tick is None:
        raise ValueError("Catalyst CSV error: missing 'ticker' column.")

    out = pd.DataFrame({
        "ticker": tick,
        "t_bull": pick(df, ["t_bull","target_bull","bull_target","bull_price"]),
        "t_base": pick(df, ["t_base","target_base","base_target","base_price"]),
        "t_bear": pick(df, ["t_bear","target_bear","bear_target","bear_price"]),
        "p_bull": pick(df, ["p_bull","prob_bull","bull_p","bull_prob"]),
        "p_base": pick(df, ["p_base","prob_base","base_p","base_prob"]),
        "p_bear": pick(df, ["p_bear","prob_bear","bear_p","bear_prob"]),
    })
    # Optional last price
    if "last" in df.columns:
        out["last"] = _num_series(df["last"])
    elif "price" in df.columns:
        out["last"] = _num_series(df["price"])
    else:
        out["last"] = np.nan

    # Drop incomplete rows (must have targets and probabilities)
    must_have = ["t_bull","t_base","t_bear","p_bull","p_base","p_bear"]
    out = out.dropna(subset=must_have, how="any")
    return out

# -------------------------
# EV Engine
# -------------------------
def normalize_probs(row, pcols=("p_bull","p_base","p_bear")):
    p = pd.to_numeric(row[list(pcols)], errors="coerce")
    s = p.sum()
    warn = False
    if not np.isfinite(s) or s <= 0:
        # fallback to priors
        p = pd.Series([0.35, 0.45, 0.20], index=pcols)
        s = 1.0
        warn = True
    elif abs(s - 1.0) > 1e-6:
        p = p / s
        warn = True
    row[list(pcols)] = p.values
    return row, warn

def compute_ev_table(positions_df, cards_df, price_source="positions",
                     assumed_price=None, L=1.0, r_annual=0.08, hold_days=5,
                     macro_weight=1.0):
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(), ["No positions loaded."]
    if cards_df is None or cards_df.empty:
        return pd.DataFrame(), ["No catalyst cards loaded."]

    pos = positions_df.copy()
    cards = cards_df.copy()
    pos["ticker"] = pos["ticker"].astype(str).str.upper().str.strip()
    cards["ticker"] = cards["ticker"].astype(str).str.upper().str.strip()

    # numeric casting
    for c in ["shares","avg","price"]:
        if c in pos.columns:
            pos[c] = _num_series(pos[c])
    for c in ["last","t_bull","t_base","t_bear","p_bull","p_base","p_bear"]:
        if c in cards.columns:
            cards[c] = _num_series(cards[c])

    df = pos.merge(cards, on="ticker", how="inner", suffixes=("_pos","_card"))
    warns = []
    if df.empty:
        return df, ["No ticker overlap between positions and catalyst cards."]

    # normalize probabilities per row
    prob_warn_rows = []
    rows = []
    for _, r in df.iterrows():
        r2 = r.copy()
        r2, w = normalize_probs(r2, pcols=("p_bull","p_base","p_bear"))
        if w:
            prob_warn_rows.append(str(r2["ticker"]))
        rows.append(r2)
    df = pd.DataFrame(rows)
    if prob_warn_rows:
        warns.append("Probabilities renormalized for: " + ", ".join(sorted(set(prob_warn_rows))))

    # choose price
    if price_source == "positions":
        use_price = df["price"]
        use_price = use_price.where(use_price.notna(), df.get("avg", np.nan))
    elif price_source == "cards_last":
        use_price = df.get("last", np.nan)
    else:  # assumed
        if assumed_price is None or not np.isfinite(assumed_price):
            return pd.DataFrame(), ["Assumed price not provided / invalid."]
        use_price = pd.Series(float(assumed_price), index=df.index, dtype=float)

    df["use_price"] = _num_series(use_price)
    miss = df["use_price"].isna()
    if miss.any():
        missing_tk = sorted(df.loc[miss, "ticker"].unique())
        warns.append(f"Missing price for: {missing_tk}")
        df = df.loc[~miss].copy()
    if df.empty:
        return df, warns if warns else ["No rows with usable price."]

    # EV target per share (weighted targets)
    df["EV_target"] = (
        df["p_bull"] * df["t_bull"] +
        df["p_base"] * df["t_base"] +
        df["p_bear"] * df["t_bear"]
    ) * float(macro_weight)

    # EV €/share and %
    df["EV_abs_sh"] = df["EV_target"] - df["use_price"]
    df["EV_pct"] = (df["EV_abs_sh"] / df["use_price"]) * 100.0

    # Position EV (unlevered)
    df["PosEV_unlev_€"] = df["EV_abs_sh"] * df["shares"]

    # Leverage & funding drag (LCM/DMCR)
    L = max(1.0, float(L))
    c = float(r_annual) * (float(hold_days)/365.0) * max(0.0, L - 1.0)
    df["funding_drag_sh"] = c * df["use_price"]
    df["EV_abs_sh_lev"] = df["EV_abs_sh"] * L - df["funding_drag_sh"]
    df["EV_pct_lev"] = (df["EV_abs_sh_lev"] / df["use_price"]) * 100.0
    df["PosEV_lev_€"] = df["EV_abs_sh_lev"] * df["shares"]

    # Outcome return %
    df["Bull_ret_%"] = ((df["t_bull"] - df["use_price"]) / df["use_price"]) * 100.0
    df["Base_ret_%"] = ((df["t_base"] - df["use_price"]) / df["use_price"]) * 100.0
    df["Bear_ret_%"] = ((df["t_bear"] - df["use_price"]) / df["use_price"]) * 100.0

    cols = [
        "ticker","shares","use_price",
        "t_bull","t_base","t_bear","p_bull","p_base","p_bear",
        "EV_target","EV_abs_sh","EV_pct","PosEV_unlev_€",
        "EV_abs_sh_lev","EV_pct_lev","PosEV_lev_€",
        "Bull_ret_%","Base_ret_%","Bear_ret_%","funding_drag_sh"
    ]
    out = df[[c for c in cols if c in df.columns]].sort_values("EV_pct", ascending=False).reset_index(drop=True)
    return out, warns

# -------------------------
# Session init
# -------------------------
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=["ticker","shares","avg","price"])
if "cards" not in st.session_state:
    st.session_state.cards = pd.DataFrame()

# -------------------------
# UI — Sidebar
# -------------------------
with st.sidebar:
    st.header("Data Upload")
    st.write("Upload your Positions CSV (from broker) and Catalyst Cards CSV.")

    pos_file = st.file_uploader("Positions CSV", type=["csv"], key="pos_up")
    if pos_file is not None:
        try:
            dfp_raw = pd.read_csv(pos_file)
        except Exception:
            pos_file.seek(0)
            dfp_raw = pd.read_csv(io.StringIO(pos_file.getvalue().decode("utf-8", errors="ignore")))
        try:
            st.session_state.positions = _map_positions_columns(dfp_raw)
            st.success(f"Loaded {len(st.session_state.positions)} positions.")
        except Exception as e:
            st.error(str(e))

    cards_file = st.file_uploader("Catalyst Cards CSV", type=["csv"], key="cards_up")
    if cards_file is not None:
        try:
            dfc_raw = pd.read_csv(cards_file)
        except Exception:
            cards_file.seek(0)
            dfc_raw = pd.read_csv(io.StringIO(cards_file.getvalue().decode("utf-8", errors="ignore")))
        try:
            st.session_state.cards = _map_cards_columns(dfc_raw)
            st.success(f"Loaded {len(st.session_state.cards)} catalyst rows.")
        except Exception as e:
            st.error(str(e))

    st.write("---")
    st.caption("Tip: CSV headers are flexible. Minimum required:\n"
               "- Positions: ticker, shares (avg/price optional)\n"
               "- Cards: ticker, t_bull/base/bear, p_bull/base/bear (last optional)")

# -------------------------
# UI — Main
# -------------------------
st.title("Catalyst EV Calculator")

colL, colR = st.columns([2, 1])
with colL:
    st.subheader("Positions (parsed)")
    if st.session_state.positions.empty:
        st.info("No positions loaded yet.")
    else:
        st.dataframe(st.session_state.positions, use_container_width=True, height=240)

    st.subheader("Catalyst Cards (parsed)")
    if st.session_state.cards.empty:
        st.info("No catalyst cards loaded yet.")
    else:
        st.dataframe(st.session_state.cards, use_container_width=True, height=280)

with colR:
    st.subheader("EV Settings")
    price_source = st.selectbox("Price source", ["positions","cards_last","assumed"])
    assumed = None
    if price_source == "assumed":
        assumed = st.number_input("Assumed price (applies to all)", min_value=0.0, value=1.0, step=0.01)
    macro_weight = st.slider("Macro weight (AdjEV multiplier)", 0.7, 1.6, 1.0, 0.05)
    L = st.number_input("Leverage L (≥1.0)", min_value=1.0, value=1.0, step=0.1)
    r_annual = st.number_input("Annual rate for funding rₐ", min_value=0.0, value=0.08, step=0.01, format="%.2f")
    hold_days = st.number_input("Expected hold days", min_value=1, value=5, step=1)

    run = st.button("Run EV math", type="primary", use_container_width=True)

st.write("---")
st.subheader("EV Results")

if run:
    ev_table, ev_warns = compute_ev_table(
        st.session_state.positions,
        st.session_state.cards,
        price_source=price_source,
        assumed_price=assumed,
        L=L, r_annual=r_annual, hold_days=hold_days,
        macro_weight=macro_weight
    )
    if ev_warns:
        for w in ev_warns:
            st.warning(w)

    if ev_table.empty:
        st.info("No EV rows to display. Check that tickers overlap between Positions and Cards.")
    else:
        st.dataframe(ev_table, use_container_width=True, height=420)

        # Portfolio totals
        tot_unlev = float(ev_table["PosEV_unlev_€"].sum()) if "PosEV_unlev_€" in ev_table else 0.0
        tot_lev = float(ev_table["PosEV_lev_€"].sum()) if "PosEV_lev_€" in ev_table else 0.0
        m1, m2 = st.columns(2)
        m1.metric("Portfolio PosEV (unlevered) €", f"{tot_unlev:,.2f}")
        m2.metric("Portfolio PosEV (levered) €", f"{tot_lev:,.2f}")

        # Download EV table
        csv_buf = ev_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download EV Table (CSV)",
            data=csv_buf,
            file_name="ev_results.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("Load both CSVs, choose settings, then click **Run EV math**.")

# Footer
st.caption("EV = Σ(p_i × target_i) − price; levered EV adjusts for funding drag. "
           "Ensure probabilities sum to 1.0 for each ticker.")
