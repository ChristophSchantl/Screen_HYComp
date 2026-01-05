# app.py — Master Scoring Model (Simplified • Pro) + Templates
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

st.set_page_config(page_title="Master Scoring Model (Simplified • Pro)", layout="wide")
st.title("📈 Master Scoring Model")
st.caption("Yahoo Finance • TTM-Kennzahlen • sektorrelative Perzentile • 52W nur aus History • schlankes Faktor-Set")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _to_float(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan

def _row(df: pd.DataFrame, keys):
    for k in keys:
        if isinstance(df, pd.DataFrame) and k in df.index:
            s = df.loc[k].dropna().astype(float)
            if len(s):
                return s
    return pd.Series(dtype=float)

def _ttm_sum(q_df: pd.DataFrame, keys, n=4):
    s = _row(q_df, keys)
    return float(s.iloc[:n].sum()) if len(s) else np.nan

def _percentile_rank(s: pd.Series) -> pd.Series:
    m = s.notna()
    if m.sum() <= 1:
        out = pd.Series(np.nan, index=s.index)
        out[m] = 0.5
        return out
    return s.rank(pct=True)

def _sector_percentile(df: pd.DataFrame, col: str, invert: bool = False) -> pd.Series:
    work = df.copy()
    work["sector"] = work.get("sector", pd.Series(index=work.index, dtype="object")).fillna("Unknown")
    p = work.groupby("sector", dropna=False, group_keys=False)[col].apply(_percentile_rank)
    if invert:
        p = 1 - p
    return (p * 100).clip(0, 100)

def _hist_close(ticker: str, period="5y", interval="1d") -> pd.Series:
    try:
        h = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if "Close" in h:
            return h["Close"].dropna()
    except Exception:
        pass
    try:
        h = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if "Close" in h:
            return h["Close"].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────
# Metrics (robust; 52W nur aus History)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60 * 30)
def fetch_metrics(tk: str) -> dict:
    t = yf.Ticker(tk)
    try:
        try:
            info = t.get_info()
        except Exception:
            info = getattr(t, "info", {}) or {}

        name = info.get("longName") or info.get("shortName") or tk

        px = _hist_close(tk, "5y", "1d")
        if px.empty:
            raise RuntimeError("no_price_history")
        price = float(px.iloc[-1])

        px1y = _hist_close(tk, "1y", "1d")
        if px1y.empty:
            px1y = px.tail(252)
        low_52w = float(px1y.min())
        high_52w = float(px1y.max())
        rng = high_52w - low_52w
        pos_52w = (price - low_52w) / (rng if rng > 0 else np.nan)

        # Dividenden (TTM + 5y Median Yield)
        try:
            div = t.get_dividends()
        except Exception:
            div = getattr(t, "dividends", pd.Series(dtype=float))

        pm = px.resample("M").last()
        dm = div.resample("M").sum().reindex(pm.index, fill_value=0.0) if isinstance(div, pd.Series) else pd.Series(0.0, index=pm.index)
        ttm_div_m = dm.rolling(12, min_periods=1).sum()
        yld_ttm = float(ttm_div_m.iloc[-1] / price) if price > 0 else np.nan
        yld_med5 = float((ttm_div_m / pm).tail(min(60, len(pm))).median()) if len(pm) else np.nan

        # Quarterlies
        q_is = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
        q_bs = t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame()
        q_cf = t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame()

        revenue = _ttm_sum(q_is, ["Total Revenue", "Revenue"])
        ebitda = _ttm_sum(q_is, ["EBITDA", "Ebitda"])
        op_cf = _ttm_sum(q_cf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
        capex_y = _ttm_sum(q_cf, ["Capital Expenditures", "Capital Expenditure"])
        capex = -capex_y if np.isfinite(capex_y) else np.nan
        fcf = op_cf - capex if np.isfinite(op_cf) and np.isfinite(capex) else np.nan

        ebitda_margin = (ebitda / revenue) if (np.isfinite(ebitda) and np.isfinite(revenue) and revenue > 0) else np.nan
        fcf_margin = (fcf / revenue) if (np.isfinite(fcf) and np.isfinite(revenue) and revenue > 0) else np.nan

        equity = _ttm_sum(q_bs, ["Total Stockholder Equity", "Total Equity Gross Minority Interest"])
        total_debt = _ttm_sum(q_bs, ["Long Term Debt"]) + _ttm_sum(q_bs, ["Short Long Term Debt", "Short Term Debt"])
        de_ratio = (total_debt / equity) if (np.isfinite(total_debt) and np.isfinite(equity) and equity > 0) else np.nan

        # Beta (2y weekly)
        try:
            spx = yf.Ticker("^GSPC").history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
            stw = t.history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
            bdf = pd.concat([stw, spx], axis=1).dropna()
            beta = float(np.polyfit(bdf.iloc[:, 1].values, bdf.iloc[:, 0].values, 1)[0]) if len(bdf) > 10 else np.nan
        except Exception:
            beta = np.nan

        # EV/EBITDA, Meta
        mcap = _to_float(info.get("marketCap"))
        pe = _to_float(info.get("trailingPE"))
        cash = _to_float(info.get("totalCash"))
        debt = _to_float(info.get("totalDebt"))
        ev = mcap if np.isfinite(mcap) else np.nan
        if np.isfinite(ev):
            ev += (debt if np.isfinite(debt) else 0) - (cash if np.isfinite(cash) else 0)
        ev_ebitda = (ev / ebitda) if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda > 0) else np.nan

        sector = (info.get("sector") or "Unknown")
        adv3 = _to_float(info.get("averageDailyVolume3Month"))

        return dict(
            ticker=tk,
            name=name,
            sector=sector,
            price=price,
            low_52w=low_52w,
            high_52w=high_52w,
            pos_52w=pos_52w,
            div_yield_ttm=yld_ttm,
            yield_5y_median=yld_med5,
            pe_ttm=pe,
            ev_ebitda_ttm=ev_ebitda,
            de_ratio=de_ratio,
            fcf_margin_ttm=fcf_margin,
            ebitda_margin_ttm=ebitda_margin,
            beta_2y_w=beta,
            market_cap=mcap,
            adv_3m=adv3,
            error=np.nan,
        )
    except Exception as e:
        return {"ticker": tk, "name": tk, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# Scoring – schlank, ohne Schwellen/Caps
# ─────────────────────────────────────────────────────────────
FACTORS = ["sc_yield", "sc_52w", "sc_pe", "sc_ev_ebitda", "sc_de", "sc_fcfm", "sc_ebitdam", "sc_beta", "sc_ygap"]

DEFAULT_W = {
    "sc_yield": 0.0,
    "sc_52w": 0.50,
    "sc_pe": 0.20,
    "sc_ev_ebitda": 0.20,
    "sc_de": 0.05,
    "sc_fcfm": 0.05,
    "sc_ebitdam": 0.00,
    "sc_beta": 0.00,
    "sc_ygap": 0.00,
}

PARAMS_DEF = dict(
    yield_floor=0.04,
    yield_scale=0.04,
    invert_52w=True,
    pos_52w_gamma=1.5,
)

# ─────────────────────────────────────────────────────────────
# Templates / Presets
# ─────────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {"weights": DEFAULT_W, "params": PARAMS_DEF},
    "Value": {
        "weights": {
            "sc_yield": 0.05,
            "sc_52w": 0.10,
            "sc_pe": 0.30,
            "sc_ev_ebitda": 0.30,
            "sc_de": 0.10,
            "sc_fcfm": 0.10,
            "sc_ebitdam": 0.05,
            "sc_beta": 0.00,
            "sc_ygap": 0.00,
        },
        "params": dict(yield_floor=0.02, yield_scale=0.05, invert_52w=True, pos_52w_gamma=1.2),
    },
    "Growth": {
        # In deinem Faktor-Set ist "Growth" indirekt (Quality-Growth): Margen/FCF + etwas Momentum/Beta
        "weights": {
            "sc_yield": 0.00,
            "sc_52w": 0.10,
            "sc_pe": 0.05,
            "sc_ev_ebitda": 0.05,
            "sc_de": 0.10,
            "sc_fcfm": 0.35,
            "sc_ebitdam": 0.25,
            "sc_beta": 0.10,
            "sc_ygap": 0.00,
        },
        "params": dict(yield_floor=0.00, yield_scale=0.04, invert_52w=False, pos_52w_gamma=1.0),
    },
    "Contrarian": {
        "weights": {
            "sc_yield": 0.00,
            "sc_52w": 0.45,
            "sc_pe": 0.15,
            "sc_ev_ebitda": 0.15,
            "sc_de": 0.05,
            "sc_fcfm": 0.10,
            "sc_ebitdam": 0.00,
            "sc_beta": 0.00,
            "sc_ygap": 0.10,
        },
        "params": dict(yield_floor=0.00, yield_scale=0.04, invert_52w=True, pos_52w_gamma=1.8),
    },
    "High Yield": {
        "weights": {
            "sc_yield": 0.40,
            "sc_52w": 0.05,
            "sc_pe": 0.10,
            "sc_ev_ebitda": 0.10,
            "sc_de": 0.10,
            "sc_fcfm": 0.15,
            "sc_ebitdam": 0.00,
            "sc_beta": 0.00,
            "sc_ygap": 0.10,
        },
        "params": dict(yield_floor=0.05, yield_scale=0.05, invert_52w=True, pos_52w_gamma=1.2),
    },
}

def _beta_score_linear(beta: float) -> float:
    # Beta 0.5 → 100, 1.0 → 50, 1.5 → 0 (linear)
    if not np.isfinite(beta):
        return 50.0
    return float(np.clip(100 - (beta - 0.5) * 100, 0, 100))

def build_scores(
    df: pd.DataFrame,
    weights: dict,
    params: dict,
    *,
    fixed_denominator: bool = True,
    missing_policy: str = "neutral50",
) -> pd.DataFrame:
    P = {**PARAMS_DEF, **(params or {})}
    d = df.copy()

    # Yield-Faktor (Floor/Scale)
    d["sc_yield"] = (np.clip((d["div_yield_ttm"] - P["yield_floor"]) / P["yield_scale"], 0, 1) * 100)

    # 52W-Faktor (invertierbar + Gamma)
    base = (1 - d["pos_52w"]) if P["invert_52w"] else d["pos_52w"]
    d["sc_52w"] = (np.clip(base, 0, 1) ** P["pos_52w_gamma"]) * 100

    # Sektorrelative Faktoren
    d["sc_pe"] = _sector_percentile(d, "pe_ttm", invert=True)
    d["sc_ev_ebitda"] = _sector_percentile(d, "ev_ebitda_ttm", invert=True)
    d["sc_de"] = _sector_percentile(d, "de_ratio", invert=True)
    d.loc[(~np.isfinite(d["de_ratio"])) | (d["de_ratio"] < 0), "sc_de"] = 0
    d["sc_fcfm"] = _sector_percentile(d, "fcf_margin_ttm", invert=False)
    d["sc_ebitdam"] = _sector_percentile(d, "ebitda_margin_ttm", invert=False)

    # Beta – simple linear score
    d["sc_beta"] = d["beta_2y_w"].apply(_beta_score_linear)

    # Yield-Gap vs 5y-Median (sektorrelativ)
    ygap = np.where(d["yield_5y_median"] > 0, d["div_yield_ttm"] / d["yield_5y_median"] - 1.0, np.nan)
    d["sc_ygap"] = _sector_percentile(pd.DataFrame({"sector": d["sector"], "ygap": ygap}), "ygap", invert=False)

    # Aggregation
    S = d[FACTORS].astype(float)

    W = pd.Series(weights).reindex(FACTORS).fillna(0.0)

    if missing_policy == "neutral50":
        S = S.fillna(50.0)
        num = (S * W).sum(axis=1)
        den = float(W.sum()) if fixed_denominator else ((~S.isna()) * W).sum(axis=1)

    elif missing_policy == "sector_median":
        for c in S.columns:
            med = d.groupby("sector")[c].transform("median")
            S[c] = S[c].fillna(med).fillna(50.0)
        num = (S * W).sum(axis=1)
        den = float(W.sum()) if fixed_denominator else ((~S.isna()) * W).sum(axis=1)

    elif missing_policy == "skip":
        # Skip = fehlende Faktoren zählen weder im Zähler noch (optional) im Nenner
        Sm = S.notna()
        num = (S.fillna(0.0) * W).sum(axis=1)
        den = float(W.sum()) if fixed_denominator else (Sm * W).sum(axis=1)

    else:
        # Fallback
        S = S.fillna(50.0)
        num = (S * W).sum(axis=1)
        den = float(W.sum()) if fixed_denominator else ((~S.isna()) * W).sum(axis=1)

    d["score"] = np.where(den > 0, num / den, np.nan).clip(0, 100)
    d["rating"] = np.select(
        [d["score"] >= 75, (d["score"] >= 60) & (d["score"] < 75)],
        ["BUY", "ACCUMULATE/WATCH"],
        "AVOID/HOLD",
    )
    return d

# ─────────────────────────────────────────────────────────────
# Sidebar — Inputs
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📥 Eingabe")
csv_file = st.sidebar.file_uploader("CSV mit Ticker-Spalte", type=["csv"])
tickers = []
if csv_file is not None:
    try:
        df_csv = pd.read_csv(csv_file)
    except Exception:
        csv_file.seek(0)
        df_csv = pd.read_csv(csv_file, sep=";")
    col = st.sidebar.selectbox("Ticker-Spalte", df_csv.columns.tolist())
    tickers = df_csv[col].astype(str).str.strip().replace({"nan": np.nan}).dropna().tolist()

manual = st.sidebar.text_area("Ticker manuell (kommasepariert)", "DHL.DE, DBK.DE, T, VZ, MO")
tickers = sorted({*tickers, *[s.strip().upper() for s in manual.split(",") if s.strip()]})
st.sidebar.caption(f"Watchlist: **{len(tickers)}**")

# ─────────────────────────────────────────────────────────────
# Sidebar — Template Auswahl (Presets)
# ─────────────────────────────────────────────────────────────
st.sidebar.header("🧠 Template")
preset_name = st.sidebar.selectbox(
    "Vorlage wählen",
    list(PRESETS.keys()),
    index=0,
    help="Setzt vordefinierte Gewichte (und optional Parameter). Danach kannst du per Slider feinjustieren.",
)

preset = PRESETS[preset_name]
preset_params = preset.get("params", PARAMS_DEF)
preset_w = preset.get("weights", DEFAULT_W)

# Preset nur dann anwenden, wenn gewechselt
if st.session_state.get("_preset_applied") != preset_name:
    st.session_state["_preset_applied"] = preset_name

    # Parameter defaults
    st.session_state["y_floor"] = float(preset_params.get("yield_floor", PARAMS_DEF["yield_floor"]))
    st.session_state["y_scale"] = float(preset_params.get("yield_scale", PARAMS_DEF["yield_scale"]))
    st.session_state["invert_52w"] = bool(preset_params.get("invert_52w", PARAMS_DEF["invert_52w"]))
    st.session_state["pos_gamma"] = float(preset_params.get("pos_52w_gamma", PARAMS_DEF["pos_52w_gamma"]))

    # Weight defaults
    for k in FACTORS:
        st.session_state[f"w_{k}"] = float(preset_w.get(k, DEFAULT_W.get(k, 0.0)))

# ─────────────────────────────────────────────────────────────
# Sidebar — Parameter & Gewichte
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Scoring-Parameter (schlank)")
y_floor = st.sidebar.number_input(
    "Yield-Floor",
    min_value=0.0,
    max_value=0.5,
    step=0.005,
    format="%.3f",
    key="y_floor",
)
y_scale = st.sidebar.number_input(
    "Yield-Scale",
    min_value=0.001,
    max_value=1.0,
    step=0.005,
    format="%.3f",
    key="y_scale",
)
invert_52w = st.sidebar.checkbox("52W invertieren (nah am Low = besser)", key="invert_52w")
pos_gamma = st.sidebar.slider("52W-Gamma", 0.3, 3.0, 1.5, 0.1, key="pos_gamma")

params = dict(yield_floor=y_floor, yield_scale=y_scale, invert_52w=invert_52w, pos_52w_gamma=pos_gamma)

st.sidebar.subheader("⚖️ Gewichte")
labels = {
    "sc_yield": "Yield",
    "sc_52w": "52W",
    "sc_pe": "PE(inv)",
    "sc_ev_ebitda": "EV/EBITDA(inv)",
    "sc_de": "D/E(inv)",
    "sc_fcfm": "FCF-Marge",
    "sc_ebitdam": "EBITDA-Marge",
    "sc_beta": "Beta",
    "sc_ygap": "Yield-Gap",
}

tmp_w = {}
for k in FACTORS:
    tmp_w[k] = st.sidebar.slider(
        labels.get(k, k),
        0.0,
        1.0,
        step=0.01,
        key=f"w_{k}",
    )

sum_w = float(np.sum(list(tmp_w.values())))
weights = {k: (v / sum_w) for k, v in tmp_w.items()} if sum_w > 0 else DEFAULT_W
st.sidebar.caption(f"Gewichtssumme (normalisiert): **{sum(weights.values()):.2f}**")

st.sidebar.subheader("🧩 Aggregation/Fehlwerte")
fixed_den = st.sidebar.checkbox("Feste Gewichtssumme (kein Re-Weighting)", value=True)
missing_policy = st.sidebar.selectbox("Missing-Policy", ["neutral50", "sector_median", "skip"], index=0)

run = st.sidebar.button("🔎 Score berechnen", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
st.subheader("Watchlist")
if tickers:
    with st.expander(f"anzeigen ({len(tickers)})"):
        st.code(", ".join(tickers), wrap_lines=True)
else:
    st.info("CSV laden oder Ticker manuell eingeben.")

if run and tickers:
    rows = []
    pbar = st.progress(0.0, text="Kennzahlen: 0/0")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_metrics, tk): tk for tk in tickers}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            pbar.progress(done / len(futs), text=f"Kennzahlen: {done}/{len(futs)}")

    df = pd.DataFrame(rows)
    base = df[df["error"].isna()].copy()
    if base.empty:
        st.warning("Keine verwertbaren Daten.")
        st.dataframe(df, use_container_width=True)
        st.stop()

    out = build_scores(
        base,
        weights=weights,
        params=params,
        fixed_denominator=fixed_den,
        missing_policy=missing_policy,
    ).copy()

    out["div_yield_%"] = out["div_yield_ttm"] * 100
    out["near_52w_low_%"] = (1 - out["pos_52w"]).clip(0, 1) * 100

    cols = [
        "ticker",
        "name",
        "sector",
        "price",
        "low_52w",
        "high_52w",
        "pos_52w",
        "near_52w_low_%",
        "pe_ttm",
        "ev_ebitda_ttm",
        "de_ratio",
        "fcf_margin_ttm",
        "ebitda_margin_ttm",
        "beta_2y_w",
        "market_cap",
        "adv_3m",
        "score",
        "rating",
        "error",
    ]

    st.subheader("Ergebnisse (Ranking)")
    st.dataframe(
        out.sort_values("score", ascending=False).reset_index(drop=True).round(3)[cols],
        use_container_width=True,
    )

    # Executive Summary (kompakt)
    st.markdown("### Executive Summary")
    vc = out["rating"].value_counts()
    kpi_total = len(out)
    kpi_buy = int(vc.get("BUY", 0))
    kpi_acc = int(vc.get("ACCUMULATE/WATCH", 0))
    kpi_hold = int(vc.get("AVOID/HOLD", 0))
    kpi_score_avg = float(out["score"].mean())
    kpi_yield_med = float(out["div_yield_%"].median())
    kpi_nearlow_med = float(out["near_52w_low_%"].median())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", kpi_total)
    c2.metric("BUY", kpi_buy)
    c3.metric("ACC/W", kpi_acc)
    c4.metric("Score ⌀", f"{kpi_score_avg:.1f}")
    c5.metric("Yield (Med.)", f"{kpi_yield_med:.2f}%")
    st.metric("Nähe 52W-Low (Med.)", f"{kpi_nearlow_med:.1f}%")

    # Top-10 Faktor-Beiträge
    st.markdown("### Top 10 – Faktor-Beitrag")
    lbl = {
        "sc_yield": "Yield",
        "sc_52w": "52W",
        "sc_pe": "PE(inv)",
        "sc_ev_ebitda": "EV/EBITDA(inv)",
        "sc_de": "D/E(inv)",
        "sc_fcfm": "FCF%",
        "sc_ebitdam": "EBITDA%",
        "sc_beta": "Beta",
        "sc_ygap": "YieldGap",
    }
    top = out.sort_values("score", ascending=False).head(10).copy()
    S = top[FACTORS].astype(float)
    Wser = pd.Series(weights).reindex(FACTORS).fillna(0.0)
    contrib = S.mul(Wser, axis=1)
    contrib.columns = [lbl.get(c, c) for c in contrib.columns]
    contrib.insert(0, "ticker", top["ticker"].values)
    contrib.insert(1, "score", top["score"].values)
    st.dataframe(contrib.set_index("ticker").round(2), use_container_width=True)

    # Faktor-Heatmap
    st.markdown("### Universum – Faktor-Median (0–100)")
    factor_meds = out[FACTORS].median().rename(lambda c: lbl.get(c, c))
    st.dataframe(
        pd.DataFrame(factor_meds, columns=["Median"]).T.style.background_gradient(axis=1),
        use_container_width=True,
    )

    # Exporte (inkl. name-Spalte)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    c_us, c_eu, c_xlsx = st.columns(3)
    with c_us:
        st.download_button(
            "⬇️ CSV (US)",
            out.to_csv(index=False).encode("utf-8"),
            file_name=f"scores_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_eu:
        st.download_button(
            "⬇️ CSV (EU)",
            out.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"scores_{ts}_eu.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_xlsx:
        buf = BytesIO()
        out.to_excel(buf, index=False, sheet_name="Scores")
        buf.seek(0)
        st.download_button(
            "⬇️ Excel",
            buf.getvalue(),
            file_name=f"scores_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Fehlerliste
    err = df[df["error"].notna()][["ticker", "name", "error"]]
    if not err.empty:
        st.warning("Hinweise/Fehler beim Laden einiger Ticker:")
        st.dataframe(err, use_container_width=True)

else:
    st.caption("Tipp: EU/UK-Suffixe (.DE, .L, .TO, .PA, .AX …).")
