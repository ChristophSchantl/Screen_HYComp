# streamlit_app.py
# -*- coding: utf-8 -*-
# High-Yield Dividend Screener (Yahoo Finance) – mit Spalten-Pruning & Whitelist/Blacklist

import warnings
warnings.filterwarnings("ignore", message=".*bottleneck.*", category=UserWarning)

import io
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# UI Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="High-Yield Dividend Screener", layout="wide")
st.title("High-Yield Dividend Screener")

st.caption("Kostenlose Daten via yfinance. Fehlende Spezial-Kennzahlen (CET1/SII/AFFO) werden neutral behandelt.")

# ─────────────────────────────────────────────────────────────
# Scoring – Standardgewichte & Richtungen
# ─────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    # Sustainability 45
    'HY_YIELD_FWD':10, 'HY_FCF_COVER':12, 'HY_PAYOUT':8,
    'HY_NETDEBT_EBITDA':6, 'HY_INT_COVERAGE':5, 'HY_DIV_STABILITY':4,
    # Quality 20
    'HY_ROIC_WACC':8, 'HY_MARGIN_STAB':4, 'HY_F_SCORE':4, 'HY_ACCRUALS':4,
    # Valuation 20
    'HY_FCF_YIELD':8, 'HY_EV_EBITDA':6, 'HY_DY_5Y_Z':6,
    # Momentum 10
    'HY_TR_MOM_6M_SR':6, 'HY_REV_3M':4,
    # Capital Return 5
    'HY_NET_BUYBACK_YIELD':3, 'HY_DILUTION_FLAG':2,
}
FACTOR_DIR = {
    'HY_YIELD_FWD': True, 'HY_FCF_COVER': True, 'HY_PAYOUT': False,
    'HY_NETDEBT_EBITDA': False, 'HY_INT_COVERAGE': True, 'HY_DIV_STABILITY': True,
    'HY_ROIC_WACC': True, 'HY_MARGIN_STAB': True, 'HY_F_SCORE': True, 'HY_ACCRUALS': False,
    'HY_FCF_YIELD': True, 'HY_EV_EBITDA': True, 'HY_DY_5Y_Z': True,
    'HY_TR_MOM_6M_SR': True, 'HY_REV_3M': True,
    'HY_NET_BUYBACK_YIELD': True, 'HY_DILUTION_FLAG': False,
    # Alternativen (werden in Free-Setup selten gefüllt, bleiben aber kompatibel)
    'HY_EPS_COVER': True, 'HY_CET1': True, 'HY_SII': True, 'HY_AFFO_COVER': True,
}

ALL_FACTORS = tuple(DEFAULT_WEIGHTS.keys())

# ─────────────────────────────────────────────────────────────
# Sidebar – Eingaben inkl. Whitelist/Blacklist & Pruning
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Einstellungen")

    default_tickers = "AZN.L,BATS.L,ENEL.MI,ORA.PA,SAN.PA,ENI.MI,ALV.DE,EOAN.DE,VNA.DE,SHELL.AS,SU.PA,RI.PA,ULVR.L,IMB.L"
    tickers_in = st.text_area(
        "Ticker (kommagetrennt, Yahoo-Format)",
        value=default_tickers,
        height=100,
        help="Suffixe beachten: .DE, .PA, .MI, .L, .SW, .AS, …",
    )

    hist_years = st.slider("Dividenden-Historie (Jahre) für 5y-Statistik", 3, 10, 5)
    price_years = st.slider("Kurs-Historie (Jahre) für Momentum", 1, 10, 3)

    st.markdown("---")
    # Pruning-Parameter
    min_cov_pct = st.slider("Min. Datenabdeckung pro Spalte (%)", 0, 100, 20,
                            help="Spalten mit geringerer Abdeckung werden entfernt.")
    drop_constant = st.checkbox("Konstante Spalten entfernen", value=True,
                                help="Entfernt Spalten mit nur einem (nicht-NaN) Wert.")

    st.markdown("---")
    st.subheader("Faktoren wählen")
    # Whitelist: initial alle (du kannst einzelne Faktoren abwählen)
    whitelist = st.multiselect(
        "Whitelist – nur diese Faktoren dürfen ins Scoring",
        options=sorted(ALL_FACTORS),
        default=sorted(ALL_FACTORS),
    )
    # Blacklist: optionale Subtraktion
    blacklist = st.multiselect(
        "Blacklist – diese Faktoren explizit ausschließen",
        options=sorted(ALL_FACTORS),
        default=[],
        help="Wird von der Whitelist abgezogen.",
    )
    st.caption("Effektiv genutzt = Whitelist MINUS Blacklist. Nicht verfügbare Spalten werden automatisch ignoriert.")

# ─────────────────────────────────────────────────────────────
# Helper: Normalisierung & Aggregation
# ─────────────────────────────────────────────────────────────
def winsorize(s: pd.Series, lower=0.005, upper=0.995) -> pd.Series:
    try:
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)
    except Exception:
        return s

def robust_zscore(s: pd.Series) -> pd.Series:
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if mad is None or not np.isfinite(mad) or mad <= 0:
        denom = s.std(skipna=True) + 1e-9
    else:
        denom = 1.4826 * mad
    out = (s - med) / (denom if denom != 0 else 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)

def to_0_100(z: pd.Series, clip=3.0) -> pd.Series:
    zc = z.clip(-clip, clip)
    return (zc + clip) * (90.0 / (2*clip)) + 5.0

def invert_if_needed(series: pd.Series, higher_is_better: bool) -> pd.Series:
    return series if higher_is_better else -series

def industry_neutralize(scores: pd.Series, industries: pd.Series) -> pd.Series:
    df = pd.DataFrame({"score": scores, "ind": industries})
    try:
        adj = df.groupby("ind", dropna=False)["score"].transform(lambda x: x - x.mean())
        return adj
    except Exception:
        return scores

# ─────────────────────────────────────────────────────────────
# Datenabdeckung messen & Spalten prunen
# ─────────────────────────────────────────────────────────────
def coverage_report(df: pd.DataFrame, min_cov: float = 0.2, drop_constant: bool = True) -> Tuple[pd.DataFrame, list[str]]:
    if df.empty:
        rep = pd.DataFrame(columns=["coverage","nunique","all_na","constant","dtype","keep"])
        return rep, []
    cov = df.notna().mean().rename("coverage")
    nunq = df.nunique(dropna=True).rename("nunique")
    all_na = df.isna().all().rename("all_na")
    const = (nunq <= 1).rename("constant")
    dtyp = df.dtypes.astype(str).rename("dtype")
    rep = pd.concat([cov, nunq, all_na, const, dtyp], axis=1)
    keep_mask = (~rep["all_na"]) & (rep["coverage"] >= float(min_cov))
    if drop_constant:
        keep_mask &= (~rep["constant"])
    rep["keep"] = keep_mask
    keep_cols = rep.index[rep["keep"]].tolist()
    return rep.sort_values("coverage", ascending=False), keep_cols

def prune_columns(df: pd.DataFrame, min_cov: float = 0.2, drop_constant: bool = True,
                  keep_always: list[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rep, keep_cols = coverage_report(df, min_cov=min_cov, drop_constant=drop_constant)
    keep_always = keep_always or []
    keep_cols = list({*keep_cols, *(c for c in keep_always if c in df.columns)})
    return df[keep_cols].copy(), rep

# ─────────────────────────────────────────────────────────────
# Yahoo Finance Fetcher
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def yf_price_hist(ticker: str, period_years: int) -> pd.DataFrame:
    period = f"{period_years}y"
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return df

@st.cache_data(show_spinner=False)
def yf_dividends(ticker: str, period_years: int) -> pd.Series:
    t = yf.Ticker(ticker)
    div = t.dividends
    if div is None or div.empty:
        return pd.Series(dtype=float)
    cutoff = pd.Timestamp.today(tz=div.index.tz) - pd.DateOffset(years=period_years)
    return div[div.index >= cutoff].sort_index()

@st.cache_data(show_spinner=False)
def yf_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    fast = getattr(t, "fast_info", None)
    price = fast.get("last_price") if fast else None
    market_cap = fast.get("market_cap") if fast else None
    shares_out = fast.get("shares") if fast else None
    currency = fast.get("currency") if fast else None

    def last_from(df, names):
        if df is None or df.empty:
            return None
        for nm in names:
            if nm in df.index:
                s = df.loc[nm]
                s = pd.to_numeric(s, errors="coerce").dropna()
                if not s.empty:
                    return float(s.iloc[0])
        return None

    try: fin = t.financials
    except Exception: fin = None
    try: bs = t.balance_sheet
    except Exception: bs = None
    try: cf = t.cashflow
    except Exception: cf = None

    ebit = last_from(fin, ["Ebit", "EBIT"])
    depam = last_from(cf, ["Depreciation And Amortization", "Depreciation", "Reconciled Depreciation"])
    ebitda = last_from(fin, ["Ebitda", "EBITDA"])
    if ebitda is None and ebit is not None and depam is not None:
        ebitda = ebit + depam

    interest_expense = last_from(fin, ["Interest Expense"])
    if interest_expense is not None:
        interest_expense = abs(interest_expense)

    total_debt = last_from(bs, ["Total Debt", "Short Long Term Debt", "Long Term Debt"])
    cash_eq = last_from(bs, ["Cash And Cash Equivalents", "Cash"])
    net_debt = (total_debt - cash_eq) if (total_debt is not None and cash_eq is not None) else None

    cfo = last_from(cf, ["Cash From Operating Activities", "Operating Cash Flow"])
    capex = last_from(cf, ["Capital Expenditure", "Investments in PPE", "Capital Expenditures"])
    fcf_ttm = (cfo - abs(capex)) if (cfo is not None and capex is not None) else None

    net_income = last_from(fin, ["Net Income"])
    total_assets = last_from(bs, ["Total Assets"])
    avg_total_assets = float(total_assets) if total_assets is not None else None

    return {
        "ticker": ticker,
        "price": price,
        "market_cap": market_cap,
        "shares_outstanding_ttm": shares_out,
        "currency": currency,
        "ebit": ebit,
        "ebitda": ebitda,
        "interest_expense": interest_expense,
        "total_debt": total_debt,
        "cash_and_equivalents": cash_eq,
        "net_debt": net_debt,
        "cfo": cfo,
        "capex": capex,
        "fcf_ttm": fcf_ttm,
        "div_cash_paid_ttm": None,  # selten zuverlässig frei verfügbar
        "net_income": net_income,
        "avg_total_assets": avg_total_assets,
    }

def compute_ttm_dps(div_series: pd.Series) -> float:
    if div_series is None or div_series.empty:
        return np.nan
    end = div_series.index.max()
    start = end - pd.Timedelta(days=365)
    return float(div_series[(div_series.index > start) & (div_series.index <= end)].sum())

def dy_5y_stats(ticker: str, years: int = 5) -> tuple[float, float]:
    px = yf_price_hist(ticker, period_years=years)
    dv = yf_dividends(ticker, period_years=years+1)
    if px is None or px.empty or dv is None:
        return (np.nan, np.nan)
    mpx = px["Close"].resample("M").last()
    dv_m = dv.resample("M").sum()
    dv_12m = dv_m.rolling(window=12, min_periods=1).sum()
    yld = (dv_12m / mpx).replace([np.inf, -np.inf], np.nan).dropna()
    if yld.empty:
        return (np.nan, np.nan)
    return (float(yld.mean()), float(yld.std(ddof=0)))

def momentum_6m(ticker: str, years: int = 3) -> tuple[float, float]:
    px = yf_price_hist(ticker, period_years=years)
    if px is None or px.empty:
        return (np.nan, np.nan)
    six_months = int(252/2)
    if len(px) < six_months + 1:
        return (np.nan, np.nan)
    ret_6m = px["Close"].iloc[-1] / px["Close"].iloc[-(six_months+1)] - 1.0
    last6 = px["Close"].pct_change().dropna().iloc[-six_months:]
    vol_6m = float(last6.std(ddof=0) * np.sqrt(252)) if not last6.empty else np.nan
    return (float(ret_6m), vol_6m)

# ─────────────────────────────────────────────────────────────
# Faktoren (Free-Daten)
# ─────────────────────────────────────────────────────────────
def compute_factors_from_free(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if set(['dps_ntm','price']).issubset(df.columns):
        out['HY_YIELD_FWD'] = (df['dps_ntm'] / df['price']).replace([np.inf, -np.inf], np.nan)

    if 'fcf_ttm' in df.columns:
        div_paid = df['div_cash_paid_ttm'] if 'div_cash_paid_ttm' in df.columns else np.nan
        out['HY_FCF_COVER'] = (df['fcf_ttm'] / pd.to_numeric(div_paid).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if set(['eps_norm_ttm','dps_ntm']).issubset(df.columns):
        out['HY_EPS_COVER'] = (df['eps_norm_ttm'] / df['dps_ntm'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['HY_PAYOUT']    = (df['dps_ntm'] / df['eps_norm_ttm'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if set(['net_debt','ebitda']).issubset(df.columns):
        out['HY_NETDEBT_EBITDA'] = (df['net_debt'] / df['ebitda'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if set(['ebit','interest_expense']).issubset(df.columns):
        out['HY_INT_COVERAGE'] = (df['ebit'] / df['interest_expense'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if set(['net_income','cfo','avg_total_assets']).issubset(df.columns):
        out['HY_ACCRUALS'] = ((df['net_income'] - df['cfo']) / df['avg_total_assets']).replace([np.inf, -np.inf], np.nan)

    if set(['fcf_ttm','market_cap']).issubset(df.columns):
        out['HY_FCF_YIELD'] = (df['fcf_ttm'] / df['market_cap'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if 'ev_to_ebitda' in df.columns:
        out['HY_EV_EBITDA'] = (1.0 / df['ev_to_ebitda'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if set(['dy_5y_avg','dy_5y_std','dps_ntm','price']).issubset(df.columns):
        out['HY_DY_5Y_Z'] = ((df['dps_ntm'] / df['price']) - df['dy_5y_avg']) / (df['dy_5y_std'] + 1e-9)

    if set(['ret_6m','vol_6m']).issubset(df.columns):
        out['HY_TR_MOM_6M_SR'] = df['ret_6m'] / (df['vol_6m'] + 1e-9)

    if set(['shares_outstanding_ttm','shares_outstanding_1y']).issubset(df.columns):
        chg = (df['shares_outstanding_ttm'] - df['shares_outstanding_1y']) / (df['shares_outstanding_1y'] + 1e-9)
        out['HY_NET_BUYBACK_YIELD'] = -chg
        out['HY_DILUTION_FLAG'] = (chg > 0.05).astype(float)

    return out

def kill_switch_free(df: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    ks = pd.Series(False, index=df.index)
    if {'HY_FCF_COVER','HY_NETDEBT_EBITDA','HY_INT_COVERAGE'}.issubset(factors.columns):
        cond_nf = (factors['HY_FCF_COVER'] < 0.8) & ((factors['HY_NETDEBT_EBITDA'] > 3.0) | (factors['HY_INT_COVERAGE'] < 2.0))
        ks = ks | cond_nf.fillna(False)
    return ks

def aggregate_scores(
    factors: pd.DataFrame,
    industries: pd.Series,
    weights: dict,
    factor_dir: dict,
) -> pd.DataFrame:
    norm = pd.DataFrame(index=factors.index)
    for f, w in weights.items():
        if w == 0 or f not in factors.columns:
            continue
        s = pd.to_numeric(factors[f], errors="coerce")
        s = winsorize(s, 0.005, 0.995)
        z = robust_zscore(s)
        s01 = to_0_100(invert_if_needed(z, factor_dir.get(f, True)))
        if f in ['HY_YIELD_FWD','HY_FCF_COVER','HY_PAYOUT','HY_NETDEBT_EBITDA','HY_FCF_YIELD','HY_EV_EBITDA']:
            s01 = industry_neutralize(s01, industries) + 50.0
        norm[f] = s01

    total_weight = float(sum([w for w in weights.values() if w > 0]))
    hy_score = sum(norm[f] * (weights[f] / total_weight) for f in norm.columns if f in weights)
    res = pd.DataFrame({'HY_Score': hy_score.clip(0, 100)}, index=factors.index)

    def subtotal(fs):
        present = [x for x in fs if x in norm.columns and weights.get(x,0) > 0]
        if not present:
            return pd.Series(index=factors.index, data=np.nan)
        wsum = sum(weights[x] for x in present)
        return sum(norm[x] * (weights[x] / wsum) for x in present)

    res['P_Sustainability'] = subtotal(['HY_YIELD_FWD','HY_FCF_COVER','HY_PAYOUT','HY_NETDEBT_EBITDA','HY_INT_COVERAGE','HY_DIV_STABILITY'])
    res['P_Quality']        = subtotal(['HY_ROIC_WACC','HY_MARGIN_STAB','HY_F_SCORE','HY_ACCRUALS'])
    res['P_Valuation']      = subtotal(['HY_FCF_YIELD','HY_EV_EBITDA','HY_DY_5Y_Z'])
    res['P_Momentum']       = subtotal(['HY_TR_MOM_6M_SR','HY_REV_3M'])
    res['P_CapitalReturn']  = subtotal(['HY_NET_BUYBACK_YIELD','HY_DILUTION_FLAG'])
    return res.join(norm.add_prefix("F_"))

# ─────────────────────────────────────────────────────────────
# Pipeline: Ticker -> Free-Daten -> Faktoren -> Pruning -> Score
# ─────────────────────────────────────────────────────────────
def build_dataset_from_yf(tickers: list[str], hist_years: int, price_years: int) -> pd.DataFrame:
    records = []
    for tk in tickers:
        tk = tk.strip()
        if not tk:
            continue
        with st.spinner(f"Lade Daten für {tk} …"):
            div = yf_dividends(tk, period_years=hist_years+1)
            dps_ttm = compute_ttm_dps(div)
            dy_avg, dy_std = dy_5y_stats(tk, years=hist_years)
            r6m, v6m = momentum_6m(tk, years=price_years)
            f = yf_fundamentals(tk)

            ev = None
            if f.get("market_cap") is not None:
                ev = f["market_cap"]
                if f.get("total_debt") is not None:
                    ev += f["total_debt"]
                if f.get("cash_and_equivalents") is not None:
                    ev -= f["cash_and_equivalents"]
            ev_to_ebitda = None
            if ev is not None and f.get("ebitda"):
                if f["ebitda"] != 0:
                    ev_to_ebitda = ev / f["ebitda"]

            eps_norm_ttm = None
            if f.get("net_income") is not None and f.get("shares_outstanding_ttm"):
                try:
                    eps_norm_ttm = f["net_income"] / float(f["shares_outstanding_ttm"])
                except Exception:
                    eps_norm_ttm = None

            records.append({
                "ticker": tk,
                "price": f.get("price"),
                "market_cap": f.get("market_cap"),
                "shares_outstanding_ttm": f.get("shares_outstanding_ttm"),
                "shares_outstanding_1y": np.nan,  # meist nicht frei verfügbar
                "currency": f.get("currency"),
                "dps_ntm": dps_ttm,  # Proxy: TTM als NTM
                "fcf_ttm": f.get("fcf_ttm"),
                "div_cash_paid_ttm": f.get("div_cash_paid_ttm"),
                "net_debt": f.get("net_debt"),
                "ebitda": f.get("ebitda"),
                "ebit": f.get("ebit"),
                "interest_expense": f.get("interest_expense"),
                "ev_to_ebitda": ev_to_ebitda,
                "dy_5y_avg": dy_avg,
                "dy_5y_std": dy_std,
                "ret_6m": r6m,
                "vol_6m": v6m,
                "eps_norm_ttm": eps_norm_ttm,
                "cfo": f.get("cfo"),
                "net_income": f.get("net_income"),
                "avg_total_assets": f.get("avg_total_assets"),
                "sector": "",
                "industry": "",
            })

    df = pd.DataFrame.from_records(records).set_index("ticker")
    return df

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
tickers = [t.strip() for t in tickers_in.split(",") if t.strip()]

if st.button("Daten laden & scoren", type="primary"):
    base = build_dataset_from_yf(tickers, hist_years=hist_years, price_years=price_years)
    if base.empty:
        st.warning("Keine Daten geladen. Prüfe Ticker/Suffixe.")
        st.stop()

    st.subheader("Rohdaten (Free-Daten, best effort)")
    st.dataframe(base, use_container_width=True)

    # Faktoren & Pruning
    factors = compute_factors_from_free(base)
    factors_pruned, fac_rep = prune_columns(
        factors,
        min_cov=min_cov_pct / 100.0,
        drop_constant=drop_constant
    )
    with st.expander("Datenabdeckung (Faktoren)"):
        st.dataframe(fac_rep.style.format({"coverage": "{:.0%}"}), use_container_width=True)

    # Effektive Faktor-Auswahl (Whitelist – Blacklist – Pruning)
    selected = [f for f in whitelist if f not in blacklist]
    available = list(factors_pruned.columns)
    effective = sorted([f for f in selected if f in available])

    st.success(f"Faktoren genutzt (nach Whitelist/Blacklist & Pruning): {', '.join(effective) if effective else '—'}")

    # Gewichte auf Basis der Auswahl setzen (nicht gewählte => Gewicht 0)
    weights_eff = DEFAULT_WEIGHTS.copy()
    for f in list(weights_eff.keys()):
        if f not in effective:
            weights_eff[f] = 0

    # Scoring
    scores = aggregate_scores(factors_pruned, industries=base.get("industry", pd.Series(index=base.index, data="")),
                              weights=weights_eff, factor_dir=FACTOR_DIR)
    kills = kill_switch_free(base, factors_pruned)

    # Ergebnis
    res = base.join(factors_pruned, how="left").join(scores, how="left")
    res["Kill_Switch"] = kills

    keep_always = ["price", "currency", "sector", "industry", "HY_Score", "Kill_Switch"]
    res_pruned, res_rep = prune_columns(
        res,
        min_cov=min_cov_pct / 100.0,
        drop_constant=drop_constant,
        keep_always=keep_always
    )
    with st.expander("Datenabdeckung (Gesamtausgabe)"):
        st.dataframe(res_rep.style.format({"coverage": "{:.0%}"}), use_container_width=True)

    st.subheader("Ranking")
    eligible = res_pruned[~res_pruned["Kill_Switch"]].copy()
    ranked = eligible.sort_values("HY_Score", ascending=False)

    st.markdown("**Top 25 (Kill-Switch gefiltert):**")
    st.dataframe(ranked.head(25), use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # Downloads (robust, ohne Absturz)
    # ─────────────────────────────────────────────────────────
    st.subheader("Download")

    def df_to_bytes(df: pd.DataFrame, kind: str = "csv"):
        if kind == "csv":
            return df.to_csv(index=True).encode("utf-8"), None
        if kind == "html":
            return df.to_html(index=True).encode("utf-8"), None
        if kind == "xlsx":
            bio = io.BytesIO()
            engine = None
            try:
                import xlsxwriter  # noqa: F401
                engine = "xlsxwriter"
            except Exception:
                pass
            if engine is None:
                try:
                    import openpyxl  # noqa: F401
                    engine = "openpyxl"
                except Exception:
                    return None, "Excel-Export deaktiviert: installiere 'xlsxwriter' oder 'openpyxl'."
            with pd.ExcelWriter(bio, engine=engine) as xw:
                df.to_excel(xw, index=True, sheet_name="scores")
            return bio.getvalue(), None
        return None, f"Unbekanntes Format: {kind}"

    c1, c2, c3 = st.columns(3)

    csv_bytes, _ = df_to_bytes(ranked, "csv")
    c1.download_button("CSV herunterladen", data=csv_bytes, file_name="hy_scores.csv", mime="text/csv")

    xlsx_bytes, xlsx_err = df_to_bytes(ranked, "xlsx")
    if xlsx_bytes is not None:
        c2.download_button(
            "Excel herunterladen",
            data=xlsx_bytes,
            file_name="hy_scores.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        c2.info(xlsx_err)

    html_bytes, _ = df_to_bytes(ranked, "html")
    c3.download_button("HTML herunterladen", data=html_bytes, file_name="hy_scores.html", mime="text/html")

    # Hinweise
    st.info(
        "Hinweise:\n"
        "- Forward DPS wird als TTM-Dividende angenähert (Europa: halb-/jährliche Zahler). "
        "Bei Sonderdividenden ggf. manuell adjustieren.\n"
        "- EV/EBITDA, FCF & weitere Fundamentals sind frei nicht immer verfügbar → Spalten-Pruning hält die Tabellen sauber.\n"
        "- Whitelist/Blacklist steuern, welche Faktoren überhaupt ins Scoring einfließen; fehlende Spalten werden automatisch ignoriert."
    )
else:
    st.caption("Bereit. Ticker prüfen und **Daten laden & scoren** klicken.")
