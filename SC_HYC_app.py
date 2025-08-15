# streamlit_eu_low_yield_screen_pro.py
# -*- coding: utf-8 -*-
# EU High-Yield Screener – 52W-Lows, D/E-Filter, FX (EUR/CHF), Index-Universen (Wikipedia/CSV),
# Presets (save/load) und Daily-Research-Dashboard mit Heatmap.
#
# Datenquellen:
# - Preise/Dividenden/Fundamentals: Yahoo Finance via yfinance (kostenlos)
# - FX: EZB eurofxref (täglich)
# - Indizes: Wikipedia (robustes Parsing + Fallback-Listen)

import warnings
warnings.filterwarnings("ignore", message=".*bottleneck.*", category=UserWarning)

import io
import re
import math
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="EU High-Yield Screener Pro", layout="wide")
st.title("EU High-Yield Screener – 52W-Lows, D/E, FX, Indizes & Heatmap")
st.caption(
    "Datenquelle: Yahoo Finance (yfinance). DivR = TTM-Dividende / letzter Preis. "
    "D/E = Total Debt / Total Equity. FX via EZB-Raten. Wikipedia (live) für Index-Universen."
)

# ─────────────────────────────────────────────────────────────
# Index-Quellen (Wikipedia) & Fallback-Listen
# ─────────────────────────────────────────────────────────────
INDEX_SOURCES = {
    "DAX 40 (.DE)": {
        "url": "https://en.wikipedia.org/wiki/DAX",
        "ticker_cols": ["Ticker symbol", "Ticker", "Symbol", "EPIC"],
        "suffix": ".DE",
    },
    "MDAX (.DE)": {
        "url": "https://en.wikipedia.org/wiki/MDAX",
        "ticker_cols": ["Ticker", "Symbol", "EPIC"],
        "suffix": ".DE",
    },
    "SBF 120 (.PA)": {
        "url": "https://en.wikipedia.org/wiki/SBF_120",
        "ticker_cols": ["Ticker", "Symbol", "EPIC"],
        "suffix": ".PA",
    },
    "FTSE 100 (.L)": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "ticker_cols": ["EPIC", "Ticker", "Symbol"],
        "suffix": ".L",
    },
    "FTSE 250 (.L)": {
        "url": "https://en.wikipedia.org/wiki/FTSE_250_Index",
        "ticker_cols": ["EPIC", "Ticker", "Symbol"],
        "suffix": ".L",
    },
    # Optional: weitere Indizes hinzufügen (IBEX .MC, FTSE MIB .MI, AEX .AS, SMI .SW, OMX .ST, …)
}

# Fallback-Ticker (falls Wikipedia mal keine brauchbare Tabelle liefert)
BACKUP_INDEX_TICKERS = {
    "DAX 40 (.DE)":  ["ALV.DE","BAS.DE","BAYN.DE","DTE.DE","EOAN.DE","SIE.DE","BMW.DE","VNA.DE"],
    "MDAX (.DE)":    ["LEG.DE","FRE.DE","HNR1.DE","1COV.DE"],
    "SBF 120 (.PA)": ["ORA.PA","ENGI.PA","SU.PA","BNP.PA","GLE.PA","EN.PA","VIV.PA","AI.PA"],
    "FTSE 100 (.L)": ["BATS.L","ULVR.L","IMB.L","VOD.L","NG.L","LGEN.L","GSK.L","BT-A.L"],
    "FTSE 250 (.L)": ["BDEV.L","PSN.L","TW.L","MGGT.L"],
}

# ─────────────────────────────────────────────────────────────
# Sidebar – Parameter & Presets
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Einstellungen")

    idx_selected = st.multiselect(
        "Index-Universen (Wikipedia live)",
        options=list(INDEX_SOURCES.keys()),
        default=["DAX 40 (.DE)", "SBF 120 (.PA)", "FTSE 100 (.L)"],
    )

    manual_tickers = st.text_area(
        "Zusätzliche Ticker (kommagetrennt, Yahoo-Format)",
        value="",
        help="z. B. ALV.DE,ENEL.MI,BATS.L",
    )
    csv_universe = st.file_uploader("Optional: CSV mit Spalte 'ticker' (weitere Titel)", type=["csv"])

    st.markdown("---")
    days_window = st.slider("Zeitfenster 52W-Tief (Tage)", 3, 60, 7)
    min_yield = st.slider("Mindest-DivR (TTM, % p.a.)", 0.0, 15.0, 5.0, step=0.5)
    max_de = st.slider("Max. D/E (%)", 20, 300, 100, step=10)
    near_low_pct = st.slider("Near-Low Schwelle (%)", 0.0, 10.0, 3.0, step=0.5,
                             help="Abstand zum 52W-Tief ≤ X %")

    st.markdown("---")
    base_ccy = st.selectbox("Berichtswährung (FX-Normalisierung)", ["EUR", "CHF"], index=0,
                            help="Preise/Geldwerte in diese Währung umrechnen (DivR ist dimensionslos).")

    st.markdown("---")
    st.subheader("Presets")

    def current_preset_dict():
        return {
            "idx_selected": idx_selected,
            "manual_tickers": manual_tickers,
            "days_window": int(days_window),
            "min_yield": float(min_yield),
            "max_de": int(max_de),
            "near_low_pct": float(near_low_pct),
            "base_ccy": base_ccy,
        }

    preset_json = json.dumps(current_preset_dict(), indent=2)
    st.download_button("Preset speichern (JSON)", data=preset_json,
                       file_name="eu_screener_preset.json", mime="application/json")

    preset_upload = st.file_uploader("Preset laden (JSON)", type=["json"], key="preset_upload")
    if preset_upload is not None:
        try:
            p = json.load(preset_upload)
            st.session_state["idx_selected"] = p.get("idx_selected", idx_selected)
            st.session_state["manual_tickers"] = p.get("manual_tickers", manual_tickers)
            st.session_state["days_window"] = p.get("days_window", days_window)
            st.session_state["min_yield"] = p.get("min_yield", min_yield)
            st.session_state["max_de"] = p.get("max_de", max_de)
            st.session_state["near_low_pct"] = p.get("near_low_pct", near_low_pct)
            st.session_state["base_ccy"] = p.get("base_ccy", base_ccy)
            st.success("Preset geladen – ändere einen Parameter oder klicke oben rechts auf 'Rerun'.")
        except Exception as e:
            st.error(f"Preset konnte nicht geladen werden: {e}")

# ─────────────────────────────────────────────────────────────
# Utils – Wikipedia: MultiIndex-Spalten robust flatten
# ─────────────────────────────────────────────────────────────
def _flatten_columns(cols) -> List[str]:
    """Konvertiert auch MultiIndex-Spalten robust in einfache Strings."""
    try:
        if isinstance(cols, pd.MultiIndex):
            flat = []
            for tup in cols.tolist():
                parts = [str(x) for x in tup if x is not None and str(x).lower() != "nan"]
                flat.append(" ".join(parts).strip())
            return flat
        return [str(c) for c in cols.tolist()]
    except Exception:
        return [str(c) for c in list(cols)]

# ─────────────────────────────────────────────────────────────
# Wikipedia-Fetcher (robust + Fallback)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_index_from_wikipedia(source: dict) -> List[str]:
    """
    Holt Ticker von Wikipedia und hängt Yahoo-Suffix an.
    Robust gegen wechselnde Tabellenlayouts und MultiIndex-Spalten.
    """
    url = source["url"]
    suffix = source["suffix"]
    cand_cols = [c.lower() for c in source.get("ticker_cols", [])]

    try:
        tables = pd.read_html(url, header=0)
    except Exception as e:
        st.warning(f"Wikipedia-Abruf fehlgeschlagen ({url}): {e}")
        return []

    tickers: List[str] = []
    for df in tables:
        flat_cols = _flatten_columns(df.columns)
        lower_map = {orig: flat.lower() for orig, flat in zip(df.columns, flat_cols)}

        # passende Spalte finden
        target_col = None
        for key in cand_cols + ["ticker", "symbol", "epic", "ric"]:
            for orig, low in lower_map.items():
                if key in low:
                    target_col = orig
                    break
            if target_col is not None:
                break
        if target_col is None:
            # Heuristik: erste Spalte nehmen
            target_col = df.columns[0]

        ser = df[target_col].astype(str)

        def clean_t(x: str) -> str:
            x = re.sub(r"\[.*?\]", "", x)        # Fußnoten
            x = x.split(",")[0]                  # "TICK, alt" -> "TICK"
            x = x.replace("\xa0", "").replace(" ", "")
            x = x.replace(".", "")               # EPIC "AV." -> "AV"
            x = re.sub(r"[^A-Za-z0-9\-]", "", x)
            return x.upper()

        vals = [clean_t(v) for v in ser.tolist() if isinstance(v, str)]
        vals = [v for v in vals if 1 <= len(v) <= 8 and v not in ("", "NAN")]
        tickers.extend([v + suffix for v in vals])

    # Deduplizieren
    seen, out = set(), []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def build_universe(idx_selected: List[str], manual_tickers: str, csv_universe) -> List[str]:
    ticks: List[str] = []
    for name in idx_selected:
        src = INDEX_SOURCES.get(name)
        tks = fetch_index_from_wikipedia(src) if src else []
        if not tks:
            tks = BACKUP_INDEX_TICKERS.get(name, [])
        ticks += tks
    if manual_tickers.strip():
        ticks += [t.strip() for t in manual_tickers.split(",") if t.strip()]
    if csv_universe is not None:
        try:
            dfu = pd.read_csv(csv_universe)
            if "ticker" in dfu.columns:
                ticks += [str(x).strip() for x in dfu["ticker"].dropna().tolist()]
        except Exception:
            st.warning("CSV konnte nicht gelesen werden – erwarte eine Spalte 'ticker'.")
    # deduplizieren
    seen, uniq = set(), []
    for t in ticks:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

# ─────────────────────────────────────────────────────────────
# FX – EZB-Raten & Umrechnung
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=6*3600)
def fetch_ecb_rates() -> Dict[str, float]:
    """
    Tägliche EZB-Raten (EUR-Basis). Rückgabe: {"USD":1.096, "CHF":0.96, ..., "EUR":1.0}
    """
    urls = [
        "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
        "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=10)
            if not r.ok:
                continue
            txt = r.text
            rates = {}
            for token in txt.split(" Cube "):
                if 'currency="' in token and 'rate="' in token:
                    try:
                        ccy = token.split('currency="',1)[1].split('"',1)[0].upper()
                        rate = float(token.split('rate="',1)[1].split('"',1)[0])
                        rates[ccy] = rate
                    except Exception:
                        pass
            if rates:
                rates["EUR"] = 1.0
                return rates
        except Exception:
            continue
    return {"EUR": 1.0}

def fx_convert(amount: Optional[float], from_ccy: Optional[str], base_ccy: str, ecb: Dict[str, float]) -> Optional[float]:
    if amount is None or from_ccy is None:
        return None
    from_ccy = from_ccy.upper()
    base_ccy = base_ccy.upper()
    if from_ccy == base_ccy:
        return float(amount)
    # ecb: 1 EUR = ecb[CCY]
    if base_ccy == "EUR":
        rate = ecb.get(from_ccy)
        if not rate:
            return None
        return float(amount) / float(rate)
    if base_ccy == "CHF":
        eur_to_chf = ecb.get("CHF")
        if not eur_to_chf:
            return None
        if from_ccy == "EUR":
            return float(amount) * eur_to_chf
        rate = ecb.get(from_ccy)
        if not rate:
            return None
        eur_amt = float(amount) / float(rate)
        return eur_amt * eur_to_chf
    return None

# ─────────────────────────────────────────────────────────────
# yfinance – Caches
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def yf_price_hist(ticker: str, years: int = 2) -> pd.DataFrame:
    return yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)

@st.cache_data(show_spinner=False)
def yf_dividends(ticker: str, years: int = 2) -> pd.Series:
    t = yf.Ticker(ticker)
    dv = t.dividends
    if dv is None or dv.empty:
        return pd.Series(dtype=float)
    cutoff = pd.Timestamp.today(tz=dv.index.tz) - pd.DateOffset(years=years)
    return dv[dv.index >= cutoff].sort_index()

@st.cache_data(show_spinner=False)
def yf_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    fast = getattr(t, "fast_info", {}) or {}
    price = fast.get("last_price")
    currency = fast.get("currency")
    market_cap = fast.get("market_cap")
    shares_out = fast.get("shares")

    def last_from(df, names):
        if df is None or df.empty:
            return None
        for nm in names:
            if nm in df.index:
                s = pd.to_numeric(df.loc[nm], errors="coerce").dropna()
                if not s.empty:
                    return float(s.iloc[0])
        return None

    try: bs = t.balance_sheet
    except Exception: bs = pd.DataFrame()

    total_debt = last_from(bs, ["Total Debt", "Short Long Term Debt", "Long Term Debt"])
    total_equity = last_from(bs, ["Total Stockholder Equity", "Stockholders Equity"])

    return {
        "price": price,
        "currency": currency,
        "market_cap": market_cap,
        "shares_outstanding_ttm": shares_out,
        "total_debt": total_debt,
        "total_equity": total_equity,
    }

# ─────────────────────────────────────────────────────────────
# Kennzahlen
# ─────────────────────────────────────────────────────────────
def ttm_dividend(div_series: pd.Series) -> Optional[float]:
    if div_series is None or div_series.empty:
        return None
    end = div_series.index.max()
    start = end - pd.Timedelta(days=365)
    return float(div_series[(div_series.index > start) & (div_series.index <= end)].sum())

def compute_52w_low(px: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[float]]:
    if px is None or px.empty:
        return None, None, None
    cutoff = px.index.max() - pd.Timedelta(days=365)
    sub = px[px.index >= cutoff]
    if sub.empty:
        return None, None, None
    low_close = float(sub["Close"].min())
    low_date = sub["Close"].idxmin()
    last_close = float(sub["Close"].iloc[-1])
    return low_date, low_close, last_close

def qualifies_low_within_window(low_date: Optional[pd.Timestamp], days: int) -> bool:
    if low_date is None:
        return False
    now = pd.Timestamp.now(tz=getattr(low_date, "tz", None))
    return (now - low_date) <= pd.Timedelta(days=days)

def compute_de_ratio(total_debt: Optional[float], total_equity: Optional[float]) -> Optional[float]:
    if total_debt is None or total_equity is None or total_equity == 0:
        return None
    if total_equity <= 0:
        return math.inf
    return float(total_debt / total_equity)

# ─────────────────────────────────────────────────────────────
# Screening-Logik
# ─────────────────────────────────────────────────────────────
def screen_tickers(tickers: List[str], days_window: int, min_yield_pct: float, max_de_pct: float,
                   base_ccy: str, ecb_rates: Dict[str,float]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        with st.spinner(f"Lade: {tk}"):
            try:
                px = yf_price_hist(tk, years=2)
                dv = yf_dividends(tk, years=2)
                f = yf_fundamentals(tk)
            except Exception:
                continue

            price = f.get("price")
            currency = f.get("currency")
            if price is None and px is not None and not px.empty:
                price = float(px["Close"].iloc[-1])

            price_base = fx_convert(price, currency, base_ccy, ecb_rates) if price is not None else None

            dps_ttm = ttm_dividend(dv)
            div_yield = None
            if dps_ttm is not None and price not in (None, 0):
                div_yield = float(dps_ttm / price)

            de_ratio = compute_de_ratio(f.get("total_debt"), f.get("total_equity"))

            low_date, low_close, last_close = compute_52w_low(px)
            low_within = qualifies_low_within_window(low_date, days_window)
            gap_to_low_pct = None
            if low_close not in (None, 0) and last_close not in (None, 0):
                gap_to_low_pct = 100.0 * (last_close - low_close) / low_close

            rows.append({
                "ticker": tk,
                "currency": currency,
                "price": price,
                f"price_{base_ccy}": price_base,
                "div_ttm": dps_ttm,
                "div_yield": div_yield,   # 0.065 = 6.5 %
                "de_ratio": de_ratio,     # 0.8 = 80 %
                "low_date": low_date,
                "low_close": low_close,
                "last_close": last_close,
                "gap_to_low_%": gap_to_low_pct,
                "low_within_window": low_within,
                "yahoo": f"https://finance.yahoo.com/quote/{tk}",
                "ft": f"https://markets.ft.com/data/equities/tearsheet/summary?s={tk.replace('.', '%3A')}",
                "reuters": f"https://www.reuters.com/markets/companies/{tk}/",
            })

    base = pd.DataFrame.from_records(rows).set_index("ticker")
    if base.empty:
        return base

    elig = base.copy()
    elig = elig[elig["low_within_window"] == True]
    if min_yield_pct is not None:
        elig = elig[(elig["div_yield"].fillna(0) >= (min_yield_pct / 100.0))]
    if max_de_pct is not None:
        elig = elig[np.isfinite(elig["de_ratio"].astype(float))]
        elig = elig[(elig["de_ratio"] <= (max_de_pct / 100.0))]
    return elig

# ─────────────────────────────────────────────────────────────
# Exporte
# ─────────────────────────────────────────────────────────────
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
                return None, "Excel-Export: installiere 'xlsxwriter' oder 'openpyxl'."
        with pd.ExcelWriter(bio, engine=engine) as xw:
            df.to_excel(xw, index=True, sheet_name="screen")
        return bio.getvalue(), None
    return None, f"Unbekanntes Format: {kind}"

# ─────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────
def build_heatmap_df(df: pd.DataFrame, near_low_thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    out = df.copy()
    out["DivR_%"] = (out["div_yield"] * 100.0).round(2)
    out["D/E_%"] = (out["de_ratio"] * 100.0).round(1)
    out["NearLow_%"] = out["gap_to_low_%"].round(2)

    def score_divr(x):
        if pd.isna(x): return np.nan
        if x >= 10: return 5
        if x >= 7:  return 4
        if x >= 5:  return 3
        if x >= 3:  return 2
        return 1

    def score_nearlow(x):
        if pd.isna(x): return np.nan
        if x <= near_low_thresh: return 5
        if x <= 5:  return 4
        if x <= 10: return 3
        if x <= 20: return 2
        return 1

    def score_de(x):
        if pd.isna(x): return np.nan
        if x <= 30:  return 5
        if x <= 60:  return 4
        if x <= 100: return 3
        if x <= 150: return 2
        return 1

    out["DivR_score"] = out["DivR_%"].apply(score_divr)
    out["NearLow_score"] = out["NearLow_%"].apply(score_nearlow)
    out["DE_score"] = out["D/E_%"].apply(score_de)

    hm = out.reset_index()[["ticker","DivR_score","NearLow_score","DE_score"]]
    hm = hm.melt(id_vars="ticker", var_name="Metric", value_name="Score")
    return out, hm

def heatmap_chart(hm_long: pd.DataFrame, title: str):
    if hm_long.empty:
        return None
    chart = alt.Chart(hm_long).mark_rect().encode(
        y=alt.Y('ticker:N', sort='-x', title="Ticker"),
        x=alt.X('Metric:N', title="Metric"),
        color=alt.Color('Score:Q', scale=alt.Scale(scheme='yellowgreenblue'),
                        legend=alt.Legend(title="Score (5=best)")),
        tooltip=['ticker:N','Metric:N','Score:Q'],
    ).properties(
        width=420,
        height=22 * max(3, hm_long["ticker"].nunique()),
        title=title
    )
    return chart

# ─────────────────────────────────────────────────────────────
# Tabs & Ablauf
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Screener", "📊 Daily Research + Heatmap"])

with st.spinner("Lade EZB-Kurse …"):
    ecb_rates = fetch_ecb_rates()

universe = build_universe(idx_selected, manual_tickers, csv_universe)

with tab1:
    st.subheader("Screening nach Regeln")
    st.write(
        f"Universum: **{len(universe)}** Ticker aus {', '.join(idx_selected) or '—'}"
        f"{' + manuell/CSV' if (manual_tickers.strip() or csv_universe is not None) else ''}"
    )

    if st.button("Screen ausführen", type="primary"):
        hits = screen_tickers(universe, days_window, min_yield, max_de, base_ccy, ecb_rates)
        if hits.empty:
            st.warning("Keine Treffer für die aktuellen Regeln.")
        else:
            view = hits.copy()
            if f"price_{base_ccy}" in view.columns:
                view[f"price_{base_ccy}"] = view[f"price_{base_ccy}"].round(4)
            view["DivR_%"] = (view["div_yield"] * 100.0).round(2)
            view["D/E_%"] = (view["de_ratio"] * 100.0).round(1)
            show_cols = [c for c in [f"price_{base_ccy}","DivR_%","D/E_%","low_date","low_close","last_close","gap_to_low_%","yahoo","ft","reuters"] if c in view.columns]
            st.dataframe(view[show_cols].sort_values("DivR_%", ascending=False), use_container_width=True)

            st.markdown("#### Download")
            c1, c2, c3 = st.columns(3)
            csv_b, _ = df_to_bytes(hits, "csv")
            c1.download_button("CSV", data=csv_b, file_name="eu_screen_hits.csv", mime="text/csv")
            x_b, x_err = df_to_bytes(hits, "xlsx")
            if x_b:
                c2.download_button("Excel", data=x_b, file_name="eu_screen_hits.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                c2.info(x_err)
            h_b, _ = df_to_bytes(hits, "html")
            c3.download_button("HTML", data=h_b, file_name="eu_screen_hits.html", mime="text/html")

        st.info("Tipp: Erhöhe das Zeitfenster (z. B. 14–30 Tage) oder setze D/E auf 150 %, wenn du mehr Treffer brauchst.")

with tab2:
    st.subheader("Daily Research Dashboard")
    if st.button("Dashboard laden / aktualisieren"):
        rows = []
        for tk in universe:
            with st.spinner(f"Berechne: {tk}"):
                try:
                    px = yf_price_hist(tk, years=2)
                    dv = yf_dividends(tk, years=2)
                    f = yf_fundamentals(tk)
                except Exception:
                    continue
                price = f.get("price")
                currency = f.get("currency")
                if price is None and px is not None and not px.empty:
                    price = float(px["Close"].iloc[-1])
                price_base = fx_convert(price, currency, base_ccy, ecb_rates) if price is not None else None

                dps_ttm = ttm_dividend(dv)
                div_yield = None
                if dps_ttm is not None and price not in (None, 0):
                    div_yield = float(dps_ttm / price)

                de_ratio = compute_de_ratio(f.get("total_debt"), f.get("total_equity"))
                low_date, low_close, last_close = compute_52w_low(px)
                gap_to_low_pct = None
                if low_close not in (None, 0) and last_close not in (None, 0):
                    gap_to_low_pct = 100.0 * (last_close - low_close) / low_close

                rows.append({
                    "ticker": tk,
                    "currency": currency,
                    "price": price,
                    f"price_{base_ccy}": price_base,
                    "div_ttm": dps_ttm,
                    "div_yield": div_yield,
                    "de_ratio": de_ratio,
                    "low_date": low_date,
                    "low_close": low_close,
                    "last_close": last_close,
                    "gap_to_low_%": gap_to_low_pct,
                    "yahoo": f"https://finance.yahoo.com/quote/{tk}",
                })

        base = pd.DataFrame.from_records(rows).set_index("ticker")
        if base.empty:
            st.warning("Keine Daten geladen.")
            st.stop()

        # KPIs (robust)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anzahl Ticker", len(base))
        
        divr_count = int(base["div_yield"].notna().sum())
        c2.metric("mit DivR", divr_count)
        
        de_series = pd.to_numeric(base["de_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        c3.metric("mit D/E", int(de_series.notna().sum()))
        
        gap_series = pd.to_numeric(base["gap_to_low_%"], errors="coerce")
        near_low_count = int(((gap_series.notna()) & (gap_series <= near_low_pct)).sum())
        c4.metric(f"Near-Lows ≤ {near_low_pct:.1f} %", near_low_count)


        st.markdown("#### Top-Yielder")
        top = base[base["div_yield"].notna()].copy()
        if f"price_{base_ccy}" in top.columns:
            top[f"price_{base_ccy}"] = top[f"price_{base_ccy}"].round(4)
        top["DivR_%"] = (top["div_yield"] * 100.0).round(2)
        st.dataframe(top.sort_values("div_yield", ascending=False).head(25)
                     [[f"price_{base_ccy}","DivR_%","low_date","yahoo"]], use_container_width=True)

        st.markdown(f"#### Near-Lows (≤ {near_low_pct:.1f} %)")
        gap_series = pd.to_numeric(base["gap_to_low_%"], errors="coerce")
        near = base[gap_series.le(near_low_pct)].copy()
        near["gap_to_low_%"] = gap_series.loc[near.index]
        st.dataframe(
            near.sort_values("gap_to_low_%", ascending=True)[[f"price_{base_ccy}","gap_to_low_%","low_date","yahoo"]],
            use_container_width=True
        )


        st.markdown("#### D/E < 100 % & DivR > 5 % (Quick-View)")
        quick = base.copy()
        quick = quick[quick["div_yield"].fillna(0) >= 0.05]
        quick = quick[np.isfinite(quick["de_ratio"].astype(float))]
        quick = quick[quick["de_ratio"] <= 1.0]
        quick["DivR_%"] = (quick["div_yield"] * 100.0).round(2)
        quick["D/E_%"] = (quick["de_ratio"] * 100.0).round(1)
        st.dataframe(quick.sort_values("DivR_%", ascending=False)
                     [[f"price_{base_ccy}","DivR_%","D/E_%","low_date","yahoo"]], use_container_width=True)

        st.markdown("#### Watchlist-Heatmap (DivR / Near-Low / D/E)")
        enriched, hm_long = build_heatmap_df(base, near_low_thresh=near_low_pct)
        chart = heatmap_chart(hm_long, title="Score-Heatmap: 5=best, 1=schwach")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Dashboard-Downloads")
        c1, c2, c3 = st.columns(3)
        b1, _ = df_to_bytes(top, "csv");  c1.download_button("Top-Yielder CSV", data=b1, file_name="dashboard_top_yielder.csv", mime="text/csv")
        b2, _ = df_to_bytes(near, "csv"); c2.download_button("Near-Lows CSV", data=b2, file_name="dashboard_near_lows.csv", mime="text/csv")
        b3, _ = df_to_bytes(quick, "csv"); c3.download_button("D/E & DivR CSV", data=b3, file_name="dashboard_de_divr.csv", mime="text/csv")
    else:
        st.info("Parameter links einstellen und **Dashboard laden / aktualisieren** klicken.")
