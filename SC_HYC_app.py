# streamlit_eu_low_yield_screen_pro.py
# -*- coding: utf-8 -*-
# EU High-Yield Screener mit:
# - Index-Universen (Wikipedia/CSV) + Yahoo-Suffix
# - 52W-Low innerhalb N Tage, DivR > Schwelle, D/E < Schwelle
# - FX-Normalisierung in EUR/CHF (EZB-Raten)
# - Presets (save/load)
# - Daily Research Dashboard + Watchlist-Heatmap
#
# Autor: ChatGPT (GPT-5 Thinking)

import warnings
warnings.filterwarnings("ignore", message=".*bottleneck.*", category=UserWarning)

import io
import math
import json
import time
from datetime import datetime
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

st.caption("Datenquelle: Yahoo Finance (yfinance). DivR = TTM-Dividende / letzter Preis. "
           "D/E = Total Debt / Total Equity. FX via EZB-Raten. Wikipedia für Index-Universen.")

# ─────────────────────────────────────────────────────────────
# Config – Indexquellen (Wikipedia) & Suffix-Mapping
# ─────────────────────────────────────────────────────────────
INDEX_SOURCES = {
    "DAX 40 (.DE)": {
        "url": "https://en.wikipedia.org/wiki/DAX",
        "ticker_cols": ["Ticker symbol", "Ticker", "Symbol"],  # heuristisch
        "suffix": ".DE",
        "table_hint": "constituents",
    },
    "MDAX (.DE)": {
        "url": "https://en.wikipedia.org/wiki/MDAX",
        "ticker_cols": ["Ticker", "Symbol"],
        "suffix": ".DE",
        "table_hint": "constituents",
    },
    "SBF 120 (.PA)": {
        "url": "https://en.wikipedia.org/wiki/SBF_120",
        "ticker_cols": ["Ticker", "Symbol"],
        "suffix": ".PA",
        "table_hint": "constituents",
    },
    "FTSE 100 (.L)": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "ticker_cols": ["EPIC", "Ticker", "Symbol"],
        "suffix": ".L",
        "table_hint": "constituents",
    },
    "FTSE 250 (.L)": {
        "url": "https://en.wikipedia.org/wiki/FTSE_250_Index",
        "ticker_cols": ["EPIC", "Ticker", "Symbol"],
        "suffix": ".L",
        "table_hint": "constituents",
    },
    # Du kannst hier weitere Indizes ergänzen: IBEX 35 (.MC), FTSE MIB (.MI), AEX (.AS), SMI (.SW), OMX Stockholm (.ST), ...
}

# ─────────────────────────────────────────────────────────────
# Sidebar – Parameter & Presets
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Einstellungen")

    # Index-Auswahl
    idx_selected = st.multiselect(
        "Index-Universen (Wikipedia live)",
        options=list(INDEX_SOURCES.keys()),
        default=["DAX 40 (.DE)", "SBF 120 (.PA)", "FTSE 100 (.L)"]
    )

    # Manuelle Ticker & CSV
    manual_tickers = st.text_area("Zusätzliche Ticker (kommagetrennt, Yahoo-Format)",
                                  value="", help="z. B. ALV.DE,ENEL.MI,BATS.L")
    csv_universe = st.file_uploader("Optional: CSV mit Spalte 'ticker' (weitere Titel)", type=["csv"])

    # Screening-Parameter
    st.markdown("---")
    days_window = st.slider("Zeitfenster für 52W-Tief (Tage)", 3, 60, 7)
    min_yield = st.slider("Mindest-DivR (TTM, % p.a.)", 0.0, 15.0, 5.0, step=0.5)
    max_de = st.slider("Max. D/E (%)", 20, 300, 100, step=10)
    near_low_pct = st.slider("Near-Low Schwelle (%)", 0.0, 10.0, 3.0, step=0.5,
                             help="Abstand zum 52W-Tief ≤ X %")

    # FX-Normalisierung
    st.markdown("---")
    base_ccy = st.selectbox("Berichtswährung (FX-Normalisierung)", ["EUR", "CHF"], index=0,
                            help="Preise & Geldwerte in diese Währung umrechnen (DivR ist dimensionslos).")

    # Presets
    st.markdown("---")
    st.subheader("Presets")
    # Save preset
    def current_preset_dict():
        return {
            "idx_selected": idx_selected,
            "manual_tickers": manual_tickers,
            "days_window": days_window,
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
            # SessionState aktualisieren, dann rerun
            st.session_state["idx_selected"] = p.get("idx_selected", idx_selected)
            st.session_state["manual_tickers"] = p.get("manual_tickers", manual_tickers)
            st.session_state["days_window"] = p.get("days_window", days_window)
            st.session_state["min_yield"] = p.get("min_yield", min_yield)
            st.session_state["max_de"] = p.get("max_de", max_de)
            st.session_state["near_low_pct"] = p.get("near_low_pct", near_low_pct)
            st.session_state["base_ccy"] = p.get("base_ccy", base_ccy)
            st.success("Preset geladen – bitte auf **Rerun** klicken (oben rechts) oder irgend­einen Parameter ändern.")
        except Exception as e:
            st.error(f"Preset konnte nicht geladen werden: {e}")

# ─────────────────────────────────────────────────────────────
# Helper – Wikipedia Indizes parsEN
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_index_from_wikipedia(source: dict) -> List[str]:
    """
    Holt Ticker/EPIC/Symbole von einer Wikipedia-Seite und hängt Yahoo-Suffix an.
    Robust: versucht mehrere Tabellen & Spaltennamen.
    """
    url = source["url"]
    suffix = source["suffix"]
    try:
        tables = pd.read_html(url)
    except Exception:
        return []
    tickers: List[str] = []
    for df in tables:
        # Heuristik: wähle Tabellen mit vielen kapitalmarktnahen Spalten
        cols_lower = [c.lower() for c in df.columns.astype(str)]
        if "constituent" in source.get("table_hint", "") or any("constituent" in c for c in cols_lower):
            pass  # ok
        # Suche mögliche Ticker-Spalten
        for colcand in source["ticker_cols"]:
            if colcand in df.columns:
                series = df[colcand].astype(str).str.strip()
                vals = series[series.str.len() > 0].tolist()
                if vals:
                    tickers += vals
                    break
        # Falls nichts gefunden, versuche generisch
        if not tickers:
            for c in df.columns:
                if str(c).lower() in ("ticker", "symbol", "epic"):
                    vals = df[c].astype(str).str.strip().tolist()
                    tickers += vals
    # Säubern, Suffix anhängen
    clean = []
    for t in tickers:
        t = t.replace(".", "").replace(" ", "").upper()
        if t and t != "NAN":
            clean.append(t + suffix)
    # deduplizieren, Reihenfolge wahren
    seen, out = set(), []
    for t in clean:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def build_universe(idx_selected: List[str], manual_tickers: str, csv_universe) -> List[str]:
    ticks: List[str] = []
    # Indizes
    for name in idx_selected:
        src = INDEX_SOURCES.get(name)
        if src:
            tks = fetch_index_from_wikipedia(src)
            ticks += tks
    # Manuell
    if manual_tickers.strip():
        ticks += [t.strip() for t in manual_tickers.split(",") if t.strip()]
    # CSV
    if csv_universe is not None:
        try:
            dfu = pd.read_csv(csv_universe)
            if "ticker" in dfu.columns:
                ticks += [str(x).strip() for x in dfu["ticker"].dropna().tolist()]
        except Exception:
            st.warning("CSV konnte nicht gelesen werden – erwarte eine Spalte 'ticker'.")
    # Deduplizieren
    seen, uniq = set(), []
    for t in ticks:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

# ─────────────────────────────────────────────────────────────
# Helper – FX-Raten (EZB) & Umrechnung
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=6*3600)
def fetch_ecb_rates() -> Dict[str, float]:
    """
    Holt tägliche EUR-FX-Raten der EZB (eurofxref-daily.xml),
    Rückgabe: Mapping CCY->EUR_RATE (z. B. {"USD":1.096, "CHF":0.96, ...} bedeutet: 1 EUR = 1.096 USD)
    """
    urls = [
        "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
        "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=10)
            if r.ok:
                txt = r.text
                # primitive XML-Parse ohne extra deps
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
    return {"EUR": 1.0}  # Fallback

def fx_convert(amount: Optional[float], from_ccy: Optional[str], base_ccy: str, ecb: Dict[str, float]) -> Optional[float]:
    if amount is None or from_ccy is None:
        return None
    from_ccy = from_ccy.upper()
    base_ccy = base_ccy.upper()
    if from_ccy == base_ccy:
        return float(amount)
    # Wir haben EUR-Rates: 1 EUR = ecb[CCY]
    if base_ccy == "EUR":
        # Betrag in CCY -> EUR: amount / (CCY per EUR)
        rate = ecb.get(from_ccy)
        if rate is None or rate == 0:
            return None
        return float(amount) / float(rate)
    if base_ccy == "CHF":
        eur_to_chf = ecb.get("CHF")
        if eur_to_chf is None or eur_to_chf == 0:
            return None
        if from_ccy == "EUR":
            return float(amount) * eur_to_chf
        # CCY -> EUR -> CHF
        rate = ecb.get(from_ccy)
        if rate is None or rate == 0:
            return None
        eur_amt = float(amount) / float(rate)
        return eur_amt * eur_to_chf
    # Erweiterbar für andere Zielwährungen
    return None

# ─────────────────────────────────────────────────────────────
# Yahoo-Fetches (Cache)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def yf_price_hist(ticker: str, years: int = 2) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    return df

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
# Kennzahlenberechnung
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
            # Fallback Preis
            if price is None and px is not None and not px.empty:
                price = float(px["Close"].iloc[-1])

            # FX-Preis in Basis
            price_base = fx_convert(price, currency, base_ccy, ecb_rates) if price is not None else None

            # DivR (CCY-neutral, da in gleicher CCY)
            dps_ttm = ttm_dividend(dv)
            div_yield = None
            if dps_ttm is not None and price not in (None, 0):
                div_yield = float(dps_ttm / price)

            # D/E
            de_ratio = compute_de_ratio(f.get("total_debt"), f.get("total_equity"))

            # 52W-Low
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
                "div_yield": div_yield,  # 0.065 = 6.5%
                "de_ratio": de_ratio,    # 0.8 = 80%
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

    # Filter anwenden
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
# Heatmap-Helper
# ─────────────────────────────────────────────────────────────
def build_heatmap_df(df: pd.DataFrame, near_low_thresh: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    # Kennzahlen (in % für Anzeige)
    out["DivR_%"] = (out["div_yield"] * 100.0).round(2)
    out["D/E_%"] = (out["de_ratio"] * 100.0).round(1)
    out["NearLow_%"] = out["gap_to_low_%"].round(2)

    # Buckets
    out["DivR_bucket"] = pd.cut(
        out["DivR_%"],
        bins=[-np.inf, 3, 5, 7, 10, np.inf],
        labels=["<3", "3–5", "5–7", "7–10", "≥10"]
    )
    out["DE_bucket"] = pd.cut(
        out["D/E_%"],
        bins=[-np.inf, 30, 60, 100, 150, np.inf],
        labels=["≤30", "30–60", "60–100", "100–150", ">150"]
    )
    out["NearLow_bucket"] = pd.cut(
        out["NearLow_%"],
        bins=[-np.inf, near_low_thresh, 5, 10, 20, np.inf],
        labels=[f"≤{near_low_thresh:.1f}", "≤5", "≤10", "≤20", ">20"]
    )

    # Für Heatmap-Legende: Scorings (höher = „besser“)
    # DivR: höher besser; NearLow: niedriger besser; D/E: niedriger besser
    def score_divr(x):
        if pd.isna(x): return np.nan
        if x >= 10: return 5
        if x >= 7: return 4
        if x >= 5: return 3
        if x >= 3: return 2
        return 1

    def score_nearlow(x):
        if pd.isna(x): return np.nan
        if x <= near_low_thresh: return 5
        if x <= 5: return 4
        if x <= 10: return 3
        if x <= 20: return 2
        return 1

    def score_de(x):
        if pd.isna(x): return np.nan
        if x <= 30: return 5
        if x <= 60: return 4
        if x <= 100: return 3
        if x <= 150: return 2
        return 1

    out["DivR_score"] = out["DivR_%"].apply(score_divr)
    out["NearLow_score"] = out["NearLow_%"].apply(score_nearlow)
    out["DE_score"] = out["D/E_%"].apply(score_de)

    # Long-Format für Altair
    hm = out.reset_index()[["ticker","DivR_score","NearLow_score","DE_score"]]
    hm = hm.melt(id_vars="ticker", var_name="Metric", value_name="Score")
    return out, hm

def heatmap_chart(hm_long: pd.DataFrame, title: str):
    if hm_long.empty:
        return None
    chart = alt.Chart(hm_long).mark_rect().encode(
        y=alt.Y('ticker:N', sort='-x', title="Ticker"),
        x=alt.X('Metric:N', title="Metric"),
        color=alt.Color('Score:Q', scale=alt.Scale(scheme='yellowgreenblue'), legend=alt.Legend(title="Score")),
        tooltip=['ticker:N','Metric:N','Score:Q']
    ).properties(width=420, height=20 * max(3, hm_long["ticker"].nunique()), title=title)
    return chart

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Screener", "📊 Daily Research + Heatmap"])

# Gemeinsame Vorarbeit: Universum & FX
with st.spinner("Lade EZB-Kurse …"):
    ecb_rates = fetch_ecb_rates()

universe = build_universe(idx_selected, manual_tickers, csv_universe)

with tab1:
    st.subheader("Screening nach Regeln")
    st.write(f"Universum: **{len(universe)}** Ticker aus {', '.join(idx_selected) or '—'} "
             f"{'+ manuell/CSV' if (manual_tickers.strip() or csv_universe is not None) else ''}")

    if st.button("Screen ausführen", type="primary"):
        hits = screen_tickers(universe, days_window, min_yield, max_de, base_ccy, ecb_rates)
        if hits.empty:
            st.warning("Keine Treffer für die aktuellen Regeln.")
        else:
            view = hits.copy()
            view[f"price_{base_ccy}"] = view[f"price_{base_ccy}"].round(4)
            view["DivR_%"] = (view["div_yield"] * 100.0).round(2)
            view["D/E_%"] = (view["de_ratio"] * 100.0).round(1)
            show_cols = [c for c in [f"price_{base_ccy}","DivR_%","D/E_%","low_date","low_close","last_close","gap_to_low_%","yahoo","ft","reuters"] if c in view.columns]
            st.dataframe(view[show_cols].sort_values("DivR_%", ascending=False), use_container_width=True)

            # Downloads
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

        st.info("Tipp: Erhöhe das Zeitfenster (z. B. 14–30 Tage) oder lockere D/E auf 150 %, falls 5+ Treffer gewünscht sind.")

with tab2:
    st.subheader("Daily Research Dashboard")
    if st.button("Dashboard laden / aktualisieren"):
        # Für das Dashboard zunächst die Basis-Metriken ohne harte Filter berechnen
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

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anzahl Ticker", len(base))
        c2.metric("mit DivR", int(base["div_yield"].notna().sum()))
        c3.metric("mit D/E", int(base["de_ratio"].replace([np.inf, -np.inf], np.nan).notna().sum()))
        c4.metric(f"Near-Lows ≤ {near_low_pct:.1f} %", int((base["gap_to_low_%"].notna()) & (base["gap_to_low_%"] <= near_low_pct)))

        # Tabellen
        st.markdown("#### Top-Yielder")
        top = base[base["div_yield"].notna()].copy()
        top[f"price_{base_ccy}"] = top[f"price_{base_ccy}"].round(4)
        top["DivR_%"] = (top["div_yield"] * 100.0).round(2)
        st.dataframe(top.sort_values("div_yield", ascending=False).head(25)[[f"price_{base_ccy}","DivR_%","low_date","yahoo"]], use_container_width=True)

        st.markdown(f"#### Near-Lows (≤ {near_low_pct:.1f} %)")
        near = base.copy()
        near = near[near["gap_to_low_%"].notna() & (near["gap_to_low_%"] <= near_low_pct)]
        st.dataframe(near.sort_values("gap_to_low_%")[[f"price_{base_ccy}","gap_to_low_%","low_date","yahoo"]], use_container_width=True)

        st.markdown("#### D/E < 100 % & DivR > 5 % (Quick-View)")
        quick = base.copy()
        quick = quick[quick["div_yield"].fillna(0) >= 0.05]
        quick = quick[np.isfinite(quick["de_ratio"].astype(float))]
        quick = quick[quick["de_ratio"] <= 1.0]
        quick["DivR_%"] = (quick["div_yield"] * 100.0).round(2)
        quick["D/E_%"] = (quick["de_ratio"] * 100.0).round(1)
        st.dataframe(quick.sort_values("DivR_%", ascending=False)[[f"price_{base_ccy}","DivR_%","D/E_%","low_date","yahoo"]], use_container_width=True)

        # Heatmap
        st.markdown("#### Watchlist-Heatmap (DivR / Near-Low / D/E – Bucket-Scores)")
        enriched, hm_long = build_heatmap_df(base, near_low_thresh=near_low_pct)
        ch = heatmap_chart(hm_long, title="Score-Heatmap: 5=best, 1=schwach")
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)

        # Downloads (Dashboard-Exports)
        st.markdown("#### Dashboard-Downloads")
        c1, c2, c3 = st.columns(3)
        b1, _ = df_to_bytes(top, "csv"); c1.download_button("Top-Yielder CSV", data=b1, file_name="dashboard_top_yielder.csv", mime="text/csv")
        b2, _ = df_to_bytes(near, "csv"); c2.download_button("Near-Lows CSV", data=b2, file_name="dashboard_near_lows.csv", mime="text/csv")
        b3, _ = df_to_bytes(quick, "csv"); c3.download_button("D_E_und_DivR CSV", data=b3, file_name="dashboard_de_divr.csv", mime="text/csv")
    else:
        st.info("Parameter links einstellen und **Dashboard laden / aktualisieren** klicken.")
