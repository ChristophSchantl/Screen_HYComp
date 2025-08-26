
# High-Yield Dividend Scoring – Streamlit (editierbare Gewichte, robuste Index-Auswahl, 403-fest)

from __future__ import annotations
import time
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
try:
    # neuere urllib3
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # fallback


# ─────────────────────────────────────────────────────────────
# Seite
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="High-Yield Dividend Scoring", layout="wide")
st.title("📈 High-Yield Dividend Scoring")
st.caption("Yahoo Finance • TTM-Kennzahlen • sektorrelative Perzentile • robuste Datenlogik")

# ─────────────────────────────────────────────────────────────
# HTTP / Parsing – gegen 403 absichern
# ─────────────────────────────────────────────────────────────
# === PATCH: HTTP-Setup (ersetzt HEADERS/_get_html/_read_tables) ===
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept-Language": "de,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://en.wikipedia.org/",
    "Cache-Control": "no-cache",
}

def _make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

_SESSION = _make_session()

def _get_html(url: str) -> str:
    r = _SESSION.get(url, headers=HEADERS, timeout=25)
    if r.status_code == 403:
        # explizite Meldung, damit du siehst welche URL blockt
        raise PermissionError(f"403 Forbidden @ {url}")
    r.raise_for_status()
    return r.text

def _read_tables(html_text: str) -> list[pd.DataFrame]:
    # erst lxml, dann bs4
    try:
        return pd.read_html(html_text, flavor="lxml")
    except Exception:
        return pd.read_html(html_text, flavor="bs4")


# === PATCH: Offline-Fallback nur für DAX (falls alles blockiert) ===
OFFLINE_TICKERS = {
    "dax": [
        # Stand 2025 – kann geringfügig abweichen; reicht als Notfall
        "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","BNR.DE","CON.DE",
        "DAI.DE","DB1.DE","DBK.DE","DHER.DE","DTE.DE","DPW.DE","DTG.DE","ENR.DE",
        "EOAN.DE","FME.DE","FRE.DE","HEI.DE","HEN3.DE","IFX.DE","LIN.DE","MBG.DE",
        "MRK.DE","MOR.DE","MUV2.DE","P911.DE","PAH3.DE","QIA.DE","RHM.DE","RWE.DE",
        "SAP.DE","SIE.DE","SRT3.DE","SY1.DE","VNA.DE","VOW3.DE","ZAL.DE","ZOE.DE"
    ],
}








# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _row(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    for k in keys:
        if isinstance(df, pd.DataFrame) and k in df.index:
            s = df.loc[k].dropna().astype(float)
            if len(s):
                return s
    return pd.Series(dtype=float)

def _ttm_sum(q_df: pd.DataFrame, keys: List[str], n: int = 4) -> float:
    s = _row(q_df, keys)
    return float(s.iloc[:n].sum()) if len(s) else np.nan

def _percentile_rank(s: pd.Series) -> pd.Series:
    m = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if m.sum() <= 1:
        out[m] = 0.5
        return out
    out[m] = s[m].rank(pct=True)
    return out

def _sector_percentile(df: pd.DataFrame, col: str, invert: bool = False) -> pd.Series:
    work = df.copy()
    work["sector"] = work.get("sector", pd.Series(index=work.index, dtype="object")).fillna("Unknown")
    p = work.groupby("sector", dropna=False, group_keys=False)[col].apply(_percentile_rank)
    if invert:
        p = 1 - p
    return (p * 100).clip(0, 100)

def _map_beta(b: float) -> float:
    # 0.4→100, 0.8→70, 1.0→50, 1.5→0
    return float(np.interp(b, [0.4, 0.8, 1.0, 1.5], [100, 70, 50, 0], left=100, right=0))

def _is_num(x) -> bool:
    try:
        f = float(x)
        return np.isfinite(f)
    except Exception:
        return False

def _to_float(x) -> float:
    try:
        f = float(x)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan

def _clean_symbols(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip().str.replace(r"[^A-Z0-9\.\-]", "", regex=True)

def _apply_suffix(symbols: List[str], suffix: str) -> List[str]:
    out = []
    for s in symbols:
        if not s:
            continue
        out.append(s if "." in s else s + suffix)
    return out

def _safe_info(t: yf.Ticker) -> Dict:
    try:
        return t.get_info()
    except Exception:
        return getattr(t, "info", {}) or {}

def _safe_fast(t: yf.Ticker) -> Dict:
    return getattr(t, "fast_info", {}) or {}

# ─────────────────────────────────────────────────────────────
# Erwartete Spalten
# ─────────────────────────────────────────────────────────────
EXPECTED_COLS = [
    "ticker","sector","price","div_yield_ttm","yield_5y_median",
    "low_52w","high_52w","pos_52w",
    "pe_ttm","ev_ebitda_ttm","de_ratio",
    "fcf_ttm","fcf_margin_ttm","ebitda_ttm","ebitda_margin_ttm",
    "beta_2y_w","market_cap","adv_3m",
    "div_cash_ttm","coverage_fcf_ttm","fcf_payout_ttm",
    "div_cut_24m","error"
]
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ─────────────────────────────────────────────────────────────
# Kurs-Historie
# ─────────────────────────────────────────────────────────────
def _hist_close(t: yf.Ticker, period="5y", interval="1d") -> pd.Series:
    """Immer Close per history() liefern."""
    try:
        h = t.history(period=period, interval=interval, auto_adjust=True)
        if isinstance(h, pd.DataFrame) and "Close" in h.columns:
            return h["Close"].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────
# Index-Mitglieder (Wikipedia + Fallbacks, 403-fest)
# ─────────────────────────────────────────────────────────────
def _pick_symbol_column(tbl: pd.DataFrame, prefer: list[str]) -> str | None:
    cols = [c for c in tbl.columns]
    for want in prefer:
        for c in cols:
            if want in str(c).lower():
                return c
    return None

# === PATCH: load_index_members komplett ersetzen ===
@st.cache_data(ttl=60*60*12, show_spinner=False)
def load_index_members(name: str) -> List[str]:
    name = name.lower().strip()

    URLS = {
        "dax": [
            "https://de.wikipedia.org/wiki/DAX",
            "https://en.wikipedia.org/wiki/DAX",
            "https://en.m.wikipedia.org/wiki/DAX",   # mobile Fallback
        ],
        "mdax": [
            "https://de.wikipedia.org/wiki/MDAX",
            "https://en.wikipedia.org/wiki/MDAX",
            "https://en.m.wikipedia.org/wiki/MDAX",
        ],
        "ftse100": ["https://en.wikipedia.org/wiki/FTSE_100_Index",
                    "https://en.m.wikipedia.org/wiki/FTSE_100_Index"],
        "ftse250": ["https://en.wikipedia.org/wiki/FTSE_250_Index",
                    "https://en.m.wikipedia.org/wiki/FTSE_250_Index"],
        "dow": ["https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                "https://en.m.wikipedia.org/wiki/Dow_Jones_Industrial_Average"],
        "djia": ["https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                 "https://en.m.wikipedia.org/wiki/Dow_Jones_Industrial_Average"],
        "dow jones 30": ["https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                         "https://en.m.wikipedia.org/wiki/Dow_Jones_Industrial_Average"],
        "sp500": ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"],
        "s&p 500": ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"],
        "sp500_diva": ["https://en.wikipedia.org/wiki/S%26P_500_Dividend_Aristocrats"],
        "sp400_diva": ["https://en.wikipedia.org/wiki/S%26P_400_Dividend_Aristocrats"],
        "tsx60": ["https://en.wikipedia.org/wiki/S%26P/TSX_60"],
        "asx200": ["https://en.wikipedia.org/wiki/S%26P/ASX_200"],
    }

    if name not in URLS:
        raise ValueError(f"Unbekannter Index: {name}")

    errors = []
    for url in URLS[name]:
        try:
            html = _get_html(url)                  # <- niemals URL direkt an read_html
            tables = _read_tables(html)
            if not tables:
                raise RuntimeError("Keine Tabellen gefunden")

            # passende Spalte finden
            def pick_col(tb: pd.DataFrame) -> str | None:
                pref = ["symbol", "ticker", "epic", "code"]
                for p in pref:
                    for c in tb.columns:
                        if p in str(c).lower():
                            return c
                return None

            tbl = None; col = None
            for tb in tables:
                col = pick_col(tb)
                if col is not None:
                    tbl = tb
                    break
            if tbl is None or col is None:
                tbl = tables[0]; col = tbl.columns[0]

            syms = _clean_symbols(tbl[col])

            # Suffix/Format
            if name in {"dax", "mdax"}:
                syms = _apply_suffix(syms.tolist(), ".DE")
            elif name in {"ftse100", "ftse250"}:
                syms = _apply_suffix(syms.tolist(), ".L")
            elif name == "tsx60":
                syms = _apply_suffix(syms.tolist(), ".TO")
            elif name == "asx200":
                syms = _apply_suffix(syms.tolist(), ".AX")
            else:
                syms = syms.str.replace(".", "-", regex=False).tolist()

            syms = sorted({s for s in syms if s and len(s) <= 12})
            if not syms:
                raise RuntimeError("Tickerliste leer")
            return syms
        except Exception as e:
            errors.append(f"{url}: {e}")
            time.sleep(0.5)
            continue

    # letzter Rettungsanker: Offline-Liste (vermeidet 403-Abbruch)
    if name in OFFLINE_TICKERS:
        return OFFLINE_TICKERS[name]
    raise RuntimeError("Index-Download fehlgeschlagen. Logs:\n" + "\n".join(errors))

# ─────────────────────────────────────────────────────────────
# Metrics (TTM, robust, GBX-fix)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60*30, show_spinner=False)
def metrics_for(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    try:
        info = _safe_info(t)
        fast = _safe_fast(t)
        currency = (fast.get("currency") or info.get("currency") or "").upper()
        is_gbx = currency in {"GBX", "GBp"} or (ticker.endswith(".L") and currency in {"", None})

        px = _hist_close(t, period="5y", interval="1d")
        if px.empty:
            try:
                px = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
            except Exception:
                px = pd.Series(dtype=float)
        if px.empty:
            raise RuntimeError("No price history")

        price_fast = _to_float(fast.get("last_price"))
        price = price_fast if np.isfinite(price_fast) and price_fast > 0 else float(px.iloc[-1])

        low_52w = fast.get("year_low"); high_52w = fast.get("year_high")
        if not (_is_num(low_52w) and _is_num(high_52w)):
            px1y = _hist_close(t, period="1y", interval="1d")
            if px1y.empty:
                px1y = px.tail(252)
            low_52w, high_52w = float(px1y.min()), float(px1y.max())
        else:
            low_52w, high_52w = float(low_52w), float(high_52w)

        if is_gbx:
            price *= 0.01
            low_52w *= 0.01
            high_52w *= 0.01
            px = px * 0.01

        rng = high_52w - low_52w
        pos_52w = (price - low_52w) / (rng if _is_num(rng) and rng > 0 else np.nan)

        try:
            div = t.get_dividends()
        except Exception:
            div = getattr(t, "dividends", pd.Series(dtype=float))
        if is_gbx and isinstance(div, pd.Series) and len(div):
            div = div * 0.01

        if isinstance(div, pd.Series) and len(div):
            div_ttm = float(div[div.index >= (div.index.max() - pd.Timedelta(days=365))].sum())
        else:
            div_ttm = 0.0
        div_yield_ttm = (div_ttm / float(price)) if price and price > 0 else np.nan

        pm = px.resample("M").last()
        dm = div.resample("M").sum().reindex(pm.index, fill_value=0.0) if isinstance(div, pd.Series) and len(div) else pd.Series(0.0, index=pm.index)
        ttm_div_m = dm.rolling(12, min_periods=1).sum()
        yld_m = (ttm_div_m / pm).dropna()
        if len(yld_m):
            months = min(60, len(yld_m))
            start = yld_m.index.max() - pd.DateOffset(months=months)
            yld_5y_med = float(yld_m.loc[yld_m.index >= start].median())
        else:
            yld_5y_med = np.nan

        div_ttm_current = float(ttm_div_m.iloc[-1]) if len(ttm_div_m) else 0.0
        div_ttm_prev = float(ttm_div_m.shift(12).iloc[-1]) if len(ttm_div_m) > 12 else np.nan
        div_cut_24m = int(np.isfinite(div_ttm_prev) and div_ttm_prev > 0 and (div_ttm_current < 0.9 * div_ttm_prev))

        q_is = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
        q_bs = t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame()
        q_cf = t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame()

        revenue = _ttm_sum(q_is, ["Total Revenue","Revenue"])
        ebitda  = _ttm_sum(q_is, ["EBITDA","Ebitda"])
        op_cf   = _ttm_sum(q_cf, ["Total Cash From Operating Activities","Operating Cash Flow"])
        capex_y = _ttm_sum(q_cf, ["Capital Expenditures","Capital Expenditure"])
        capex   = -capex_y if np.isfinite(capex_y) else np.nan
        fcf     = op_cf - capex if np.isfinite(op_cf) and np.isfinite(capex) else np.nan

        div_paid_cf = _ttm_sum(q_cf, ["Dividends Paid"])
        div_cash_ttm = abs(div_paid_cf) if np.isfinite(div_paid_cf) else div_ttm_current

        equity = _ttm_sum(q_bs, ["Total Stockholder Equity","Total Equity Gross Minority Interest"])
        total_debt_cf = _ttm_sum(q_bs, ["Long Term Debt"]) + _ttm_sum(q_bs, ["Short Long Term Debt","Short Term Debt"])
        total_debt_cf = total_debt_cf if np.isfinite(total_debt_cf) else np.nan

        sector = (info.get("sector") or "Unknown")
        mcap = _to_float(info.get("marketCap", fast.get("market_cap")))
        adv3 = _to_float(info.get("averageDailyVolume3Month", fast.get("three_month_average_volume")))

        total_debt = _to_float(info.get("totalDebt", total_debt_cf))
        cash = _to_float(info.get("totalCash", np.nan))

        de_ratio = (total_debt / equity) if (np.isfinite(total_debt) and np.isfinite(equity) and equity > 0) else np.nan
        pe_ttm = _to_float(info.get("trailingPE", np.nan))

        ev_num = (mcap if np.isfinite(mcap) else np.nan)
        if np.isfinite(ev_num):
            ev_num += (total_debt if np.isfinite(total_debt) else 0) - (cash if np.isfinite(cash) else 0)
        ev_ebitda = (ev_num / ebitda) if (np.isfinite(ev_num) and np.isfinite(ebitda) and ebitda > 0) else np.nan

        fcf_margin = (fcf / revenue) if (np.isfinite(fcf) and np.isfinite(revenue) and revenue > 0) else np.nan
        ebitda_margin = (ebitda / revenue) if (np.isfinite(ebitda) and np.isfinite(revenue) and revenue > 0) else np.nan

        try:
            spx = yf.Ticker("^GSPC").history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
            stw = t.history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
            bdf = pd.concat([stw, spx], axis=1).dropna()
            beta = float(np.polyfit(bdf.iloc[:,1].values, bdf.iloc[:,0].values, 1)[0]) if len(bdf) > 10 else np.nan
        except Exception:
            beta = np.nan

        coverage_fcf = (fcf / div_cash_ttm) if (np.isfinite(fcf) and div_cash_ttm > 0) else np.inf
        fcf_payout   = (div_cash_ttm / fcf) if (np.isfinite(fcf) and fcf > 0) else np.inf

        err = np.nan
        if np.isfinite(div_yield_ttm) and div_yield_ttm > 0.25:
            err = "yield_outlier_check_currency"

        return {
            "ticker": ticker, "sector": sector, "price": float(price),
            "div_yield_ttm": float(div_yield_ttm),
            "yield_5y_median": float(yld_5y_med) if np.isfinite(yld_5y_med) else np.nan,
            "low_52w": float(low_52w), "high_52w": float(high_52w),
            "pos_52w": float(pos_52w) if np.isfinite(pos_52w) else np.nan,
            "pe_ttm": float(pe_ttm) if np.isfinite(pe_ttm) else np.nan,
            "ev_ebitda_ttm": float(ev_ebitda) if np.isfinite(ev_ebitda) else np.nan,
            "de_ratio": float(de_ratio) if np.isfinite(de_ratio) else np.nan,
            "fcf_ttm": float(fcf) if np.isfinite(fcf) else np.nan,
            "fcf_margin_ttm": float(fcf_margin) if np.isfinite(fcf_margin) else np.nan,
            "ebitda_ttm": float(ebitda) if np.isfinite(ebitda) else np.nan,
            "ebitda_margin_ttm": float(ebitda_margin) if np.isfinite(ebitda_margin) else np.nan,
            "beta_2y_w": float(beta) if np.isfinite(beta) else np.nan,
            "market_cap": float(mcap) if np.isfinite(mcap) else np.nan,
            "adv_3m": float(adv3) if np.isfinite(adv3) else np.nan,
            "div_cash_ttm": float(div_cash_ttm) if np.isfinite(div_cash_ttm) else np.nan,
            "coverage_fcf_ttm": float(coverage_fcf) if np.isfinite(coverage_fcf) else np.nan,
            "fcf_payout_ttm": float(fcf_payout) if np.isfinite(fcf_payout) else np.nan,
            "div_cut_24m": int(div_cut_24m), "error": err
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# Scoring (editierbare Gewichte)
# ─────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "sc_yield": 0.22, "sc_52w": 0.18, "sc_pe": 0.12, "sc_ev_ebitda": 0.12,
    "sc_de": 0.12, "sc_fcfm": 0.08, "sc_ebitdam": 0.06, "sc_beta": 0.06, "sc_ygap": 0.04,
}

def build_scores(df: pd.DataFrame, weights: Dict[str, float] | None = None) -> pd.DataFrame:
    wdict = weights or DEFAULT_WEIGHTS
    d = df.copy()
    d["sc_yield"]     = (np.clip((d["div_yield_ttm"] - 0.05) / 0.05, 0, 1) * 100)
    d["sc_52w"]       = ((1 - d["pos_52w"]).clip(0, 1) * 100)
    d["sc_pe"]        = _sector_percentile(d, "pe_ttm", invert=True)
    d["sc_ev_ebitda"] = _sector_percentile(d, "ev_ebitda_ttm", invert=True)
    d["sc_de"]        = _sector_percentile(d, "de_ratio", invert=True)
    d.loc[~np.isfinite(d["de_ratio"]) | (d["de_ratio"] < 0), "sc_de"] = 0
    d["sc_fcfm"]      = _sector_percentile(d, "fcf_margin_ttm", invert=False)
    d["sc_ebitdam"]   = _sector_percentile(d, "ebitda_margin_ttm", invert=False)
    d["sc_beta"]      = d["beta_2y_w"].apply(lambda b: _map_beta(b) if np.isfinite(b) else 50.0)

    ygap = np.where(d["yield_5y_median"] > 0,
                    d["div_yield_ttm"] / d["yield_5y_median"] - 1.0,
                    np.nan)
    d["sc_ygap"] = _sector_percentile(pd.DataFrame({"sector": d["sector"], "ygap": ygap}), "ygap", invert=False)

    S = d[list(DEFAULT_WEIGHTS.keys())].astype(float)
    w = pd.Series(wdict).reindex(S.columns).fillna(0.0)
    num = (S * w).sum(axis=1, skipna=True)
    den = ((~S.isna()) * w).sum(axis=1)
    d["score_raw"] = np.where(den > 0, num / den, np.nan)

    cap = d["score_raw"].copy()
    cap = np.where(d["div_cut_24m"] == 1, np.minimum(cap, 59), cap)
    cap = np.where((d["fcf_payout_ttm"] > 1.0) | (d["coverage_fcf_ttm"] < 1.0), np.minimum(cap, 49), cap)
    cap = np.where((d["de_ratio"] > 2.5), cap - 15, cap)
    cap = np.where((d["beta_2y_w"] > 1.5), cap - 10, cap)
    cap = np.where((d["pos_52w"] < 0.10) & ((d["fcf_margin_ttm"] <= 0) | (d["ebitda_margin_ttm"] <= 0)), np.minimum(cap, 49), cap)

    d["score"] = pd.Series(cap, index=d.index).clip(0, 100)
    d["rating"] = np.select([d["score"] >= 75, (d["score"] >= 60) & (d["score"] < 75)],
                            ["BUY", "ACCUMULATE/WATCH"], default="AVOID/HOLD")
    return d

# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
def run_scoring(
    tickers: Iterable[str],
    min_yield: float = 0.05,
    min_mcap: float = 1_000_000_000,
    min_adv: float = 1_500_000,
    exclude_financials: bool = True,
    drop_prefilter_fails: bool = True,
    max_workers: int = 6,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    tickers = sorted({t.strip().upper() for t in tickers if t and isinstance(t, str)})
    if not tickers:
        return pd.DataFrame(columns=EXPECTED_COLS)

    rows, errors = [], []
    pbar = st.progress(0.0, text="Kennzahlen: 0/0")
    total = len(tickers)
    done = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(metrics_for, tk): tk for tk in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                errors.append((tk, str(e)))
                rows.append({"ticker": tk, "error": str(e)})
            done += 1
            pbar.progress(done / total, text=f"Kennzahlen: {done}/{total}")

    df = _ensure_columns(pd.DataFrame(rows))

    df["pf_yield"]  = df["div_yield_ttm"] >= min_yield
    df["pf_mcap"]   = np.isfinite(df["market_cap"]) & (df["market_cap"] >= min_mcap)
    df["pf_adv"]    = np.isfinite(df["adv_3m"]) & (df["adv_3m"] >= min_adv)
    df["pf_sector"] = ~(df["sector"].fillna("Unknown").str.contains("Financial", na=False)) \
                      if exclude_financials else pd.Series(True, index=df.index)
    df["prefilter_pass"] = df[["pf_yield","pf_mcap","pf_adv","pf_sector"]].all(axis=1)

    base = df[df["prefilter_pass"]].copy() if drop_prefilter_fails else df.copy()
    if base.empty:
        out = df.copy()
        out["score"] = np.nan
        out["rating"] = "PF-FAIL (check filters)"
        out["div_yield_%"] = out["div_yield_ttm"] * 100
        out["pos_52w_%"] = out["pos_52w"] * 100
        out["near_52w_low_%"] = (1 - out["pos_52w"]).clip(0, 1) * 100
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].round(2)
        return out

    scored = build_scores(base, weights=weights)
    out = scored.copy()
    out["div_yield_%"] = out["div_yield_ttm"] * 100
    out["pos_52w_%"] = out["pos_52w"] * 100
    out["near_52w_low_%"] = (1 - out["pos_52w"]).clip(0, 1) * 100

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(2)
    return out

# ─────────────────────────────────────────────────────────────
# Sidebar – Inputs
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📥 Eingabedaten")

csv_file = st.sidebar.file_uploader("CSV mit Tickern hochladen", type=["csv"])
uploaded_syms = []
if csv_file is not None:
    try:
        df_csv = pd.read_csv(csv_file)
    except Exception:
        csv_file.seek(0); df_csv = pd.read_csv(csv_file, sep=";")
    col = st.sidebar.selectbox("Spalte mit Tickern auswählen", df_csv.columns.tolist())
    uploaded_syms = df_csv[col].astype(str).str.strip().replace({"nan": np.nan}).dropna().tolist()
    st.sidebar.success(f"{len(uploaded_syms)} Ticker aus CSV")

manual = st.sidebar.text_area("Ticker manuell (kommasepariert)", placeholder="z.B. T, VZ, MO, RIO, BTI")
manual_syms = [s.strip().upper() for s in manual.split(",") if s.strip()] if manual else []

st.sidebar.subheader("📚 Index hinzufügen")
index_choice = st.sidebar.selectbox(
    "Index",
    [
        "– auswählen –",
        "DAX", "MDAX",
        "FTSE 100", "FTSE 250",
        "S&P 500",
        "S&P 500 Div. Aristocrats",
        "S&P 400 Div. Aristocrats",
        "S&P/TSX 60",
        "S&P/ASX 200",
        "Dow Jones 30",
    ]
)
index_syms = []
if index_choice != "– auswählen –":
    try:
        key_map = {
            "ftse 100": "ftse100",
            "ftse 250": "ftse250",
            "dow jones 30": "dow jones 30",
            "s&p 500": "s&p 500",
            "s&p 500 div. aristocrats": "sp500_diva",
            "s&p 400 div. aristocrats": "sp400_diva",
            "s&p/tsx 60": "tsx60",
            "s&p/asx 200": "asx200",
        }
        idx_key = key_map.get(index_choice.lower(), index_choice)
        index_syms = load_index_members(idx_key)
        st.sidebar.info(f"{index_choice}: {len(index_syms)} Werte geladen")
    except Exception as e:
        st.sidebar.error(f"Index-Fehler: {e}")

watchlist = sorted({*uploaded_syms, *manual_syms, *index_syms})
st.sidebar.caption(f"Gesamt-Watchlist: **{len(watchlist)}** Ticker")

# ─────────────────────────────────────────────────────────────
# Sidebar – Filter & editierbare Gewichte
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Filter & Optionen")
min_yield = st.sidebar.number_input("Min. Dividendenrendite", min_value=0.0, max_value=0.3, value=0.05, step=0.005, format="%.3f")
min_mcap  = st.sidebar.number_input("Min. Market Cap (USD)", min_value=0.0, value=1_000_000_000.0, step=100_000_000.0, format="%.0f")
min_adv   = st.sidebar.number_input("Min. 3M ADV (Shares)", min_value=0.0, value=1_500_000.0, step=100_000.0, format="%.0f")
exclude_financials = st.sidebar.checkbox("Finanzsektor ausschließen", value=True)
drop_pf = st.sidebar.checkbox("Nur Pre-Filter-Pass zeigen", value=True)
max_workers = st.sidebar.slider("Parallel-Worker", 1, 12, 6)

st.sidebar.subheader("⚖️ Gewichte (Score-Komponenten)")
if "weights" not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()

c_reset, c_norm = st.sidebar.columns([1,1])
with c_reset:
    if st.button("↩️ Standard", use_container_width=True):
        st.session_state.weights = DEFAULT_WEIGHTS.copy()
with c_norm:
    auto_norm = st.checkbox("Normieren", value=True, help="Skaliert die Gewichte so, dass die Summe 1.0 ergibt")

label_map = {
    "sc_yield": "Yield",
    "sc_52w": "52W-Position (umgekehrt)",
    "sc_pe": "P/E (Sektor, invertiert)",
    "sc_ev_ebitda": "EV/EBITDA (invertiert)",
    "sc_de": "Debt/Equity (invertiert)",
    "sc_fcfm": "FCF-Marge",
    "sc_ebitdam": "EBITDA-Marge",
    "sc_beta": "Beta",
    "sc_ygap": "Yield-/Median-Gap",
}

tmp_weights = {}
for k, default in DEFAULT_WEIGHTS.items():
    tmp_weights[k] = st.sidebar.slider(
        label_map.get(k, k),
        0.0, 1.0,
        float(st.session_state.weights.get(k, default)),
        0.01
    )

total_w = sum(tmp_weights.values())
if auto_norm and total_w > 0:
    weights = {k: v / total_w for k, v in tmp_weights.items()}
else:
    weights = tmp_weights

# persist current slider positions
st.session_state.weights = tmp_weights
st.sidebar.caption(f"Gewichtssumme: **{sum(weights.values()):.2f}**")

run_btn = st.sidebar.button("🔎 Score berechnen", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Main – Output
# ─────────────────────────────────────────────────────────────
st.subheader("Watchlist")
if watchlist:
    with st.expander(f"Watchlist anzeigen ({len(watchlist)} Ticker)"):
        st.code(", ".join(map(str, watchlist)), wrap_lines=True)
else:
    st.info("Lade eine CSV hoch, füge Ticker manuell hinzu oder wähle einen Index.")

if run_btn and watchlist:
    df = run_scoring(
        watchlist,
        min_yield=min_yield, min_mcap=min_mcap, min_adv=min_adv,
        exclude_financials=exclude_financials, drop_prefilter_fails=drop_pf,
        max_workers=max_workers,
        weights=weights,
    )

    st.subheader("Ergebnisse")
    if not df.empty:
        counts = df["rating"].value_counts(dropna=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BUY", int(counts.get("BUY", 0)))
        c2.metric("ACCUMULATE/WATCH", int(counts.get("ACCUMULATE/WATCH", 0)))
        c3.metric("AVOID/HOLD", int(counts.get("AVOID/HOLD", 0)))
        c4.metric("Total", len(df))

        prefer_cols = [
            "ticker","sector","price","div_yield_%","near_52w_low_%",
            "pe_ttm","ev_ebitda_ttm","de_ratio","fcf_margin_ttm","ebitda_margin_ttm",
            "beta_2y_w","market_cap","adv_3m","score","rating","error"
        ]
        show_cols = [c for c in prefer_cols if c in df.columns]
        st.dataframe(
            df[show_cols],
            use_container_width=True,
            column_config={
                "div_yield_%": st.column_config.NumberColumn("DivR %", format="%.2f"),
                "near_52w_low_%": st.column_config.NumberColumn("Nähe 52W-Low %", format="%.2f"),
                "price": st.column_config.NumberColumn("Price", format="%.2f"),
                "pe_ttm": st.column_config.NumberColumn("PE (TTM)", format="%.2f"),
                "ev_ebitda_ttm": st.column_config.NumberColumn("EV/EBITDA", format="%.2f"),
                "de_ratio": st.column_config.NumberColumn("Debt/Equity", format="%.2f"),
                "fcf_margin_ttm": st.column_config.NumberColumn("FCF-Marge", format="%.2f"),
                "ebitda_margin_ttm": st.column_config.NumberColumn("EBITDA-Marge", format="%.2f"),
                "beta_2y_w": st.column_config.NumberColumn("Beta (2Y W)", format="%.2f"),
                "market_cap": st.column_config.NumberColumn("Market Cap", format="%.0f"),
                "adv_3m": st.column_config.NumberColumn("ADV 3M", format="%.0f"),
                "score": st.column_config.NumberColumn("Score", format="%.2f"),
            },
        )

        st.download_button(
            "⬇️ Ergebnisse als CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="high_yield_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

        err_df = df[df["error"].notna()][["ticker","error"]]
        if not err_df.empty:
            st.warning("Hinweise/Fehler beim Laden einiger Ticker:")
            st.dataframe(err_df, use_container_width=True)
else:
    st.caption("Tipp: Bei EU/UK-Werten Yahoo-Suffixe nutzen (.DE, .L, .VI, .PA, .TO, .AX, …).")

