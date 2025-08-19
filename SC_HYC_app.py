# app.py
# High-Yield-Dividend Scoring – Streamlit App
# -------------------------------------------
# pip install streamlit yfinance pandas numpy lxml

import time
from typing import Dict, List, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="High-Yield Dividend Scoring", layout="wide")
st.title("📈 High-Yield Dividend Scoring")
st.caption("Yahoo Finance • TTM-basierte Kennzahlen • sektorrelative Perzentile • robuste Fetch-Logik")

# =========================
# Core Helpers (aus deinem Code, leicht angepasst)
# =========================
def _row(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    for k in keys:
        if isinstance(df, pd.DataFrame) and k in df.index:
            s = df.loc[k].dropna().astype(float)
            if len(s): return s
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
    work['sector'] = work.get('sector', pd.Series(index=work.index, dtype='object')).fillna('Unknown')
    p = work.groupby('sector', dropna=False, group_keys=False)[col].apply(_percentile_rank)
    if invert: p = 1 - p
    return (p * 100).clip(0, 100)

def _map_beta(b: float) -> float:
    return float(np.interp(b, [0.4, 0.8, 1.0, 1.5], [100, 70, 50, 0], left=100, right=0))

def _coerce_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cols = df.columns
    if 'Close' in cols:
        close = df['Close']
        if isinstance(close, pd.Series):
            return close.dropna()
        if isinstance(close, pd.DataFrame):
            try:
                if ticker in close.columns:
                    return close[ticker].squeeze().dropna()
            except Exception:
                pass
            return close.iloc[:, 0].squeeze().dropna()
    if isinstance(cols, pd.MultiIndex):
        try:
            return df.xs(('Close', ticker), axis=1).squeeze().dropna()
        except Exception:
            close_cols = [c for c in cols if isinstance(c, tuple) and c[0] == 'Close']
            if close_cols:
                return df[close_cols[0]].squeeze().dropna()
    return df.iloc[:, -1].squeeze().dropna()

def _download_close_series(ticker: str, period: str = "5y", interval: str = "1d",
                           retries: int = 3, pause: float = 1.2) -> pd.Series:
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            close = _coerce_close_series(df, ticker)
            if not close.empty:
                return close
            last_err = "empty close series"
        except Exception as e:
            last_err = str(e)
        time.sleep(pause)
    raise RuntimeError(f"download failed for {ticker} ({period}/{interval}) – {last_err}")

def _sp_fast_info(t: yf.Ticker):
    fast = getattr(t, "fast_info", {}) or {}
    mcap = fast.get("market_cap")
    adv3 = fast.get("three_month_average_volume") or fast.get("three_month_average_volume_shares")
    return mcap, adv3

EXPECTED_COLS = [
    "ticker","sector","price",
    "div_yield_ttm","yield_5y_median","low_52w","high_52w","pos_52w",
    "pe_ttm","ev_ebitda_ttm","de_ratio",
    "fcf_ttm","fcf_margin_ttm","ebitda_ttm","ebitda_margin_ttm",
    "beta_2y_w","market_cap","adv_3m",
    "div_cash_ttm","coverage_fcf_ttm","fcf_payout_ttm","div_cut_24m","error"
]
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLS:
        if c not in df.columns: df[c] = np.nan
    return df

# =========================
# Index-Constituents (DAX, Dow 30)
# =========================
@st.cache_data(ttl=60*60*12)  # 12h
def load_index_members(name: str) -> List[str]:
    name = name.lower().strip()
    if name in {"dax", "dax40"}:
        # Wikipedia: DAX – Spalte "Ticker symbol" (ohne .DE)
        url = "https://en.wikipedia.org/wiki/DAX"
        tables = pd.read_html(url, flavor="lxml")
        # suche Tabelle mit 'Ticker' oder 'Ticker symbol'
        table = None
        for tb in tables:
            cols = [c.lower() for c in tb.columns.astype(str)]
            if any("ticker" in c for c in cols):
                table = tb
                break
        if table is None:
            raise RuntimeError("DAX constituents not found on page")
        # Spalte ermitteln
        col = [c for c in table.columns if "Ticker" in str(c) or "ticker" in str(c).lower()][0]
        syms = table[col].astype(str).str.replace(r"\W+", "", regex=True).str.upper().tolist()
        return [s + ".DE" if "." not in s else s for s in syms]
    if name in {"dow", "djia", "dow jones 30", "dow jones"}:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tables = pd.read_html(url, flavor="lxml")
        table = None
        for tb in tables:
            cols = [c.lower() for c in tb.columns.astype(str)]
            if any(("symbol" in c) for c in cols):
                table = tb
                break
        if table is None:
            raise RuntimeError("Dow 30 constituents not found on page")
        col = [c for c in table.columns if "Symbol" in str(c) or "symbol" in str(c).lower()][0]
        syms = table[col].astype(str).str.replace(r"\W+", "", regex=True).str.upper().tolist()
        return syms
    raise ValueError("Unbekannter Index: " + name)

# =========================
# Metrics + Scoring
# =========================
@st.cache_data(ttl=60*30)  # 30 min Cache je Ticker
def metrics_for(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    try:
        px = _download_close_series(ticker, period="5y", interval="1d")
        price = float(px.iloc[-1])

        cutoff = px.index.max() - pd.Timedelta(days=365)
        px1y = px.loc[px.index >= cutoff]
        if px1y.empty and len(px) >= 252:
            px1y = px.tail(252)

        low_52w = float(px1y.min())
        high_52w = float(px1y.max())
        rng = max(1e-9, high_52w - low_52w)
        pos_52w = (price - low_52w) / rng

        div = t.dividends if hasattr(t, "dividends") else pd.Series(dtype=float)
        if isinstance(div, pd.Series) and len(div):
            div_ttm = float(div[div.index >= (div.index.max() - pd.Timedelta(days=365))].sum())
        else:
            div_ttm = 0.0
        div_yield_ttm = div_ttm / price if price > 0 else np.nan

        pm = px.resample("ME").last()
        dm = div.resample("ME").sum().reindex(pm.index, fill_value=0.0) if len(div) else pd.Series(0.0, index=pm.index)
        ttm_div_m = dm.rolling(12, min_periods=1).sum()
        yld_m = (ttm_div_m / pm).dropna()
        if len(yld_m):
            start_5y = yld_m.index.max() - pd.DateOffset(months=60)
            yld_5y_med = float(yld_m.loc[yld_m.index >= start_5y].median())
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
        if not np.isfinite(total_debt_cf): total_debt_cf = np.nan

        try:
            info = t.get_info()
        except Exception:
            info = getattr(t, "info", {}) or {}
        mcap_fast, adv_fast = _sp_fast_info(t)

        sector = (info.get("sector") or "Unknown")
        mcap = info.get("marketCap", mcap_fast)
        adv3 = info.get("averageDailyVolume3Month", adv_fast)

        total_debt = info.get("totalDebt", total_debt_cf)
        cash = info.get("totalCash", np.nan)

        de_ratio = (total_debt / equity) if (np.isfinite(total_debt) and np.isfinite(equity) and equity > 0) else np.nan
        pe_ttm = info.get("trailingPE", np.nan)

        ev = (mcap if np.isfinite(mcap) else np.nan) + (total_debt if np.isfinite(total_debt) else 0) - (cash if np.isfinite(cash) else 0)
        ev_ebitda = (ev / ebitda) if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda > 0) else np.nan

        fcf_margin = (fcf / revenue) if (np.isfinite(fcf) and np.isfinite(revenue) and revenue > 0) else np.nan
        ebitda_margin = (ebitda / revenue) if (np.isfinite(ebitda) and np.isfinite(revenue) and revenue > 0) else np.nan

        spx = _download_close_series("^GSPC", period="2y", interval="1wk").pct_change().dropna()
        stw = _download_close_series(ticker, period="2y", interval="1wk").pct_change().dropna()
        bdf = pd.concat([stw, spx], axis=1).dropna()
        beta = float(np.polyfit(bdf.iloc[:,1].values, bdf.iloc[:,0].values, 1)[0]) if len(bdf) > 10 else np.nan

        coverage_fcf = (fcf / div_cash_ttm) if (np.isfinite(fcf) and div_cash_ttm > 0) else np.inf
        fcf_payout   = (div_cash_ttm / fcf) if (np.isfinite(fcf) and fcf > 0) else np.inf

        return {
            "ticker": ticker, "sector": sector, "price": price,
            "div_yield_ttm": div_yield_ttm, "yield_5y_median": yld_5y_med,
            "low_52w": low_52w, "high_52w": high_52w, "pos_52w": pos_52w,
            "pe_ttm": pe_ttm, "ev_ebitda_ttm": ev_ebitda, "de_ratio": de_ratio,
            "fcf_ttm": fcf, "fcf_margin_ttm": fcf_margin,
            "ebitda_ttm": ebitda, "ebitda_margin_ttm": ebitda_margin,
            "beta_2y_w": beta, "market_cap": mcap, "adv_3m": adv3,
            "div_cash_ttm": div_cash_ttm, "coverage_fcf_ttm": coverage_fcf, "fcf_payout_ttm": fcf_payout,
            "div_cut_24m": div_cut_24m, "error": np.nan
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

WEIGHTS: Dict[str, float] = {
    'sc_yield': 0.22, 'sc_52w': 0.18, 'sc_pe': 0.12, 'sc_ev_ebitda': 0.12,
    'sc_de': 0.12, 'sc_fcfm': 0.08, 'sc_ebitdam': 0.06, 'sc_beta': 0.06, 'sc_ygap': 0.04,
}

def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['sc_yield']    = (np.clip((d['div_yield_ttm'] - 0.05) / 0.05, 0, 1) * 100)
    d['sc_52w']      = ((1 - d['pos_52w']).clip(0, 1) * 100)
    d['sc_pe']       = _sector_percentile(d, 'pe_ttm', invert=True)
    d['sc_ev_ebitda']= _sector_percentile(d, 'ev_ebitda_ttm', invert=True)
    d['sc_de']       = _sector_percentile(d, 'de_ratio', invert=True)
    d.loc[~np.isfinite(d['de_ratio']) | (d['de_ratio'] < 0), 'sc_de'] = 0
    d['sc_fcfm']     = _sector_percentile(d, 'fcf_margin_ttm', invert=False)
    d['sc_ebitdam']  = _sector_percentile(d, 'ebitda_margin_ttm', invert=False)
    d['sc_beta']     = d['beta_2y_w'].apply(_map_beta)
    ygap             = (d['div_yield_ttm'] / d['yield_5y_median']) - 1.0
    d['sc_ygap']     = _sector_percentile(pd.DataFrame({'sector': d['sector'], 'ygap': ygap}), 'ygap', invert=False)

    S = d[list(WEIGHTS.keys())].astype(float)
    w = pd.Series(WEIGHTS)
    num = (S * w).sum(axis=1, skipna=True)
    den = ((~S.isna()) * w).sum(axis=1)
    d['score_raw'] = np.where(den > 0, num / den, np.nan)

    cap = d['score_raw'].copy()
    cap = np.where(d['div_cut_24m'] == 1, np.minimum(cap, 59), cap)
    cap = np.where((d['fcf_payout_ttm'] > 1.0) | (d['coverage_fcf_ttm'] < 1.0), np.minimum(cap, 49), cap)
    cap = np.where((d['de_ratio'] > 2.5), cap - 15, cap)
    cap = np.where((d['beta_2y_w'] > 1.5), cap - 10, cap)
    cap = np.where((d['pos_52w'] < 0.10) & ((d['fcf_margin_ttm'] <= 0) | (d['ebitda_margin_ttm'] <= 0)),
                   np.minimum(cap, 49), cap)

    d['score'] = pd.Series(cap, index=d.index).clip(0, 100)
    d['rating'] = np.select([d['score'] >= 75, (d['score'] >= 60) & (d['score'] < 75)],
                            ['BUY', 'ACCUMULATE/WATCH'], default='AVOID/HOLD')
    return d

def run_scoring(
    tickers: Iterable[str],
    min_yield: float = 0.05,
    min_mcap: float = 1_000_000_000,
    min_adv: float = 1_500_000,
    exclude_financials: bool = True,
    drop_prefilter_fails: bool = True,
    max_workers: int = 6,
) -> pd.DataFrame:
    tickers = sorted({t.strip().upper() for t in tickers if t and isinstance(t, str)})
    if not tickers:
        return pd.DataFrame(columns=EXPECTED_COLS)

    # Parallel fetchen mit Progress
    rows, errors = [], []
    pbar = st.progress(0.0, text="Lade Kennzahlen …")
    total = len(tickers)
    done = 0

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

    # Pre-Filter
    df['pf_yield']  = df['div_yield_ttm'] >= min_yield
    df['pf_mcap']   = np.isfinite(df['market_cap']) & (df['market_cap'] >= min_mcap)
    df['pf_adv']    = np.isfinite(df['adv_3m']) & (df['adv_3m'] >= min_adv)
    df['pf_sector'] = ~(df['sector'].fillna('Unknown').str.contains('Financial', na=False)) if exclude_financials else True
    df['prefilter_pass'] = df[['pf_yield','pf_mcap','pf_adv','pf_sector']].all(axis=1)

    base = df[df['prefilter_pass']].copy() if drop_prefilter_fails else df.copy()
    if base.empty:
        out = df.copy()
        out['score'] = np.nan
        out['rating'] = 'PF-FAIL (check filters)'
        out['div_yield_%'] = out['div_yield_ttm'] * 100
        out['pos_52w_%'] = out['pos_52w'] * 100
        # Darstellung runden
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].round(2)
        return out

    scored = build_scores(base)
    out = scored.copy()
    out['div_yield_%'] = out['div_yield_ttm'] * 100
    out['pos_52w_%'] = out['pos_52w'] * 100

    # Sortieren + Runden (nur Darstellung)
    out = out.sort_values('score', ascending=False).reset_index(drop=True)
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(2)
    return out

# =========================
# Sidebar – Data Input
# =========================
st.sidebar.header("📥 Eingabedaten")

# 1) CSV Upload
csv_file = st.sidebar.file_uploader("CSV mit Tickern hochladen", type=["csv"])
uploaded_syms = []
if csv_file is not None:
    df_csv = pd.read_csv(csv_file)
    # Spalten anbieten
    col = st.sidebar.selectbox("Spalte mit Tickern auswählen", df_csv.columns.tolist())
    uploaded_syms = (
        df_csv[col].astype(str).str.strip().replace({"nan": np.nan}).dropna().tolist()
    )
    st.sidebar.success(f"{len(uploaded_syms)} Ticker aus CSV geladen")

# 2) Manuell
manual = st.sidebar.text_area("Ticker manuell (kommasepariert)", placeholder="z.B. T, VZ, MO, RIO, BTI")
manual_syms = [s.strip().upper() for s in manual.split(",") if s.strip()] if manual else []

# 3) Index Loader
st.sidebar.subheader("📚 Index hinzufügen")
index_choice = st.sidebar.selectbox("Index", ["– auswählen –", "DAX", "Dow Jones 30"])
index_syms = []
if index_choice != "– auswählen –":
    try:
        index_syms = load_index_members(index_choice)
        st.sidebar.info(f"{index_choice}: {len(index_syms)} Werte geladen")
    except Exception as e:
        st.sidebar.error(f"Index-Fehler: {e}")

# 4) Watchlist zusammenführen
watchlist = sorted({*uploaded_syms, *manual_syms, *index_syms})
st.sidebar.caption(f"Gesamt-Watchlist: **{len(watchlist)}** Ticker")

# 5) Filter & Optionen
st.sidebar.header("⚙️ Filter & Optionen")
min_yield = st.sidebar.number_input("Min. Dividendenrendite", min_value=0.0, max_value=0.2, value=0.05, step=0.005, format="%.3f")
min_mcap  = st.sidebar.number_input("Min. Market Cap (USD)", min_value=0.0, value=1_000_000_000.0, step=100_000_000.0, format="%.0f")
min_adv   = st.sidebar.number_input("Min. 3M ADV (Shares)", min_value=0.0, value=1_500_000.0, step=100_000.0, format="%.0f")
exclude_financials = st.sidebar.checkbox("Finanzsektor ausschließen", value=True)
drop_pf = st.sidebar.checkbox("Nur Pre-Filter-Pass zeigen", value=True)
max_workers = st.sidebar.slider("Parallel-Worker", 1, 12, 6)

run_btn = st.sidebar.button("🔎 Score berechnen", use_container_width=True)

# =========================
# Main – Execution & Output
# =========================
st.subheader("Watchlist")
if watchlist:
    st.code(", ".join(watchlist), wrap=True)
else:
    st.info("Lade eine CSV hoch, füge Ticker manuell hinzu oder wähle einen Index.")

if run_btn and watchlist:
    df = run_scoring(
        watchlist,
        min_yield=min_yield,
        min_mcap=min_mcap,
        min_adv=min_adv,
        exclude_financials=exclude_financials,
        drop_prefilter_fails=drop_pf,
        max_workers=max_workers,
    )

    # Kennzahlen/Übersicht
    st.subheader("Ergebnisse")
    if not df.empty:
        # Rating-Zusammenfassung
        counts = df['rating'].value_counts(dropna=False)
        cols = st.columns(4)
        cols[0].metric("BUY", int(counts.get("BUY", 0)))
        cols[1].metric("ACCUMULATE/WATCH", int(counts.get("ACCUMULATE/WATCH", 0)))
        cols[2].metric("AVOID/HOLD", int(counts.get("AVOID/HOLD", 0)))
        cols[3].metric("Total", len(df))

        # Anzeige-Tabelle
        prefer_cols = [
            'ticker','sector','price','div_yield_%','pos_52w_%',
            'pe_ttm','ev_ebitda_ttm','de_ratio','fcf_margin_ttm','ebitda_margin_ttm',
            'beta_2y_w','market_cap','adv_3m','score','rating','error'
        ]
        show_cols = [c for c in prefer_cols if c in df.columns]
        st.dataframe(
            df[show_cols],
            use_container_width=True,
            column_config={
                "div_yield_%": st.column_config.NumberColumn("DivR %", format="%.2f"),
                "pos_52w_%": st.column_config.NumberColumn("Nähe 52W-Low %", format="%.2f"),
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

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Ergebnisse als CSV", data=csv, file_name="high_yield_scores.csv", mime="text/csv")

        # Fehlerliste (falls vorhanden)
        err_df = df[df['error'].notna()][['ticker','error']]
        if not err_df.empty:
            st.warning("Einige Ticker hatten Fehler beim Laden:")
            st.dataframe(err_df, use_container_width=True)

else:
    st.caption("Tipp: Bei europäischen Werten ggf. Yahoo-Suffixe nutzen (z. B. **.DE**, **.L**, **.PA**).")
