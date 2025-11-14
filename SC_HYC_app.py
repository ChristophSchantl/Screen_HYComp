# app.py — Master Scoring Model (Streamlit • Pro)
from __future__ import annotations
import numpy as np, pandas as pd, streamlit as st, yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

st.set_page_config(page_title="Master Scoring Model (Pro)", layout="wide")
st.title("📈 Master Scoring Model (Pro)")
st.caption("Yahoo Finance • TTM-Kennzahlen • sektorrelative Perzentile • robuste Datenlogik • PM-Summary")

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
            if len(s): return s
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
    if invert: p = 1 - p
    return (p * 100).clip(0, 100)

def _hist_close(ticker: str, period="5y", interval="1d") -> pd.Series:
    try:
        h = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if "Close" in h: return h["Close"].dropna()
    except Exception: pass
    try:
        h = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if "Close" in h: return h["Close"].dropna()
    except Exception: pass
    return pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────
# Metrics (robust; 52W nur aus History)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60*30)
def fetch_metrics(tk: str) -> dict:
    t = yf.Ticker(tk)
    try:
        try:
            info = t.get_info()
        except Exception:
            info = getattr(t, "info", {}) or {}

        px = _hist_close(tk, "5y", "1d")
        if px.empty: raise RuntimeError("no_price_history")
        price = float(px.iloc[-1])

        # 52W aus ADJUSTED history
        px1y = _hist_close(tk, "1y", "1d")
        if px1y.empty: px1y = px.tail(252)
        low_52w  = float(px1y.min())
        high_52w = float(px1y.max())
        rng = high_52w - low_52w
        pos_52w = (price - low_52w) / (rng if rng > 0 else np.nan)

        # Dividenden
        try: div = t.get_dividends()
        except Exception: div = getattr(t, "dividends", pd.Series(dtype=float))
        pm = px.resample("M").last()
        dm = div.resample("M").sum().reindex(pm.index, fill_value=0.0) if isinstance(div, pd.Series) else pd.Series(0.0, index=pm.index)
        ttm_div_m = dm.rolling(12, min_periods=1).sum()
        yld_ttm  = float(ttm_div_m.iloc[-1] / price) if price > 0 else np.nan
        yld_med5 = float((ttm_div_m / pm).tail(min(60, len(pm))).median()) if len(pm) else np.nan

        # Quarterlies
        q_is = t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame()
        q_bs = t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame()
        q_cf = t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame()

        revenue=_ttm_sum(q_is,["Total Revenue","Revenue"])
        ebitda =_ttm_sum(q_is,["EBITDA","Ebitda"])
        op_cf  =_ttm_sum(q_cf,["Total Cash From Operating Activities","Operating Cash Flow"])
        capex_y=_ttm_sum(q_cf,["Capital Expenditures","Capital Expenditure"])
        capex  = -capex_y if np.isfinite(capex_y) else np.nan
        fcf    = op_cf - capex if np.isfinite(op_cf) and np.isfinite(capex) else np.nan

        ebitda_margin = (ebitda/revenue) if (np.isfinite(ebitda) and np.isfinite(revenue) and revenue>0) else np.nan
        fcf_margin    = (fcf/revenue) if (np.isfinite(fcf) and np.isfinite(revenue) and revenue>0) else np.nan

        equity=_ttm_sum(q_bs,["Total Stockholder Equity","Total Equity Gross Minority Interest"])
        total_debt = _ttm_sum(q_bs,["Long Term Debt"]) + _ttm_sum(q_bs,["Short Long Term Debt","Short Term Debt"])
        de_ratio = (total_debt/equity) if (np.isfinite(total_debt) and np.isfinite(equity) and equity>0) else np.nan

        # Beta (2y w)
        try:
            spx = yf.Ticker("^GSPC").history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
            stw = t.history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
            bdf = pd.concat([stw, spx], axis=1).dropna()
            beta = float(np.polyfit(bdf.iloc[:,1].values, bdf.iloc[:,0].values, 1)[0]) if len(bdf)>10 else np.nan
        except Exception:
            beta = np.nan

        # EV/EBITDA, Meta
        mcap=_to_float(info.get("marketCap"))
        pe  =_to_float(info.get("trailingPE"))
        cash=_to_float(info.get("totalCash")); debt=_to_float(info.get("totalDebt"))
        ev = mcap if np.isfinite(mcap) else np.nan
        if np.isfinite(ev): ev += (debt if np.isfinite(debt) else 0) - (cash if np.isfinite(cash) else 0)
        ev_ebitda = (ev/ebitda) if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda>0) else np.nan

        # Div-Qualität
        div_paid = _ttm_sum(q_cf,["Dividends Paid"])
        div_cash_ttm = abs(div_paid) if np.isfinite(div_paid) else float(ttm_div_m.iloc[-1] if len(ttm_div_m) else 0.0)
        coverage_fcf = (fcf/div_cash_ttm) if (np.isfinite(fcf) and div_cash_ttm>0) else np.inf
        fcf_payout   = (div_cash_ttm/fcf) if (np.isfinite(fcf) and fcf>0) else np.inf
        prev = float(ttm_div_m.shift(12).iloc[-1]) if len(ttm_div_m)>12 else np.nan
        div_cut = int(np.isfinite(prev) and prev>0 and (float(ttm_div_m.iloc[-1]) < 0.9*prev))

        sector = (info.get("sector") or "Unknown")
        adv3   = _to_float(info.get("averageDailyVolume3Month"))

        return dict(ticker=tk, sector=sector, price=price,
            low_52w=low_52w, high_52w=high_52w, pos_52w=pos_52w,
            div_yield_ttm=yld_ttm, yield_5y_median=yld_med5,
            pe_ttm=pe, ev_ebitda_ttm=ev_ebitda, de_ratio=de_ratio,
            fcf_margin_ttm=fcf_margin, ebitda_margin_ttm=ebitda_margin,
            beta_2y_w=beta, market_cap=mcap, adv_3m=adv3,
            div_cash_ttm=div_cash_ttm, coverage_fcf_ttm=coverage_fcf, fcf_payout_ttm=fcf_payout,
            div_cut_24m=div_cut, error=np.nan)
    except Exception as e:
        return {"ticker": tk, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────
FACTORS = ["sc_yield","sc_52w","sc_pe","sc_ev_ebitda","sc_de","sc_fcfm","sc_ebitdam","sc_beta","sc_ygap"]
DEFAULT_W = {"sc_yield":0.25,"sc_52w":0.22,"sc_pe":0.12,"sc_ev_ebitda":0.12,"sc_de":0.12,"sc_fcfm":0.07,"sc_ebitdam":0.04,"sc_beta":0.04,"sc_ygap":0.02}

PARAMS_DEF = dict(
    yield_floor=0.04, yield_scale=0.04,
    invert_52w=True, pos_52w_gamma=1.5,
    beta_knots=[0.30,0.80,1.00,1.60], beta_scores=[100,75,55,10],
    de_threshold=2.0, high_de_penalty=20.0,
    beta_threshold=1.6, high_beta_penalty=12.0,
    cap_max_after_cut=55.0, cap_max_after_cov=45.0
)

def _beta_map(b, knots, scores):
    if not np.isfinite(b): return 50.0
    return float(np.interp(b, knots, scores, left=scores[0], right=scores[-1]))

def build_scores(df: pd.DataFrame, weights: dict, params: dict,
                 *, fixed_denominator: bool=True, missing_policy: str="neutral50") -> pd.DataFrame:
    P = {**PARAMS_DEF, **(params or {})}
    d = df.copy()

    d["sc_yield"] = (np.clip((d["div_yield_ttm"]-P["yield_floor"])/P["yield_scale"],0,1)*100)
    base = (1-d["pos_52w"]) if P["invert_52w"] else d["pos_52w"]
    d["sc_52w"] = (np.clip(base,0,1)**P["pos_52w_gamma"])*100
    d["sc_pe"]        = _sector_percentile(d,"pe_ttm",invert=True)
    d["sc_ev_ebitda"] = _sector_percentile(d,"ev_ebitda_ttm",invert=True)
    d["sc_de"]        = _sector_percentile(d,"de_ratio",invert=True)
    d.loc[(~np.isfinite(d["de_ratio"]))|(d["de_ratio"]<0),"sc_de"]=0
    d["sc_fcfm"]      = _sector_percentile(d,"fcf_margin_ttm",invert=False)
    d["sc_ebitdam"]   = _sector_percentile(d,"ebitda_margin_ttm",invert=False)
    d["sc_beta"]      = d["beta_2y_w"].apply(lambda b: _beta_map(b,P["beta_knots"],P["beta_scores"]))

    ygap=np.where(d["yield_5y_median"]>0, d["div_yield_ttm"]/d["yield_5y_median"]-1.0, np.nan)
    d["sc_ygap"]=_sector_percentile(pd.DataFrame({"sector":d["sector"],"ygap":ygap}),"ygap",invert=False)

    S=d[FACTORS].astype(float)
    if missing_policy=="neutral50":
        S=S.fillna(50.0)
    elif missing_policy=="sector_median":
        for c in S.columns:
            med=d.groupby("sector")[c].transform("median")
            S[c]=S[c].fillna(med).fillna(50.0)

    W=pd.Series(weights).reindex(FACTORS).fillna(0.0)
    num=(S*W).sum(axis=1)
    den=float(W.sum()) if fixed_denominator else ((~S.isna())*W).sum(axis=1)
    d["score_raw"]=np.where(den>0,num/den,np.nan)

    cap=d["score_raw"].copy()
    cap=np.where(d["div_cut_24m"]==1, np.minimum(cap,P["cap_max_after_cut"]), cap)
    cap=np.where((d["fcf_payout_ttm"]>1.0)|(d["coverage_fcf_ttm"]<1.0), np.minimum(cap,P["cap_max_after_cov"]), cap)
    cap=np.where((d["de_ratio"]>P["de_threshold"]), cap-P["high_de_penalty"], cap)
    cap=np.where((d["beta_2y_w"]>P["beta_threshold"]), cap-P["high_beta_penalty"], cap)

    d["score"]=pd.Series(cap,index=d.index).clip(0,100)
    d["rating"]=np.select([d["score"]>=75,(d["score"]>=60)&(d["score"]<75)],["BUY","ACCUMULATE/WATCH"],"AVOID/HOLD")
    return d

# ─────────────────────────────────────────────────────────────
# Sidebar — Inputs
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📥 Eingabe")
csv_file=st.sidebar.file_uploader("CSV mit Ticker-Spalte", type=["csv"])
tickers=[]
if csv_file is not None:
    try: df_csv=pd.read_csv(csv_file)
    except Exception: csv_file.seek(0); df_csv=pd.read_csv(csv_file, sep=";")
    col=st.sidebar.selectbox("Ticker-Spalte", df_csv.columns.tolist())
    tickers=df_csv[col].astype(str).str.strip().replace({"nan":np.nan}).dropna().tolist()
manual = st.sidebar.text_area("Ticker manuell (kommasepariert)", "DHL.DE, DBK.DE, T, VZ, MO")
tickers=sorted({*tickers, *[s.strip().upper() for s in manual.split(",") if s.strip()]})
st.sidebar.caption(f"Watchlist: **{len(tickers)}**")

# ─────────────────────────────────────────────────────────────
# Sidebar — Parameter & Gewichte
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Scoring-Parameter")
P0=PARAMS_DEF
y_floor=st.sidebar.number_input("Yield-Floor", min_value=0.0, max_value=0.5, value=float(P0["yield_floor"]), step=0.005, format="%.3f")
y_scale=st.sidebar.number_input("Yield-Scale", min_value=0.001, max_value=1.0, value=float(P0["yield_scale"]), step=0.005, format="%.3f")
invert_52w=st.sidebar.checkbox("52W invertieren (nah am Low = besser)", value=True)
pos_gamma =st.sidebar.slider("52W-Gamma", 0.3, 3.0, float(P0["pos_52w_gamma"]), 0.1)

c1,c2,c3,c4=st.sidebar.columns(4)
beta_lo=c1.number_input("β lo", 0.0, 3.0, float(P0["beta_knots"][0]), 0.05)
beta_m1=c2.number_input("β m1", 0.0, 3.0, float(P0["beta_knots"][1]), 0.05)
beta_m2=c3.number_input("β m2", 0.0, 3.0, float(P0["beta_knots"][2]), 0.05)
beta_hi=c4.number_input("β hi", 0.0, 3.0, float(P0["beta_knots"][3]), 0.05)
knots=[beta_lo,beta_m1,beta_m2,beta_hi]
if any(np.diff(knots)<0):
    knots=sorted(knots)
    st.sidebar.info("β-Knoten wurden aufsteigend sortiert.")

c5,c6,c7,c8=st.sidebar.columns(4)
pts_lo=c5.number_input("Pts lo", 0.0, 100.0, float(P0["beta_scores"][0]), 1.0)
pts_m1=c6.number_input("Pts m1", 0.0, 100.0, float(P0["beta_scores"][1]), 1.0)
pts_m2=c7.number_input("Pts m2", 0.0, 100.0, float(P0["beta_scores"][2]), 1.0)
pts_hi=c8.number_input("Pts hi", 0.0, 100.0, float(P0["beta_scores"][3]), 1.0)
scores=[pts_lo,pts_m1,pts_m2,pts_hi]

de_thr =st.sidebar.number_input("D/E-Schwelle", 0.0, 10.0, float(P0["de_threshold"]), 0.1)
de_pen =st.sidebar.number_input("Strafe D/E", 0.0, 50.0, float(P0["high_de_penalty"]), 1.0)
beta_thr=st.sidebar.number_input("Beta-Schwelle", 0.0, 3.0, float(P0["beta_threshold"]), 0.1)
beta_pen=st.sidebar.number_input("Strafe Beta", 0.0, 50.0, float(P0["high_beta_penalty"]), 1.0)
cap_cut=st.sidebar.number_input("Cap nach Div-Cut (max Score)", 0.0, 100.0, float(P0["cap_max_after_cut"]), 1.0)
cap_cov=st.sidebar.number_input("Cap bei schwacher Coverage (max Score)", 0.0, 100.0, float(P0["cap_max_after_cov"]), 1.0)

params=dict(
    yield_floor=y_floor, yield_scale=y_scale,
    invert_52w=invert_52w, pos_52w_gamma=pos_gamma,
    beta_knots=knots, beta_scores=scores,
    de_threshold=de_thr, high_de_penalty=de_pen,
    beta_threshold=beta_thr, high_beta_penalty=beta_pen,
    cap_max_after_cut=cap_cut, cap_max_after_cov=cap_cov
)

st.sidebar.subheader("⚖️ Gewichte")
tmp_w={}
labels={"sc_yield":"Yield","sc_52w":"52W","sc_pe":"PE(inv)","sc_ev_ebitda":"EV/EBITDA(inv)","sc_de":"D/E(inv)","sc_fcfm":"FCF-Marge","sc_ebitdam":"EBITDA-Marge","sc_beta":"Beta","sc_ygap":"Yield-Gap"}
for k in FACTORS:
    tmp_w[k]=st.sidebar.slider(labels.get(k,k), 0.0, 1.0, float(DEFAULT_W.get(k,0.0)), 0.01)
sum_w=sum(tmp_w.values())
weights={k:(v/sum_w) for k,v in tmp_w.items()} if sum_w>0 else DEFAULT_W
st.sidebar.caption(f"Gewichtssumme (normalisiert): **{sum(weights.values()):.2f}**")

st.sidebar.subheader("🧩 Aggregation/Fehlwerte")
fixed_den=st.sidebar.checkbox("Feste Gewichtssumme (kein Re-Weighting)", value=True)
missing_policy=st.sidebar.selectbox("Missing-Policy", ["neutral50","sector_median","skip"], index=0)

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
    rows=[]; pbar=st.progress(0.0, text="Kennzahlen: 0/0")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(fetch_metrics, tk): tk for tk in tickers}
        done=0
        for fut in as_completed(futs):
            rows.append(fut.result()); done+=1
            pbar.progress(done/len(futs), text=f"Kennzahlen: {done}/{len(futs)}")

    df=pd.DataFrame(rows)
    base=df[df["error"].isna()].copy()
    if base.empty:
        st.warning("Keine verwertbaren Daten."); st.dataframe(df, use_container_width=True); st.stop()

    scored=build_scores(base, weights=weights, params=params, fixed_denominator=fixed_den, missing_policy=missing_policy)
    out=scored.copy()
    out["div_yield_%"]=out["div_yield_ttm"]*100
    out["near_52w_low_%"]=(1-out["pos_52w"]).clip(0,1)*100

    cols=["ticker","sector","price","low_52w","high_52w","pos_52w","near_52w_low_%",
          "pe_ttm","ev_ebitda_ttm","de_ratio","fcf_margin_ttm","ebitda_margin_ttm",
          "beta_2y_w","market_cap","adv_3m","score","rating","error"]
    st.subheader("Ergebnisse (Ranking)")
    st.dataframe(out.sort_values("score", ascending=False).reset_index(drop=True).round(3)[cols], use_container_width=True)

    # Executive Summary
    st.markdown("### Executive Summary")
    vc=out["rating"].value_counts()
    kpi_total=len(out); kpi_buy=int(vc.get("BUY",0)); kpi_acc=int(vc.get("ACCUMULATE/WATCH",0)); kpi_hold=int(vc.get("AVOID/HOLD",0))
    kpi_score_avg=float(out["score"].mean()); kpi_yield_med=float(out["div_yield_%"].median())
    kpi_nearlow_med=float(out["near_52w_low_%"].median()); kpi_de_med=float(out["de_ratio"].median(skipna=True)) if np.isfinite(out["de_ratio"].median(skipna=True)) else np.nan
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Total",kpi_total); c2.metric("BUY",kpi_buy); c3.metric("ACC/W",kpi_acc)
    c4.metric("Score ⌀",f"{kpi_score_avg:.1f}"); c5.metric("Yield (Med.)",f"{kpi_yield_med:.2f}%")
    c6,c7=st.columns(2)
    c6.metric("Nähe 52W-Low (Med.)", f"{kpi_nearlow_med:.1f}%")
    c7.metric("D/E (Med.)", f"{kpi_de_med:.2f}" if np.isfinite(kpi_de_med) else "n/a")

    # Top-10 Faktor-Beiträge
    st.markdown("### Top 10 – Faktor-Beitrag")
    label_short={"sc_yield":"Yield","sc_52w":"52W","sc_pe":"PE(inv)","sc_ev_ebitda":"EV/EBITDA(inv)","sc_de":"D/E(inv)","sc_fcfm":"FCF%","sc_ebitdam":"EBITDA%","sc_beta":"Beta","sc_ygap":"YieldGap"}
    top=out.sort_values("score",ascending=False).head(10).copy()
    S=top[FACTORS].astype(float); Wser=pd.Series(weights).reindex(FACTORS).fillna(0.0)
    contrib=S.mul(Wser,axis=1); contrib.columns=[label_short.get(c,c) for c in contrib.columns]
    contrib.insert(0,"ticker",top["ticker"].values); contrib.insert(1,"score",top["score"].values)
    st.dataframe(contrib.set_index("ticker").round(2), use_container_width=True)

    # Faktor-Heatmap
    st.markdown("### Universum – Faktor-Heatmap (Median 0–100)")
    factor_meds=out[FACTORS].median().rename(lambda c: label_short.get(c,c))
    st.dataframe(pd.DataFrame(factor_meds,columns=["Median"]).T.style.background_gradient(axis=1), use_container_width=True)

    # Kurz-Kommentar (NaN-sicher)
    bias = "Contrarian-Bias (nah am 52W-Low belohnt)" if params["invert_52w"] else "Momentum-Bias (Nähe 52W-High belohnt)"
    score_str=f"{kpi_score_avg:.1f}" if np.isfinite(kpi_score_avg) else "n/a"
    yield_str=f"{kpi_yield_med:.2f}%" if np.isfinite(kpi_yield_med) else "n/a"
    nearlow_str=f"{kpi_nearlow_med:.1f}%" if np.isfinite(kpi_nearlow_med) else "n/a"
    de_str=f"{kpi_de_med:.2f}" if np.isfinite(kpi_de_med) else "n/a"
    st.markdown("### Zusammenfassung")
    st.write(f"Coverage: {kpi_total} Titel; BUY {kpi_buy}, ACC/W {kpi_acc}, AVOID/HOLD {kpi_hold}. "
             f"Score Ø {score_str}; Yield (Med.) {yield_str}. {bias}. "
             f"Median Nähe 52W-Low {nearlow_str}; D/E (Med.) {de_str}.")

    # Exporte
    ts=pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    c_us,c_eu,c_xlsx=st.columns(3)
    with c_us:
        st.download_button("⬇️ CSV (US)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"scores_{ts}.csv", mime="text/csv", use_container_width=True)
    with c_eu:
        st.download_button("⬇️ CSV (EU)", out.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"scores_{ts}_eu.csv", mime="text/csv", use_container_width=True)
    with c_xlsx:
        buf=BytesIO(); out.to_excel(buf, index=False, sheet_name="Scores"); buf.seek(0)
        st.download_button("⬇️ Excel", buf.getvalue(),
                           file_name=f"scores_{ts}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    # Fehlerliste
    err=df[df["error"].notna()][["ticker","error"]]
    if not err.empty:
        st.warning("Hinweise/Fehler beim Laden einiger Ticker:")
        st.dataframe(err, use_container_width=True)

else:
    st.caption("Tipp: EU/UK-Suffixe (.DE, .L, .TO, .PA, .AX …).")
