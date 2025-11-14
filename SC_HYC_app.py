# app.py  — Minimal Pro Scoring (contrarian ready)
from __future__ import annotations
import numpy as np, pandas as pd, streamlit as st, yfinance as yf
from io import BytesIO

st.set_page_config(page_title="Master Scoring (Pro-Minimal)", layout="wide")
st.title("📈 Master Scoring (Pro-Minimal)")

# ---------- helpers ----------
def _to_float(x):
    try: f=float(x); return f if np.isfinite(f) else np.nan
    except: return np.nan

def _hist_close(tk, period="5y", interval="1d"):
    try:
        h = yf.Ticker(tk).history(period=period, interval=interval, auto_adjust=True)
        return h["Close"].dropna() if "Close" in h else pd.Series(dtype=float)
    except: return pd.Series(dtype=float)

def _row(df, keys):
    for k in keys:
        if isinstance(df, pd.DataFrame) and k in df.index:
            s = df.loc[k].dropna().astype(float)
            if len(s): return s
    return pd.Series(dtype=float)

def _ttm_sum(q_df, keys, n=4):
    s = _row(q_df, keys); return float(s.iloc[:n].sum()) if len(s) else np.nan

def _sector_pct(df, col, invert=False):
    x = df.copy(); x["sector"] = x.get("sector", "Unknown").fillna("Unknown")
    def pct(s): 
        m=s.notna()
        return s.rank(pct=True) if m.sum()>1 else pd.Series(0.5, index=s.index)
    p = x.groupby("sector", group_keys=False)[col].apply(pct)
    p = 1-p if invert else p
    return (p*100).clip(0,100)

# ---------- metrics (robust) ----------
@st.cache_data(ttl=1800)
def fetch_metrics(tk: str) -> dict:
    t = yf.Ticker(tk)
    try:
        info = t.get_info() if hasattr(t, "get_info") else getattr(t,"info",{}) or {}
    except: info = getattr(t,"info",{}) or {}
    px = _hist_close(tk, "5y", "1d")
    if px.empty:
        try: px = yf.download(tk, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
        except: pass
    if px.empty: return {"ticker": tk, "error":"no_price_history"}

    price = float(px.iloc[-1])

    # 52W — immer aus adjustierter History
    px1y = _hist_close(tk, "1y", "1d")
    if px1y.empty: px1y = px.tail(252)
    low_52w, high_52w = float(px1y.min()), float(px1y.max())
    rng = high_52w - low_52w
    pos_52w = (price - low_52w) / (rng if (rng>0) else np.nan)

    # Dividenden (TTM + 5y Median der Monatsrendite)
    try: div = t.get_dividends()
    except: div = getattr(t,"dividends", pd.Series(dtype=float))
    pm = px.resample("M").last()
    dm = (div*1.0).resample("M").sum().reindex(pm.index, fill_value=0.0) if isinstance(div,pd.Series) else pd.Series(0.0, index=pm.index)
    ttm_div_m = dm.rolling(12, min_periods=1).sum()
    yld_ttm = float(ttm_div_m.iloc[-1] / price) if price>0 else np.nan
    yld_med5 = float((ttm_div_m/pm).tail(min(60,len(pm))).median()) if len(pm) else np.nan

    # Finanzsheets
    q_is = t.quarterly_financials if hasattr(t,"quarterly_financials") else pd.DataFrame()
    q_bs = t.quarterly_balance_sheet if hasattr(t,"quarterly_balance_sheet") else pd.DataFrame()
    q_cf = t.quarterly_cashflow if hasattr(t,"quarterly_cashflow") else pd.DataFrame()
    revenue=_ttm_sum(q_is,["Total Revenue","Revenue"])
    ebitda =_ttm_sum(q_is,["EBITDA","Ebitda"])
    op_cf  =_ttm_sum(q_cf,["Total Cash From Operating Activities","Operating Cash Flow"])
    capex_y=_ttm_sum(q_cf,["Capital Expenditures","Capital Expenditure"])
    capex = -capex_y if np.isfinite(capex_y) else np.nan
    fcf   = op_cf - capex if np.isfinite(op_cf) and np.isfinite(capex) else np.nan
    ebitda_margin = (ebitda/revenue) if (np.isfinite(ebitda) and np.isfinite(revenue) and revenue>0) else np.nan
    fcf_margin    = (fcf/revenue) if (np.isfinite(fcf) and np.isfinite(revenue) and revenue>0) else np.nan
    equity=_ttm_sum(q_bs,["Total Stockholder Equity","Total Equity Gross Minority Interest"])
    total_debt = _ttm_sum(q_bs,["Long Term Debt"]) + _ttm_sum(q_bs,["Short Long Term Debt","Short Term Debt"])
    de_ratio = (total_debt/equity) if (np.isfinite(total_debt) and np.isfinite(equity) and equity>0) else np.nan

    # Beta (2y weekly)
    try:
        spx = yf.Ticker("^GSPC").history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
        stw = t.history(period="2y", interval="1wk", auto_adjust=True)["Close"].pct_change().dropna()
        bdf = pd.concat([stw, spx], axis=1).dropna()
        beta = float(np.polyfit(bdf.iloc[:,1].values, bdf.iloc[:,0].values, 1)[0]) if len(bdf)>10 else np.nan
    except: beta = np.nan

    sector = (info.get("sector") or "Unknown")
    mcap   = _to_float(info.get("marketCap"))
    pe     = _to_float(info.get("trailingPE"))
    # EV/EBITDA
    cash=_to_float(info.get("totalCash")); debt=_to_float(info.get("totalDebt"))
    ev = (mcap if np.isfinite(mcap) else np.nan)
    if np.isfinite(ev): ev += (debt if np.isfinite(debt) else 0) - (cash if np.isfinite(cash) else 0)
    ev_ebitda = (ev/ebitda) if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda>0) else np.nan

    # Div-Qualität
    div_paid = _ttm_sum(q_cf,["Dividends Paid"])
    div_cash_ttm = abs(div_paid) if np.isfinite(div_paid) else float(ttm_div_m.iloc[-1] if len(ttm_div_m) else 0.0)
    coverage_fcf = (fcf/div_cash_ttm) if (np.isfinite(fcf) and div_cash_ttm>0) else np.inf
    fcf_payout   = (div_cash_ttm/fcf) if (np.isfinite(fcf) and fcf>0) else np.inf
    prev = float(ttm_div_m.shift(12).iloc[-1]) if len(ttm_div_m)>12 else np.nan
    div_cut = int(np.isfinite(prev) and prev>0 and (float(ttm_div_m.iloc[-1]) < 0.9*prev))

    return dict(ticker=tk, sector=sector, price=price,
                low_52w=low_52w, high_52w=high_52w, pos_52w=pos_52w,
                div_yield_ttm=yld_ttm, yield_5y_median=yld_med5,
                pe_ttm=pe, ev_ebitda_ttm=ev_ebitda, de_ratio=de_ratio,
                fcf_margin_ttm=fcf_margin, ebitda_margin_ttm=ebitda_margin,
                beta_2y_w=beta, market_cap=mcap, adv_3m=_to_float(info.get("averageDailyVolume3Month")),
                div_cash_ttm=div_cash_ttm, coverage_fcf_ttm=coverage_fcf, fcf_payout_ttm=fcf_payout,
                div_cut_24m=div_cut, error=np.nan)

# ---------- scoring ----------
FACTORS = ["sc_yield","sc_52w","sc_pe","sc_ev_ebitda","sc_de","sc_fcfm","sc_ebitdam","sc_beta","sc_ygap"]
DEFAULT_W = {"sc_yield":0.25,"sc_52w":0.22,"sc_pe":0.12,"sc_ev_ebitda":0.12,"sc_de":0.12,"sc_fcfm":0.07,"sc_ebitdam":0.04,"sc_beta":0.04,"sc_ygap":0.02}

def build_scores(df, *, invert_52w=True, gamma=1.5, yield_floor=0.04, yield_scale=0.04,
                 beta_knots=(0.30,0.80,1.00,1.60), beta_scores=(100,75,55,10),
                 cap_cut=55, cap_cov=45, de_thr=2.0, de_pen=20, beta_thr=1.6, beta_pen=12,
                 weights=None, missing_policy="neutral50", fixed_den=True):
    d=df.copy()
    # factors
    d["sc_yield"] = (np.clip((d["div_yield_ttm"]-yield_floor)/yield_scale,0,1)*100)
    base = (1-d["pos_52w"]) if invert_52w else d["pos_52w"]
    d["sc_52w"] = (np.clip(base,0,1)**gamma)*100
    d["sc_pe"]        = _sector_pct(d,"pe_ttm",invert=True)
    d["sc_ev_ebitda"] = _sector_pct(d,"ev_ebitda_ttm",invert=True)
    d["sc_de"]        = _sector_pct(d,"de_ratio",invert=True)
    d.loc[(~np.isfinite(d["de_ratio"]))|(d["de_ratio"]<0),"sc_de"]=0
    d["sc_fcfm"]      = _sector_pct(d,"fcf_margin_ttm",invert=False)
    d["sc_ebitdam"]   = _sector_pct(d,"ebitda_margin_ttm",invert=False)
    # beta mapping
    def map_beta(b):
        if not np.isfinite(b): return 50.0
        x=np.interp(b,beta_knots,beta_scores,left=beta_scores[0],right=beta_scores[-1])
        return float(x)
    d["sc_beta"]=d["beta_2y_w"].apply(map_beta)
    # yield gap
    ygap=np.where(d["yield_5y_median"]>0,d["div_yield_ttm"]/d["yield_5y_median"]-1.0,np.nan)
    d["sc_ygap"]=_sector_pct(pd.DataFrame({"sector":d["sector"],"ygap":ygap}),"ygap",invert=False)
    # aggregate
    W=pd.Series((weights or DEFAULT_W)).reindex(FACTORS).fillna(0.0)
    S=d[FACTORS].astype(float)
    if missing_policy=="neutral50": S=S.fillna(50.0)
    elif missing_policy=="sector_median":
        for c in S.columns:
            med=d.groupby("sector")[c].transform("median"); S[c]=S[c].fillna(med).fillna(50.0)
    num=(S*W).sum(axis=1,skipna=True)
    den=float(W.sum()) if fixed_den else ((~S.isna())*W).sum(axis=1)
    d["score_raw"]=np.where(den>0,num/den,np.nan)
    # caps/penalties
    cap=d["score_raw"].copy()
    cap=np.where(d["div_cut_24m"]==1,np.minimum(cap,cap_cut),cap)
    cap=np.where((d["fcf_payout_ttm"]>1.0)|(d["coverage_fcf_ttm"]<1.0),np.minimum(cap,cap_cov),cap)
    cap=np.where((d["de_ratio"]>de_thr),cap-de_pen,cap)
    cap=np.where((d["beta_2y_w"]>beta_thr),cap-beta_pen,cap)
    d["score"]=pd.Series(cap,index=d.index).clip(0,100)
    d["rating"]=np.select([d["score"]>=75,(d["score"]>=60)&(d["score"]<75)],["BUY","ACCUMULATE/WATCH"],"AVOID/HOLD")
    return d

# ---------- UI ----------
st.sidebar.header("Eingabe")
csv = st.sidebar.file_uploader("CSV mit Ticker-Spalte", type=["csv"])
tickers=[]
if csv is not None:
    try: dfc=pd.read_csv(csv)
    except: csv.seek(0); dfc=pd.read_csv(csv,sep=";")
    col=st.sidebar.selectbox("Ticker-Spalte", dfc.columns)
    tickers = dfc[col].astype(str).str.strip().replace({"nan":np.nan}).dropna().tolist()
manual = st.sidebar.text_area("Ticker manuell (kommagetrennt)", "DHL.DE, DBK.DE")
tickers = sorted({*tickers, *[s.strip().upper() for s in manual.split(",") if s.strip()]})
st.sidebar.caption(f"Watchlist: {len(tickers)}")

st.sidebar.header("Contrarian/52W")
invert_52w = st.sidebar.checkbox("Invertieren (nah am Low = besser)", True)
gamma = st.sidebar.slider("52W-Gamma", 0.5, 2.5, 1.5, 0.1)

st.sidebar.header("Gewichte")
W={}
for k,lab in [("sc_52w","52W"),("sc_yield","Yield"),("sc_pe","PE inv"),("sc_ev_ebitda","EV/EBITDA inv"),
              ("sc_de","D/E inv"),("sc_fcfm","FCF-Marge"),("sc_ebitdam","EBITDA-Marge"),("sc_beta","Beta"),("sc_ygap","Yield-Gap")]:
    W[k]=st.sidebar.slider(lab,0.0,1.0,float(DEFAULT_W.get(k,0.0)),0.01)
if sum(W.values())==0: st.stop()
# normalisieren
s=sum(W.values()); W={k:v/s for k,v in W.items()}

st.sidebar.header("Fehlwerte/Denominator")
fixed_den = st.sidebar.checkbox("Feste Gewichtssumme", True)
missing_policy = st.sidebar.selectbox("Missing-Policy", ["neutral50","sector_median","skip"], index=0)

run = st.sidebar.button("🔎 Score berechnen", use_container_width=True)

st.subheader("Watchlist")
if tickers:
    with st.expander("anzeigen"):
        st.code(", ".join(tickers), wrap_lines=True)
else:
    st.info("Ticker eingeben oder CSV laden.")

if run and tickers:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows=[]; p=st.progress(0.0, text="Lade Kennzahlen …")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(fetch_metrics, tk): tk for tk in tickers}
        done=0
        for fut in as_completed(futs):
            rows.append(fut.result()); done+=1; p.progress(done/len(futs), text=f"Kennzahlen: {done}/{len(futs)}")
    df=pd.DataFrame(rows)
    base=df[df["error"].isna()].copy()
    scored=build_scores(base, invert_52w=invert_52w, gamma=gamma, weights=W,
                        missing_policy=missing_policy, fixed_den=fixed_den)
    out=scored.copy()
    out["div_yield_%"]=out["div_yield_ttm"]*100
    out["near_52w_low_%"]=(1-out["pos_52w"])*100
    cols=["ticker","sector","price","low_52w","high_52w","pos_52w","near_52w_low_%",
          "pe_ttm","ev_ebitda_ttm","de_ratio","fcf_margin_ttm","ebitda_margin_ttm",
          "beta_2y_w","market_cap","score","rating","error"]
    st.subheader("Ergebnisse")
    st.dataframe(out[cols].sort_values("score", ascending=False).round(3), use_container_width=True)
    # exports
    ts=pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    c1,c2=st.columns(2)
    with c1:
        st.download_button("⬇️ CSV", out.to_csv(index=False).encode("utf-8"), f"scores_{ts}.csv", "text/csv", use_container_width=True)
    with c2:
        buf=BytesIO(); out.to_excel(buf, index=False); buf.seek(0)
        st.download_button("⬇️ Excel", buf.getvalue(), f"scores_{ts}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
