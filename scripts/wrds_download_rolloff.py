#!/usr/bin/env python3
"""
wrds_download_rolloff.py — generalized scheduled-gamma-roll-off panel (any ticker)
================================================================================
Ticker-parametrized version of wrds_download_spx_rolloff.py: builds the roll-off instrument
(dollar-gamma scheduled to expire by the next monthly OPEX) for single names, so the
roll-off IV (Paper A) can be run per name (NVDA, TSLA, AAPL, ...).

Run:  python3 wrds_download_rolloff.py NVDA TSLA AAPL
Out:  data/real/{TICKER}_rolloff.csv  (same schema as spx_rolloff.csv)
"""
import sys
import pathlib
import pandas as pd
import wrds_lib as w

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"; OUT.mkdir(parents=True, exist_ok=True)

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(days=14)

def next_opex(dt):
    cands = [third_friday(dt.year, dt.month), third_friday(dt.year + (dt.month == 12), (dt.month % 12) + 1)]
    return min(c for c in cands if c > dt)

def resolve(ticker):
    df = w.query("SELECT secid, COUNT(*) n FROM optionm.secnmd WHERE ticker=%s GROUP BY secid ORDER BY n DESC LIMIT 1;", (ticker,))
    if df.empty: raise SystemExit(f"no secid for {ticker}")
    return int(df["secid"].iloc[0])

def build(ticker, y0=2016, y1=2024):
    secid = resolve(ticker)
    frames = [w.query(f"""
        SELECT date, exdate,
               SUM(gamma*open_interest*contract_size) AS gg,
               SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS ng
        FROM optionm.opprcd{y} WHERE secid={secid} AND gamma IS NOT NULL AND open_interest>0
        GROUP BY date, exdate;""") for y in range(y0, y1 + 1)]
    de = pd.concat(frames, ignore_index=True)
    de["date"] = pd.to_datetime(de["date"]); de["exdate"] = pd.to_datetime(de["exdate"])
    de["gg"] = pd.to_numeric(de["gg"], errors="coerce"); de["ng"] = pd.to_numeric(de["ng"], errors="coerce")
    dd = pd.DataFrame({"date": de["date"].unique()}); dd["nopex"] = dd["date"].map(next_opex)
    de = de.merge(dd, on="date"); de["in_win"] = (de["exdate"] > de["date"]) & (de["exdate"] <= de["nopex"])
    g = de.groupby("date").apply(lambda x: pd.Series({
        "gross_all": x["gg"].sum(), "net_all": x["ng"].sum(),
        "scheduled_rolloff_gross": x.loc[x["in_win"], "gg"].sum(),
        "scheduled_rolloff_net": x.loc[x["in_win"], "ng"].sum()})).reset_index()
    sec = pd.concat([w.query(f"SELECT date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid={secid} ORDER BY date;")
                     for y in range(y0, y1 + 1)])
    sec["date"] = pd.to_datetime(sec["date"])
    p = g.merge(sec, on="date").sort_values("date").reset_index(drop=True)
    for c in ("spot", "ret"): p[c] = pd.to_numeric(p[c], errors="coerce")
    p["realized_vol"] = p["ret"].rolling(5).std()
    out = OUT / f"{ticker}_rolloff.csv"; p.to_csv(out, index=False)
    print(f"  {ticker} (secid {secid}): {len(p)} days -> {out.name}")

def main():
    tickers = sys.argv[1:] or ["NVDA", "TSLA", "AAPL"]
    print("Building roll-off panels for:", ", ".join(tickers))
    for tk in tickers:
        build(tk)

if __name__ == "__main__":
    main()
