#!/usr/bin/env python3
"""
wrds_download_gamma.py — generalized dealer-gamma panel downloader (any ticker)
==============================================================================
Same construction as wrds_download_spx_gamma.py, but resolves the OptionMetrics
secid from a ticker and works for any optionable name. Used to extend E4 from SPX
to single names (esp. the AI-trade leaders: NVDA, TSLA, AAPL, ...).

Run:  python3 wrds_download_gamma.py NVDA TSLA AAPL [--years 2016 2024]
Out:  data/real/{TICKER}_gamma_panel.csv   (same schema as the SPX panel; see DATA_GUIDE.md)

⚠️ Same caveats as the SPX build: call/put dealer sign is a PROXY; standardize +
stationarize before analysis (see data/real/DATA_GUIDE.md).
"""
import sys
import pathlib
import pandas as pd
import wrds_lib as w

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"; OUT.mkdir(parents=True, exist_ok=True)

def resolve_secid(ticker):
    df = w.query("SELECT secid, COUNT(*) n FROM optionm.secnmd WHERE ticker=%s GROUP BY secid ORDER BY n DESC LIMIT 1;",
                 (ticker,))
    if df.empty:
        raise SystemExit(f"no secid for ticker {ticker}")
    return int(df["secid"].iloc[0])

def download(ticker, y0, y1):
    secid = resolve_secid(ticker)
    frames = []
    for y in range(y0, y1 + 1):
        opt = w.query(f"""
            SELECT date,
                   SUM(gamma*open_interest*contract_size) AS gross_gamma_oi,
                   SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS net_gamma_oi,
                   SUM(open_interest) AS total_oi
            FROM optionm.opprcd{y}
            WHERE secid={secid} AND gamma IS NOT NULL AND open_interest > 0
            GROUP BY date ORDER BY date;""")
        sec = w.query(f"SELECT date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid={secid} ORDER BY date;")
        if opt.empty or sec.empty:
            continue
        frames.append(opt.merge(sec, on="date", how="inner"))
    if not frames:
        print(f"  {ticker}: no data"); return
    p = pd.concat(frames).sort_values("date").reset_index(drop=True)
    for col in ("gross_gamma_oi", "net_gamma_oi", "total_oi", "spot", "ret"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["net_gex_dollar_pct"] = p["net_gamma_oi"] * p["spot"] ** 2 * 0.01
    p["gross_gex_dollar_pct"] = p["gross_gamma_oi"] * p["spot"] ** 2 * 0.01
    p["realized_vol"] = p["ret"].rolling(5).std()
    out = OUT / f"{ticker}_gamma_panel.csv"
    p.to_csv(out, index=False)
    sh = (p["net_gamma_oi"] < 0).mean()
    print(f"  {ticker} (secid {secid}): {len(p)} days {p['date'].min()}..{p['date'].max()}, "
          f"net-short-gamma {sh:.0%} of days -> {out.name}")

def main():
    args = sys.argv[1:]
    y0, y1 = 2016, 2024
    if "--years" in args:
        i = args.index("--years"); y0, y1 = int(args[i+1]), int(args[i+2]); args = args[:i]
    if not args:
        args = ["NVDA", "TSLA", "AAPL"]
    print(f"Downloading gamma panels {y0}-{y1} for: {', '.join(args)}")
    for tk in args:
        download(tk, y0, y1)

if __name__ == "__main__":
    main()
