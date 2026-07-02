#!/usr/bin/env python3
"""
wrds_download_spx_gamma.py — REAL data for E4: SPX dealer-gamma panel from OptionMetrics
=======================================================================================
Builds a daily panel for the S&P 500 index (OptionMetrics secid 108105):
  • gross_gamma_oi  = Σ gamma·OI·contract_size  over all SPX options that day (gamma concentration)
  • net_gamma_oi    = Σ sign·gamma·OI·contract_size,  sign=+1 call / −1 put
  • net/gross GEX in $ per 1% move = (·)·spot²·0.01
  • spot, daily index return, realized_vol (rolling 5d)

⚠️ Sign convention: net_gamma_oi uses the common SqueezeMetrics assumption that dealers
are long calls / short puts. That is a PROXY — the true dealer position needs signed
volume (à la Ni-Pearson-Poteshman-White 2021). Documented so it isn't mistaken for ground truth.

Aggregation is done server-side (GROUP BY date) so each year returns ~252 rows, not millions.

Run:  python3 wrds_download_spx_gamma.py [start_year] [end_year]   (default 2016 2024)
Out:  data/real/spx_gamma_panel.csv
"""
import sys
import pathlib
import pandas as pd
import wrds_lib as w

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "real"; OUT.mkdir(parents=True, exist_ok=True)
SECID = 108105   # SPX

def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    frames = []
    for y in range(y0, y1 + 1):
        print(f"  {y}: querying optionm.opprcd{y} / secprd{y} ...", flush=True)
        opt = w.query(f"""
            SELECT date,
                   SUM(gamma*open_interest*contract_size)                         AS gross_gamma_oi,
                   SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)
                        *gamma*open_interest*contract_size)                       AS net_gamma_oi,
                   SUM(open_interest)                                             AS total_oi
            FROM optionm.opprcd{y}
            WHERE secid={SECID} AND gamma IS NOT NULL AND open_interest > 0
            GROUP BY date ORDER BY date;""")
        sec = w.query(f"""SELECT date, close AS spot, return AS ret
                          FROM optionm.secprd{y} WHERE secid={SECID} ORDER BY date;""")
        if opt.empty or sec.empty:
            print(f"    (no rows for {y}, skipping)"); continue
        frames.append(opt.merge(sec, on="date", how="inner"))
    if not frames:
        raise SystemExit("No data returned.")
    p = pd.concat(frames).sort_values("date").reset_index(drop=True)
    for col in ("gross_gamma_oi", "net_gamma_oi", "total_oi", "spot", "ret"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["net_gex_dollar_pct"] = p["net_gamma_oi"] * p["spot"] ** 2 * 0.01
    p["gross_gex_dollar_pct"] = p["gross_gamma_oi"] * p["spot"] ** 2 * 0.01
    p["realized_vol"] = p["ret"].rolling(5).std()
    out = OUT / "spx_gamma_panel.csv"
    p.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(p)} trading days, {p['date'].min()}..{p['date'].max()})")
    print(p[["date", "spot", "ret", "gross_gamma_oi", "net_gamma_oi", "realized_vol"]].tail(4).to_string(index=False))
    # quick sign read: how often are dealers (proxy) net SHORT gamma?
    sh = (p["net_gamma_oi"] < 0).mean()
    print(f"\nDealer (proxy) net SHORT-gamma share of days: {sh:.1%}  "
          f"(Dim et al.: dealers on avg LONG → expect this to be a minority)")

if __name__ == "__main__":
    main()
