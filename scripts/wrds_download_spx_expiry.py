#!/usr/bin/env python3
"""
wrds_download_spx_expiry.py — SPX gamma split by time-to-expiry (for the near-expiry IV)
==============================================================================
Richer version of the SPX panel: aggregates gamma·OI by days-to-expiry bucket to build a cleaner instrument
for the causal test. Near-expiry gamma (≤7d) is the
component that mechanically rolls off on the expiration calendar.

Buckets:  gamma_le7 (≤7d), gamma_8_30 (8–30d), gamma_gt30 (>30d); plus gross_all/net_all.
Run:  python3 wrds_download_spx_expiry.py   →  data/real/spx_gamma_expiry.csv
"""
import pathlib
import pandas as pd
import wrds_lib as w

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"; OUT.mkdir(parents=True, exist_ok=True)
SECID = 108105

def main():
    frames = []
    for y in range(2016, 2025):
        print(f"  {y} ...", flush=True)
        opt = w.query(f"""
            SELECT date,
              SUM(gamma*open_interest*contract_size) AS gross_all,
              SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS net_all,
              SUM(CASE WHEN (exdate-date)<=7  THEN gamma*open_interest*contract_size ELSE 0 END) AS gamma_le7,
              SUM(CASE WHEN (exdate-date) BETWEEN 8 AND 30 THEN gamma*open_interest*contract_size ELSE 0 END) AS gamma_8_30,
              SUM(CASE WHEN (exdate-date)>30 THEN gamma*open_interest*contract_size ELSE 0 END) AS gamma_gt30
            FROM optionm.opprcd{y}
            WHERE secid={SECID} AND gamma IS NOT NULL AND open_interest > 0
            GROUP BY date ORDER BY date;""")
        sec = w.query(f"SELECT date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid={SECID} ORDER BY date;")
        if not opt.empty and not sec.empty:
            frames.append(opt.merge(sec, on="date", how="inner"))
    p = pd.concat(frames).sort_values("date").reset_index(drop=True)
    for c in ("gross_all", "net_all", "gamma_le7", "gamma_8_30", "gamma_gt30", "spot", "ret"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["realized_vol"] = p["ret"].rolling(5).std()
    p["near_share"] = p["gamma_le7"] / p["gross_all"]
    out = OUT / "spx_gamma_expiry.csv"
    p.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(p)} days). near-expiry gamma share: "
          f"mean {p['near_share'].mean():.2f}, 2016 {p['near_share'][:50].mean():.2f} -> 2024 {p['near_share'][-50:].mean():.2f}")

if __name__ == "__main__":
    main()
