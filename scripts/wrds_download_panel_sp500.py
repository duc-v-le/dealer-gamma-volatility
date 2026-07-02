#!/usr/bin/env python3
"""
wrds_download_panel_sp500.py — FULL S&P 500 gamma+roll-off panel (parallel, resumable)
==========================================================================================
Scales the liquid-names panel to the real S&P 500 (members 2016-2024 linked to OptionMetrics
secids; sp500_secids.csv). Strategy:
  • DOWNLOAD BY YEAR (each year independent, bounds memory).
  • PARALLEL across years via a thread pool (the bottleneck is WRDS server-side scans +
    network = I/O-bound, so threads/concurrent connections help; capped to respect WRDS
    per-user connection limits — NOT 50).
  • RESUMABLE: each year writes data/real/_sp500_parts/{y}.csv; a re-run skips finished
    years and just re-concatenates. So a timeout/interrupt never loses work.

Run:  python3 wrds_download_panel_sp500.py [workers=6]
Out:  data/real/panel_sp500.csv
"""
import sys
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import wrds_lib as w

REAL = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"
PARTS = REAL / "_sp500_parts"; PARTS.mkdir(parents=True, exist_ok=True)

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(days=14)

def next_opex(dt):
    cands = [third_friday(dt.year, dt.month), third_friday(dt.year + (dt.month == 12), (dt.month % 12) + 1)]
    return min(c for c in cands if c > dt)

def download_year(y, ids):
    part = PARTS / f"{y}.parquet"
    if part.exists():
        return f"{y}: cached"
    tot = w.query(f"""SELECT secid, date,
          SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS net_all
        FROM optionm.opprcd{y} WHERE secid IN ({ids}) AND gamma IS NOT NULL AND open_interest>0
        GROUP BY secid, date;""")
    near = w.query(f"""SELECT secid, date, exdate,
          SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS ng
        FROM optionm.opprcd{y} WHERE secid IN ({ids}) AND gamma IS NOT NULL AND open_interest>0
          AND (exdate-date)<=45 GROUP BY secid, date, exdate;""")
    px = w.query(f"SELECT secid, date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid IN ({ids});")
    for d in (tot, near, px):
        d["date"] = pd.to_datetime(d["date"])
    near["exdate"] = pd.to_datetime(near["exdate"]); near["ng"] = pd.to_numeric(near["ng"], errors="coerce")
    dd = pd.DataFrame({"date": near["date"].unique()}); dd["nopex"] = dd["date"].map(next_opex)
    near = near.merge(dd, on="date")
    roll = (near[(near["exdate"] > near["date"]) & (near["exdate"] <= near["nopex"])]
            .groupby(["secid", "date"])["ng"].sum().reset_index().rename(columns={"ng": "scheduled_rolloff_net"}))
    yr = tot.merge(px, on=["secid", "date"]).merge(roll, on=["secid", "date"], how="left")
    yr.to_parquet(part, index=False)
    return f"{y}: {len(yr):,} firm-days -> {part.name}"

def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ids = ",".join(str(int(s)) for s in pd.read_csv(REAL / "sp500_secids.csv")["secid"])
    years = list(range(2016, 2025))
    print(f"S&P 500 panel: {len(years)} years, {workers} parallel workers (resumable).")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_year, y, ids): y for y in years}
        for f in as_completed(futs):
            print("  " + f.result(), flush=True)
    # concatenate parts; compute net_gex + per-secid realized vol on the full series
    p = pd.concat([pd.read_parquet(PARTS / f"{y}.parquet") for y in years])
    p = p.sort_values(["secid", "date"]).reset_index(drop=True)
    p["scheduled_rolloff_net"] = p["scheduled_rolloff_net"].fillna(0.0)
    for c in ("net_all", "spot", "ret"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["net_gex"] = p["net_all"] * p["spot"]**2 * 0.01
    p["realized_vol"] = p.groupby("secid")["ret"].transform(lambda s: s.rolling(5).std())
    out = REAL / "panel_sp500.parquet"; p.to_parquet(out, index=False)
    print(f"\nWrote {out}: {p['secid'].nunique()} names × {p['date'].nunique()} days = {len(p):,} firm-days")

if __name__ == "__main__":
    main()
