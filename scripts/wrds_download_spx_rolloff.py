#!/usr/bin/env python3
"""
wrds_download_spx_rolloff.py — build the "scheduled gamma roll-off" instrument
==================================================================================
The OPEX *calendar* is a weak instrument for the policy-relevant SHORT-gamma.
This builds a finer, mechanically-direct instrument: the dollar-gamma of options
SCHEDULED to expire by the next monthly OPEX — i.e., gamma whose roll-off date is
already fixed by the option calendar, plausibly exogenous to future vol innovations.

Pulls SPX gamma aggregated by (date, exdate), then for each trading day t computes:
  scheduled_rolloff_gross/net = Σ gamma·OI·cs over options with t < exdate ≤ next_OPEX(t)
  rolloff_share               = scheduled_rolloff_gross / gross_all

Run:  python3 wrds_download_spx_rolloff.py  →  data/real/spx_rolloff.csv
"""
import pathlib
import numpy as np
import pandas as pd
import wrds_lib as w

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"; OUT.mkdir(parents=True, exist_ok=True)
SECID = 108105

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(days=14)

def next_opex(dt):
    cands = [third_friday(dt.year, dt.month), third_friday(dt.year + (dt.month == 12), (dt.month % 12) + 1)]
    fut = [c for c in cands if c > dt]
    return min(fut)

def main():
    frames = []
    for y in range(2016, 2025):
        print(f"  {y} ...", flush=True)
        de = w.query(f"""
            SELECT date, exdate,
                   SUM(gamma*open_interest*contract_size) AS gg,
                   SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS ng
            FROM optionm.opprcd{y}
            WHERE secid={SECID} AND gamma IS NOT NULL AND open_interest > 0
            GROUP BY date, exdate;""")
        frames.append(de)
    de = pd.concat(frames, ignore_index=True)
    de["date"] = pd.to_datetime(de["date"]); de["exdate"] = pd.to_datetime(de["exdate"])
    de["gg"] = pd.to_numeric(de["gg"], errors="coerce"); de["ng"] = pd.to_numeric(de["ng"], errors="coerce")

    # next OPEX per unique date, then flag options expiring in (date, next_opex]
    dates = pd.DataFrame({"date": de["date"].unique()})
    dates["nopex"] = dates["date"].map(next_opex)
    de = de.merge(dates, on="date", how="left")
    de["in_win"] = (de["exdate"] > de["date"]) & (de["exdate"] <= de["nopex"])

    g = de.groupby("date").apply(lambda x: pd.Series({
        "gross_all": x["gg"].sum(),
        "net_all": x["ng"].sum(),
        "scheduled_rolloff_gross": x.loc[x["in_win"], "gg"].sum(),
        "scheduled_rolloff_net": x.loc[x["in_win"], "ng"].sum(),
    })).reset_index()
    g["rolloff_share"] = g["scheduled_rolloff_gross"] / g["gross_all"]

    # spot / ret
    sec = pd.concat([w.query(f"SELECT date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid={SECID} ORDER BY date;")
                     for y in range(2016, 2025)])
    sec["date"] = pd.to_datetime(sec["date"])
    p = g.merge(sec, on="date", how="inner").sort_values("date").reset_index(drop=True)
    p["spot"] = pd.to_numeric(p["spot"], errors="coerce"); p["ret"] = pd.to_numeric(p["ret"], errors="coerce")
    p["realized_vol"] = p["ret"].rolling(5).std()
    out = OUT / "spx_rolloff.csv"
    p.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(p)} days). rolloff_share mean {p['rolloff_share'].mean():.2f}")

if __name__ == "__main__":
    main()
