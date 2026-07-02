#!/usr/bin/env python3
"""
wrds_download_panel.py — liquid-options PANEL (dealer gamma + roll-off, top ~50 names)
=========================================================================================
Builds a firm×day panel over the most liquid optionable names for the Paper-A panel IV.
All aggregation is server-side; only aggregated rows are transferred.

Universe: top N optionable secids by 2023 option volume (excl. VIX). For each: daily net
dealer gamma, the scheduled gamma roll-off (net gamma expiring by next monthly OPEX),
return, realized vol. Output a long panel.

Run:  python3 wrds_download_panel.py [N=50]  →  data/real/panel_gamma.csv
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

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rank = w.query("""SELECT secid, SUM(volume) vol FROM optionm.opprcd2023
                      WHERE volume IS NOT NULL GROUP BY secid ORDER BY vol DESC LIMIT 80;""")
    rank = rank[rank["secid"] != 117801].head(N)              # drop VIX
    ids = ",".join(str(int(s)) for s in rank["secid"])
    print(f"Universe: {N} liquid optionable names. Downloading {2016}-{2024} ...")

    tot, near, px = [], [], []
    for y in range(2016, 2025):
        print(f"  {y} ...", flush=True)
        tot.append(w.query(f"""SELECT secid, date,
              SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS net_all
            FROM optionm.opprcd{y} WHERE secid IN ({ids}) AND gamma IS NOT NULL AND open_interest>0
            GROUP BY secid, date;"""))
        near.append(w.query(f"""SELECT secid, date, exdate,
              SUM((CASE WHEN cp_flag='C' THEN 1 ELSE -1 END)*gamma*open_interest*contract_size) AS ng
            FROM optionm.opprcd{y} WHERE secid IN ({ids}) AND gamma IS NOT NULL AND open_interest>0
              AND (exdate-date)<=45 GROUP BY secid, date, exdate;"""))
        px.append(w.query(f"SELECT secid, date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid IN ({ids});"))
    tot = pd.concat(tot); near = pd.concat(near); px = pd.concat(px)
    for d in (tot, near, px):
        d["date"] = pd.to_datetime(d["date"])
    near["exdate"] = pd.to_datetime(near["exdate"])
    for col, d in [("net_all", tot), ("ng", near), ("spot", px), ("ret", px)]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # scheduled roll-off (net) by next OPEX, per (secid,date)
    dd = pd.DataFrame({"date": near["date"].unique()}); dd["nopex"] = dd["date"].map(next_opex)
    near = near.merge(dd, on="date")
    roll = (near[(near["exdate"] > near["date"]) & (near["exdate"] <= near["nopex"])]
            .groupby(["secid", "date"])["ng"].sum().reset_index().rename(columns={"ng": "scheduled_rolloff_net"}))

    p = tot.merge(px, on=["secid", "date"], how="inner").merge(roll, on=["secid", "date"], how="left").sort_values(["secid", "date"])
    p["scheduled_rolloff_net"] = p["scheduled_rolloff_net"].fillna(0.0)
    p["net_gex"] = p["net_all"] * p["spot"]**2 * 0.01
    p["realized_vol"] = p.groupby("secid")["ret"].transform(lambda s: s.rolling(5).std())
    p = p.merge(rank.assign(liq_rank=range(1, len(rank)+1))[["secid", "liq_rank"]], on="secid", how="left")
    out = OUT / "panel_gamma.csv"
    p.to_csv(out, index=False)
    print(f"\nWrote {out}: {p['secid'].nunique()} names × {p['date'].nunique()} days = {len(p):,} firm-days")

if __name__ == "__main__":
    main()
