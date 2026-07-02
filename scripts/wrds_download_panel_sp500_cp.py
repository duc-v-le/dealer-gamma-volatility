#!/usr/bin/env python3
"""
wrds_download_panel_sp500_cp.py — S&P 500 panel with CALL/PUT-split gamma
=============================================================================
Re-download the S&P 500 gamma panel but keep CALL and PUT gamma separate (level and
scheduled roll-off), so the dealer-gamma SIGN CONVENTION can be stress-tested:
  • baseline (call-long/put-short):  net = call_g − put_g
  • call-only, put-only, gross (= short-all-customer-buying)
This is the feasible substitute for signed customer demand (CBOE/ISE Open-Close), which
is not accessible on this WRDS account.

Parallel-by-year, parquet, resumable.  Run:  python3 wrds_download_panel_sp500_cp.py [workers=6]
Out:  data/real/panel_sp500_cp.parquet
"""
import sys, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import wrds_lib as w

REAL = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"
PARTS = REAL / "_sp500cp_parts"; PARTS.mkdir(parents=True, exist_ok=True)

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(days=14)
def next_opex(dt):
    c = [third_friday(dt.year, dt.month), third_friday(dt.year + (dt.month == 12), (dt.month % 12) + 1)]
    return min(x for x in c if x > dt)

def dl(y, ids):
    part = PARTS / f"{y}.parquet"
    if part.exists(): return f"{y}: cached"
    tot = w.query(f"""SELECT secid, date,
        SUM(CASE WHEN cp_flag='C' THEN gamma*open_interest*contract_size ELSE 0 END) AS call_g,
        SUM(CASE WHEN cp_flag='P' THEN gamma*open_interest*contract_size ELSE 0 END) AS put_g
      FROM optionm.opprcd{y} WHERE secid IN ({ids}) AND gamma IS NOT NULL AND open_interest>0
      GROUP BY secid, date;""")
    near = w.query(f"""SELECT secid, date, exdate,
        SUM(CASE WHEN cp_flag='C' THEN gamma*open_interest*contract_size ELSE 0 END) AS call_ng,
        SUM(CASE WHEN cp_flag='P' THEN gamma*open_interest*contract_size ELSE 0 END) AS put_ng
      FROM optionm.opprcd{y} WHERE secid IN ({ids}) AND gamma IS NOT NULL AND open_interest>0
        AND (exdate-date)<=45 GROUP BY secid, date, exdate;""")
    px = w.query(f"SELECT secid, date, close AS spot, return AS ret FROM optionm.secprd{y} WHERE secid IN ({ids});")
    for d in (tot, near, px): d["date"] = pd.to_datetime(d["date"])
    near["exdate"] = pd.to_datetime(near["exdate"])
    for c in ("call_ng", "put_ng"): near[c] = pd.to_numeric(near[c], errors="coerce")
    dd = pd.DataFrame({"date": near["date"].unique()}); dd["nopex"] = dd["date"].map(next_opex)
    near = near.merge(dd, on="date"); win = (near["exdate"] > near["date"]) & (near["exdate"] <= near["nopex"])
    roll = near[win].groupby(["secid", "date"]).agg(
        roll_call=("call_ng", "sum"), roll_put=("put_ng", "sum")).reset_index()
    yr = tot.merge(px, on=["secid", "date"]).merge(roll, on=["secid", "date"], how="left")
    yr.to_parquet(part, index=False); return f"{y}: {len(yr):,} rows"

def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ids = ",".join(str(int(s)) for s in pd.read_csv(REAL / "sp500_secids.csv")["secid"])
    years = list(range(2016, 2025))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed({ex.submit(dl, y, ids): y for y in years}):
            print("  " + f.result(), flush=True)
    p = pd.concat([pd.read_parquet(PARTS / f"{y}.parquet") for y in years]).sort_values(["secid", "date"]).reset_index(drop=True)
    for c in ("roll_call", "roll_put"): p[c] = p[c].fillna(0.0)
    for c in ("call_g", "put_g", "spot", "ret"): p[c] = pd.to_numeric(p[c], errors="coerce")
    p["realized_vol"] = p.groupby("secid")["ret"].transform(lambda s: s.rolling(5).std())
    out = REAL / "panel_sp500_cp.parquet"; p.to_parquet(out, index=False)
    print(f"\nWrote {out}: {p['secid'].nunique()} names × {p['date'].nunique()} days = {len(p):,} firm-days")

if __name__ == "__main__":
    main()
