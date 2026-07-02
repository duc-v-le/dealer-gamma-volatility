#!/usr/bin/env python3
"""
e4_earnings_exclusion.py — harden the exclusion restriction against name-level,
forward-looking option demand. An imminent earnings announcement can raise BOTH near-dated option
positioning (the instrument Z) AND subsequent realized volatility, through a channel unrelated to
dealer hedging — the most demanding threat to exclusion. Re-run the baseline panel IV EXCLUDING
earnings windows (±5 trading days ≈ ±7 calendar days around each Compustat report date rdq). If the
causal effect survives off-earnings with similar magnitude, that threat is bounded.

Link chain: panel secid -> permno (opcrsphist) -> gvkey (CCM lnkhist) -> rdq (comp.fundq).
Two-way (firm+date) clustered. Out: data/e4_earnings_exclusion.csv
"""
import pathlib
import numpy as np, pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

REAL = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
WIN = 7  # calendar-day half-window around each earnings date

def rz(s, win=60): return (s - s.rolling(win).mean()) / s.rolling(win).std()

def main():
    ids = pd.read_csv(REAL / "sp500_secids.csv")["secid"].astype(int).tolist()
    idstr = ",".join(map(str, ids))
    s2p = w.query(f"SELECT DISTINCT secid, permno FROM wrdsapps_link_crsp_optionm.opcrsphist "
                  f"WHERE secid IN ({idstr});")
    s2p = s2p.dropna(); s2p["permno"] = s2p["permno"].astype(int)
    permnos = ",".join(str(x) for x in s2p["permno"].unique())
    p2g = w.query(f"SELECT DISTINCT lpermno AS permno, gvkey FROM crsp.ccmxpf_lnkhist "
                  f"WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C') AND lpermno IN ({permnos});")
    gvkeys = ",".join("'" + str(g) + "'" for g in p2g["gvkey"].dropna().unique())
    rdq = w.query(f"SELECT DISTINCT gvkey, rdq FROM comp.fundq WHERE rdq IS NOT NULL "
                  f"AND rdq>='2016-01-01' AND rdq<='2024-12-31' AND gvkey IN ({gvkeys});")
    ann = (s2p.merge(p2g, on="permno").merge(rdq, on="gvkey")[["secid", "rdq"]]
           .dropna().drop_duplicates())
    ann["rdq"] = pd.to_datetime(ann["rdq"])
    print(f"earnings dates linked: {len(ann):,} for {ann['secid'].nunique()} names")

    p = pd.read_parquet(REAL / "panel_sp500_cp.parquet")
    p["date"] = pd.to_datetime(p["date"]); p = p.sort_values(["secid", "date"]).reset_index(drop=True)
    sp2 = p["spot"] ** 2 * 0.01
    p["net"] = (p["call_g"] - p["put_g"]) * sp2; p["roll"] = (p["roll_call"] - p["roll_put"])
    p["c"] = p.groupby("secid")["net"].transform(rz)
    p["Z"] = p.groupby("secid")["roll"].transform(rz)
    p["Y"] = p.groupby("secid")["realized_vol"].transform(lambda s: rz(s.shift(-5)))

    anng = ann.groupby("secid")["rdq"].apply(lambda s: np.array(s.values, dtype="datetime64[D]")).to_dict()
    def mark(grp):
        rds = anng.get(grp.name)
        d = grp["date"].values.astype("datetime64[D]")
        flag = np.zeros(len(grp), bool)
        if rds is not None:
            for rd in rds:
                flag |= np.abs((d - rd).astype("timedelta64[D]").astype(int)) <= WIN
        return pd.Series(flag, index=grp.index)
    p["earn"] = p.groupby("secid", group_keys=False).apply(mark)

    def run(df):
        for v in ("Y", "c", "Z"): df[v + "_d"] = df[v] - df.groupby("date")[v].transform("mean")
        df = df.dropna(subset=["Y_d", "c_d", "Z_d"]); df["const"] = 1.0
        cl = np.column_stack([df["secid"].astype("category").cat.codes,
                              df["date"].astype("category").cat.codes])
        m = IV2SLS(df["Y_d"], df[["const"]], df["c_d"], df["Z_d"]).fit(cov_type="clustered", clusters=cl)
        return (len(df), round(float(m.params["c_d"]), 4), round(float(m.tstats["c_d"]), 2))

    base = p[["date", "secid", "Y", "c", "Z", "earn"]].dropna(subset=["Y", "c", "Z"])
    print(f"earnings-window share: {base['earn'].mean():.1%} of firm-days")
    rows = [("spec", "firm_days", "beta", "t_twoway"),
            ("all firm-days",) + run(base.copy()),
            (f"EXCL earnings window (±{WIN}d)",) + run(base[~base["earn"]].copy()),
            ("earnings window only",) + run(base[base["earn"]].copy())]
    for r in rows[1:]: print("  ", r)
    (DATA / "e4_earnings_exclusion.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_earnings_exclusion.csv")

if __name__ == "__main__":
    main()
