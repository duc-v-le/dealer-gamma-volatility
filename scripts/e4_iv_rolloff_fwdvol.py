#!/usr/bin/env python3
"""
e4_iv_rolloff_fwdvol.py — UNIFIED Table 2 (reconciles the SPX spec across Table 2 & Table 3)
========================================================================================
The original e4_iv_rolloff.py (Table 2) used a different forward-vol construction than
e4_paperA_outcomes.py (Table 3), so the SPX net-gamma IV read -0.133 in one and -0.185 in
the other. This script re-runs the FULL Table-2 battery (OLS benchmarks + roll-off IV for
short/net/gross + the coarse OPEX-share instrument) under the SAME outcome the draft §2.4
defines and Table 3 uses: the forward 5-day realized vol = std of daily returns over the
NEXT five trading days, then rolling-z stationarized. After this, Table 2's IV net-gamma row
equals Table 3's SPX realized_vol_5d row by construction.

Run: python3 e4_iv_rolloff_fwdvol.py  ->  data/e4_iv_rolloff_fwdvol.csv
"""
import pathlib
import numpy as np, pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def rz(s, win=60):
    s = pd.Series(s); return (s - s.rolling(win).mean()) / s.rolling(win).std()

def fwd_std(ret, H=5):
    out = np.full(len(ret), np.nan)
    for t in range(len(ret) - H - 1):
        out[t] = np.nanstd(ret[t+1:t+1+H])      # SAME as e4_paperA_outcomes.fwd(ret,5,np.nanstd)
    return out

def iv(df, endog, instr, label):
    m = IV2SLS(df["Y"], df[["const", "vix"]], df[endog], df[instr]).fit(cov_type="kernel", kernel="bartlett")
    F = m.first_stage.diagnostics.loc[endog, "f.stat"]
    fs = m.first_stage.individual[endog].params[instr]   # first-stage slope pi_1
    return (label, round(m.params[endog], 4), round(m.tstats[endog], 2),
            round(m.params["vix"], 3), round(m.tstats["vix"], 2), round(fs, 3), round(F, 1))

def main():
    p = pd.read_csv(REAL / "spx_rolloff.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="inner")
    p["net_gex"] = pd.to_numeric(p["net_all"], errors="coerce") * pd.to_numeric(p["spot"], errors="coerce")**2 * 0.01
    p["short_intensity"] = np.maximum(-p["net_gex"], 0.0)
    ret = pd.to_numeric(p["ret"], errors="coerce").values

    df = pd.DataFrame({
        "c_short": rz(p["short_intensity"]),
        "c_net":   rz(p["net_gex"]),
        "c_gross": rz(p["gross_all"]),
        "vix":     rz(p["vix"]),
        "Zg":      rz(p["scheduled_rolloff_gross"]),
        "Zn":      rz(p["scheduled_rolloff_net"]),
        "Zs":      rz(p["rolloff_share"]),
    })
    df["Y"] = rz(fwd_std(ret, 5))          # <-- unified outcome (draft §2.4 / Table 3)
    df["const"] = 1.0
    df = df.dropna().reset_index(drop=True)

    out = []
    for endo in ("c_short", "c_net", "c_gross"):
        m = IV2SLS(df["Y"], df[["const", endo, "vix"]], None, None).fit(cov_type="kernel", kernel="bartlett")
        out.append((f"OLS {endo}|vix", round(m.params[endo], 4), round(m.tstats[endo], 2),
                    round(m.params["vix"], 3), round(m.tstats["vix"], 2), "-", "-"))
    out.append(iv(df, "c_short", "Zn", "IV short-gamma | rolloff_net"))
    out.append(iv(df, "c_net",   "Zn", "IV net-gamma   | rolloff_net"))
    out.append(iv(df, "c_gross", "Zg", "IV gross-gamma | rolloff_gross"))
    out.append(iv(df, "c_short", "Zs", "IV short-gamma | rolloff_share"))

    print(f"UNIFIED Table 2 — roll-off IV, draft-§2.4 outcome (linearmodels HAC), n={len(df)}\n")
    print(f"  {'spec':<32}{'beta':>9}{'t':>7}{'vixcoef':>9}{'vix_t':>7}{'fs_pi1':>8}{'firstF':>9}")
    for r in out:
        print(f"  {r[0]:<32}{r[1]:>9}{r[2]:>7}{str(r[3]):>9}{str(r[4]):>7}{str(r[5]):>8}{str(r[6]):>9}")
    rows = [("spec", "beta", "t", "vix_coef", "vix_t", "firststage_pi1", "first_stage_F")] + out
    (DATA / "e4_iv_rolloff_fwdvol.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_iv_rolloff_fwdvol.csv")

if __name__ == "__main__":
    main()
