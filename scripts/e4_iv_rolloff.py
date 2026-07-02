#!/usr/bin/env python3
"""
e4_iv_rolloff.py — IV with the SCHEDULED GAMMA ROLL-OFF instrument
=====================================================================
Tests whether the mechanically-direct "scheduled roll-off" instrument (dollar-gamma
expiring by the next OPEX, from wrds_download_spx_rolloff.py) instruments the
policy-relevant dealer gamma more strongly than the calendar instrument (which was weak,
F≈1.6 for short-gamma). Proper 2SLS via linearmodels (kernel/HAC SEs, first-stage F).

A strong first-stage F (≫10) together with a significant,
correctly-signed 2SLS beyond VIX would identify the causal channel; a still-weak
first stage indicates that even a finer mechanical instrument cannot
separate the truncated/net short-gamma regime from VIX.
"""
import pathlib
import numpy as np
import pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def rz(s, win=60):
    s = pd.Series(s); return ((s - s.rolling(win).mean()) / s.rolling(win).std())

def iv(df, endog, instr, label):
    m = IV2SLS(df["Y"], df[["const", "vix"]], df[endog], df[instr]).fit(cov_type="kernel", kernel="bartlett")
    F = m.first_stage.diagnostics.loc[endog, "f.stat"]
    return (label, round(m.params[endog], 4), round(m.tstats[endog], 2), round(F, 1))

def main():
    p = pd.read_csv(REAL / "spx_rolloff.csv", parse_dates=["date"]).dropna(subset=["realized_vol"]).reset_index(drop=True)
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="inner")
    p["net_gex"] = p["net_all"] * p["spot"]**2 * 0.01
    p["short_intensity"] = np.maximum(-p["net_gex"], 0.0)

    df = pd.DataFrame({
        "c_short": rz(p["short_intensity"]),
        "c_net":   rz(p["net_gex"]),
        "c_gross": rz(p["gross_all"]),
        "vix":     rz(p["vix"]),
        "Zg":      rz(p["scheduled_rolloff_gross"]),
        "Zn":      rz(p["scheduled_rolloff_net"]),
        "Zs":      rz(p["rolloff_share"]),
    })
    df["Y"] = rz(p["realized_vol"]).shift(-1).rolling(5).mean().shift(-4)
    df["const"] = 1.0
    df = df.dropna().reset_index(drop=True)

    out = []
    # OLS benchmarks (with VIX control)
    for endo in ("c_short", "c_net", "c_gross"):
        m = IV2SLS(df["Y"], df[["const", endo, "vix"]], None, None).fit(cov_type="kernel", kernel="bartlett")
        out.append((f"OLS {endo}|vix", round(m.params[endo], 4), round(m.tstats[endo], 2), "-"))
    # IV with the scheduled-rolloff instruments
    out.append(iv(df, "c_short", "Zn", "IV short-gamma | rolloff_net"))
    out.append(iv(df, "c_net",   "Zn", "IV net-gamma   | rolloff_net"))
    out.append(iv(df, "c_gross", "Zg", "IV gross-gamma | rolloff_gross"))
    out.append(iv(df, "c_short", "Zs", "IV short-gamma | rolloff_share"))

    print(f"IV with scheduled gamma roll-off (linearmodels, HAC), n={len(df)}\n")
    print(f"  {'spec':<32}{'beta':>9}{'t':>7}{'firstF':>9}")
    for r in out:
        print(f"  {r[0]:<32}{r[1]:>9}{r[2]:>7}{str(r[3]):>9}")
    rows = [("spec", "beta", "t", "first_stage_F")] + out
    (DATA / "e4_iv_rolloff.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_iv_rolloff.csv")
    print("  Compare first-stage F to the calendar instrument (F≈1.6 for short-gamma).")

if __name__ == "__main__":
    main()
