#!/usr/bin/env python3
"""
e4_sign_robust.py — is the result robust to the dealer-gamma SIGN CONVENTION?
================================================================================
Signed customer demand (CBOE/ISE Open-Close) is not accessible on this WRDS account, so gamma is
instead DECOMPOSED into call and put components (panel_sp500_cp) and the IV is tested under
several conventions. Crucially this lets the data *validate* the standard convention:
  • CALL gamma alone: if dealers are LONG calls, more call gamma ⇒ more dealer long gamma ⇒
    LOWER future vol ⇒ expect β<0.
  • PUT gamma alone: if dealers are SHORT puts, more put gamma ⇒ more dealer SHORT gamma ⇒
    HIGHER future vol ⇒ expect β>0.
  • NET (call−put): the baseline dealer-gamma measure; expect β<0.
If call→β<0 and put→β>0 (opposite signs, as the convention predicts), the call-long/put-short
assumption is EMPIRICALLY SUPPORTED, not merely assumed.

Panel IV: date-FE, date-clustered, roll-off-instrumented. Run: python3 e4_sign_robust.py
Out: data/e4_sign_robust.csv
"""
import pathlib
import pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def rz(s, win=60):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()

def panel_iv(p, cval, zval, label):
    df = pd.DataFrame({"date": p["date"], "secid": p["secid"], "Y": p["Y"],
                       "c": cval, "Z": zval, "vixz": p["vixz"]}).dropna()
    for v in ("Y", "c", "Z", "vixz"):
        df[v + "_d"] = df[v] - df.groupby("date")[v].transform("mean")
    df = df.dropna(subset=["Y_d", "c_d", "Z_d", "vixz_d"]); df["const"] = 1.0
    m = IV2SLS(df["Y_d"], df[["const", "vixz_d"]], df["c_d"], df["Z_d"]).fit(
        cov_type="clustered", clusters=df["date"].astype("category").cat.codes)
    return (label, len(df), round(m.params["c_d"], 4), round(m.tstats["c_d"], 2),
            round(m.first_stage.diagnostics.loc["c_d", "f.stat"], 0))

def main():
    p = pd.read_parquet(REAL / "panel_sp500_cp.parquet"); p["date"] = pd.to_datetime(p["date"])
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="left").sort_values(["secid", "date"]).reset_index(drop=True)
    sp2 = p["spot"] ** 2 * 0.01
    g = p.groupby("secid", group_keys=False)
    p["Y"] = g.apply(lambda x: rz(x["realized_vol"].shift(-5))).reset_index(level=0, drop=True)
    p["vixz"] = rz(p["vix"])

    # build the four gamma measures (×spot²×0.01) and their roll-off instruments, then per-secid rolling-z
    def Z(series): return p.groupby("secid", group_keys=False).apply(lambda x: rz(series.loc[x.index])).reset_index(level=0, drop=True)
    measures = {
        "NET (call−put) [baseline]": (p["call_g"] - p["put_g"]) * sp2,
        "CALL gamma only":            p["call_g"] * sp2,
        "PUT gamma only":             p["put_g"] * sp2,
        "GROSS (call+put)":          (p["call_g"] + p["put_g"]) * sp2,
    }
    instr = {
        "NET (call−put) [baseline]":  p["roll_call"] - p["roll_put"],
        "CALL gamma only":            p["roll_call"],
        "PUT gamma only":             p["roll_put"],
        "GROSS (call+put)":           p["roll_call"] + p["roll_put"],
    }
    rows = [("convention", "firm_days", "beta", "t", "firstF")]
    for k in measures:
        rows.append(panel_iv(p, Z(measures[k]), Z(instr[k]), k))

    print("sign-convention robustness (panel IV, date-FE, date-clustered)\n")
    print(f"  {'convention':<28}{'firm-days':>11}{'beta':>9}{'t':>7}{'F':>9}")
    for r in rows[1:]:
        print(f"  {r[0]:<28}{r[1]:>11,}{r[2]:>9}{r[3]:>7}{r[4]:>9}")
    (DATA / "e4_sign_robust.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_sign_robust.csv")
    print("  Expectation: CALL β<0 (dealer long-gamma dampens) and PUT β>0 (dealer short-gamma")
    print("  amplifies) ⇒ the call-long/put-short convention is EMPIRICALLY SUPPORTED; NET (baseline)")
    print("  carries the directional signal. If call & put have the SAME sign, the convention is")
    print("  the wrong frame — report honestly.")

if __name__ == "__main__":
    main()
