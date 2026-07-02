#!/usr/bin/env python3
"""
e4_iv_robust_excl.py — harden the roll-off IV against the exclusion restriction
=============================================================================
The scheduled gamma roll-off is a strong instrument (F≈1400) giving a significant,
correctly-signed causal effect of dealer net gamma on future vol beyond VIX. The open
worry is the EXCLUSION RESTRICTION: OPEX roll-off coincides with volume/hedging-unwind/
pinning, which could move vol through non-gamma channels. This stress-tests it:

  (1) Volume control — add SPX option-volume (the OPEX activity spike) to the 2SLS.
      If the IV effect survives, the volume channel is unlikely to be the violation.
  (2) ΔVIX control — guard against the instrument proxying for vol-of-vol moves.
  (3) Over-identification (Hansen/Wooldridge J) — two roll-off instruments; H0 = valid.
  (4) Falsification placebos:
      (a) shuffle the instrument → first-stage F must collapse (genuine strength check);
      (b) does the instrument "predict" PAST vol? (it shouldn't, if it's a clean forward effect.)

Survival under volume/ΔVIX controls + a non-rejected J + clean
placebos ⇒ the causal effect is robust. Failure under controls ⇒ exclusion likely violated.
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

def main():
    p = pd.read_csv(REAL / "spx_rolloff.csv", parse_dates=["date"]).dropna(subset=["realized_vol"]).reset_index(drop=True)
    # controls from WRDS: VIX level + SPX daily total option volume
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    optvol = pd.concat([w.query(f"SELECT date, SUM(volume) AS optvol FROM optionm.opprcd{y} "
                                f"WHERE secid=108105 AND volume IS NOT NULL GROUP BY date;")
                        for y in range(2016, 2025)])
    for d in (vix, optvol):
        d["date"] = pd.to_datetime(d["date"])
    p = p.merge(vix, on="date").merge(optvol, on="date")
    p["vix"] = p["vix"].astype(float); p["optvol"] = pd.to_numeric(p["optvol"], errors="coerce")
    p["net_gex"] = p["net_all"] * p["spot"]**2 * 0.01

    df = pd.DataFrame({
        "c":    rz(p["net_gex"]),                 # endogenous: signed net dealer gamma
        "vix":  rz(p["vix"]),
        "dvix": rz(p["vix"].diff()),              # ΔVIX
        "vol":  rz(np.log(p["optvol"].clip(lower=1))),  # log option volume (OPEX activity)
        "Z":    rz(p["scheduled_rolloff_net"]),   # instrument
        "Zg":   rz(p["scheduled_rolloff_gross"]), # 2nd instrument (for over-id)
    })
    Yz = rz(p["realized_vol"])
    df["Y"] = Yz.shift(-1).rolling(5).mean().shift(-4)     # future 5d vol
    df["Ypast"] = Yz.rolling(5).mean()                     # past 5d vol (placebo outcome)
    df["const"] = 1.0
    df = df.dropna().reset_index(drop=True)
    n = len(df)

    def run(exog, instr, dep="Y"):
        m = IV2SLS(df[dep], df[["const"] + exog], df["c"], df[instr]).fit(cov_type="kernel", kernel="bartlett")
        F = m.first_stage.diagnostics.loc["c", "f.stat"]
        j = ""
        if isinstance(instr, list) and len(instr) > 1:
            try: j = f"J p={m.wooldridge_overid.pval:.2f}"
            except Exception: j = ""
        return round(m.params["c"], 4), round(m.tstats["c"], 2), round(F, 1), j

    rows = [("spec", "beta_c", "t", "firstF", "overid")]
    rows.append(("baseline  Y~c|vix",) + run(["vix"], "Z"))
    rows.append(("+ option volume",) + run(["vix", "vol"], "Z"))
    rows.append(("+ volume + dVIX",) + run(["vix", "vol", "dvix"], "Z"))
    rows.append(("over-id [Z,Zg]",) + run(["vix", "vol"], ["Z", "Zg"]))
    # placebo (a): shuffle instrument -> first stage should collapse
    dfx = df.copy(); dfx["Zsh"] = np.random.default_rng(0).permutation(df["Z"].values)
    fs = IV2SLS(dfx["Y"], dfx[["const", "vix"]], dfx["c"], dfx["Zsh"]).fit(cov_type="kernel", kernel="bartlett")
    rows.append(("PLACEBO shuffled-Z", round(fs.params["c"], 4), round(fs.tstats["c"], 2),
                 round(fs.first_stage.diagnostics.loc["c", "f.stat"], 2), "should be weak"))
    # placebo (b): does Z predict PAST vol? (reduced form on Ypast)
    rf = IV2SLS(df["Ypast"], df[["const", "vix", "Z"]], None, None).fit(cov_type="kernel", kernel="bartlett")
    rows.append(("PLACEBO Z->PAST vol(RF)", round(rf.params["Z"], 4), round(rf.tstats["Z"], 2), "-", "want ~0"))

    print(f"exclusion-restriction hardening of the roll-off IV (n={n})\n")
    print(f"  {'spec':<26}{'beta_c':>9}{'t':>7}{'firstF':>9}{'overid':>16}")
    for r in rows[1:]:
        print(f"  {r[0]:<26}{str(r[1]):>9}{str(r[2]):>7}{str(r[3]):>9}{str(r[4]):>16}")
    (DATA / "e4_iv_robust_excl.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_iv_robust_excl.csv")
    print("  Survives volume+dVIX with strong F + non-rejected J + weak shuffled-Z + ~0 past-vol")
    print("  placebo ⇒ the causal effect is robust to the main exclusion-restriction threats.")

if __name__ == "__main__":
    main()
