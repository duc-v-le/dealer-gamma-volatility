# Does Dealer Gamma Move Volatility? Causal Evidence from the Option-Expiration Roll-off

Replication code for the working paper by **Duc V. Le** (Georgetown University).

The study asks whether option-dealer gamma hedging causally moves the realized volatility of the
underlying market. Because dealer positioning and volatility are jointly determined, the dollar gamma
scheduled to expire at each monthly option expiration serves as a calendar-determined instrument for
dealer gamma, in the S&P 500 and its constituents over 2016–2024.

The code spans **data acquisition** (WRDS / OptionMetrics) and **econometrics in Python** —
instrumental variables (hand-rolled and `linearmodels` 2SLS), panel fixed effects, Newey–West (HAC)
inference, and over-identification and falsification tests.

## Repository layout

```
scripts/   Python pipeline: WRDS downloaders (wrds_*.py) + analyses (e4_*.py)
figures/   Generated result figures (PNG)
data/      Aggregated result tables (small CSVs)
docs/      code_documentation.pdf — scripts, data flow, and pipeline diagram
```

## Data availability

The analysis uses dealer-gamma and price panels derived from **WRDS / OptionMetrics**, which are
licensed and **not redistributed here**. The `wrds_*.py` scripts rebuild every panel directly from
WRDS for users with their own access; credentials are read at run time from `~/.pgpass`
(or the `WRDS_USER` environment variable) and are never stored in the code. The `data/` directory
contains only **aggregated result tables** (regression coefficients and summary statistics) — no
licensed records.

## Reproduce

Requires a WRDS account. Run an acquisition script to build a panel, then the matching analysis:

```bash
pip install -r requirements.txt
cp .env.example .env                           # then set WRDS_USER (password lives in ~/.pgpass)
python3 scripts/wrds_download_spx_rolloff.py   # build the scheduled-roll-off panel
python3 scripts/e4_iv_rolloff.py               # the headline roll-off instrumental-variables result
```

See `docs/code_documentation.pdf` for the full script map and the panel → analysis dependency order.

## License

Released under the MIT License (see `LICENSE`).
