# Model comparisons

Open `../notebooks/22_model_comparisons.ipynb` to regenerate the current manuscript tables and their cross-checks. It reads the included October results in `published/`, shows each dataframe, and saves that same table in `tables/`. It does not retrain models or use the older flexible-head benchmark tables.

The estimator code and fixed inputs are included for rerunning the comparisons. Model fitting starts from the analysis-ready partitions used in the paper. The compact country archives contain the original training matrices, validation/test choices, participant/pair mappings and external-classification folds. Their bytes are verified before use. No Dropbox account, server directory or earlier checkout is required.

## Install

Activate the manuscript's Python 3.10 environment, then run from this folder:

```sh
python -m pip install -r requirements.txt
Rscript install_packages.R
python unpack_inputs.py
```

R and an R package build toolchain must be installed first. `install_packages.R` installs/checks pscl 1.5.9, wnominate 1.5 and emIRT 0.0.15, with jsonlite and digest. The native code also uses `sha256sum` or `shasum`, normally supplied by Linux/macOS. Set `DYNAMAP_RSCRIPT` if the desired Rscript is not on PATH. The separate one-dimensional timing run needs GNU time (`brew install gnu-time` on macOS).

## Two-dimensional comparison

Run these stages in order. Replace `brazil` with `france` or `chile` to repeat the other country. Parameters remain explicit in the source, and the protocol describes their scientific role.

```sh
python run_comparison.py --country brazil --stage prepare
python run_comparison.py --country brazil --stage tune
python run_comparison.py --country brazil --stage fit
python run_comparison.py --country brazil --stage native
python run_comparison.py --country brazil --stage evaluate
```

`--stage all` runs the same sequence. Preparation rebuilds the common support using training responses only. Tuning uses the original six configurations and two seeds. Final neural fits use three seeds, and native estimation includes W-NOMINATE plus the four-chain Bayesian fits and prior/anchor sensitivities. Test evaluation occurs only after the country's neural configurations and final fits are fixed.

Outputs go to `prepared/` and `results/`; the included publication results are never replaced. The native stage can take days and substantial memory. Its default wall budget is seven days from invocation (`--days` changes that wall budget); the statistical maximum remains 30,000 Bayesian iterations. A nonconverged fit is recorded and excluded from predictive scoring. Inspect an interrupted native fit's lock and saved status before restarting. W-NOMINATE cannot resume an interrupted Fortran call, although a completed saved native object can be postprocessed again.

## One-dimensional EM-IRT comparison

First run the two-dimensional `prepare` stage for the chosen country. Then, from `benchmarks/emirt/`:

```sh
python -m src.prepare --source .. --out data --country brazil
python -m src.build_jobs --root . --country brazil
python -m src.scheduler --root .
```

This recreates the matched 1D inputs from the same support masks, selects neural configurations, and runs the original primary and supplementary EM-IRT comparisons. It records new validation, time and memory results in this directory. By default its wall budget is seven days from startup; set `DYNAMAP_BENCHMARK_DAYS` to change it. Keep a completed run in its own copy before rebuilding a queue for another country. The supplied original measurements remain in `published/emirt_scaling/`.

## Verify without a full refit

```sh
python unpack_inputs.py
python tests/test_evaluate_smoke.py
```

The input check verifies all archived file hashes. The evaluation check exercises common-row matching, held-out target validation, per-fit aggregation, fixed external folds, bootstrap direction, exact ties and overwrite guards. Notebook 22 verifies all three publication tables against the supplied fit-level summaries, including runtime totals and EM-IRT medians. These checks do not substitute for rerunning the long native fits.

The [two-dimensional protocol](PROTOCOL.md) and [one-dimensional protocol](emirt/PROTOCOL.md) state the scope and limitations. `source_provenance.json` identifies the original estimator-source hashes and portability changes. `published/` holds immutable numerical records; timings depend on the original hardware and will vary on a new machine. The published Bayesian diagnostic results describe failure to meet the full convergence criterion, not a missing or zero predictive score.
