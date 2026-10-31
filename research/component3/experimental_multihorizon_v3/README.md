# Multi-horizon v3 — isolated, actually executed research

Synthetic Azure benchmark, inclusive intervals `(0,7]`, `(7,14]`, `(14,30]`.
No backend, frontend, APIs, deployed models or N2 integration.

**Result:** unpooled HGB is promising for further independent validation but loses
30-day Top-10 performance. Anchored PAV preserves inclusive-reference P7 exactly;
standard PAV changes 141/1,300 P7 values and slightly improves aggregate 7-day ranking.
Keep the deployed system unchanged pending review.

- [Complete report](run_003/report.md)
- [All metrics including ROC-AUC, ECE, raw variants](run_003/comparison.csv)
- [Per-date ranking changes](run_003/ranking_changes_by_date.csv)
- [Adjustment frequencies/magnitudes](run_003/adjustments.csv)

## Exact Windows CMD commands

Existing Python environment; no packages installed. V2 files and its local research
artifacts must remain available. Reproduction reads them without modifying them.

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
python -B research\component3\experimental_multihorizon_v3\evaluate_v3.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --run-dir research\component3\experimental_multihorizon_v3\run_004
python -B research\component3\experimental_multihorizon_v3\make_report_v3.py research\component3\experimental_multihorizon_v3\run_004
set "C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -m pytest research\component3\experimental_multihorizon_v3\test_v3.py research\component3\experimental_multihorizon_v2\test_v2.py research\component3\experimental_multihorizon\test_hazard_model.py research\component3\experimental_algorithm_comparison\test_comparison.py -q -p no:cacheprovider
```

New run directories only; existing outputs cannot be overwritten. Runs 001, 002
and 003 were executed. Run 002 uses numerically stable cumulative recurrence with
exact first-horizon preservation; differences from run 001 are only floating-point
arithmetic in extended unpooled horizons. Main test metrics/ranking agree.
Run 003 verifies exact numerical reproduction of run 002, with standard sklearn
BaseEstimator metadata support. Tests read the reviewed run_002 saved results
(identical to run_003) as well as deterministic fixtures. The final combined
command above passed **64 tests**, with raw-data integration enabled and no skips
or warnings. Earlier collection exposed sklearn estimator-tag deprecation warnings;
inheriting BaseEstimator corrected them before the final run.

## Implementation and outputs

- `methods_v3.py`: deterministic equal-weight PAV, exact-P7 constrained projection,
  interval-specific LR/HGB risk-set fits and validation-only frozen calibration.
- `evaluate_v3.py`: source/cohort/artifact v2 reproduction; training-only development;
  complete evaluations, adjustment/ranking diagnostics, bootstrap, plots and bundles.
- `test_v3.py`: projection optimality vs independent QP solver, all nested targets,
  counterexamples, risk-set preprocessing, freeze/load/repeatability, numerical edge
  cases and actual saved-cohort/metric verification.
- `make_report_v3.py`: completed-experiment report with actual CSV/JSON tables.
- Per run: `plan.json`, `source_hashes.json`, `summary.json`, `development.csv`,
  `comparison.csv`, `metrics_all_splits.csv`, `calibration_bins.csv`,
  `top10_by_date.csv`, `ranking_changes_by_date.csv`, `ranking_changes_total.csv`,
  `adjustments.csv`, `heldout_predictions.csv`, `paired_cluster_deltas.csv`,
  `calibration.png`, `precision_recall.png`, and local Git-ignored research joblibs.

Raw data and v2 artifacts must be present; missing/mismatched source cohorts fail
explicitly. No artificial replacement predictions or hidden case dropping.
Reports describe this fixed completed experiment, not arbitrary future grid changes;
review narrative if the protocol changes. No test-driven hyperparameter expansion.
