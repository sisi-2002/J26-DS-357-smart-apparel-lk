# Coherent multi-horizon v2 — isolated research

**Completed negative/mixed result: no application integration recommended.**
See [the full report](run_003/report.md) and [comparison CSV](run_003/comparison.csv).
This is synthetic Azure industrial-machine data, not validated apparel-factory data.

The new experiment implements upper-inclusive intervals `(0,7]`, `(7,14]`,
`(14,30]`. Previous experiments use upper-exclusive horizons. Old reference scores
are verified separately; fresh independent LR references use the new labels on
identical machine-time keys. Do not compare old/new Briers as a common target.

## Windows CMD reproduction

Use the existing Python environment (run_003 records exact versions). No packages
were installed. Raw input requires machines/maint/failures/errors CSVs; no telemetry.

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
python -B research\component3\experimental_multihorizon_v2\train_evaluate_v2.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --run-dir research\component3\experimental_multihorizon_v2\run_004
python -B research\component3\experimental_multihorizon_v2\make_report.py research\component3\experimental_multihorizon_v2\run_004
set "C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -m pytest research\component3\experimental_multihorizon_v2\test_v2.py research\component3\experimental_multihorizon\test_hazard_model.py research\component3\experimental_algorithm_comparison\test_comparison.py -q -p no:cacheprovider
```

Use a new run directory each time: existing results are never overwritten.
Default output is run_001, so an explicit new name is needed after initial execution.
Missing data, changed cohorts, reference mismatch or incomplete evaluation follow-up
fail explicitly. The existing research sources are read-only imports and must remain
available; v2 is not an independent installable package. Python `-B` protects caches.

## Files and outputs

- `hazard_v2.py`: inclusive event-time/risk sets, pooled LR/HGB, schema-checked loader.
- `train_evaluate_v2.py`: source/cohort verification, development folds, fits,
  calibration, metrics, cluster comparisons, charts and local research artifacts.
- `test_v2.py`: deterministic known outcomes, censoring, leakage, schema/loading,
  repeatability, purging, real numerical outputs and protected deployed artifact.
- `make_report.py`: renders saved CSV/JSON data without fitting models.
- `run_001/`, `run_002/`, `run_003/`: actual repeated runs. Metric/development/
  per-instance prediction CSVs were identical across runs; runtime varies.
- Per run: `plan.json`, `summary.json`, `comparison.csv`, `metrics_all_splits.csv`,
  `development.csv`, `calibration_bins.csv`, `top10_by_date.csv`,
  `heldout_predictions.csv`, `boundary_differences.csv`, `runtime.csv`,
  `paired_cluster_deltas.csv`, `feature_shift.csv`, `legacy_algorithm_reference.csv`,
  calibration and precision–recall PNGs. Newer runs include `source_hashes.json`.
- Local `hazard_v2_*_research.joblib` bundles are Git-ignored. No model replaces
  `models/c3_7day_sigmoid.joblib`; no backend/frontend reads these bundles.

The first run established results. Runs 2 and 3 verified numerical reproducibility;
the runner was uniquely named to avoid a Python module-name collision when
collecting older experiment tests together. No parameter grid was expanded after
examining test metrics. All previous research and uncommitted frontend work is retained.
