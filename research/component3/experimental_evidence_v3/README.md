# Function 2 — evidence sufficiency against the frozen v3 system

Research-only adaptation to the actual calibrated unpooled HGB conditional-hazard
artifact from multi-horizon v3/run_003. Earlier LR/N2 results are preserved and
are not treated as v3 evidence. The application, contracts, artifacts and model
probabilities were not changed. Synthetic Azure industrial benchmark data is
not validated apparel-factory data.

Completed results: [report](run_002/report.md), [all 120 metric rows](run_002/metrics.csv),
[paired clustered intervals](run_002/paired_cluster_intervals.csv),
[baseline profiles](run_002/baseline_evidence_profiles.csv),
[secondary feasibility](run_002/secondary_feasibility.csv).

## Findings

Full-history Brier at 7/14/30 days reproduced exactly:
**0.092593 / 0.179541 / 0.233211**; AP **0.354306 / 0.374760 / 0.631971**.
The test cohort has 1,300 cases and 167/332/631 positive machine-date windows.

- Deleting maintenance records generally harmed the frozen model more than
  deleting errors. At 75% expected deletion, five-seed mean Brier became
  **0.114982 / 0.190817 / 0.254466** for maintenance and
  **0.092939 / 0.182366 / 0.238105** for errors.
- Joint 90-day truncation modestly lowered 7/14-day Brier but worsened 30-day
  Brier; ranking and calibration trade-offs prevent a universal shorter-is-better
  interpretation. Maintenance-only 90-day truncation worsened Brier at all horizons.
- Existing fitted imputers can transform numeric masks; that does not make
  missing fields harmless. Maintenance-field masking worsened seven-day Brier
  to **0.154898**. All outputs remained coherent, illustrating that monotonicity
  does not establish accuracy or evidence sufficiency.
- R had a strong descriptive seven-day loss association; F a moderate fourteen-day
  association. H was weak and C constant. These are confounded observations,
  not causal relationships or validated sufficiency thresholds.
- **C0/C1 was not estimated.** Base-training predictions are in-sample;
  calibration predictions reused labels fitted by the sigmoid calibrators;
  test labels cannot train secondary models. No independent earlier 30-day-purged
  period remains. The report documents the exact dates and future design needed.

Recommendation: **descriptive evidence display only**. No reliability percentage,
confidence score, maintenance recommendation or sufficiency category is justified.

## Method and scope

Predeclared 40 conditions: full; joint, maintenance-only and error-only truncation
at 180/90 days; maintenance/error thinning at expected removal rates 25/50/75%,
five seeds each; numeric masking of maintenance fields, error fields and recencies.
No optional 30-day truncation or combined-source thinning was added after inspecting
results. Sparse/zero event histories remain explicit. Random deletion masks persist
across later snapshots of the same record; future events never enter features.
True inclusive future labels stay unchanged, including under historical failure
truncation. Numeric masks use only the artifact's fitted preprocessing, not zeros
or new imputers. H/C remain descriptive metadata, not v3 predictors.

Machine-cluster and date-cluster bootstrap views use 1,000 draws separately.
They account for repeated observations within each grouping, not all dependence
simultaneously. Ranking bootstrap holds original memberships fixed and weights
their contributions; it does not rerank duplicate physical machines. Counts,
denominators, score ties, exact group ranges and seed replicates are retained.
Neither reused test results nor percentile intervals are independent validation.

## Existing environment and exact Windows CMD commands

No installation is necessary in the audited environment (NumPy 2.2.6, pandas 2.3.2,
scikit-learn 1.7.1, SciPy 1.16.1, joblib 1.5.1, matplotlib as recorded in summary).
Do not run old training CLIs to regenerate charts. Use a **new** output directory:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
set OMP_NUM_THREADS=1
set C3_EVIDENCE_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw
python -B -m pytest research\component3\experimental_evidence_v3\test_study.py -q -p no:cacheprovider
python -B research\component3\experimental_evidence_v3\run_study.py --data-dir "%C3_EVIDENCE_DATA_DIR%" --out research\component3\experimental_evidence_v3\run_003 --bootstrap-draws 1000
python -B research\component3\experimental_evidence_v3\report_study.py research\component3\experimental_evidence_v3\run_003
```

For saved-output fidelity tests against the completed run:

```cmd
set C3_EVIDENCE_RUN_DIR=E:\J26-DS-357-smart-apparel-lk\research\component3\experimental_evidence_v3\run_002
python -B -m pytest research\component3\experimental_evidence_v3\test_study.py -q -p no:cacheprovider
```

Keep the original audited v3 artifact at
`research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib`,
the original LR artifact, all five checksum-pinned research source modules, the
experimental manifest, v3 `heldout_predictions.csv`/`comparison.csv`, and original
`outputs_corrected/machine_prediction_examples.csv`. The runner verifies these
sources and raw files, aborting on missing/inconsistent data. Git LF integrity
rules remain applicable. Raw files and model bundles remain local. No package
installation, refitting or exported model is part of this study.

## Files and outputs

Only this new directory is created/changed:

- `evidence.py`: historical reconstruction, persistent perturbations, split/C audit.
- `evaluation.py`: metrics, deterministic grouped Top 10, tie-aware weighted AP,
  paired machine/date bootstrap calculations.
- `run_study.py`: audit, no-fitting guard, exact cohort/reference checks, executed
  sensitivity experiment and full-precision outputs.
- `report_study.py`: saved-result tables/interpretation, numerical plots/provenance;
  no model loading or fitting.
- `test_study.py`, `README.md`, `.gitignore`.

The completed local `run_002/` contains plan/summary/provenance JSON; baseline
evidence profiles; 40 per-condition prediction CSVs; metric, calibration,
per-date ranking, evidence distribution/strata/correlation, frozen imputer,
secondary feasibility, condition audit, replicate and cluster-stability tables;
four PNG charts; and report.md. CSV predictions include unchanged true labels.
Conditions remain matched; replicates are not pooled as extra observations.
`run_001/` is retained as incomplete with failure.json (duplicate export keyword,
fixed before sensitivity evaluation); do not use it as a completed result.

Generated `run_*/` files are locally ignored and not committed. report_study.py
regenerates only derived files inside this experiment; run_study.py refuses
existing output directories. The initial completed run took about 191 seconds.
No previous experiment, FastAPI, React, model artifact or Function 1/2 UI was edited.

## Verification record — 2026-10-23

**28 tests passed, zero skipped**, with real data and completed-run paths set.
Tests cover strict-past/window boundaries, inclusive event boundaries, incomplete
follow-up, missing recencies, field masks, persistent thinning keys/seeds,
chronological embargo and secondary-period infeasibility, known metrics/ties,
weighted AP against sklearn, paired cluster calculations, exact 1,300-case model
parity, all 40 saved cohorts/labels and 120 metric rows, actual seeded prediction
reproduction, no-fitting guards and unchanged source/model hashes. The completed
run produced 61 output files, including 40 per-condition prediction files.

The before/after protection audit checked **328 pre-existing files** across the
research/backend/frontend trees: **none changed**. Existing uncommitted work was
preserved. No packages installed, no model/calibrator refits, no new joblib files,
and no commits or pushes. Charts were visually reviewed; CSV tables accompany
their descriptive groups/means and uncertainty summaries.
