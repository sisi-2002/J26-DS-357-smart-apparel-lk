# Maintenance history and historical-error ablation

**Research only. Synthetic Azure industrial-machine benchmark; not validated
apparel-factory prediction. Maintenance-log-first, with no telemetry features.**

Read the generated [numerical report](results/report.md),
[precision–recall curves](results/precision_recall.png) and
[calibration plot](results/calibration.png).

## Important correction to the requested baseline

The existing baseline already includes `error_events_prior`, `error_events_7d`,
`error_events_30d`, `error_events_90d` and `days_since_last_error`. Consequently,
the reference Brier 0.091485 and AP 0.252146 are **not history-only results**.
Adding those same columns again would not be a meaningful incremental-error
experiment.

This experiment preserves the exact reference and adds a clean control:

| Model | Feature set | Raw feature count |
| --- | --- | ---: |
| History-only control | Original features minus all five generic error features; machine age/type, maintenance history and failure history retained | 13 |
| A: existing reference | Exact original 17 numeric features plus machine-model category, already including the five error summaries | 18 |
| B: category extension | A plus four category-specific 30-day counts | 22 |

Control versus A estimates the value of generic error history. A versus B
estimates additional category detail. This prevents attributing the deployed
reference metrics to an error-free model.

## Verified error data

The local `PdM_errors.csv` has exactly `datetime`, `machineID`, `errorID`:
3,919 records, no missing fields, no duplicate complete rows, all 100 known
machine IDs, and no unknown IDs. Timestamps range from `2015-01-01 06:00:00` to
`2016-01-01 05:00:00`. The original rows are not globally time-sorted; the builder
sorts within each machine.

Generic category record counts are `error1=1010`, `error2=988`, `error3=838`,
`error4=727`, `error5=356`. The first four appear on all 100 machines; error5
appears on 93. There are 303 additional records at already occupied machine-time
keys: distinct simultaneous categories are retained as distinct records, matching
the original generic-count semantics. Exact duplicate records would fail rather
than be silently removed.

These categories are **not confirmed failures**, validated sewing-machine fault
types, physical severity classes or maintenance recommendations. Complete CSV
fields do not establish comprehensive real-world event logging. Future records,
including those in 2016, cannot affect any historical snapshot.

## Exact error features

The builder independently reconstructs the existing five features and checks
parity across all 3,600 original machine-time cases:

- `error_events_prior`: all available recorded error rows strictly before `t`.
- `error_events_7d`, `error_events_30d`, `error_events_90d`: counts in
  `[t - window, t)`; lower endpoints included, prediction time excluded.
- `days_since_last_error`: elapsed days from the latest record strictly before
  `t`; **NaN** if no prior event exists.

B additionally uses:

- `error_category_error2_30d`
- `error_category_error3_30d`
- `error_category_error4_30d`
- `error_category_error5_30d`

The category vocabulary is learned only from records strictly before the final
training landmark (`2015-05-31 06:00:00`). All five categories have training
support: 440, 386, 360, 288 and 159 records respectively. `error1` is the fixed
reference category: its 30-day count is recoverable from total 30-day count minus
the four added counts, avoiding exact linear redundancy. Historical categories
outside the frozen vocabulary fail explicitly; future categories do not expand
the vocabulary or affect past features.

Zero prior counts mean no recorded past errors, not confirmed error-free operation
or verified logging coverage. Error recency means time since an event, **not
database freshness**. Missing recency is handled by the original training-only
median imputer and missingness indicator; no arbitrary large or zero recency is
invented. Three training cases have no prior errors; every calibration/test case
has prior errors, so missing-error-log behavior is not validated by this test set.

## Training, calibration and temporal controls

Original chronological cases are retained: 1,800 training, 500 later
validation/calibration, 1,300 test, with 100 machines and 13 test dates. The full
30-day separation between splits is checked. Features use strictly past events;
targets preserve `t < failure < t+7 days`. The legacy run-summary metadata says
`>= t`, but executable code and verified saved labels use `> t`; existing files
are not changed.

Every model reuses the original Logistic Regression factory/configuration:
L2, `C=1`, `max_iter=2000`, seed 42, no class balancing. Only the explicitly
declared numeric-column list changes. Median imputation, missingness indicators,
scaling and machine-model one-hot encoding fit training cases only. Sigmoid fits
the designated 500 later cases with the base pipeline frozen. A fitted-state hash
checks that calibration cannot modify preprocessing or base parameters.

A must reproduce the original chronological calibration CSV and the deployed
artifact's per-instance scores before either ablation model is interpreted.
Observed parity: Brier `0.09148540592410336`, AP `0.2521463037932662`, maximum
deployed-score difference `0`. No uncalibrated or stratified-CV3 comparison is
mixed into the reported results.

Three new research Logistic Regression models and three validation sigmoids are
fitted in memory per run. The deployed 7-day artifact is loaded only for prediction
parity. No model artifacts are serialized or replaced, and no hyperparameters,
feature set or thresholds are tuned against test labels.

## Observed result and interpretation

| Model | Brier | AP | Precision@10 | Recall@10 | Top 10 captured windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| History-only control | 0.091757 | 0.237720 | 0.246154 | 0.219178 | 32 |
| A: existing reference | 0.091485 | 0.252146 | 0.284615 | 0.253425 | 37 |
| B: category extension | 0.090813 | 0.251217 | 0.276923 | 0.246575 | 36 |

Test prevalence is 146/1,300 (11.2308%); the constant training-prevalence baseline
Brier is 0.099731. Generic errors give modest preliminary group-level signal:
Brier improves by 0.000272, AP by 0.014427 and Top 10 captures five more windows.
The conditional machine-cluster Brier-difference interval includes zero, so a
general improvement is not established.

Category detail improves Brier by another 0.000672 and log loss by 0.001892, but
slightly worsens AP, Top 10 and ten-bin ECE. At fixed budgets 10/20/30 per date,
B has 94/196/302 false-positive windows versus A's 93/198/301: no consistent
false-alarm reduction. No operational acceptability is inferred from these counts.

The error features are not uniformly noise, but neither group-level ablation nor
correlated training coefficients establishes which individual feature generalizes.
Category error2/error3 have the largest category-coefficient magnitudes in this
fit; this is not causal importance. Error5's 30-day count is zero in 72.69% of test
cases. Total prior errors also grow with accumulated observation (training mean
10.00, test mean 30.85), creating calendar/log-duration confounding and possible
distribution drift despite strictly historical calculation.

**Keep the existing reference for the current demo.** Continue investigating
category detail as an optional research source rather than replacing the model.
For factories without error logs, validate a separate history-only fallback;
do not silently feed invented zero histories into the current artifact, which
was trained with error features.

## Evaluation definitions and limitations

Top 10/20/30 is selected separately within each date, by unrounded probability
descending and machine ID ascending for ties. Micro precision divides captured
positive windows by all selected windows; micro recall divides by all positive
machine-time windows. At Top 10 the denominators are 130 selections and 146
positive windows. These are **not unique physical failures**. Repeated machines,
overlapping prediction windows and shared calendar effects create dependence.

Fixed budgets compare false positives at the same number of selections without
using test outcomes to choose a threshold. Separately exported validation-derived
50/70/80% recall thresholds reuse sigmoid calibration cases and are explicitly
exploratory, not independent threshold validation. They are applied unchanged to
test; achieved test recalls differ, so their FP counts are not an exact
matched-recall comparison. No inspection-capacity or maintenance-cost rule is
validated here.

AP is average precision, not trapezoidal PR area. Calibration plots show actual
mean probabilities and observed fractions in ten equal-width bins, with empty
bins omitted; counts remain in the numerical table. No smoothing or invented
significance is added. Brier measures total probability error, and ECE depends
on the bins; lower Brier alone does not prove better calibration.

The 500 seed-42 machine-cluster bootstrap resamples retain all snapshots for a
sampled machine, conditional on fitted models. They exclude fitting and shared-date
uncertainty, lack multiplicity correction and are not external confirmation.
Training/calibration resubstitution metrics are labelled in
`split_fit_diagnostics.csv`; apparent gaps do not by themselves prove overfitting.
The same test period has already been examined repeatedly. Genuinely fresh
chronological evaluation and real factory validation are still required.

## Run and verify

Use the existing research Python environment; `requirements.txt` reuses shared
research dependencies and documents matplotlib. No installation was needed for
the recorded run. Required inputs are the four original event/machine CSVs,
`outputs_corrected/machine_prediction_examples.csv`, the chronological calibration
CSV and the trusted local 7-day artifact. The existing
`experimental_thresholds/threshold_metrics.py` supplies read-only verified metric
utilities; the old threshold experiment is not run to generate these outputs.

From the repository root, Windows CMD:

```bat
python -B research\component3\experimental_error_features\run_experiment.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
set "C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -W error::DeprecationWarning -m unittest discover -s research\component3\experimental_error_features -p "test_*.py" -v
```

Missing/inconsistent sources and reference-parity failures stop the run. Without
the environment variable, 13 unit tests run and four real-data integration tests
explicitly skip. Raw data and existing model artifacts remain local/Git-ignored.

Files in this directory:

- `error_features.py`: schema validation, frozen training vocabulary and
  historical feature builder.
- `run_experiment.py`: reference reproduction, ablations, evaluation and figures.
- `test_error_features.py`: boundary/leakage, missing history, preprocessing,
  calibration, reproducibility and actual-benchmark regression checks.
- `results/report.md`, `results/summary.json`: full numerical report and provenance.
- `results/model_metrics.csv`, `results/comparisons.csv`,
  `results/split_fit_diagnostics.csv`, `results/calibration_bins.csv`.
- `results/top10_by_date.csv`, `results/matched_alert_budgets.csv`,
  `results/exploratory_recall_thresholds.csv`.
- `results/error_availability.csv`, `results/error_feature_distribution.csv`,
  `results/training_coefficients.csv`, `results/paired_brier_differences.csv`.
- `results/precision_recall.*`, `results/calibration.*`: PNG/SVG figures.
- `outputs/heldout_predictions.csv`, `outputs/historical_error_features.csv`:
  detailed local exports ignored by existing Git rules.

No existing research scripts/outputs, hazard/threshold experiments, Function 2,
FastAPI, React or saved model files are modified. Nothing is committed, pushed
or integrated into the demo.
