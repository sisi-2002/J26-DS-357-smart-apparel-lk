# Exploratory calibrated 7-day algorithm comparison

**Synthetic Azure industrial-machine benchmark, not validated apparel-factory
data. Research only; the existing deployed Logistic Regression stays unchanged.**

See the generated [numerical report](results/report.md),
[precision–recall curves](results/precision_recall.png),
[calibration plot](results/calibration.png) and
[full-precision metrics](results/model_metrics.csv).

## Scope and identical features

Every model uses exactly the same original **18 raw features**: the 17 numeric
columns and `machine_model` category from `scripts/build_and_evaluate.py`.
Machine age/type, maintenance history, failure history and all five existing
generic error-history features are retained:
`error_events_prior`, `error_events_7d`, `error_events_30d`, `error_events_90d`,
`days_since_last_error`.

No telemetry, error-category extensions, new engineered features, machine-ID
predictor or future aggregates are added. The original median imputation with
missingness indicators, scaling and one-hot category encoding are reused for
every algorithm. Each fitting subset fits its own preprocessing. Dense output
for trees changes only matrix representation; tests check identical transformed
values across algorithms. Missingness indicators are the original preprocessing,
not new features added for a particular classifier.

The original full panel is reproduced and checked against saved feature rows,
labels and machine-time keys before fitting. Features use events strictly before
prediction time. Targets remain `t < failure < t+7 days`; the legacy metadata's
`>= t` wording is not used. The original benchmark observation-through-2015
assumption is retained; individual operating/downtime coverage is not proven.

## Chronological development and calibration

Outer cohort: **1,800 training, 500 later calibration, 1,300 test** cases from the
same 100 machines. The original 30-day separation between outer splits is checked.
Test contains 13 complete dates and 146 positive machine-time windows.

Conservative tuning is confined to the original training portion:

| Development fold | Fitting dates | Fitting cases / positives | Scoring dates | Scoring cases / positives |
| --- | --- | --- | --- | --- |
| 1 | Feb 1–Mar 1 | 500 / 59 | Apr 5–Apr 19 | 300 / 32 |
| 2 | Feb 1–Mar 22 | 800 / 91 | Apr 26–May 10 | 300 / 33 |
| 3 | Feb 1–Apr 12 | 1,100 / 135 | May 17–May 31 | 300 / 36 |

All dates are in 2015 at 06:00. Every fitting prefix ends more than 30 days
before its scoring block, conservatively exceeding the seven-day target horizon.
Each complete date keeps all 100 machines together. Scoring blocks are disjoint;
expanding fitting prefixes reuse earlier observations only once their outcome
windows are entirely in the past.

Same-machine overlap is intentional for future predictions on known machines.
It is not unseen-machine validation. Random row CV, shuffled validation splits,
test-set early stopping and validation-set hyperparameter tuning are not used.
RF bootstrap and XGBoost subsampling occur only inside their fitting data.

Three configurations per tree algorithm are fixed in `models.py` before test
evaluation. Mean **uncalibrated** development Brier chooses the configuration;
mean AP breaks exact Brier ties, then candidate index. LR's existing settings are
fixed and its development scores are descriptive, not retuned. All candidates
and fold scores are exported. `selection_plan.json` is written before tuning and
includes the factory/runner hashes.

After selection, each base pipeline is fitted on all 1,800 training cases.
**All 500 designated later cases are reserved for sigmoid calibration**, with
the base pipeline frozen. Hash checks verify that calibration cannot alter
training-fitted preprocessing or model state. No decision thresholds are selected
on those cases in this experiment. Test labels are used only for evaluation.

Selection deliberately does not use calibrated inner-fold loss: reserving the
whole later calibration set avoids recycling it for hyperparameters. Raw-Brier
selection may rank candidates differently from calibrated performance. The small
folds and fixed LR versus lightly tuned trees are limitations, not a universal
best-algorithm comparison.

## Availability and fixed settings

All four models ran successfully. **XGBoost 3.0.5 was already installed; no
packages were installed.** If XGBoost is absent or cannot import, the script
records its reason and runs the other algorithms without installation.

Selected configurations:

- LR: original L2 Logistic Regression, `C=1`, `max_iter=2000`, seed 42,
  no class balancing.
- RF: 300 trees, depth 6, minimum leaf 20, `max_features=0.7`, bootstrap,
  no class balancing, seed 42, one worker.
- HistGradientBoosting: 150 iterations, learning rate 0.05, 15 leaf nodes,
  minimum leaf 50, L2 regularization 3, seed 42, **early stopping disabled**.
- XGBoost: 150 trees, learning rate 0.05, depth 3, minimum child weight 10,
  lambda 5, row/column sampling 0.8, histogram method, binary logistic objective,
  log-loss metric, seed 42, one worker, **no evaluation set or early stopping**.

Every attempted configuration is in the fixed plan and development CSVs.
Full classifier parameters/versions are in `summary.json`; NaN library defaults
are explicitly marked rather than presented as fitted values. Reproduction uses
the factory source, not literal reapplication of display metadata.

## Actual calibrated test results

| Metric | LR | RF | HistGradientBoosting | XGBoost |
| --- | ---: | ---: | ---: | ---: |
| Brier | 0.091485 | 0.085914 | 0.081555 | 0.085714 |
| AP | 0.252146 | 0.330092 | 0.362913 | 0.346426 |
| Precision@10 | 0.284615 | 0.330769 | 0.376923 | 0.376923 |
| Recall@10 | 0.253425 | 0.294521 | 0.335616 | 0.335616 |
| Captured Top 10 positive windows | 37 | 43 | 49 | 49 |
| Log loss | 0.292293 | 0.267467 | 0.252327 | 0.273796 |
| ECE (10 bins) | 0.027172 | 0.048417 | 0.024965 | 0.043930 |
| Mean probability minus prevalence | +0.020894 | +0.011979 | +0.000087 | +0.010745 |

The reference reproduces Brier `0.09148540592410336`, AP
`0.2521463037932662`, and exactly the saved deployed probabilities. The constant
training-prevalence Brier baseline is `0.09973098290598291`; test prevalence is
`146/1300 = 0.1123076923`.

All trees improve Brier, AP and aggregate Top 10 capture over LR in this
exploratory test. HistGradientBoosting improves Brier by 0.009931 and AP by
0.110766, captures 12 additional positive Top 10 windows, and has lower log loss
and slightly lower ECE. It improves per-date capture on 9 dates, ties on 3 and
worsens on 1. XGBoost ties aggregate Top 10 capture, but has worse Brier, AP,
log loss and ECE than HistGradientBoosting.

Calibration remains a trade-off: RF and XGBoost have **worse** ten-bin ECE than LR.
Their own sigmoid calibration also worsens raw test Brier: RF 0.081987→0.085914,
XGBoost 0.081521→0.085714. HGB improves 0.083068→0.081555. Raw diagnostics are
labelled separately and are not mixed into the calibrated main comparison or
selected after observing test results.

The standard sklearn sigmoid protocol uses `decision_function` when available
(LR/HGB) and positive-class `predict_proba` otherwise (RF/XGBoost). Thus native
score responses differ even with the same calibration method. No alternate
calibration transformations or methods were searched after seeing test results.
Calibration uses only 59 positive cases, so its stability needs fresh assessment.

## Alert burden and overfitting

Each date ranks all 100 machines by unrounded probability descending, with
ascending machine ID for ties. Top 10 selects 130 machine-time windows across
13 dates; micro precision is captured/130 and recall is captured/146. These are
not unique physical failures or prevented breakdowns. Numerical per-date tables
retain selected/captured machine IDs.

| Per-date budget | LR FP | RF FP | HGB FP | XGBoost FP |
| --- | ---: | ---: | ---: | ---: |
| 10 | 93 | 87 | 81 | 81 |
| 20 | 198 | 176 | 177 | 168 |
| 30 | 301 | 275 | 275 | 276 |

Budget-specific rankings differ: XGBoost captures most at budget 20. No factory
inspection capacity or acceptable false-alarm cost has been validated.

Results have internal support but do not establish robustness outside this
benchmark. The selected raw development mean Briers are LR 0.097668, RF 0.080270,
HGB 0.082736 and XGBoost 0.083507. Development favored RF by raw Brier, whereas
calibrated test favored HGB; algorithm ordering is not identical across stages.

There are substantial training/test gaps: HGB training AP is 0.899200 versus
0.362913 on test; RF 0.609281 versus 0.330092; XGBoost 0.632643 versus 0.346426.
Such gaps indicate capacity/overfitting concerns, while temporal distribution
changes also contribute. In-sample and calibration-resubstitution metrics are
explicitly labelled; they are not independent validation.

Paired 500-resample machine-cluster diagnostics retain all snapshots per machine
and show conditional Brier/AP improvements over LR. They condition on fitted
scores, omit training/shared-date uncertainty and multiple-comparison adjustment,
and cannot establish independent significance. Repeated machines/history windows
and the repeatedly inspected test period limit confirmation.

**Prioritize HistGradientBoosting for genuinely fresh chronological validation;
keep LR deployed during research review.** The small search, same-machine cohort
and synthetic source do not establish a universal winner, factory capability,
reliability score, risk threshold or maintenance recommendation.

## Reproduce and test

Use the existing research environment. `requirements.txt` documents shared
dependencies plus plotting; XGBoost remains optional and is not automatically
installed. The recorded environment uses Python 3.13.2, sklearn 1.7.1 and
XGBoost 3.0.5.

Required local inputs: the four original machine/maintenance/failure/error CSVs,
`outputs_corrected/machine_prediction_examples.csv`, the chronological calibration
CSV and trusted 7-day artifact. The existing error-ablation calibration/reference
helpers and threshold ranking helpers are imported read-only; neither prior
experiment is executed or modified to generate these results. Missing inputs,
changed cohorts, inadequate development folds or reference-parity failure abort
the experiment.

From repository root, Windows CMD:

```bat
python -B research\component3\experimental_algorithm_comparison\run_comparison.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
set "C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -W error::DeprecationWarning -m unittest discover -s research\component3\experimental_algorithm_comparison -p "test_*.py" -v
```

There are 10 unit tests and four opt-in benchmark integration tests. The latter
require saved results and the environment variable; otherwise they explicitly
skip. They verify final fitted copies and probabilities without writing files.
The runner fits 30 development pipelines, four final base pipelines and four
validation sigmoids when all algorithms are available. No models are serialized;
the deployed artifact is loaded only for read-only probability parity.

Files:

- `models.py`: fixed algorithm configurations, purged date folds and training-only selection.
- `run_comparison.py`: original cohort verification, frozen calibration, evaluation and export.
- `test_comparison.py`: fold separation, whole-date grouping, feature parity,
  training-only preprocessing, frozen calibration, optional dependency behavior,
  early-stopping guards, deterministic fitting, score/ranking/result fidelity.
- `results/selection_plan.json`, `results/development_scores.csv`,
  `results/development_summary.csv`: complete search audit.
- `results/model_metrics.csv`, `results/calibration_bins.csv`,
  `results/split_fit_diagnostics.csv`, `results/matched_alert_budgets.csv`,
  `results/top10_by_date.csv`, `results/paired_differences.csv`,
  `results/date_consistency.csv`: numerical evaluation.
- `results/summary.json`, `results/report.md`: provenance, parameters and report.
- `results/precision_recall.*`, `results/calibration.*`: PNG/SVG figures.
- `outputs/heldout_predictions.csv`: full-precision detailed predictions, ignored
  by existing Git rules.

No existing research outputs, model artifacts, Function 2, backend or frontend
files are changed. No deployment, commit or push is performed.
