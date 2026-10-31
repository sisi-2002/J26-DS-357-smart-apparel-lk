# Experimental coherent multi-horizon hazard baseline

**Synthetic Azure industrial-machine benchmark research only. This is not a
validated apparel-factory model and is not connected to FastAPI or React.**

The experiment guarantees ordered cumulative probabilities, but did **not**
improve Brier scores over the existing independently fitted, chronologically
sigmoid-calibrated Logistic Regression models. Keep the deployed 7-day artifact
and PP1 UI unchanged.

## Feasibility and target

The existing benchmark has machine IDs, dated failures and historical features,
so it supports a landmark **time to next recorded failure** formulation. Each
original machine-time snapshot starts a new prediction task. Recurrent failures
can therefore contribute to different landmarks; this is not a model of first
lifetime failure, confirmed operating uptime or component-specific survival.
Simultaneous failure records are one event time for this target; historical
feature counts retain the original research implementation unchanged.

The required administrative observation end is exclusive. The run below assumes
all 100 benchmark machines are observed throughout 2015, ending at
`2016-01-01 00:00:00`. Individual entry/exit, downtime and repair-duration records
do not establish that coverage independently. The last recorded failure is
**never** used as a censoring date. The latest test outcome window ends on
`2015-12-25 06:00:00`, before the declared end. Evaluation aborts if any original
landmark lacks full 30-day follow-up; no held-out cases are silently dropped.

The original labels use **strict** future bounds: `t < failure < t + horizon`.
To preserve those labels exactly, interval membership is:

| Interval | Event delay in days | Event exactly at upper boundary |
| --- | --- | --- |
| 1 | `(0, 7)` | Day 7 belongs to interval 2 |
| 2 | `[7, 14)` | Day 14 belongs to interval 3 |
| 3 | `[14, 30)` | Day 30 is outside the prediction window |

Events at prediction time are excluded. These boundary choices matter: among
test landmarks, the first future failure occurs exactly at days 7, 14 and 30 in
21, 28 and 19 cases respectively. The code verifies all reconstructed labels
against the saved original labels, including nesting `Y7 <= Y14 <= Y30`.

There is a pre-existing documentation inconsistency: the original
`outputs_corrected/run_summary.json` describes the lower target bound as `>= t`,
but `build_panel()` uses `searchsorted(..., side="right")`, implementing `> t`.
This experiment follows the executable code and verified saved labels; it does
not alter that existing metadata file.

## Model and leakage controls

`hazard_model.py` implements one pooled L2 Logistic Regression (`C=1`,
`max_iter=2000`, seed 42, no class balancing), with shared feature slopes and
categorical interval intercepts. The original 17 numeric temporal features and
machine-model category are reused. Features are frozen at the landmark and use
events strictly before that time; future failures only determine training
outcomes and at-risk eligibility.

Training expands each landmark into person-period records until its first
future event or censoring. No record follows an event. An incomplete event-free
interval is omitted rather than labelled negative; a recorded event contributes
its interval even if follow-up ends within it. All cases in the actual comparison
have complete 30-day observation. Supporting partial censoring in the utility
does not validate an informative-censoring analysis.

The original imputation, scaling and categorical preprocessing are fitted on
the **1,800 unique training landmarks**, before period expansion. Model
parameters are fitted on their 4,740 eligible person-period records. Neither
validation nor test covariates fit preprocessing. Inference evaluates all three
conditional hazards for every landmark without reading its future outcome or
filtering prediction rows by its observed survival.

The primary model uses raw conditional hazards. A fixed secondary experiment
fits **one pooled sigmoid on conditional hazards** using only the later
validation risk sets and a frozen training classifier. It does not calibrate
cumulative probabilities independently. The original preprocessor and base
classifier are checked to remain unchanged during calibration. Both variants
compose hazards as:

```text
P7  = h1
P14 = 1 - (1-h1)(1-h2)
P30 = 1 - (1-h1)(1-h2)(1-h3)
```

Consequently `0 <= P7 <= P14 <= P30 <= 1` is structural. The 16-day third
interval has its own intercept; these are interval probabilities, not daily
hazard rates or continuous-time hazard ratios. This is a pooled discrete-time
logistic baseline, not an elaborate neural survival model. Settings were fixed
before this evaluation; no test-driven search, threshold selection or model
replacement was performed.

## Original chronological cohort

| Split | Landmarks | Snapshots x machines | Eligible person-periods | Interval 1 events / at risk | Interval 2 events / at risk | Interval 3 events / at risk |
| --- | ---: | --- | ---: | --- | --- | --- |
| Training | 1,800 | 18 x 100 | 4,740 | 213 / 1,800 | 234 / 1,587 | 409 / 1,353 |
| Validation / calibration | 500 | 5 x 100 | 1,320 | 59 / 500 | 62 / 441 | 112 / 379 |
| Test | 1,300 | 13 x 100 | 3,450 | 146 / 1,300 | 158 / 1,154 | 308 / 996 |

Train: February 1 to May 31, validation: July 2 to July 30, test: September 2 to
November 25, 2015; all snapshots are at 06:00. The original 30-day separation is
checked between splits. The 3,450 test person-period rows are descriptive only:
all predictions and horizon metrics use the same **1,300 machine-time keys**.
Train, validation and test contain the same machines. Overlapping landmark
windows and recurrent failures create dependent observations, so the pooled
likelihood is a working composite likelihood, not justification for iid
coefficient inference or claims about previously unseen machines.

## Reproduction and inputs

Use the existing research Python environment and dependencies in
`../requirements.txt` (the recorded run uses Python 3.13.2, NumPy 2.2.6,
pandas 2.3.2 and scikit-learn 1.7.1). No model download or retraining of the
deployed artifact is required. The experiment intentionally fits new models
in memory and does not serialize them.

Required local inputs, read without alteration:

- `PdM_machines.csv`, `PdM_maint.csv`, `PdM_failures.csv`, `PdM_errors.csv` in the
  supplied raw-data directory.
- `../outputs_corrected/machine_prediction_examples.csv` to verify all original
  feature rows, labels and splits.
- `../outputs_corrected/heldout_predictions_with_evidence.csv` to verify saved
  uncalibrated per-instance probabilities and test keys.
- `../outputs_calibration/calibration_brier_comparison.csv` for the original
  chronological calibration metrics. The stratified CV3 diagnostic is not used.

The original outputs and raw data may be Git-ignored; regeneration requires
those existing local files. Missing or inconsistent inputs abort the run. The
small saved summaries in `results/` remain readable without those local inputs.

From the repository root, Windows CMD:

```bat
python -B research\component3\experimental_multihorizon\train_evaluate.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --observation-end "2016-01-01 00:00:00"
set "C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -W error::DeprecationWarning -m unittest discover -s research\component3\experimental_multihorizon -p "test_*.py" -v
```

The evaluator reuses the original feature builder and baseline factory. It fits
one experimental pooled model and one validation-only hazard sigmoid, plus
three independent Logistic Regression reference models and their three
validation-only sigmoids. Reference raw predictions must match existing saved
probabilities within `1e-10`; reference Brier and AP must match the saved
chronological results at their six-decimal precision. The saved deployed 7-day
model is neither loaded for fitting nor altered. It is distinct from these
reproduced evaluation models.

## Observed results

These values are rounded for reading; full precision is in `results/metrics.csv`.
Lower Brier is better. The prevalence baseline uses **training** label prevalence,
not test prevalence. AP means average precision, not trapezoidal PR area.

| Horizon | Train-prevalence Brier | Independent raw Brier | Independent sigmoid Brier | Pooled hazard Brier | Pooled hazard + sigmoid Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 days | 0.099731 | 0.092460 | 0.091485 | 0.102455 | 0.098464 |
| 14 days | 0.179372 | 0.181707 | 0.172319 | 0.189545 | 0.174108 |
| 30 days | 0.249168 | 0.292497 | 0.239737 | 0.290512 | 0.245737 |

| Model | Horizon | AP | Precision at 0.5 | Recall at 0.5 | ECE (10 equal-width bins) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled hazard | 7 | 0.166301 | 0.000000* | 0.000000 | 0.067566 |
| Pooled hazard | 14 | 0.336726 | 0.500000 | 0.006579 | 0.133266 |
| Pooled hazard | 30 | 0.629156 | 0.732558 | 0.102941 | 0.243287 |
| Pooled hazard + sigmoid | 7 | 0.166301 | 0.000000* | 0.000000 | 0.022454 |
| Pooled hazard + sigmoid | 14 | 0.336726 | 0.300000 | 0.009868 | 0.045816 |
| Pooled hazard + sigmoid | 30 | 0.629156 | 0.706731 | 0.240196 | 0.120950 |

*No positive predictions at this threshold; precision is undefined and stored as
zero by `zero_division=0`, with predicted-positive counts included in the CSV.
The common `p >= 0.5` diagnostic threshold was fixed, not optimized on test data.
It is not a maintenance-action recommendation. Existing reports that optimize
thresholds on validation data have a different precision/recall setup.

| Model | P7 > P14 | P14 > P30 | Either / 1,300 | Percentage |
| --- | ---: | ---: | ---: | ---: |
| Independent raw | 620 | 146 | 660 | 50.77% |
| Independent sigmoid | 267 | 14 | 272 | 20.92% |
| Pooled hazard | 0 | 0 | 0 | 0.00% |
| Pooled hazard + sigmoid | 0 | 0 | 0 | 0.00% |

The coherent baseline sacrifices probability accuracy in this comparison.
Validation-only sigmoid improves its Brier score at every horizon and beats the
training-prevalence baseline, but remains worse than the independent sigmoid
models by 0.006978, 0.001789 and 0.006000 respectively. AP is lower at 7 and 14
days and only slightly higher at 30 days (0.629156 versus 0.628533). Zero ordering
violations do not establish an overall performance improvement.

Calibration diagnostics include mean predicted probability, observed prevalence,
their difference, log loss, ECE and the full reliability-bin table. Bins are
left-closed/right-open except the final bin includes 1; empty bins have missing
means rather than fabricated observations. ECE is bin-dependent and Brier also
reflects discrimination, so neither alone proves calibration. Conditional-hazard
calibration does not guarantee perfect cumulative calibration.

Paired machine-cluster bootstrap differences use 500 seed-42 resamples, retaining
all snapshots for each sampled machine. These conditional-on-fitted-model
intervals account for within-machine grouping but exclude fitting uncertainty,
shared calendar effects and multiplicity adjustments. For the sigmoid hazard
model, Brier-delta intervals versus independent sigmoid are approximately
`[0.002614, 0.010947]`, `[-0.000858, 0.004370]`,
`[0.002613, 0.009441]`. The 14-day interval includes zero; these exploratory
diagnostics are not a claim of universally inferior performance or formal
deployment validation.

## Files, tests and limitations

- `hazard_model.py`: reusable construction, fitting and feature-only inference.
- `train_evaluate.py`: feasibility checks, frozen comparison, diagnostics and export.
- `test_hazard_model.py`: 12 unit tests plus 4 raw-benchmark integration tests.
  Without `C3_EXPERIMENT_DATA_DIR`, integration tests explicitly skip.
- `results/summary.json`: cohort, settings, metrics, ordering, versions, fit counts
  and SHA-256 source provenance; no model object or raw dataset.
- `results/metrics.csv`, `results/calibration_bins.csv`,
  `results/paired_brier_deltas.csv`: compact, reviewable numerical outputs.
- `outputs/heldout_predictions.csv`: full-precision, 1,300-case predictions and
  labels, ignored by the existing Git rules. No serialized model artifacts.

Tests cover strict event boundaries, censoring and first-event risk sets;
training-only preprocessing; frozen validation calibration; feature selection
that ignores future labels; single/batch and original inference feature parity;
invariance to events added at/after prediction time; probability bounds and
ordering; deterministic fitting; complete follow-up; and saved-result fidelity.

The same test period has already informed prior research, so it is not untouched
external validation. Shared slopes may underfit interval-specific effects.
Physical uptime is unverified, and performance on real apparel machines is
unknown. Further work could predefine interval interactions or alternative
coherent models, compare with grouped temporal validation, examine cumulative
calibration and validate on new data. No such tuning or correction is implemented
here. This module is suitable as an experimental research baseline, **not yet
as a justified replacement for the deployed model**.

The Component 3 proposal's survival-analysis/evaluation discussion motivates the
comparison, but does not itself validate this target or benchmark assumptions.
For the general person-period likelihood and survival-product construction, see
[Suresh et al., Survival prediction models: an introduction to discrete-time modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC9316420/).
The present strict-upper-bound convention intentionally follows this repository's
labels rather than changing them to the article's interval convention.
