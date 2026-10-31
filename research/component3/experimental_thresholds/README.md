# Exploratory calibrated-model threshold and Top 10 study

**Research only: synthetic Azure industrial-machine benchmark, not validated
apparel-factory data. No thresholds or model changes are integrated into the
backend, frontend or deployed 7-day artifact.**

Read [the generated numerical report](results/report.md),
[precision–recall curves](results/precision_recall.png) and
[recall versus alert burden](results/recall_alert_burden.png).
The report and tables are generated directly from verified scores, without
manually copied result values.

## Protocol and independence

The existing 1,800 training, 500 July calibration and 1,300 September–November
test machine-time cases are reused. All models use the existing strictly
historical features, strict-future labels and full 30-day separation between
splits. The prior hazard experiment's explicit 2015 administrative observation
assumption and recurrent-landmark limitations still apply.

**All 500 validation cases fitted the sigmoid calibrators. There is no separate
threshold-selection holdout in this cohort.** The five validation snapshots span
28 days, so they cannot be partitioned into calibration and a later threshold
set with the existing full 30-day target separation. This experiment does not
silently refit on a different split or claim independent threshold validation.

Thresholds selected on those reused validation labels are explicitly
**exploratory candidates**. They are then applied unchanged to test scores.
No test labels select candidates. The full test sweeps and PR curves are
retrospective descriptions, not instructions to select the best-looking test
threshold. Because the test period has been inspected repeatedly, even candidate
test performance is not untouched final validation.

The study compares only:

1. Independent chronological sigmoid-calibrated Logistic Regression at 7/14/30
   days, using the original baseline factory and settings.
2. Pooled hazard + one validation-person-period sigmoid, composed into coherent
   7/14/30-day cumulative probabilities with the unchanged experimental settings.

There is no uncalibrated/CV3 mixture. Missing validation probabilities require
fitting evaluation copies **in memory**: three independent base models, three
validation sigmoids, one pooled hazard model and its validation sigmoid per run.
No models are serialized. All reproduced test probabilities must match the prior
saved experiment within `1e-10`; Brier and AP must match within `1e-12`.
The deployed 7-day artifact is loaded only to check prediction parity. It is
never fitted, recalibrated, replaced or written.

## Threshold definitions

- A case is flagged when `probability >= threshold`; score ties are included.
- Sweeps include every unique score, a 0.01 grid, extra low-threshold diagnostic
  points, 0 and 1. Test and validation sweeps are labelled separately.
- Precision is `TP/(TP+FP)` and recall is `TP/(TP+FN)`.
- F1 is `2*TP/(2*TP+FP+FN)`; F2 is `5*TP/(5*TP+4*FN+FP)`.
- Alert burden is flagged machine-time cases divided by all evaluated cases.
- False positives per 100 cases is `100*FP/N`, **not** `FP/(FP+TN)`.
- Precision is blank/null when nothing is flagged; recall is blank/null with no
  positive labels. F scores use zero for a zero denominator. Undefined precision
  is not evidence of perfect classification.

Five candidates per model/horizon maximize validation F1, maximize validation
F2, or target validation recall of 50%, 70% or 80%. For a recall target, select
the highest threshold that reaches it, minimizing alerts. Objective ties also
choose the highest threshold. Targets are always attainable here by sufficiently
lowering the threshold; the measured alert/false-positive burden must accompany
that statement. The utility explicitly reports targets unattainable when no
positive selection labels exist. Validation recall targets are not guarantees
about test recall.

These objectives do not encode validated factory costs. Flagging every case
gives 100% recall and precision equal to prevalence. Nothing in this study defines
an operationally acceptable number of alerts or recommends maintenance actions.

Threshold changes leave scores, AP, Brier and probability ordering unchanged.
The reported PR-AUC statistic is **average precision**, not trapezoidal PR area.
Charts use empirical steps without smoothing. PR terminal precision 1 at recall
0 is a plotting convention, not a usable perfect-precision operating point.
Even coherent probabilities can produce non-nested binary decisions if different
thresholds are chosen for different horizons.

## Top 10 definitions

At each of the 13 test timestamps, rank all 100 machines by unrounded score,
descending, with ascending machine ID as a deterministic tie-break. Select
exactly 10 machines each time, matching the existing dashboard workflow.

- Captured: number of selected machines whose binary outcome window is positive.
- Micro Precision@10: total captured / 130 selected machine-time windows.
- Micro Recall@10: total captured / all positive machine-time windows.
- Macro Recall@10: average per-date recall; dates without positives are excluded
  and their number is reported.
- Uniform-random baseline: exact expectation for selecting 10 of 100 uniformly
  without replacement each date, with no Monte Carlo variability.
- Historical heuristic baseline: rank the existing past-90-day failure count,
  breaking ties by machine ID. A bounded monotone count transform is used only
  to reuse ranking validation; it is not claimed to be a probability model.

Repeated snapshots can label the same physical failure in multiple windows.
These are **positive machine-time windows**, not unique failures, unique alerted
machines, repairs, prevented breakdowns or independent observations. No alert
suppression, inspection scheduling or operational benefit is simulated. Ranking
quality does not change merely because a classification threshold changes.

## Reproduce

Use the existing research Python environment. The installed environment already
has the required packages; `requirements.txt` documents the extra plotting
dependency without changing the shared requirements.

Required local inputs:

- The four `PdM_*.csv` files in the supplied raw-data directory.
- Existing `outputs_corrected/machine_prediction_examples.csv`.
- Existing `experimental_multihorizon/outputs/heldout_predictions.csv` and
  `experimental_multihorizon/results/summary.json`.
- Existing `models/c3_7day_sigmoid.joblib` for read-only dashboard parity.
- Existing chronological calibration CSV and original source files for provenance.

Missing or inconsistent inputs fail explicitly. No regeneration of old outputs
is performed. Small results are readable from this directory without raw inputs;
full reproduction requires the local, Git-ignored research inputs.

Windows CMD, from the repository root:

```bat
python -B research\component3\experimental_thresholds\evaluate_thresholds.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
set "C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -W error::DeprecationWarning -m unittest discover -s research\component3\experimental_thresholds -p "test_*.py" -v
```

Without the data environment variable, the 12 pure tests run and the three
real-data integration tests explicitly skip. Integration tests reproduce model
copies but never modify existing files. Generate this study's results before
running its complete integration suite.

## Files

- `threshold_metrics.py`: pure metrics, candidate selection, chronology guards
  and per-date ranking calculations.
- `evaluate_thresholds.py`: verified model reproduction, comparison, plots,
  provenance and generated report. Writes only inside this new directory.
- `test_thresholds.py`: numerical agreement with scikit-learn, exact ties,
  missing/invalid cases, metric denominators, chronological separation, test-label
  perturbation checks for threshold independence, actual score/result fidelity.
- `results/report.md`, `results/summary.json`: readable report and full metadata.
- `results/model_metrics.csv`, `results/diagnostic_thresholds.csv`,
  `results/validation_candidates.csv`: exact scores and threshold comparison
  tables, including all confusion counts, F1/F2 and burden.
- `results/top10_by_date.csv`, `results/top10_summary.csv`: all 13 dates, three
  horizons and both models plus the historical heuristic; includes selected and
  captured machine IDs and exact random-baseline expectations.
- `results/precision_recall.*`, `results/recall_alert_burden.*`: PNG/SVG figures.
- `outputs/full_threshold_sweeps.csv`, `outputs/pr_curve_points.csv`,
  `outputs/validation_predictions.csv`: detailed local exports, ignored by the
  existing Git rules. No raw datasets or model artifacts are copied here.

Before deployment, obtain a genuinely new chronological threshold-selection
period separate from calibration, then a genuinely fresh holdout; establish
factory inspection capacity and missed-failure/false-alarm costs, and validate
on real factory data. These results do not establish practical usefulness merely
by raising recall or guaranteeing probability ordering.
