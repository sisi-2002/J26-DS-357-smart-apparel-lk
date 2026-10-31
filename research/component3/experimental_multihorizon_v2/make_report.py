"""Render saved results without training or modifying experiment inputs."""
import argparse
import json
from pathlib import Path
import pandas as pd

def table(frame):
    def value(v): return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+ '\n'.join(
        '| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def report(folder):
    summary=json.loads((folder/'summary.json').read_text())
    metrics=pd.read_csv(folder/'comparison.csv')
    # Narrative below describes this completed fixed experiment. Fail rather than
    # attach stale conclusions to a later changed dataset/model protocol.
    expected={'independent_sigmoid':[.097660,.182473,.238296],
              'lr_sigmoid':[.109141,.184291,.246014],
              'hgb_sigmoid':[.107927,.181588,.235270]}
    for method, values in expected.items():
        actual=metrics[metrics.method.eq(method)].sort_values('horizon_days').brier.tolist()
        if len(actual)!=3 or any(abs(a-b)>0.0000006 for a,b in zip(actual,values)):
            raise ValueError('Results differ from the verified fixed experiment; review narrative before exporting')
    selected=metrics[metrics.method.isin(['independent_sigmoid','lr_sigmoid','hgb_sigmoid','hgb_raw','train_prevalence'])]
    columns=['method','horizon_days','cases','positive_cases','brier','average_precision','roc_auc',
             'precision_at_10','recall_at_10','top10_captured','ece_10_equal_width']
    text='''# Multi-horizon v2: completed exploratory comparison

## Decision

**Do not integrate either pooled candidate. Keep the existing deployed 7-day LR unchanged.**
The boosted hazard is the stronger coherent candidate, but loses substantial 7-day
probability quality, discrimination and Top-10 performance against the same-label
independent LR. Its modest longer-horizon Brier gains do not establish a robust
overall improvement. No backend, frontend or experimental feature flag was added.
Research joblib bundles are local, versioned by run directory, Git-ignored and not
deployment approvals. Random Survival Forest was not run: scikit-survival is absent.
No dependencies were installed. Optional monotonic post-processing was not run.

## Critical target-definition distinction

Existing executable labels are `t < failure < t+h`, despite a legacy summary saying
`>= t`. Required v2 labels are `t < failure <= t+h`: (0,7], (7,14], (14,30].
The same 1,300 test machine-date keys now have 167 / 332 / 631 positive windows,
versus 146 / 304 / 612 previously: 21 / 28 / 19 negatives change to positives.
Consequently, **old numerical scores are not direct comparators for v2**.
The table below uses freshly fitted independent LR with exactly the v2 targets,
features and cohorts. No deployed model was refitted or overwritten.

Legacy chronological calibration was reproduced before changing labels and matched
saved metrics: Brier 0.091485405924 / 0.172318573489 / 0.239736680361 and
272/1,300 ordering violations. Its CV3 diagnostic variant was not used.
Prior pooled-sigmoid Briers 0.098463842971 / 0.174107823673 / 0.245736716815
were verified in previous saved results, not rerun as v2 models.
The separate classification HGB result 0.081554901476 is verified in the saved
algorithm comparison; it uses the OLD 7-day labels and a different objective.
It does not imply a boosted shared hazard must perform better.

## Data and scientific feasibility

100 machines; 3,286 maintenance rows; 761 component-failure rows; 3,919 error rows.
Raw fields have no missing values or exact duplicate rows. Categories are generic
comp1–comp4 / error1–error5, not validated apparel or sewing-machine fault types.
Component failure timestamps collapse to the first distinct future machine event;
simultaneous component failures form one outcome time. Original historical event
COUNTS retain original definitions. Every new landmark restarts the next-event task;
there is no known downtime/repair/uptime risk set, so this is next RECORDED failure,
not physical lifetime survival. Features are frozen at the landmark.

The four CSVs lack machine entry/exit or independent logging-coverage information.
Administrative follow-up assumes all benchmark machines observed until exclusive
2016-01-01 00:00. It is not inferred from the last failure. All evaluated 30-day
windows end by 2015-12-25 06:00. Unknown censored negatives remain missing;
incomplete event-free intervals are excluded, and observed partial-interval events
are retained. An exclusive censoring end exactly at day h does NOT establish the
inclusive day-h outcome. No evaluated cases were dropped or silently imputed.

## Chronology and models

Train: 1,800 landmarks / 18 dates, Feb 1–May 31. Calibration: 500 / 5 dates,
July 2–July 30. Test: 1,300 / 13 dates, Sep 2–Nov 25; all 100 machines per date.
Train+30d < calibration start; calibration+30d < test start.
V2 person-period rows: 4,671 train, 1,303 calibration, 3,401 test (test risk sets
are not used for inference or fitting). First-event totals: 885 / 239 / 631.
At-risk membership ends at the first future event; post-failure intervals never
become event-free training records.

The original 18 features include maintenance/failure/error history and static
machine age/model, using only events strictly before t. No telemetry, illustrative
production lines, display names, future labels or future events enter predictors.
Machine age is reused as published static metadata, not reconstructed operating age.
Preprocessing is fitted once per unique fitting landmark, before risk-set expansion.

A: pooled L2 Logistic Regression, shared slopes + interval dummies and interval
duration / 30. B: one pooled HistGradientBoosting classifier with the same inputs.
This is a working discrete-time composite likelihood over repeated landmarks,
not a neural joint survival model or a continuous daily hazard-rate model.
Durations are interval probabilities, with interval 3 lasting 16 days.

Conditional hazards compose as P=1-cumprod(1-h), ensuring valid cumulative ordering.
One pooled sigmoid is fitted to conditional hazard risk sets on later calibration
data with the entire training state frozen. No cumulative horizon is calibrated
independently. Sigmoid improves some scores and worsens others; both raw and
calibrated versions are reported rather than selecting retrospectively on test.

HGB selection: three prespecified conservative settings, three expanding
training-only whole-date folds, 30-day purge; 300 scored cases per fold and
500/800/1,100 fitting cases. Same-machine recurrence across periods is intentional
for future assessments of existing machines, not evidence for new-machine transfer.
The objective is mean RAW cumulative Brier across 3 horizons and 3 folds.
No random internal holdout (`early_stopping=False`), test tuning, feature search,
class balancing or threshold optimization. Fixed seed 42. Plan saved before fits.
'''
    text+='\nSelected parameters: `'+json.dumps(summary['selected_hgb_parameters'])+'`.\n\n'
    text+='## Same-cohort inclusive-label test metrics\n\n'+table(selected[columns])+'\n\n'
    text+='''Top-10 selects 10 per date (130 total flags), with full-precision descending
probabilities and ascending machine-ID ties. Precision@10=captured/130;
Recall@10=captured/positive windows, not a count of unique physical failures.
All 13 dates are in top10_by_date.csv. Random ranking expects 16.7 / 33.2 / 63.1
positive windows at 7/14/30 days. Constant-score prevalence Top-10 uses deterministic
ID ties and is NOT the random-ranking expectation.

The sigmoid HGB captures 35/62/95 positive windows vs LR 44/58/98. At the fixed
130-flag budget, false positives are therefore 95/68/35 versus 86/72/32.
No factory inspection capacity or operational threshold has been validated.
AP, Brier and ranking are not improved merely by lowering a decision threshold.
Precision/recall at 0.5 are included only as diagnostics in comparison.csv.

ECE uses ten equal-width bins; sparse bins, 100 repeated machines and only 13
dates limit interpretation. Calibration plots are descriptive, not significance
tests. Brier is overall probability loss, not pure calibration error.

## Ordering, numerical checks and stability

All 3,600 landmarks across splits for BOTH raw/calibrated hazard candidates satisfy
0 <= P7 <= P14 <= P30 <= 1; zero nonfinite/out-of-range predictions or dropped
evaluation cases. Fresh inclusive-label independent LR has 144/1,300 violations;
the OLD-label calibrated LR's 272 violations are a separate comparison.

The calibrated HGB 7-day Brier deterioration is +0.010267; 14-day change -0.000885;
30-day change -0.003026. Machine-cluster and date-cluster fixed-prediction bootstrap
intervals both span zero for the longer-horizon differences. Machine-cluster 95%
intervals are +[0.006856,0.013555], [-0.007465,0.005690], [-0.014396,0.008671].
These are exploratory uncertainty summaries, not confirmatory significance claims;
they omit refitting uncertainty, temporal dependence beyond clusters and multiplicity.

Overfitting/generalization is a concern: HGB+sigmoid training AP 0.323810/0.561997/
0.778728 falls to 0.236575/0.388543/0.633514 on test. Its 30-day training Brier
0.196664 becomes 0.235270 on test. Feature-history count/span growth creates an
expected temporal distribution shift; feature_shift.csv records split means,
medians and missingness. No adjustment was selected after seeing these test results.

There are sufficient observations to fit a small baseline and detect these large
descriptive trade-offs, but positive windows are dependent and not unique failures.
Five calibration dates and thirteen test dates cannot establish robust operational
validity. The repeatedly inspected test period is exploratory, not untouched or
external validation. No genuinely independent fresh period was used.

Best Brier: independent LR at 7 days; RAW HGB hazard at 14 and 30 days.
Best AP: independent LR at 7 and 30 days; raw HGB hazard at 14 days.
Best Top-10: independent LR at 7 and 30 days; HGB hazard at 14 days.
Thus no overall winner meeting both requested objectives was established.

## Artifacts, reproduction and remaining work

Research model loading was round-trip checked with exact repeated probabilities;
bundle schema/features are checked by load_research_model(). Artifacts depend on
the existing research modules and matching sklearn runtime; they are not a
self-contained production serving contract. runtime.csv reports measured fit/
calibration/evaluation elapsed time and one 1,300-case batch inference timing;
serialized_bytes there is a pickle estimate, actual joblib sizes are below.
Model sizes and timings are measurements of this local run, not service SLAs.
'''
    text+='\n'+table(pd.DataFrame(summary['saved_artifacts']))+'\n\n'
    text+='''Prior proposal sections 3.7 and 3.11–3.12 support survival only with defensible
event/censoring structure and chronological rare-event/calibration evaluation;
they do not establish benchmark-to-apparel validity. This experiment is that
limited recorded-event baseline, not completion of all proposed survival analyses.

Before guarded integration: agree horizon boundary contracts; obtain genuinely
fresh data with entry/exit, logging coverage, repair and uptime semantics; evaluate
new factories/machines and temporal shifts; pre-register fresh model-selection and
calibration validation; establish alert costs/capacity and monitoring. Future
model-design changes need a new development protocol, not repeated test-driven
grid expansion. Keep the current calibrated 7-day application as-is.

See ../README.md for exact Windows CMD commands, tests and output inventory.
'''
    (folder/'report.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);report(p.parse_args().run_dir)
