# Coherent multi-horizon v3 — completed exploratory evaluation

## Recommendation

Keep the deployed 7-day system unchanged. **Unpooled HGB is the strongest new
candidate for further independent validation**, not an established operational
replacement: it improves Brier at all horizons and 7-day AP/Top-10, but reduces
30-day AP and Top-10 capture. Anchored PAV is the simplest coherent alternative
if EXACT preservation of inclusive-reference 7-day predictions is required;
it slightly worsens 14-day loss/ranking. Neither is an unequivocal winner.
No backend, frontend, APIs, deployed artifact or Function 2 was changed.

## Cohort and faithful v2 reproduction

Same original machine-date keys: 1,800 train / 500 later calibration / 1,300 test,
100 machines, 18/5/13 dates. Inputs are the original 18 historical features;
no telemetry, display names or illustrative line metadata. Strictly past events
only. Train+30d < calibration start and calibration+30d < test start.
Intervals are (0,7], (7,14], (14,30]; events at the landmark are excluded.
Test positive windows: 167 / 332 / 631 (12.846% / 25.538% / 48.538%).

Feasibility checks rebuild the original panel and compare it with the saved panel
before replacing labels with v2 inclusive targets. Identical test keys and labels
are checked against v2 per-instance predictions. V2 pooled development folds,
selected model refits, saved artifact inference and independent calibrated LR
were reproduced without writing to v2. Fidelity differences are in summary.json.
The v2 independent reference has Briers 0.097660 / 0.182473 / 0.238296 and
144/1,300 ordering violations (141 P7>P14; 6 P14>P30; 3 both).
Do not substitute the old deployed evaluation's 0.091485 or 272 violations:
those use different, upper-exclusive labels. The deployed artifact is not refitted.

Administrative observation is assumed through exclusive 2016-01-01, not inferred
from the last event. Last evaluated 30-day window ends Dec 25. Known incomplete
negatives remain unlabelled; incomplete event-free intervals are omitted and
post-failure intervals never become negative training rows. Simultaneous component
failures represent one future machine event time. Recurrent snapshots restart a
next-recorded-failure prediction, not a physical lifetime/uptime survival clock.
Individual observation entry/exit, repair duration and logging coverage are unknown.

## Experiment A: projections, not learned calibration

A0 uses the reproduced inclusive calibrated LR triplets. A1 runs equal-weight PAV
on each triplet using scores alone. A2 fixes P7 and projects only its tail: if
P14>P30, pool their ORIGINAL mean, then floor both tail coordinates at P7.
Clipping inputs to P7 before pooling would solve the wrong optimization problem.
Both methods are deterministic and fit nothing to test labels or other cases.

For the closed convex set C={0<=x1<=x2<=x3<=1}, equal-weight PAV is the unique
Euclidean projection q=Proj_C(p). Optimality gives (p-q)·(y-q)<=0 for every y in C.
Expanding squared norms gives ||p-y||² >= ||p-q||²+||q-y||². Complete nested
binary outcome vectors 000,001,011,111 are in C, so TOTAL squared error across
the three horizons cannot increase for any complete case. Unequal per-horizon
loss weights would require the corresponding weighted projection/guarantee.
This is not a separate-horizon Brier guarantee: p=(.9,.8,.1), y=111 projects to
(.6,.6,.6), making the 7-day loss worse while reducing total loss.

Anchoring changes the feasible set to x1=p1, which generally excludes the true
binary outcome vector. It has NO unrestricted total-loss guarantee. For example,
p=(.9,.8,.1), y=000 becomes (.9,.9,.9), increasing total loss from 1.46 to 2.43.
Tests compare both projections with an independent constrained quadratic optimizer
and check 10,000 random triplets against all four nested binary labels.

Actual summed test squared error: 673.9572328681845 → 673.8016910326701.

## Main comparison: all calibrated models, inclusive labels

| method | horizon_days | brier | average_precision | roc_auc | ece_10_equal_width | precision_at_10 | recall_at_10 | top10_captured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent_lr | 7 | 0.097660 | 0.314204 | 0.797617 | 0.013915 | 0.338462 | 0.263473 | 44 |
| independent_lr | 14 | 0.182473 | 0.374546 | 0.641797 | 0.033136 | 0.446154 | 0.174699 | 58 |
| independent_lr | 30 | 0.238296 | 0.652423 | 0.661057 | 0.094297 | 0.753846 | 0.155309 | 98 |
| standard_pav | 7 | 0.097505 | 0.316102 | 0.799187 | 0.010570 | 0.346154 | 0.269461 | 45 |
| standard_pav | 14 | 0.182555 | 0.372958 | 0.639261 | 0.031950 | 0.438462 | 0.171687 | 57 |
| standard_pav | 30 | 0.238249 | 0.652429 | 0.661081 | 0.094209 | 0.753846 | 0.155309 | 98 |
| anchored_pav | 7 | 0.097660 | 0.314204 | 0.797617 | 0.013915 | 0.338462 | 0.263473 | 44 |
| anchored_pav | 14 | 0.182684 | 0.371895 | 0.636983 | 0.030675 | 0.438462 | 0.171687 | 57 |
| anchored_pav | 30 | 0.238136 | 0.652575 | 0.661313 | 0.093938 | 0.753846 | 0.155309 | 98 |
| pooled_lr | 7 | 0.109141 | 0.209866 | 0.649280 | 0.026889 | 0.246154 | 0.191617 | 32 |
| pooled_lr | 14 | 0.184291 | 0.371523 | 0.638501 | 0.057319 | 0.430769 | 0.168675 | 56 |
| pooled_lr | 30 | 0.246014 | 0.650130 | 0.660126 | 0.127526 | 0.738462 | 0.152139 | 96 |
| pooled_hgb | 7 | 0.107927 | 0.236575 | 0.667334 | 0.026164 | 0.269231 | 0.209581 | 35 |
| pooled_hgb | 14 | 0.181588 | 0.388543 | 0.649647 | 0.041671 | 0.476923 | 0.186747 | 62 |
| pooled_hgb | 30 | 0.235270 | 0.633514 | 0.645845 | 0.057587 | 0.730769 | 0.150555 | 95 |
| unpooled_lr | 7 | 0.097660 | 0.314204 | 0.797617 | 0.013915 | 0.338462 | 0.263473 | 44 |
| unpooled_lr | 14 | 0.188523 | 0.347895 | 0.614900 | 0.061833 | 0.384615 | 0.150602 | 50 |
| unpooled_lr | 30 | 0.247895 | 0.642325 | 0.648438 | 0.124404 | 0.738462 | 0.152139 | 96 |
| unpooled_hgb | 7 | 0.092593 | 0.354306 | 0.837982 | 0.017403 | 0.361538 | 0.281437 | 47 |
| unpooled_hgb | 14 | 0.179541 | 0.374760 | 0.657345 | 0.029885 | 0.453846 | 0.177711 | 59 |
| unpooled_hgb | 30 | 0.233211 | 0.631971 | 0.652205 | 0.030653 | 0.684615 | 0.141046 | 89 |

Raw unpooled models are additional diagnostics in comparison.csv, not selected
after test inspection. Raw unpooled HGB has the lowest observed 30-day Brier
among the raw/calibrated variants, but was not retrospectively chosen for serving.

Top-10 uses raw full-precision scores within each timestamp; equal scores break
ties by ascending machine ID. There are 130 flagged machine-date windows per
horizon. Precision@10=captured/130; Recall@10=captured/all positive windows.
Repeated windows may refer to the same underlying event; these are NOT unique
physical failures. No alert cost or factory inspection capacity is assumed.
Per-date capture is in top10_by_date.csv. Rankings do not change by lowering a
classification threshold; here changed rankings arise only from changed scores.

## Projection adjustment frequency and magnitude

Exact changed counts use nonzero full-precision deltas, not rounded UI values.
Mean absolute adjustment is over all 1,300 cases; magnitudes are probabilities.

| method | horizon | changed | changed_pct | mean_absolute | max_absolute | increases | decreases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard_pav | 7 | 141 | 10.846154 | 0.001275 | 0.059727 | 0 | 141 |
| standard_pav | 14 | 144 | 11.076923 | 0.001229 | 0.059727 | 141 | 3 |
| standard_pav | 30 | 9 | 0.692308 | 0.000088 | 0.025773 | 9 | 0 |
| anchored_pav | 7 | 0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |
| anchored_pav | 14 | 144 | 11.076923 | 0.002504 | 0.119455 | 141 | 3 |
| anchored_pav | 30 | 16 | 1.230769 | 0.000359 | 0.060871 | 16 | 0 |

Standard PAV changed 141 P7 values, all downward. It improved overall 7-day
Brier/AP and captured 45 instead of 44 positive windows in Top-10; this does not
mean no individual machine prediction/rank was harmed. Anchored PAV changed zero
P7 values and zero 7-day ranks, including exact preservation of Top-10 membership.
At 14 days both projections captured 57 instead of 58 positive windows; at 30 days
both captured the same 98 and preserved Top-10 membership/order.

Ranking changes below are aggregated over 13 dates, not unique machine IDs.
Changed rank positions counts machine-date positions, total shift sums absolute
rank differences, replacements counts new Top-10 members, and order positions
counts mismatched slots among the first 10. Full per-date rows are preserved.

| method | horizon | changed_rank_positions | total_absolute_rank_shift | top10_replacements | top10_order_positions_changed |
| --- | --- | --- | --- | --- | --- |
| anchored_pav | 7 | 0 | 0 | 0 | 0 |
| anchored_pav | 14 | 583 | 1510 | 12 | 72 |
| anchored_pav | 30 | 122 | 220 | 0 | 0 |
| standard_pav | 7 | 190 | 314 | 2 | 68 |
| standard_pav | 14 | 423 | 850 | 9 | 34 |
| standard_pav | 30 | 25 | 38 | 0 | 0 |

## Experiment B: interval-specific conditional risk sets

| split | interval | positives | negatives | cases |
| --- | --- | --- | --- | --- |
| train | 1 | 247 | 1553 | 1800 |
| train | 2 | 235 | 1318 | 1553 |
| train | 3 | 403 | 915 | 1318 |
| validation | 1 | 67 | 433 | 500 |
| validation | 2 | 63 | 370 | 433 |
| validation | 3 | 109 | 261 | 370 |
| test | 1 | 167 | 1133 | 1300 |
| test | 2 | 165 | 968 | 1133 |
| test | 3 | 299 | 669 | 968 |

Each interval fits a separate original-preprocessing LR (C=1) on only eligible
TRAIN landmarks. Its median imputer/scaler/category encoder never sees later
data. Frozen sigmoid calibration fits on that interval's eligible later calibration
cases, never test labels. H1 is exactly the inclusive independent LR 7-day task;
its scores match bit-for-bit. H2/H3 have different at-risk populations and their
own preprocessing/slopes, unlike the previous pooled model.

The optional HGB was feasible with 247/235/403 training interval positives and
67/63/109 calibration positives. The prespecified v2 three-setting conservative
grid was reused; parameters were selected using three whole-date, expanding
TRAIN-only folds with >30-day purging and mean RAW cumulative Brier across
all horizons. No internal random validation split (`early_stopping=False`), test
tuning or grid expansion occurred. Selected parameters:

`{"max_leaf_nodes": 7, "max_depth": 3, "min_samples_leaf": 60, "l2_regularization": 10.0, "learning_rate": 0.03, "max_iter": 150}`

All three hazards are evaluated from the same frozen historical feature vector
for every test case; actual future survival is NOT used to choose inference rows.
Cumulative risks use P1=H1; P2=P1+(1-P1)H2; P3=P2+(1-P2)H3. This equals the
requested survival-product formula, while avoiding cancellation that could change
P7 or cause tiny floating-point ordering violations when later hazards are zero.

All projected and hazard variants have ZERO violations, nonfinite or out-of-range
scores across all 3,600 landmarks. A0 alone remains inconsistent (144 test cases).
The unpooled models are standard conditional interval classifiers composed into
survival probabilities, not an elaborate jointly optimized survival network.

## Calibration, overfitting and uncertainty

Among the seven main rows, unpooled HGB has lowest Brier at all three horizons.
Standard PAV has lowest 7-day 10-bin ECE; unpooled HGB has lowest 14/30-day ECE.
Best AP/Top-10: unpooled HGB at 7 days, previous pooled HGB at 14 days,
anchored PAV AP at 30 days, and independent/PAV rows tie for 30-day Top-10.
Thus best Brier is not synonymous with best calibration or ranking.

Unpooled HGB captured 47/59/89 positive Top-10 windows vs independent LR 44/58/98.
Its 30-day ranking loss is meaningful for the dashboard despite better Brier/ECE.
At 130 flags, the corresponding false-positive windows are 83/71/41 vs 86/72/32.
Unpooled LR preserves 7-day performance but loses longer-horizon Brier/AP/ranking.

Training-versus-test unpooled HGB AP is .515224/.557605/.778985 versus
.354306/.374760/.631971; training Brier .087685/.163921/.198260 versus
.092593/.179541/.233211. These gaps warrant overfitting/shift concern, not claims
of robust generalization. Calibration-period scores are resubstitution for the
fitted calibrators and are not independent model-selection evidence.

Paired machine-cluster and date-cluster bootstrap intervals use FIXED predictions,
not model refits. Machine clusters preserve each machine's 13 snapshots; date
clusters preserve all 100 machines at a date. They do not jointly solve both
dependencies or capture fitting/model-selection uncertainty. Only 13 dates and
100 machines, overlapping windows and recurrent failures limit inferential strength.
ECE uses ten equal-width bins and can move with binning/small counts; calibration
curves are descriptive, not significance evidence. No confirmatory p-values.

| cluster | method | horizon | delta | low | high |
| --- | --- | --- | --- | --- | --- |
| machineID | standard_pav | 7 | -0.000156 | -0.000435 | 0.000078 |
| machineID | standard_pav | 14 | 0.000083 | -0.000133 | 0.000331 |
| machineID | standard_pav | 30 | -0.000046 | -0.000139 | 0.000000 |
| machineID | anchored_pav | 7 | 0.000000 | 0.000000 | 0.000000 |
| machineID | anchored_pav | 14 | 0.000212 | -0.000247 | 0.000745 |
| machineID | anchored_pav | 30 | -0.000160 | -0.000451 | 0.000032 |
| machineID | unpooled_lr | 7 | 0.000000 | 0.000000 | 0.000000 |
| machineID | unpooled_lr | 14 | 0.006050 | 0.003545 | 0.008559 |
| machineID | unpooled_lr | 30 | 0.009600 | 0.006339 | 0.013183 |
| machineID | unpooled_hgb | 7 | -0.005067 | -0.008493 | -0.002004 |
| machineID | unpooled_hgb | 14 | -0.002932 | -0.009181 | 0.002520 |
| machineID | unpooled_hgb | 30 | -0.005085 | -0.017716 | 0.006243 |
| prediction_time | standard_pav | 7 | -0.000156 | -0.000438 | 0.000106 |
| prediction_time | standard_pav | 14 | 0.000083 | -0.000108 | 0.000294 |
| prediction_time | standard_pav | 30 | -0.000046 | -0.000125 | 0.000025 |
| prediction_time | anchored_pav | 7 | 0.000000 | 0.000000 | 0.000000 |
| prediction_time | anchored_pav | 14 | 0.000212 | -0.000196 | 0.000672 |
| prediction_time | anchored_pav | 30 | -0.000160 | -0.000386 | 0.000039 |
| prediction_time | unpooled_lr | 7 | 0.000000 | 0.000000 | 0.000000 |
| prediction_time | unpooled_lr | 14 | 0.006050 | 0.003054 | 0.008709 |
| prediction_time | unpooled_lr | 30 | 0.009600 | 0.005742 | 0.013119 |
| prediction_time | unpooled_hgb | 7 | -0.005067 | -0.009518 | -0.000734 |
| prediction_time | unpooled_hgb | 14 | -0.002932 | -0.008450 | 0.002868 |
| prediction_time | unpooled_hgb | 30 | -0.005085 | -0.009623 | -0.000845 |

The 7-day unpooled HGB Brier improvement remains negative in both cluster
intervals. Fourteen-day intervals include zero; 30-day machine-cluster interval
includes zero even though the date-cluster interval does not. PAV improvements
are small and mostly have intervals including zero. These findings are exploratory
because this historical test period has been repeatedly inspected across several
experiments. No fresh independent holdout or factory validation was obtained.

## Integration decision

Unpooled HGB deserves further fresh-cohort validation and, after review, a possible
guarded experimental integration test. It is not yet a justified replacement,
especially where 30-day Top-10 ranking matters. Anchored PAV is a transparent
score-only fallback for coherent outputs with exact inclusive 7-day preservation,
but its 14-day trade-off must be explicit. PAV outputs must be labelled post-processed,
not naturally coherent survival estimates or outcome-fitted isotonic calibration.

Retain the deployed 7-day LR. Its older target semantics and artifact remain
unchanged; none of these inclusive research probabilities was substituted into
the existing application. More experiments on this same test set cannot establish
an independently validated winner. Need fresh observation coverage/repair semantics,
new dates/factories, predeclared selection and calibration, and validated alert costs.
No serving/UI changes, dependency installs, commits or pushes were performed.

See ../README.md for exact reproduction and test commands. Summary/source hashes,
per-instance predictions, per-date ranking changes and raw/calibrated diagnostics
are saved alongside this report. All artifacts are research-only and local.
