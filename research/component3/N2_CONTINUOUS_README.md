# Component 3 / N2: Continuous Evidence–Prediction-Loss Analysis (PP1 pilot)

This adds a **separate** reproducible experiment to your already-committed C3 scripts. Do **not** replace `build_and_evaluate.py` or `n2_history_pilot.py`.

## Research question

For the 7-day held-out machine-time predictions, how are the maintenance-evidence dimensions associated with observed squared prediction error, **without assuming an arbitrary sufficiency threshold**?

- **H** — days since earliest *observed* maintenance event within the permitted history window; this is not confirmed CMMS logging coverage.
- **M** — prior recorded maintenance-event count in that window.
- **F** — prior recorded failures in that window.
- **C** — completeness of source-published fields, 100% for the audited Azure dataset; **constant, so excluded from relationship fitting**.
- **R** — days since latest recorded maintenance event in that window (missing if no event).
- **Machine model/type** — included as a categorical *control*, not a sufficiency dimension.

**Outcome:** Individual held-out 7-day prediction squared loss `(y - predicted_probability)**2` from the **original uncalibrated** N2 pilot models. Your separate sigmoid-calibration experiment is NOT mixed into this analysis.

### Why the join is condition-specific

The existing N2 pilot provides losses for three separately retrained classifiers: **Full**, **180 days**, and **90 days**. This script rebuilds the exact historical evidence variables separately for every condition, matches them using `machineID` + `prediction_time`, and verifies strict-future labels. A 90-day loss is **never** paired with full-history evidence.

## Add these files to the existing repo

```text
J26-DS-357-smart-apparel-lk/
  research/component3/
    scripts/
      build_and_evaluate.py                <-- already committed; leave unchanged
      n2_history_pilot.py                  <-- already committed; leave unchanged
      n2_continuous_analysis.py            <-- NEW
      test_n2_continuous_analysis.py       <-- NEW
    N2_CONTINUOUS_README.md                <-- NEW
    requirements_n2.txt                   <-- NEW
```

## Run on Windows Command Prompt

Open Command Prompt and enter:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk

git branch --show-current
cd research\component3
python -m pip install -r requirements_n2.txt

python scripts\n2_history_pilot.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --out-dir outputs_corrected

python scripts\test_n2_continuous_analysis.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --losses outputs_corrected\n2_paired_test_prediction_losses.csv

python scripts\n2_continuous_analysis.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --losses outputs_corrected\n2_paired_test_prediction_losses.csv --out-dir outputs_n2_continuous
```

**Run commands from `research\component3`, not the repository root.** Verify your active Git branch is `feature/component3` first. If raw data is in another folder, replace the quoted path. `requirements_n2.txt` installs plotting and references dependencies already used by the base scripts; it does not install `FrozenEstimator` directly.

## Results generated (intentionally ignored by .gitignore)

```text
outputs_n2_continuous/
  n2_prediction_evidence_and_losses.csv      3,900 rows = 1,300 held-out cases x 3 conditions
  n2_descriptive_spearman_correlations.csv   descriptive rho (no naive p-values)
  n2_evidence_loss_quantile_summary.csv      5-ish value-quantile groups; includes outcome prevalence
  n2_grouped_cv_relationship_metrics.csv     cross-fitted loss regression vs simple loss-mean baseline
  n2_grouped_cv_loss_predictions.csv         machine-grouped out-of-fold loss estimates
  n2_methodology_and_limitations.json        assumptions and interpretation warnings
  figures/
    n2_maintenance_count_vs_loss.png
    n2_history_span_vs_loss.png
    n2_recency_vs_loss.png
    n2_window_brier_comparison.png
```

## What the analysis DOES and DOES NOT establish

1. **Descriptive association:** Spearman correlations and mean squared errors in naturally formed evidence quantile groups. The original test cases repeat machines and share overlapping seven-day windows. These correlations are descriptive, not independently sampled hypothesis tests.
2. **Exploratory learned relationship:** Ridge regression of prediction loss on H/M/F/R and machine model. Five-fold **GroupKFold by machineID** generates *out-of-fold* loss predictions. Compare its MAE/RMSE with the training-fold average-loss predictor. A negative/low R² should be reported, not hidden.
3. **No real completeness effect:** C is constant at 100% in this benchmark. The method **cannot** estimate a learned completeness relationship until simulated masking or suitable real records are added.
4. **No single-machine reliability percentage:** The estimated loss relationship, even if useful, does not justify saying a future prediction is e.g. “88% reliable.”
5. **Not apparel validation:** Azure is a generic **synthetic** machine dataset. This experiment does not validate sewing-machine failure risk or natural CMMS missingness.
6. **No causal claim:** Loss also changes with failure prevalence/case difficulty and other confounders. Controlled history-window models were **retrained**; differences reflect changed inputs and refitting.
7. **Reused test period:** The first-stage held-out period has been inspected during previous C3 development. This is therefore a *PP1 exploratory second-stage analysis*, **not an untouched final test**.
8. **Group CV scope:** Grouped CV isolates machine identities *within the loss-relation stage*, but the original failure-prediction model previously trained on earlier records from these same machines. It does not establish performance for wholly unseen factory machines.

## PP1 explanation

> “To address the proposal panel's recommendation, I extended N2 beyond fixed lookback rules. I matched each held-out seven-day prediction's squared probability error to its own pre-prediction evidence profile (history span, maintenance-event count, prior failures, recency, and available completeness). I tested continuous associations and a machine-grouped cross-fitted Ridge relationship model. These preliminary synthetic-benchmark findings do not establish a universal sufficiency threshold or individual-machine reliability percentage; factory data and stronger independent validation are still needed.”

## Suggested Git commit

Once the tests pass and generated output CSV/PNG are verified **not staged**, run from the repository root:

```cmd
git add research/component3/scripts/n2_continuous_analysis.py research/component3/scripts/test_n2_continuous_analysis.py research/component3/N2_CONTINUOUS_README.md research/component3/requirements_n2.txt
git diff --cached --name-only
git diff --cached --check
git commit -m "feat(component3): analyze continuous maintenance evidence and held-out prediction loss"
git push origin feature/component3
```

## What to develop after this PP1 pilot

- Controlled event thinning and **simulated** field masking for C (source fields are not naturally missing).
- Sensitivity to machine model and failure prevalence, grouped uncertainty intervals, smoother nonlinearity where warranted.
- New chronological or external domain evaluation without repeatedly tuning on the present test period.
- Actual sewing-machine maintenance data, component/fault taxonomies, and technician feedback.
