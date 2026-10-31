# Component 3 — Saved-model inference milestone (PP1)

This release adds a real *training → saved model → later inference* workflow. It is based on the **corrected** `scripts/build_and_evaluate.py` already in your Component 3 repository. **Do not overwrite that existing script.**

## Copy into the Git repository

Extract this ZIP into `E:\J26-DS-357-smart-apparel-lk\research\component3\`.

It adds **four** Python scripts in `scripts/` and this README. Keep your existing files unchanged.

## Install

Use your existing environment (Python 3.13, pandas, NumPy, scikit-learn >= 1.6, joblib). If needed, from `research\component3`:

```cmd
python -m pip install -r requirements.txt
```

## Run from `E:\J26-DS-357-smart-apparel-lk\research\component3`

**A. Save the fitted model (runs once):**

```cmd
python scripts\export_7day_model.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --model-out models\c3_7day_sigmoid.joblib
```

It fits a Logistic Regression preprocessing Pipeline on the earlier **training** period and applies sigmoid calibration on the **later validation** period via `FrozenEstimator`. The exported pipeline contains preprocessing and calibration. It is *not* refitted when a prediction is requested.

**B. Make a historical 7-day prediction (loads saved artifact):**

```cmd
python scripts\predict_historical_machine.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --model models\c3_7day_sigmoid.joblib --machine-id 1 --prediction-time "2015-09-02 06:00:00"
```

**C. Run verification tests:**

```cmd
python scripts\test_saved_inference.py --data-dir "E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw" --model models\c3_7day_sigmoid.joblib
```

## Critical details

- Input date is deliberately limited to one of the **13 historical test timestamps** from September–November 2015. Do not claim the model is ready to predict current apparel-machinery failures.
- A prediction uses real fitted model probabilities from the synthetic Azure benchmark and a **single machine's strictly past maintenance, failure and error events**, with the exact feature specification used for training. Future labels are **not constructed or read** by the inference module.
- Model output has `predicted_failure_probability`, `horizon_days`, and H/M/F/C/R evidence characteristics. These evidence characteristics are **not** a validated reliability score or a maintenance decision rule.
- H is time since the first *observed* maintenance event, **not** proven data collection coverage. C is a synthetic benchmark's constant 100% of published source fields, **not** validated completeness of a factory CMMS.
- The synthetic `model1`...`model4` values must **not** be described as lockstitch, overlock or coverstitch machines.
- No user-facing maintenance threshold/action is returned. A defensible threshold must consider independent validation, calibration, false alerts and actual maintenance costs.
- Joblib files can execute code on load. **Only load your own locally created model**. Avoid committing `.joblib` artifacts, raw CSV datasets, secrets or generated outputs. The existing Component 3 `.gitignore` should cover `*.joblib`; verify with `git status`.
- The original failure model has been evaluated repeatedly on this historical test period. Future research requires a new untouched period/real apparel-domain validation.
- This is a **demo inference module**; the next step is to call `predict_saved(...)` from a Component 3 FastAPI router, then add a minimal React page.

## Git workflow (after local verification)

From repository root:

```cmd
git branch --show-current
git status --short
git add research/component3/scripts/model_inference.py research/component3/scripts/export_7day_model.py research/component3/scripts/predict_historical_machine.py research/component3/scripts/test_saved_inference.py research/component3/SAVED_INFERENCE_README.md
git diff --cached --name-only
git diff --cached --check
git commit -m "feat(component3): add saved calibrated historical inference"
git push origin feature/component3
```

Stage and commit only after all tests succeed and the staged-file list contains exactly the five files above. No generated models should be staged.
