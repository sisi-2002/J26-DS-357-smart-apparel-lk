# Component 3 FastAPI historical demonstration

This minimal backend serves the **synthetic Microsoft Azure Predictive Maintenance
benchmark**. It is not validated for apparel factories or current machinery.
It reuses the unchanged `research/component3/scripts/model_inference.py` and your
existing trusted calibrated 7-day artifact. No endpoint trains, calibrates,
constructs future labels, or runs N2 degradation experiments.

## Windows CMD setup and startup

Python 3.13 is the locally verified interpreter. Run from the repository root:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
python -m venv backend\.venv
call backend\.venv\Scripts\activate.bat
python -m pip install -r backend\requirements-dev.txt
set "C3_MODEL_PATH=E:\J26-DS-357-smart-apparel-lk\research\component3\models\c3_7day_sigmoid.joblib"
set "C3_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Replace the dataset path if necessary. The directory must contain `PdM_machines.csv`,
`PdM_maint.csv`, `PdM_failures.csv`, and `PdM_errors.csv`. Both variables are required;
relative values resolve from the repository root. `.env.example` documents them;
the app does **not** automatically load `.env`. Never point at an untrusted joblib
file: loading it can execute code. Do not commit models, datasets, secrets or outputs.

The pinned ML versions match the inspected local inference environment. Saved
scikit-learn objects require a compatible export environment. Install runtime-only
dependencies with `backend\requirements.txt`; `requirements-dev.txt` adds tests.

The loader caches model and data once per process. Restart after changing paths or
artifacts. Missing, corrupt or incompatible resources keep the component unavailable
with HTTP 503, while OpenAPI remains accessible. No automatic retraining occurs.

Open http://127.0.0.1:8000/docs for Swagger, http://127.0.0.1:8000/openapi.json
for the contract, and http://127.0.0.1:8000/api/v1/health for readiness.

## Request and output

From another Windows CMD window:

```cmd
curl.exe -X POST "http://127.0.0.1:8000/api/v1/component3/predict-risk" -H "Content-Type: application/json" -d "{\"machine_id\":1,\"prediction_time\":\"2015-09-02T06:00:00\"}"
```

The body requires a positive integer `machine_id` and a timezone-naive historical
`prediction_time` string. Only the 13 original weekly test timestamps are accepted:
September 2 through November 25, 2015, at 06:00:00. `Z` and timezone offsets are
rejected. This preserves the benchmark's unspecified timezone rather than inventing
a UTC or Sri Lankan timezone. Current timestamps and arbitrary historical days are
also rejected. Only the saved 7-day horizon is supported by this API.

The real local artifact returns probability `0.196381` for the example above:

```json
{
  "machine_id": 1,
  "prediction_time": "2015-09-02T06:00:00",
  "horizon_days": 7,
  "predicted_failure_probability": 0.196381,
  "calibration": "sigmoid fitted on later chronological validation period",
  "evidence_profile": {
    "H_observed_maintenance_span_days": 458.0,
    "M_prior_maintenance_events": 24,
    "F_prior_failure_events": 4,
    "R_days_since_latest_maintenance": 30.0,
    "C_published_field_completeness_pct": 100.0
  },
  "machine_type": "model3",
  "research_only": true,
  "note": "Synthetic Azure benchmark; 2015 historical test snapshot. Not an apparel machine, real-time prediction, or validated maintenance decision. C=100% constant and H is not verified logging coverage.",
  "request_id": "<unique UUID per request>",
  "model_schema_version": "c3-pdm-7day-v1",
  "model_version": "sha256:20908cc0c0466ddc5563d542ff2068223cddd0ed2f6cded620a65f9fbc7c3bcc",
  "data_status": "synthetic_azure_historical_demo"
}
```

Probability retains the research module's six-decimal rounding. Model identity is
computed from the actual artifact bytes; the UUID varies. The H/M/F/C/R field names
are deliberately retained for research-output compatibility. H is elapsed time
since the first **observed** maintenance event, not verified logging coverage.
M/F are prior maintenance/failure counts. R is event recency, not database freshness.
C is the published benchmark's constant completeness, not factory-CMMS quality.
No reliability score, recommendation, or maintenance threshold is returned.
Missing H/R history is JSON `null`. The synthetic `model3` label is not an apparel
machine category. Research limitations include reused evaluation periods and
unvalidated apparel-domain generalization.

Errors use FastAPI's `detail` field: invalid input is 422, an unknown benchmark
machine is 404, unavailable model/data is 503, and an inference failure is 500.
Internal failure details are logged server-side and not returned to callers.

## Automated verification without retraining

With the environment variables above set, run from the repository root:

```cmd
python -B -m pytest backend\tests -p no:cacheprovider -q
python -B research\component3\scripts\test_prediction_time_boundary.py
```

The API suite blocks training/calibration `fit` methods, including preprocessing
fitting. Portable tests use an explicitly named test double serialized into a
temporary directory, plus temporary benchmark-shaped CSV fixtures. Integration
tests compare the **real configured artifact** with direct research inference,
check repeatability and artifact identity, and inject at/future-time events in
memory to check strict-past behavior. They never write the original artifact or
CSV files. If environment variables are absent, actual-artifact integration tests
are explicitly skipped; configured paths that are wrong cause failure.

Do not run `scripts/test_saved_inference.py` for a no-retraining check: its final
comparison fits an independent model. Export, training, calibration and N2 research
commands are not required to test or start this API.

## Architecture and later group integration

`app/main.py` is a small app factory and component-lifecycle registration point.
`core/config.py` owns environment resolution. Component 3 owns its router,
Pydantic schemas, synchronous endpoint, prediction service and cached ML loader.
Its import adapter supplies the research module's flat dependency temporarily
without changing research files or adding global search paths. Deployment must
include the existing research scripts alongside the backend.

Later integration must reconcile shared startup/configuration, dependencies,
authentication/authorization, readiness reporting, response conventions and the
documented benchmark-timezone exception. The current server is a localhost research
demonstration with no authentication, persistence, frontend, survival/SHAP outputs,
or validated N2 reliability mapping. Group migration and database ownership can
remain separate from this inference milestone. Register the Component 3 router
only once: it already includes `/api/v1/component3`.
