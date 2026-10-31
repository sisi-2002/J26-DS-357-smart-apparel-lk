# Isolated v3 multi-horizon preview

**Current product behavior:** the later [unified dashboard implementation](UNIFIED_MULTIHORIZON.md)
uses `/component3/predict-risk` for v3 cards, v3 ranking and descriptive evidence
when the existing frontend flag is enabled. `/component3/experimental-risk` is
now a redirect to that primary route. The separate preview UI and test counts
described below are historical implementation records. The checksum, local-data,
runtime and fresh-machine checklist remain applicable. Both flags still default
OFF; the original LR APIs and flag-off dashboard remain intact.

This opt-in demonstration loads the **existing**
`research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib`.
It never trains, refits a calibrator, exports a model, or substitutes the deployed
7-day Logistic Regression. Both flags default to **off**. The existing prediction,
Top-10, health and evidence-study contracts remain unchanged.

## Audit and provenance

- Release: `v3-run_003-unpooled-hgb-sigmoid`, schema `unpooled-inclusive-v3`.
- SHA-256: `28d55ba04e18a8358afc3fe786da8f39db1436513261b20a2385f2a2042a2a61`.
- Artifact contains three fitted HGB pipelines (including imputers, scaler and
  one-hot encoder) and three fitted sigmoid calibrators wrapping those exact
  frozen pipelines. No fitting is required to serve predictions.
- Feature schema: the original 18 historical features, including machine model,
  age, previous maintenance/failure/error counts and event recencies. Display
  names and illustrative production lines never enter inference.
- The unchanged `model_inference.machine_features` builds strict-past features.
  Tests rebuilding these features reproduced **all 1,300 saved machine-date
  triplets exactly**, using round-trip CSV parsing and exact array equality,
  not just rounded percentages or a tolerance. There were zero ordering violations.
- API parity is checked for five machines across all 13 timestamps (65 inputs).
  Browser tests also compare actual responses with the saved v3 CSV.
- No label generation, fitting or calibration occurs in the new runtime.
  Backend tests prohibit estimator/preprocessor/calibrator fitting, including HGB.

The loader checks the artifact bytes **before joblib deserialization**, the
scikit-learn version, the five research inference source hashes and the four raw
CSV hashes against `app/components/component3/experimental/v3_manifest.json`.
It checks model/calibrator types, frozen estimator identity, the feature schema,
all 100 IDs and a real prediction before becoming available. Hashes pin this
specific audited release; they do not make arbitrary pickle files safe to load.
Research imports are scoped to artifact loading and restore the app's module
registry and search path, preserving the original inference loader.

## Semantics and scope

Inputs are IDs 1–100 and the existing 13 Wednesday snapshots, 06:00, from
2015-09-02 through 2015-11-25. Timestamps are timezone-naive. Historical features
use events **strictly before** prediction time. Targets are the next machine-level
component failure in `(0,7]`, `(7,14]` and `(14,30]` days. Events at prediction time
are excluded. Simultaneous component failures represent the same next-failure
time; later intervals condition on survival through the earlier interval.

Frozen conditional sigmoid outputs `h1,h2,h3` accumulate as
`P7=h1; P14=P7+(1-P7)*h2; P30=P14+(1-P14)*h3`.
This is the actual v3 computation, including its numerically stable accumulation;
it is not PAV projection, independently calibrated cumulative outputs, or a mix
with the deployed LR. Both service and response schema reject nonfinite,
out-of-range or unordered output. Frontend independently checks these conditions
and the echoed machine, timestamp, schema and release before displaying results.

Windows end within the published observation period (assumed exclusive end
2016-01-01). This does not verify machine uptime or complete event reporting.
Only the audited synthetic Azure dataset is supported, not arbitrary new factory
data or live dates. No thresholds, urgency or repair recommendations are produced.

## Windows CMD: opt-in backend

Use the project's existing Python environment and installed dependencies.
This release requires **scikit-learn 1.7.1** (audit runtime also had NumPy 2.2.6
and pandas 2.3.2). No packages were installed for this preview.

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\backend
set C3_MODEL_PATH=research/component3/models/c3_7day_sigmoid.joblib
set C3_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw
set C3_EXPERIMENTAL_MULTIHORIZON=true
set C3_EXPERIMENTAL_MODEL_PATH=research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib
set OMP_NUM_THREADS=1
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Paths resolve from the repository root. `C3_MODEL_PATH` still identifies only the
existing deployed LR. The separate experimental path never replaces it.
`.env.example` documents these settings; Python does not auto-load that file.

In a second CMD window:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\frontend
set VITE_API_BASE_URL=/api/v1
set API_PROXY_TARGET=http://127.0.0.1:8002
set VITE_C3_EXPERIMENTAL_MULTIHORIZON=true
npm run dev -- --port 5175
```

These alternate local ports allow the existing app on 8000/5173 to keep running
unchanged. Visit `http://127.0.0.1:5175/component3/experimental-risk` or the new
"Experimental horizons" link. Vite proxies same-origin requests; no CORS changes
are needed. The frontend flag is public build-time configuration, not access
control. Rebuild a production bundle after changing it. No public deployment or
authorization system is provided by this feature flag.

To disable, restart the backend with `C3_EXPERIMENTAL_MULTIHORIZON=false` and
restart Vite (or rebuild) with `VITE_C3_EXPERIMENTAL_MULTIHORIZON=false`.
Unset flags have the same effect. The API is then absent (404 and absent from
OpenAPI); the link is hidden and the experimental page redirects to the existing
7-day dashboard without making experimental API calls.

## Actual request and response

```cmd
curl.exe -X POST http://127.0.0.1:8002/api/v1/component3/experimental/predict-risk -H "Content-Type: application/json" -d "{\"machine_id\":1,\"prediction_time\":\"2015-09-02T06:00:00\"}"
```

Actual response observed during verification (request ID changes per request):

```json
{
  "machine_id": 1,
  "prediction_time": "2015-09-02T06:00:00",
  "p7": 0.27501825406882163,
  "p14": 0.3349944223020458,
  "p30": 0.7244866651273162,
  "model_version": "sha256:28d55ba04e18a8358afc3fe786da8f39db1436513261b20a2385f2a2042a2a61",
  "model_release": "v3-run_003-unpooled-hgb-sigmoid",
  "model_schema": "unpooled-inclusive-v3",
  "experimental": true,
  "research_only": true,
  "data_status": "synthetic_azure_historical_demo",
  "interval_semantics": "(0,7], (7,14], (14,30]; features strictly before prediction time",
  "note": "Synthetic Azure 2015 benchmark; exploratory, repeatedly inspected test data. Not validated for apparel factories or maintenance alerts. All horizons use the same v3 conditional hazard system; they do not use the deployed seven-day LR. Thirty-day Top-10 ranking was weaker than the independent LR research reference.",
  "request_id": "6744af24-53df-40bd-a2a6-6404ca7463de"
}
```

Machine 001 is shown as Line 01. All percentages derive from these actual API
values; formatting to two decimals does not change the stored probabilities.

## Reproduce verification (no research retraining)

With the backend CMD variables above, from `backend`:

```cmd
python -B -m pytest tests/component3 -q -p no:cacheprovider
```

This includes actual model loading, full saved-prediction reproduction, schema
checks, no-training guards, fail-closed missing/wrong artifacts and changed
data, invalid outputs, original API regression and identical LR prediction/ranking
JSON with the feature on/off (excluding random request IDs).

From `frontend`:

```cmd
npm test -- --maxWorkers=2
npm run typecheck
npm run build
set E2E_BASE_URL=http://127.0.0.1:5175
set E2E_EXPERIMENTAL_ENABLED=true
npm run test:e2e -- --output=test-results/experimental-enabled
```

With **both** running servers restarted with flags false, repeat E2E with
`E2E_EXPERIMENTAL_ENABLED=false` and output `test-results/experimental-disabled`.
Both modes run all the existing dashboard/N2 tests; enabled mode adds three
actual-v3 tests at desktop/tablet/mobile sizes, while disabled mode checks the
preview is inaccessible and the original ranking and prediction still work.
Mode-specific tests are explicitly skipped in the opposite mode.

Existing research tests were also run separately with the real data:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
set C3_EXPERIMENT_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw
python -B -m pytest research/component3/experimental_multihorizon_v3/test_v3.py research/component3/experimental_multihorizon_v2/test_v2.py research/component3/experimental_multihorizon/test_hazard_model.py research/component3/experimental_algorithm_comparison/test_comparison.py -q -p no:cacheprovider
```

These existing research tests fit their own deterministic test estimators; they
do not refit or overwrite the served artifacts or saved experimental outputs.

## Availability and scientific limitations

- Models/datasets remain local and Git-ignored; a clean checkout without them
  works with the flags off. Enabling without the exact audited artifact, data,
  source modules or runtime fails closed (503), without affecting the LR service.
  No new artifact or raw data is copied into the application or committed.
- The artifact is a research pickle tied to the repo's custom modules and runtime,
  not a self-contained portable serving package. Do not silently change the
  manifest to permit a new artifact; re-audit any new release first.
- Both flags must be enabled. Backend health intentionally continues reporting
  only the original service. Experimental availability is determined by its own
  endpoint; safe failure never returns invented probabilities.
- Zero ordering violations and exact saved-prediction parity establish correct
  integration of this model, not validated accuracy. The test period was reused
  repeatedly, machine observations repeat, and 30-day ranking was weaker than
  the independent LR research reference. The v3 inclusive-label evaluation must
  not be directly compared with old strict-upper-bound deployed LR metrics.
- This is a synthetic industrial benchmark demonstration, not a validated apparel
  factory maintenance system. It remains off by default pending review and fresh
  independent validation. No operational prediction flow was replaced.

## Verification record (2026-10-22)

| Check | Outcome |
| --- | --- |
| Existing v3 + v2 + hazard + algorithm research tests | 64 passed |
| Backend suite, including 29 experimental tests | 121 passed |
| Frontend unit tests | 135 passed |
| TypeScript | passed |
| Production builds, experimental flag true and false | passed |
| Full Playwright suite, both flags false | 14 passed, 3 enabled-only tests skipped |
| Full Playwright suite, both flags true | 16 passed, 1 disabled-only test skipped |
| Final preview browser checks after wording refinement | 3 passed, 1 disabled-only test skipped |
| Saved v3 prediction reproduction | exact for all 1,300 triplets; zero violations |
| Existing files compared with pre-task SHA-256 inventory | 321 checked; only the 8 intended configuration/navigation/documentation files changed |

No unresolved test failures. Two pandas date-inference warnings come from the
existing malformed-data regression tests. Playwright emitted the environment's
NO_COLOR/FORCE_COLOR notice; it did not affect results.

Reviewed empty, populated and error states at desktop 1440px, tablet 820px and
mobile 390px. Browser checks cover normal-text AA contrast, visible keyboard
focus, no horizontal overflow, exact saved probabilities, multiple selections,
network failures, and return to the unchanged LR dashboard. The final production
bundle was built with the experimental flag **false**.

Screenshots are local, ignored verification artifacts under
`frontend/test-results/experimental-preview-final/` (the per-test directories
contain `experimental-{desktop,tablet,mobile}-{empty,populated,audit,error}.png`).
The two full regression run outputs are under `experimental-disabled/` and
`experimental-enabled/` within the same `test-results` directory.

Existing uncommitted frontend and research work was preserved. No datasets,
research files, artifacts, Function 1 or N2 business logic changed; no commits,
pushes or public deployments were performed.

## Exact file inventory for the initial preview implementation

Modified existing files (8):

```text
backend/.env.example
backend/app/core/config.py
backend/app/main.py
frontend/.env.example
frontend/README.md
frontend/src/components/component3/routes.tsx
frontend/src/shared/layouts/MainLayout.tsx
frontend/src/vite-env.d.ts
```

Created files (13):

```text
backend/EXPERIMENTAL_MULTIHORIZON.md
backend/app/components/component3/experimental/__init__.py
backend/app/components/component3/experimental/v3_manifest.json
backend/app/components/component3/experimental/service.py
backend/app/components/component3/experimental/router.py
backend/tests/component3/test_experimental_preview.py
frontend/src/components/component3/experimental/featureFlag.ts
frontend/src/components/component3/experimental/experimentalApi.ts
frontend/src/components/component3/experimental/useExperimentalPrediction.ts
frontend/src/components/component3/experimental/ExperimentalRiskPage.tsx
frontend/src/components/component3/experimental/experimental-preview.css
frontend/src/components/component3/experimental/ExperimentalRiskPage.test.tsx
frontend/e2e/experimental-preview.spec.ts
```

Generated, ignored verification artifacts: `frontend/dist/` (flag off),
`frontend/test-results/` screenshots and Playwright traces/results. These are not
research output changes and are not intended for Git. Temporary test fixtures
were created by pytest in the operating system's temporary directory.

## Windows portability fix and fresh-machine checklist

The follow-up portability task adds only root `.gitattributes`,
`backend/tests/component3/test_source_portability.py`, and this documentation
update. Runtime inference, manifest checksums, model artifacts and research code
are unchanged. No broad renormalization or staging in the main repository was
performed. Both flags still default OFF.

Follow-up verification on the existing Windows environment:

| Check | Result |
| --- | --- |
| Source hashes before/after LF policy | All five match the original manifest exactly; no source bytes changed |
| Real Git checkout tests, LF/CRLF inputs, policy on/off | 4 passed; unrelated sentinel still CRLF |
| Full backend regression suite with real artifact/data | 125 passed, no skips; two existing malformed-date warnings |
| Exhaustive FastAPI TestClient audit with fitting prohibited | All 1,300 triplets exactly equal saved v3 CSV; zero ordering violations |
| Frontend unit suite / TypeScript | 135 passed / passed |
| Loader, manifest and both model artifacts | Original bytes and SHA-256 values unchanged |

No retraining, package installation, browser test rerun or production build was
needed for this attributes/documentation change. Existing backend regressions
verify flag-off behavior, loading/failure handling and identical deployed LR
prediction/Top-10 responses with the experimental feature on/off. Prior browser
and build results above describe the initial preview implementation, not a new
browser/build run during this follow-up.

### Exact source integrity and line endings

Root `.gitattributes` enforces `text eol=lf` on **only** these five paths:

| Path (relative to repository root) | Expected SHA-256, unchanged |
| --- | --- |
| `research/component3/scripts/build_and_evaluate.py` | `dea4900cf7890e3128ce92cb0f937ff0fe61a2f1fd5823b059234aa11ebec745` |
| `research/component3/scripts/model_inference.py` | `763912c37d5243a6e56b63cff3f83b4953c94ecd894afeec23c845ed111ffaa7` |
| `research/component3/experimental_multihorizon/hazard_model.py` | `1b96726edd88b75c7109e94f6e418c84cb953c11733d91aa0d0223c165025612` |
| `research/component3/experimental_multihorizon_v2/hazard_v2.py` | `0a9f9714b9bbfe5f589ed9cf9f220076e30bd72b121ab38fabb9d11a9b93e0fc` |
| `research/component3/experimental_multihorizon_v3/methods_v3.py` | `e26db66d16c1afc053a3e6f6d0ce9e678755b1e30bdbf052b774c721a3d8cf15` |

All five already contained LF bytes and matched the manifest before the fix.
Attributes prevent future checkout conversion; they do not repair an already
modified working copy. Do not change expected hashes to accommodate a mismatch.
For a fresh clone of a future reviewed revision that includes these files and
attributes, Git will check out LF even with `core.autocrlf=true`. Editors must
also preserve LF when saving these files. Avoid `git add --renormalize .`,
`git reset`, or overwriting local work to repair a checkout.

The new tests use isolated temporary Git repositories and indices (no commits,
no access to the project's index). They stage LF/CRLF inputs with
`core.autocrlf=true`, then materialize a checkout with `git checkout-index`:
with the actual attributes all five hashes match; without them they mismatch.
An unrelated text sentinel still checks out as CRLF in both cases. This verifies
the line-ending policy without pretending that the current HEAD contains all
uncommitted preview code.

### Transfer inventory (status observed on 2026-10-22)

An ordinary clone of the **current HEAD is insufficient**. This task intentionally
does not commit or stage any project files. Before a second-machine rehearsal,
transfer an explicitly reviewed working-tree source snapshot, including its
untracked source files and tracked modifications, or later obtain a reviewed
revision containing them. Never assume that copying a Git commit also copies
ignored artifacts or untracked files.

| Required item | Current Git status / transfer requirement |
| --- | --- |
| Two `research/component3/scripts/` files listed above | Tracked; preserve exact LF bytes |
| Three hazard Python modules listed above | Untracked; absent from current HEAD; retain these exact relative paths |
| `backend/app/` | Existing scaffold tracked; config/main have uncommitted changes; entire `components/component3/experimental/` directory untracked (including manifest, router, service, `__init__.py`) |
| `backend/tests/component3/` | Existing tests tracked; experimental and portability test modules untracked; needed for verification |
| Root `.gitattributes`, this guide | Untracked; include in source snapshot |
| `research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib` | Ignored, local; supply the existing trusted artifact separately, SHA-256 `28d55ba04e18a8358afc3fe786da8f39db1436513261b20a2385f2a2042a2a61` |
| `research/component3/models/c3_7day_sigmoid.joblib` | Ignored, local; required for the existing LR service/regressions; SHA-256 `20908cc0c0466ddc5563d542ff2068223cddd0ed2f6cded620a65f9fbc7c3bcc` |
| `research/component3/experimental_multihorizon_v3/run_003/heldout_predictions.csv` | Untracked saved reference; needed for parity tests and preview E2E, not serving |
| `PdM_machines.csv`, `PdM_maint.csv`, `PdM_failures.csv`, `PdM_errors.csv` | External local raw-data directory, outside Git; retain exact original bytes and filenames; hashes are in the manifest |
| `frontend/src/` | Existing pages/shared layout include pre-existing uncommitted work; experimental directory and `data/machineDisplay.ts` plus its test are untracked; include the complete reviewed source snapshot |
| `frontend/e2e/` | Existing tests have uncommitted changes; new experimental test untracked; include for browser verification |
| Backend requirements, frontend package/lock files and Vite configuration | Tracked; include the matching source-snapshot versions; do not transfer `node_modules` or a Python environment between machines |

The five Python modules are sufficient research-code dependencies for inference;
training/evaluation/report scripts and previous experiment artifacts are not
imported by this endpoint. Keep all required files under a common repository
root: backend resolves it from `app/core/config.py`, not from a hard-coded drive.
The raw directory may live anywhere; configure its absolute path. Illustrative
machine names and lines are frontend display metadata only.

### Windows CMD preflight

These are instructions for a new machine, **not installations performed by this
task**. Choose the drive paths appropriate to that machine. Reference runtime:
Windows AMD64, CPython **3.13.2**, Node **22.13.0**, npm **11.2.0** (frontend
requires Node >=22.12.0). The backend requirements pin NumPy 2.2.6, pandas 2.3.2,
scikit-learn **1.7.1**, joblib 1.5.1, FastAPI 0.116.1, Pydantic 2.13.4 and
Uvicorn 0.35.0; dev requirements add pytest 8.4.1 and httpx 0.28.1.
Audited numerical transitive dependencies were SciPy **1.16.1** and threadpoolctl
**3.6.0**. They are not pinned by the current requirements file; the optional
explicit pins below reproduce those versions without editing project files.
Other transitive dependencies are not fully locked; a fresh-environment parity
run is mandatory before claiming exact cross-machine reproducibility.

After supplying the source snapshot, two artifacts, reference CSV and raw files:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
py -3.13 -m venv backend\.venv
call backend\.venv\Scripts\activate.bat
python -m pip install -r backend\requirements-dev.txt scipy==1.16.1 threadpoolctl==3.6.0
python -m pip check
python -c "import sys,sklearn,numpy,pandas,scipy,joblib; print(sys.version); print(sklearn.__version__,numpy.__version__,pandas.__version__,scipy.__version__,joblib.__version__); assert sklearn.__version__=='1.7.1'"
set C3_MODEL_PATH=research/component3/models/c3_7day_sigmoid.joblib
set C3_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw
set C3_EXPERIMENTAL_MODEL_PATH=research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib
set C3_EXPERIMENTAL_MULTIHORIZON=false
set OMP_NUM_THREADS=1
```

From the repository root, check the policy and exact bytes (no deserialization,
normalization or file rewriting):

```cmd
git check-attr text eol -- research/component3/scripts/build_and_evaluate.py research/component3/scripts/model_inference.py research/component3/experimental_multihorizon/hazard_model.py research/component3/experimental_multihorizon_v2/hazard_v2.py research/component3/experimental_multihorizon_v3/methods_v3.py
python -B -c "import json,hashlib; from pathlib import Path; m=json.loads(Path('backend/app/components/component3/experimental/v3_manifest.json').read_text()); checks=[(p,hashlib.sha256(Path(p).read_bytes()).hexdigest()==h) for p,h in m['source_sha256'].items()]; print(checks); assert all(ok for p,ok in checks), 'Source checksum mismatch: stop'"
python -B -c "import os,json,hashlib; from pathlib import Path; m=json.loads(Path('backend/app/components/component3/experimental/v3_manifest.json').read_text()); files=[(Path(os.environ['C3_DATA_DIR'])/n,h) for n,h in m['data_sha256'].items()]+[(Path(os.environ['C3_EXPERIMENTAL_MODEL_PATH']),m['artifact_sha256']),(Path(os.environ['C3_MODEL_PATH']),'20908cc0c0466ddc5563d542ff2068223cddd0ed2f6cded620a65f9fbc7c3bcc')]; checks=[(str(p),hashlib.sha256(p.read_bytes()).hexdigest()==h) for p,h in files]; print(checks); assert all(ok for p,ok in checks), 'Artifact/data mismatch: stop'"
cd backend
python -B -m pytest tests/component3 -q -p no:cacheprovider
```

All five attributes must report `text: set`, `eol: lf`; all checksum checks must
be True. The full configured backend suite must run with **no skips**. Its
fixtures prohibit model fitting. Without `C3_MODEL_PATH`/`C3_DATA_DIR`, real-data
checks deliberately skip, so an unconfigured green run does not prove readiness.
The saved CSV is verification evidence only; missing it fails the parity tests.

Once checks pass, use the earlier opt-in server commands (8002/5175). Backend
`.env.example` is documentation, not an auto-loaded environment file. In a second
CMD window activate the same Python environment only if running Python; set the
frontend variables there explicitly, since CMD environments do not propagate
between independently opened windows. Run `npm ci` in `frontend` to reproduce
its lockfile, then `npm test -- --maxWorkers=2`, `npm run typecheck` and
`npm run build`. For browser tests use the installed Edge channel on Windows or
explicitly provision a supported Playwright browser on the rehearsal machine.

Finally POST the earlier actual request to the experimental endpoint and compare
its probabilities. A healthy `/api/v1/health` reports **only LR availability**;
it cannot establish experimental readiness. Missing inputs, incompatible runtime
or source bytes must yield an experimental 503, with no substitute predictions.
Frontend opt-in plus backend off is expected to show an unavailable preview.
Leave both flags unset/false outside the isolated demonstration.

### Remaining limits

The Git checkout conversion defect is fixed by the narrow attributes policy.
The current working tree still needs review and a deliberate source transfer or
later approved commit before an ordinary clone contains this feature. Model and
raw-data delivery remains manual and local, with strict checksums. No second
physical Windows machine or newly installed environment has been tested here;
temporary Git checkout tests validate line endings, not dependency installation.
No integrity checks were relaxed, no manifest values changed and no model was
retrained. This remains a synthetic research demo with the scientific limitations
already described above.
