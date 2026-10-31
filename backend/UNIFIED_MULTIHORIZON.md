# Component 3 unified multi-horizon product prototype

The **existing** frontend opt-in `VITE_C3_EXPERIMENTAL_MULTIHORIZON=true`
switches `/component3/predict-risk` to the unified v3 workflow. It shows one
selected machine's 7/14/30-day cumulative probabilities, a v3 Top-10 comparison
and strictly historical descriptive evidence. `/component3/experimental-risk`
redirects to the primary route; the sidebar has one Machine risk entry.
The standalone `/component3/evidence-study` is unchanged.

Both existing flags default **OFF**. The normal bundle/flag-off route retains
the stable LR dashboard. Backend flag-off omits both experimental endpoints.
The deployed LR artifact, existing APIs and research scripts were not changed.
Enabling the UI while v3 is unavailable displays an error; it never silently
uses an LR probability/ranking. Rollback is an explicit flag change/restart.

## API additions

- Existing experimental `POST /api/v1/component3/experimental/predict-risk`
  retains its probabilities and provenance and now also returns
  `evidence_profile`. This is an additive response change. Its request is unchanged.
- New opt-in `POST /api/v1/component3/experimental/top-risk-machines` accepts
  `prediction_time` and `horizon_days` (7, 14 or 30; default 7). It scores all 100
  machines and returns ten rows with `rank`, `machine_id`, `p7`, `p14`, `p30`,
  the selected horizon, counts and the same v3 artifact identity/provenance.
- Existing `/api/v1/component3/predict-risk`, `/top-risk-machines` and health
  endpoints retain their prior contracts and behavior.

Inputs remain IDs 1–100 and the 13 timezone-naive Wednesday snapshots at 06:00,
September 2 through November 25, 2015. No live/new dates or factory data are
supported. Invalid requests return 422; missing/invalid resources return 503;
invalid inference returns 500. Both experimental URLs are 404 when disabled.

## Inference, ranking and caching

The original v3 artifact is loaded without fitting or exporting. Strict source,
artifact and dataset checksum validation remains active, including the narrow
LF `.gitattributes` rules. Frozen preprocessing and conditional sigmoid calibration
are reused. The cumulative computation and predictions are unchanged.

For each date, the service builds 100 feature rows using the unchanged historical
feature function. One batch call to the v3 system invokes each of its three
conditional calibrators once on all 100 rows. All triplets are checked for shape,
finiteness, range and ordering before caching. Sorting uses the selected horizon's
**unrounded** probability descending and machine ID ascending for deterministic
ties. No LR inference is involved.

The service caches immutable triplets, not mutable response dictionaries, keyed
by canonical date. There are at most 13 supported keys, shared across the three
horizons. A per-service lock prevents duplicate concurrent fills. Each request
gets fresh metadata/request ID and newly constructed Top-10 rows. Refresh ranking
reloads the API response; fixed benchmark scores may come from this cache.
Restarting the process clears it. Artifacts and data are fixed and validated at
startup; changing files on disk requires restart and a successful integrity check.

Changing machines preserves the visible ranking. Changing the ranking horizon
preserves the selected machine's three-card result, since it is the same system
and date. Date changes clear both results, cancel pending requests and load the
new ranking while preserving the manually selected machine. Late responses are
ignored even if transport ignores cancellation. Page reload returns to the first
snapshot/machine, default 7-day comparison and no individual prediction.

View requests actual individual inference, then verifies **all three** full-precision
values and model identity against the returned ranking row. Probabilities are
formatted to two decimals only at display time. Probability bars use a fixed
0–100% scale. Position 1–3 coral accents denote relative rank only.

## Evidence context

The additive evidence fields are computed from the same strictly historical
feature row used by the selected v3 prediction:

| Field | Meaning |
| --- | --- |
| H | Time since first previously recorded maintenance; observed span, not guaranteed log coverage |
| M | Number of previously recorded maintenance events |
| F | Number of previously recorded component-failure events; not unique physical failures |
| C | Published benchmark field completeness, constant 100%; not factory data quality |
| R | Days since last previously recorded maintenance; event recency, not database freshness |
| Error history | Total prior errors; counts in past 7/30/90 days; days since last recorded error |

All event records must satisfy `datetime < prediction_time`; window lower bounds
follow the original feature implementation. Missing recencies/spans remain null,
displayed as no prior event observed, never substituted with zero. Generic Azure
error categories are not validated apparel fault types or confirmed failures.
Machine names and Line 01–10 assignments remain deterministic display metadata.

This evidence is **descriptive**. No evidence-derived reliability percentage,
sufficiency threshold, alert category, maintenance urgency or repair action is
produced. The collapsed Model & Data Info explains the offline N2 experiments
used original uncalibrated LR models. Their findings are not validated evidence-loss
relationships for v3. Future research must evaluate sufficiency against v3 at
each horizon. The original N2 page, snapshot and experiments remain unchanged.

## Actual example: September 2, 2015, 06:00

Selected **Machine 001 · Line 01**:

```json
{
  "machine_id": 1,
  "prediction_time": "2015-09-02T06:00:00",
  "p7": 0.27501825406882163,
  "p14": 0.3349944223020458,
  "p30": 0.7244866651273162,
  "evidence_profile": {
    "H_observed_maintenance_span_days": 458.0,
    "M_prior_maintenance_events": 24,
    "F_prior_failure_events": 4,
    "C_published_field_completeness_pct": 100.0,
    "R_days_since_latest_maintenance": 30.0,
    "error_events_prior": 26,
    "error_events_7d": 2,
    "error_events_30d": 3,
    "error_events_90d": 6,
    "days_since_last_error": 1.0
  }
}
```

This is a subset of the actual API response; full responses also include model
release `v3-run_003-unpooled-hgb-sigmoid`, schema `unpooled-inclusive-v3`,
artifact SHA-256 `28d55ba04e18a8358afc3fe786da8f39db1436513261b20a2385f2a2042a2a61`,
experimental/research flags, data status, inclusive interval semantics, model note
and a fresh request ID. UI cards show **27.50%, 33.50%, 72.45%**.

Actual v3 7-day Top 10 (display-rounded percentages only):

| Rank | Machine | Line | 7-day probability |
| --- | --- | --- | --- |
| 1 | Machine 071 | Line 08 | 49.75% |
| 2 | Machine 094 | Line 10 | 46.60% |
| 3 | Machine 013 | Line 02 | 44.49% |
| 4 | Machine 021 | Line 03 | 43.10% |
| 5 | Machine 097 | Line 10 | 42.23% |
| 6 | Machine 007 | Line 01 | 41.45% |
| 7 | Machine 022 | Line 03 | 39.95% |
| 8 | Machine 030 | Line 03 | 38.72% |
| 9 | Machine 032 | Line 04 | 37.43% |
| 10 | Machine 049 | Line 05 | 32.83% |

The 14-day leader is Machine 017 (P14 `0.5658543137119392`); the 30-day leader is
Machine 071 (P30 `0.7976607981354883`). These rankings differ from the LR
dashboard and from one another. No probabilities were adjusted to produce them.

## Windows CMD demonstration

Use the existing Python environment and the exact local artifacts/data. See
[the portability and fresh-machine checklist](EXPERIMENTAL_MULTIHORIZON.md#windows-portability-fix-and-fresh-machine-checklist)
before using a new workstation. Scikit-learn **1.7.1** and all five LF source
hashes must match. No dependencies were installed during this implementation.

Backend CMD:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\backend
set C3_MODEL_PATH=research/component3/models/c3_7day_sigmoid.joblib
set C3_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw
set C3_EXPERIMENTAL_MODEL_PATH=research/component3/experimental_multihorizon_v3/run_003/unpooled_hgb_research.joblib
set C3_EXPERIMENTAL_MULTIHORIZON=true
set OMP_NUM_THREADS=1
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Frontend CMD:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\frontend
set VITE_API_BASE_URL=/api/v1
set API_PROXY_TARGET=http://127.0.0.1:8002
set VITE_C3_EXPERIMENTAL_MULTIHORIZON=true
npm run dev -- --port 5175
```

Open `http://127.0.0.1:5175/component3/predict-risk`.
These alternate local ports leave existing servers on 8000/5173 untouched.
The Vite proxy uses same-origin API requests; no CORS change is needed.
Example ranking request:

```cmd
curl.exe -X POST http://127.0.0.1:8002/api/v1/component3/experimental/top-risk-machines -H "Content-Type: application/json" -d "{\"prediction_time\":\"2015-09-02T06:00:00\",\"horizon_days\":14}"
```

Rollback: restart the backend with `C3_EXPERIMENTAL_MULTIHORIZON=false` and Vite
with `VITE_C3_EXPERIMENTAL_MULTIHORIZON=false`. Unset flags also mean false.
Production frontend flags are build-time: rebuild to switch. The backend flag is
startup configuration. No public deployment or default model replacement occurred.

## Validation

From backend, with the real local model/data environment configured:

```cmd
python -B -m pytest tests/component3 -q -p no:cacheprovider
```

From frontend:

```cmd
npm test -- --maxWorkers=2
npm run typecheck
npm run build
set E2E_BASE_URL=http://127.0.0.1:5175
set E2E_EXPERIMENTAL_ENABLED=true
npm run test:e2e -- --output=test-results/unified-enabled
```

For rollback verification run both servers with flags false, then E2E with
`E2E_EXPERIMENTAL_ENABLED=false`. The test-mode variable only selects tests; it
does not configure either application. Backend tests prohibit estimator fitting.
Frontend unit tests use HTTP fixtures for behavior; Playwright success scenarios
use the actual API/models and compare with saved v3 CSV values. Only network-error
scenarios abort transport. No research models were retrained.

Results:

- Backend: **135 passed** with real artifacts/data, including all 1,300 individual
  API cases exactly reproducing saved v3 triplets; zero ordering violations.
- All **39** date/horizon rankings match full-precision saved predictions; selected
  rows match individual inference and artifact identity. Tests verify shared cache,
  one call per conditional calibrator for all 100 machines, ties, invalid outputs,
  missing artifacts/schema/data, flag-off and unchanged deployed LR APIs.
- Existing Windows checkout tests pass; source/manifest/artifact hashes remain intact.
- Frontend: **149 passed**, TypeScript and both flag-on/flag-off builds pass.
- Full flag-off Playwright suite: **14 passed, 3 enabled-only skips**.
- Full flag-on suite: **6 passed, 11 flag-off-only skips** (v3 workflow plus N2
  at desktop/tablet/mobile). The ten stable LR browser tests run in flag-off mode;
  the disabled-route test also runs only there. Neither workflow is left untested.
- Screenshots cover populated, ranking-only, audit and network-error states at
  1440px desktop, 1024px tablet and 390px mobile. Contrast, keyboard focus,
  overflow, all horizon selectors, matching evidence, date changes/reloads and
  absence of LR API requests in the v3 workflow are browser-checked.

Two existing pandas malformed-date warnings and the Playwright NO_COLOR notice
are non-failing diagnostics. Screenshots/build/test outputs remain local and
ignored. Final browser screenshots are in `frontend/test-results/unified-final/`.

## Scientific and portability limits

Synthetic Azure industrial-machine data is not validated apparel-factory data.
The reused test period and repeated machine snapshots make performance findings
exploratory. Complete event reporting/uptime is not established by benchmark
coverage. Guaranteed cumulative ordering and faithful integration do not establish
calibration/generalization in a factory. The v3 30-day Top-10 ranking was weaker
than the independent LR research reference; exposing it is a research preview,
not a claim of improved maintenance decisions.

Required research modules and UI additions remain uncommitted; artifacts remain
local/ignored and the raw directory remains external. A plain clone of current
HEAD is insufficient. Transfer a reviewed source snapshot and trusted artifacts
or use a later approved revision, then follow the checksum/runtime preflight.
The backend health check still reports only the original LR service; use the
experimental endpoints to establish v3 readiness. No check was weakened.

## Exact files changed for the unified workflow

Modified existing working-tree files (18):

```text
backend/EXPERIMENTAL_MULTIHORIZON.md
backend/app/components/component3/experimental/service.py
backend/app/components/component3/experimental/router.py
backend/tests/component3/test_experimental_preview.py
frontend/README.md
frontend/src/components/component3/components/PredictionControls.tsx
frontend/src/components/component3/components/TopRiskMachinesPanel.tsx
frontend/src/components/component3/experimental/ExperimentalRiskPage.tsx
frontend/src/components/component3/experimental/ExperimentalRiskPage.test.tsx
frontend/src/components/component3/experimental/experimentalApi.ts
frontend/src/components/component3/experimental/useExperimentalPrediction.ts
frontend/src/components/component3/routes.tsx
frontend/src/shared/layouts/MainLayout.tsx
frontend/src/test/setup.ts
frontend/e2e/experimental-preview.spec.ts
frontend/e2e/dashboard-layout.spec.ts
frontend/e2e/prediction.spec.ts
frontend/e2e/top-risk-machines.spec.ts
```

Created (4):

```text
backend/UNIFIED_MULTIHORIZON.md
frontend/src/components/component3/experimental/UnifiedRiskPage.tsx
frontend/src/components/component3/experimental/unified-risk.css
frontend/src/components/component3/experimental/useExperimentalRanking.ts
```

The previous separate preview page became a small bookmark redirect; its old
stylesheet remains untouched. Shared comparison/control components gained
optional props whose defaults preserve the LR presentation. Legacy browser
tests now explicitly run in flag-off mode; new workflow tests run in flag-on
mode. Unit setup defaults to flag-off and v3 tests explicitly opt in.

SHA-256 comparison of 317 existing backend/frontend/research files against the
start-of-task inventory found only the intended changes listed above within those
trees. All research files, saved outputs, model artifacts, Function 2 source,
snapshot and styles, and the stable LR page/services remain byte-identical.
The two README/guide edits sit outside that inventory. Root `.gitattributes`,
the manifest, backend default settings and all existing LR API contracts remain
unchanged. Existing uncommitted work was preserved.

After the small final layout refinement (placing the horizon selector inside
the comparison card), unit tests and both builds were rerun successfully and
the enabled preview browser checks passed again: **3 passed, 1 disabled-only
skip**. Final build output has the frontend experimental flag **false**.
Test/build/screenshot outputs are ignored local verification artifacts. No
dependencies were installed; no models were retrained, committed, pushed or
deployed. The temporary local verification servers were stopped after review.
