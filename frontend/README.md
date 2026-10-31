# Component 3 frontend — historical machine risk

React + TypeScript + Vite, with one modular feature at `/component3/predict-risk`.
This is a **synthetic Microsoft Azure benchmark historical demonstration**, not a
live monitoring system or validated apparel-factory predictor. The browser calls
the existing FastAPI endpoint; it contains no model, predictions, training code,
maintenance recommendations or individual reliability scores.

## Windows CMD: run both servers

Use Node.js 22.12 or newer (Node 22.13 is available on the development machine).
Use two separate Windows CMD windows.

Backend window, using a Python environment with the existing backend dependencies
installed and the existing artifact:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk
set "C3_MODEL_PATH=E:\J26-DS-357-smart-apparel-lk\research\component3\models\c3_7day_sigmoid.joblib"
set "C3_DATA_DIR=E:\Downloads\Research_Com_3\C3_PP1_Code_Review_Corrections\raw"
python -B -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

If backend dependencies are not installed in your active Python environment,
follow the isolated-environment setup in `backend/README.md` first. Adjust the
dataset path if necessary. No model export or retraining is needed.

Frontend window, first setup:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\frontend
npm ci
if not exist .env.local copy .env.example .env.local
npm run dev
```

On later runs, use `npm run dev`; keep your `.env.local`. In PowerShell use
`npm.cmd` if script execution policy blocks `npm.ps1`.

The supplied lockfile supports `npm ci` with the installed npm 11.2. When refreshing
dependencies, older npm versions can hit the optional-peer `edgesOut` resolver
bug. The dependency refresh here used a temporary npm 11.6.2 without replacing
global npm: `npm exec --yes --package=npm@11.6.2 -- npm install`.

Open http://127.0.0.1:5173/component3/predict-risk.

## Official shared visual theme

`src/app/styles.css` defines the Smart Apparel-LK Modern Textile Industrial theme
in Light Mode: Deep Navy `#142D4E`, Textile Teal `#087F8C`, Soft White `#F5F7F8`,
Sage Mist `#D9E9E8`, and Warm Amber `#E4A75A`. Shared semantic tokens control the
shell, cards, controls, selections and notices on both Component 3 pages; feature
styles consume them without defining competing brand palettes.

White text on primary teal, navy text on selected sage rows, and darker teal link
text preserve AA contrast. Muted blue research/chart accents and the existing coral
ordinal markers remain distinct from alert or severity classifications. Typography,
responsive grids, chart scales, routes and prediction/research behavior are unchanged.
Browser checks verify rendered palette usage, text contrast, focus and both pages
at desktop, tablet and mobile widths. No Component 1 source was present for a more
specific visual comparison; alignment follows the supplied official palette.

## API configuration and CORS

`.env.local` is ignored. Only `.env.example` should be tracked:

```dotenv
VITE_API_BASE_URL=/api/v1
API_PROXY_TARGET=http://127.0.0.1:8000
```

The shared API client combines the configured base with
`component3/predict-risk`. Vite proxies `/api` to FastAPI without rewriting the
path. The browser uses the frontend origin, so no backend CORS change is required.
The proxy is configured for both development and local production preview. Restart
Vite after changing environment variables. An unset base URL produces a clear
configuration error; there is no fallback prediction or hardcoded service URL in
application code. The proxy target belongs to Vite's server environment.

`VITE_` values are public build-time configuration: never place credentials, raw
dataset paths or model paths there. Frontend runtime does not need those files.
Vite's proxy is not included in the static production bundle: production hosting
must route `/api` to FastAPI and serve `index.html` for frontend routes. If hosting
frontend and API on separate origins later, an explicit FastAPI CORS allowlist must
be agreed separately. Setting an absolute API URL alone does not enable CORS.

## Interaction and interpretation

- The ID list is the current benchmark's 100 synthetic machines, numbered 1–100.
  Thirteen permitted snapshots run weekly from September 2 to November 25, 2015,
  at 06:00. These are public input metadata; no source CSVs are bundled.
- Timestamp strings are sent unchanged without `Date`, `toISOString`, `Z` or a
  timezone offset, preserving the backend's timezone-naive contract.
- Individual prediction happens only through **Predict selected machine** or **View**. The backend remains authoritative
  for available IDs and timestamps. Its six-decimal probability is displayed as
  a percentage to two decimals, with the returned number also shown.
- H is elapsed time since the earliest observed maintenance, not verified logging
  coverage. M/F count prior maintenance/failures. C is constant published-field
  completeness, not factory CMMS completeness or prediction reliability. R is
  maintenance-event recency, not database freshness. Missing H/R is displayed as
  “No prior maintenance observed”, never substituted with zero.
- Buttons expose pending state; duplicate requests are prevented. Changing inputs
  clears the old result and aborts pending work. A request-generation check also
  ignores late replies if the transport fails to honor cancellation. Unmounting
  aborts pending work. Network requests time out after 30 seconds.
- HTTP errors, Pydantic validation arrays, network failures and incompatible
  response bodies are handled without substituting results. The client checks
  the returned machine/time, seven-day horizon, probability, evidence and
  synthetic/research flags before displaying it.
- The model's `model1`–`model4` labels remain synthetic labels, not apparel-machine
  categories. Backend limitations, calibration, artifact identity and request ID
  remain available with each result. No severity thresholds are invented.

## Tests and builds

From `frontend`:

```cmd
npm test
npm run typecheck
npm run build
```

Vitest + React Testing Library tests use clearly identified **test-only** fixtures
and mocked HTTP transport. Fixtures are not imported by runtime application code.
Tests cover the exact payload and choices, response validation/rendering, loading,
null evidence, failures, cancellation, stale results, unmount cleanup and routing.

For real end-to-end verification, leave both servers running and run:

```cmd
npm run test:e2e
```

Playwright uses installed Microsoft Edge on Windows, headlessly. No inference or
network response is mocked in this suite. It opens the React route, submits the
form, checks the real proxied response and rendered values, and verifies desktop
and mobile layouts. Backend model/data must be configured. Screenshots and failed
traces go into ignored `test-results/`. Override the frontend origin with
`E2E_BASE_URL`; override the browser channel with `PLAYWRIGHT_CHANNEL`. On systems
without Edge, install Playwright Chromium (`npx playwright install chromium`) and
set `PLAYWRIGHT_CHANNEL=chromium`.

To inspect the production build locally, stop the dev server and run
`npm run preview`. Open http://127.0.0.1:4173/component3/predict-risk; the same
configured proxy is available for this local preview.

## Group integration

`src/app` owns the small shared shell and route registry. `src/shared` owns the
layout, JSON API client and error component. Component 3 exports its route objects
and owns its API contracts, inputs, request hook and prediction displays. Future
group integration can register `component3Routes` inside the shared React Router
without copying business logic or nesting a second `BrowserRouter`.

This milestone adds no authentication, database storage, SHAP/survival outputs or
new backend endpoints. Merge the shared shell, dependency lockfile, configuration
and API client with the group's chosen owners before final integration. If the
benchmark changes, update the input metadata together with the backend contract.
Keep `.env.local`, `node_modules`, builds, test outputs, source datasets and model
artifacts out of Git. No Git commit or push is part of the frontend workflow.

## Created source files

```text
frontend/
  .env.example
  .gitignore
  README.md
  index.html
  package.json
  package-lock.json
  tsconfig.json
  vite.config.ts
  playwright.config.ts
  e2e/dashboard-layout.spec.ts
  e2e/prediction.spec.ts
  e2e/top-risk-machines.spec.ts
  src/main.tsx
  src/vite-env.d.ts
  src/app/App.tsx
  src/app/AppRoutes.tsx
  src/app/styles.css
  src/shared/api/apiClient.ts
  src/shared/api/apiClient.test.ts
  src/shared/components/ErrorMessage.tsx
  src/shared/layouts/MainLayout.tsx
  src/components/component3/component3.css
  src/components/component3/routes.tsx
  src/components/component3/api/failureRiskApi.ts
  src/components/component3/api/failureRiskApi.test.ts
  src/components/component3/api/topRiskMachinesApi.ts
  src/components/component3/api/topRiskMachinesApi.test.ts
  src/components/component3/types/component3.types.ts
  src/components/component3/data/demoOptions.ts
  src/components/component3/hooks/useFailureRisk.ts
  src/components/component3/hooks/useTopRiskMachines.ts
  src/components/component3/pages/FailureRiskPredictionPage.tsx
  src/components/component3/pages/FailureRiskPredictionPage.test.tsx
  src/components/component3/pages/TopRiskMachines.test.tsx
  src/components/component3/pages/DashboardLayout.test.tsx
  src/components/component3/components/PredictionControls.tsx
  src/components/component3/components/ResearchDetails.tsx
  src/components/component3/components/MachineSelector.tsx
  src/components/component3/components/PredictionDateSelector.tsx
  src/components/component3/components/FailureRiskCard.tsx
  src/components/component3/components/EvidenceSufficiencyPanel.tsx
  src/components/component3/components/SyntheticDataNotice.tsx
  src/components/component3/components/TopRiskMachinesPanel.tsx
  src/test/setup.ts
  src/test/fixtures.ts
```

Local `.env.local`, installed packages, production assets and browser-test outputs
are ignored generated/local files, not part of the source inventory above.

## Top 10 Highest-Risk Machines

The prediction page automatically loads **Highest Failure-Risk Machines (Next 7 Days)**
on entry and whenever Prediction snapshot changes. **Refresh ranking** retries or
reloads the selected date. Each comparison sends one timestamp-only request to
`POST /api/v1/component3/top-risk-machines`; it does not issue 100 individual
requests. All results come from the saved calibrated model on the backend.

The table retains the backend's ordering and displays the returned probabilities
as percentages to two decimal places. Bars and selection retain the returned values.
Backend ordering uses unrounded probabilities, so equal display
values need not be sorted by machine ID. No severity thresholds are added. The
ranking is explicitly relative to the 100 machines in the synthetic Azure
benchmark at this historical date, not a reliability assessment or maintenance
recommendation.

Each **View** button selects the machine and requests its full prediction and
H/M/F/C/R profile through the existing endpoint. All ten ranking rows remain while
switching machines on the same date. Date changes clear stale outputs, automatically
reload the ranking, preserve the machine selector and abort outdated requests.
Late ranking replies are ignored even if cancellation is not
honored. Detail responses must match the ranked probability and model identity;
otherwise the page asks for a refreshed ranking instead of associating mismatched
evidence. Null H/R handling and the original individual workflow remain intact.

New unit tests cover ranking response validation, ordering, request count, row
selection, loading, retry, cancellation and model mismatch. The existing real
browser command also runs the new ranking suite: desktop verifies evidence for
all ten ranked machines, and mobile verifies switching between two while retaining
the list. Both suites use the real API and no model mocks. Existing individual
prediction browser tests remain included.

The Vite proxy and environment variables require no changes. Restart the backend
to load the new route. If standard ports are already occupied, a separate test
instance can use `uvicorn ... --port 8001`, `API_PROXY_TARGET=http://127.0.0.1:8001`,
`npm run dev -- --port 5174`, and `E2E_BASE_URL=http://127.0.0.1:5174`; this avoids
stopping another developer's servers.

## Dashboard layout and visual verification

The page title is **Machine Failure-Risk Explorer**. Its full-width controls share
one historical date between direct prediction and comparison, with a fixed 7-day
horizon. Direct prediction works even while ranking is loading or unavailable. Comparison never
automatically changes the machine selector or requests an individual prediction.

On desktop, Highest Failure-Risk Machines and Selected Machine Prediction occupy approximately
40% and 60% of the result area. At tablet/mobile widths they stack, with compact
empty states and no stretched blank prediction card. View selects a ranked machine,
requests its existing detailed prediction, and highlights the selector's machine
in the ranking. Ranking rows remain when switching machines at the same date.
On stacked layouts, explicit View actions focus and reveal the prediction heading.
Changing dates retains the existing reset/cancellation/stale-response behavior.

The five evidence indicators are metric cards, including unchanged H/C/R caveats
and null handling. A compact prototype note identifies synthetic Azure 2015 data
and illustrative production lines. Full benchmark provenance and factory-validation
limitations remain inside **Model & Data Info**, a native keyboard-operable
disclosure, initially collapsed, containing separate prediction/ranking notes and
provenance. Model hashes and request IDs are not shown outside that disclosure.
Shared sidebar branding remains in MainLayout; feature presentation is scoped in
component3.css. Existing API clients, request hooks and contracts are unchanged.

`data/machineDisplay.ts` supplies neutral names and stable illustrative line labels
for all 100 machines: IDs 001–010 map to Line 01, 011–020 to Line 02, through
091–100 to Line 10. The selector, ranking and prediction card reuse this mapping.
These are presentation labels, not verified factory assignments or ML features.
Requests still contain only the existing numeric machine ID and historical timestamp.

The dashboard browser suite checks empty, ranking-only and selected-prediction
states at 1440px, 1024px and 390px widths. It also checks that comparison sends no
individual prediction request, selected rows are highlighted, the disclaimer stays
visible after scrolling, Model & Data Info supports keyboard expansion, and the
page has no horizontal overflow. Existing real prediction/ranking tests remain.
Run `npm run test:e2e` with both servers running. Its generated screenshots are:

```text
test-results/dashboard/desktop-empty.png
test-results/dashboard/desktop-ranking-only.png
test-results/dashboard/desktop-selected-prediction.png
test-results/dashboard/tablet-empty.png
test-results/dashboard/tablet-ranking-only.png
test-results/dashboard/tablet-selected-prediction.png
test-results/dashboard/mobile-empty.png
test-results/dashboard/mobile-ranking-only.png
test-results/dashboard/mobile-selected-prediction.png
```

These screenshots and all browser outputs remain ignored by Git. The next browser
run replaces them. No backend, research, model or API changes are needed for this
presentation redesign.

## Function 2 / N2 offline evidence study

Open `/component3/evidence-study` or **Evidence study** in the sidebar. This page
uses the small, source-controlled aggregate snapshot
`src/components/component3/data/n2Study.snapshot.json`. It works with the normal
frontend start/build commands on a clean checkout: no ignored experiment CSVs,
raw dataset, Python environment, saved ML model or FastAPI server is required to
view N2. The reviewed JSON must be included with the new source files when you
eventually commit; it is intentionally **not** ignored. No per-machine research
rows are copied into the frontend and no model is fitted or invoked by the page.

N2 uses the **original uncalibrated, separately refitted pilot classifiers**.
Function 1 continues to use its separate calibrated 7-day inference API. Brier
score measures the original failure predictions' mean squared loss. The later
MAE/RMSE/R² comparisons evaluate **second-stage loss estimation**. Risk-only had
slightly lower MAE than full risk-plus-evidence in all three conditions; this
negative finding is shown. H does not establish logging coverage, R is maintenance
event recency rather than database freshness, and C is constant 100%, excluded
from relationship fitting. Four quantile groups for 90-day F are preserved.

Only regeneration requires the existing local saved results under
`research/component3/outputs_corrected`, `outputs_n2_continuous`, and
`outputs_n2_incremental`. From the repository root in Windows CMD:

```cmd
python -B frontend\scripts\export_n2_study.py --check
python -B frontend\scripts\export_n2_study.py
python -B -m unittest discover -s frontend\scripts -p test_export_n2_study.py -v
```

The first command verifies sources and the current snapshot without writing. The
second regenerates it; review the resulting diff. `--source-dir` accepts another
Component 3 research-results directory, and `--output` selects a different JSON
destination. The exporter uses only the Python standard library (also works with
`python -S -B ...`), imports no research or ML code, and validates paired labels,
squared losses, evidence aliases, constant C, quantile groups, descriptive
correlations, OOF machine folds, saved metrics and deltas. Missing or inconsistent
sources fail with a nonzero exit code before replacing the output. Provenance
records source-relative filenames, row counts, byte sizes and SHA-256 hashes.

Normal `npm test` checks the included snapshot without local research outputs.
The separate Python regeneration tests explicitly skip when sources are absent.
N2 browser tests block API access and verify all 12 history/evidence selections,
numbers, four-group F, provenance, navigation, keyboard focus and no page overflow
at desktop/tablet/mobile widths. Screenshots are generated under
`test-results/evidence-study/{desktop,tablet,mobile}-{overview,90day-f}.png`.
Run against Vite with both servers for the full Function 1 + N2 regression suite:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\frontend
npm test
npm run build
npm run test:e2e
```

To verify N2 on the **production bundle without backend access**, start
`npm run preview` in one CMD window, then use another:

```cmd
cd /d E:\J26-DS-357-smart-apparel-lk\frontend
set E2E_BASE_URL=http://127.0.0.1:4173
npm run test:e2e -- evidence-study.spec.ts
```

## Optional isolated v3 multi-horizon preview

Default: **off**. The existing 7-day dashboard, ranking and N2 page remain the
normal application. For the separately audited v3 conditional HGB preview, set
`VITE_C3_EXPERIMENTAL_MULTIHORIZON=true` before starting Vite or building, and
enable the separate backend flag with its audited local artifact. Visit
`/component3/predict-risk` (the former `/component3/experimental-risk` redirects
there). The enabled main dashboard includes v3-only 7/14/30-day ranking and
descriptive evidence. All three cards use that one v3 system; none uses
the deployed LR. No fallback prediction is substituted when unavailable.

Full Windows CMD setup, actual request/response, audit, safe-disable instructions
and test commands: [experimental preview guide](../backend/EXPERIMENTAL_MULTIHORIZON.md).
Current product behavior and API additions: [unified dashboard guide](../backend/UNIFIED_MULTIHORIZON.md).
The public frontend flag controls visibility, not authentication. Restart/rebuild
after changing it. Leave both flags unset/false for the normal app.
