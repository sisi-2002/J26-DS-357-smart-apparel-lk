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
- Prediction happens only on form submission. The backend remains authoritative
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
  e2e/prediction.spec.ts
  src/main.tsx
  src/vite-env.d.ts
  src/app/App.tsx
  src/app/AppRoutes.tsx
  src/app/styles.css
  src/shared/api/apiClient.ts
  src/shared/api/apiClient.test.ts
  src/shared/components/ErrorMessage.tsx
  src/shared/layouts/MainLayout.tsx
  src/components/component3/routes.tsx
  src/components/component3/api/failureRiskApi.ts
  src/components/component3/api/failureRiskApi.test.ts
  src/components/component3/types/component3.types.ts
  src/components/component3/data/demoOptions.ts
  src/components/component3/hooks/useFailureRisk.ts
  src/components/component3/pages/FailureRiskPredictionPage.tsx
  src/components/component3/pages/FailureRiskPredictionPage.test.tsx
  src/components/component3/components/MachineSelector.tsx
  src/components/component3/components/PredictionDateSelector.tsx
  src/components/component3/components/FailureRiskCard.tsx
  src/components/component3/components/EvidenceSufficiencyPanel.tsx
  src/components/component3/components/SyntheticDataNotice.tsx
  src/test/setup.ts
  src/test/fixtures.ts
```

Local `.env.local`, installed packages, production assets and browser-test outputs
are ignored generated/local files, not part of the source inventory above.
