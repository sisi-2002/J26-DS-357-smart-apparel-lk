import type { FailureRiskResponse } from '../components/component3/types/component3.types';

/** Test-only data. Never imported by application code or used as a fallback. */
export function riskFixture(overrides: Partial<FailureRiskResponse> = {}): FailureRiskResponse {
  return {
    machine_id: 1, prediction_time: '2015-09-02T06:00:00', horizon_days: 7,
    predicted_failure_probability: 0.271828,
    calibration: 'Test fixture: chronological sigmoid calibration',
    evidence_profile: { H_observed_maintenance_span_days: 42, M_prior_maintenance_events: 3,
      F_prior_failure_events: 2, C_published_field_completeness_pct: 100, R_days_since_latest_maintenance: 1.5 },
    machine_type: 'model2', research_only: true,
    note: 'Synthetic Azure benchmark; test fixture only. Not validated for apparel factories.',
    request_id: 'test-request', model_schema_version: 'c3-pdm-7day-v1',
    model_version: 'sha256:test-only', data_status: 'synthetic_azure_historical_demo',
    ...overrides,
  };
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
