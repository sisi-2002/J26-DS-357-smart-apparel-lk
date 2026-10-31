import type { FailureRiskResponse, TopRiskMachinesResponse } from '../components/component3/types/component3.types';

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

/** Contrived test-only ranking, unrelated to the real benchmark probabilities. */
export function rankingFixture(overrides: Partial<TopRiskMachinesResponse> = {}): TopRiskMachinesResponse {
  return {
    prediction_time: '2015-09-02T06:00:00', horizon_days: 7,
    scored_machine_count: 100, returned_machine_count: 10,
    machines: Array.from({ length: 10 }, (_, index) => ({
      rank: index + 1, machine_id: 20 - index, predicted_failure_probability: Number((0.8 - index * 0.05).toFixed(6)),
    })),
    request_id: 'test-ranking', model_schema_version: 'c3-pdm-7day-v1', model_version: 'sha256:test-only',
    research_only: true, data_status: 'synthetic_azure_historical_demo',
    note: 'Relative ranking of 100 synthetic Azure machines. Test fixture only; no maintenance advice.',
    ...overrides,
  };
}

export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
