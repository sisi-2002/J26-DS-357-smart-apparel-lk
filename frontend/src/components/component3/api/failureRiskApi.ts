import { ApiError, postJson } from '../../../shared/api/apiClient';
import { PREDICTION_TIMESTAMPS } from '../data/demoOptions';
import type { FailureRiskRequest, FailureRiskResponse } from '../types/component3.types';

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function nonnegative(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}
function validResponse(value: unknown): value is FailureRiskResponse {
  if (!record(value) || !record(value.evidence_profile)) return false;
  const evidence = value.evidence_profile;
  return Number.isInteger(value.machine_id) && nonnegative(value.machine_id) && value.machine_id > 0
    && typeof value.prediction_time === 'string'
    && PREDICTION_TIMESTAMPS.some(time => time === value.prediction_time?.toString().replace(' ', 'T'))
    && value.horizon_days === 7 && value.research_only === true
    && value.data_status === 'synthetic_azure_historical_demo'
    && nonnegative(value.predicted_failure_probability) && value.predicted_failure_probability <= 1
    && ['calibration', 'machine_type', 'note', 'request_id', 'model_schema_version', 'model_version']
      .every(key => typeof value[key] === 'string' && value[key].length > 0)
    && (evidence.H_observed_maintenance_span_days === null || nonnegative(evidence.H_observed_maintenance_span_days))
    && (evidence.R_days_since_latest_maintenance === null || nonnegative(evidence.R_days_since_latest_maintenance))
    && Number.isInteger(evidence.M_prior_maintenance_events) && nonnegative(evidence.M_prior_maintenance_events)
    && Number.isInteger(evidence.F_prior_failure_events) && nonnegative(evidence.F_prior_failure_events)
    && evidence.C_published_field_completeness_pct === 100;
}

export async function predictFailureRisk(body: FailureRiskRequest, signal?: AbortSignal): Promise<FailureRiskResponse> {
  const response = await postJson('component3/predict-risk', body, signal);
  if (!validResponse(response)) throw new ApiError('The API returned an incompatible prediction response. No result has been displayed.');
  if (response.machine_id !== body.machine_id || response.prediction_time.replace(' ', 'T') !== body.prediction_time) {
    throw new ApiError('The API returned a different machine or timestamp. Please retry your selection.');
  }
  return response;
}
