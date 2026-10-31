import { ApiError, postJson } from '../../../shared/api/apiClient';
import type { FailureRiskRequest, EvidenceProfile } from '../types/component3.types';

export type Horizon = 7 | 14 | 30;
export interface DescriptiveEvidence extends EvidenceProfile {
  error_events_prior: number;
  error_events_7d: number;
  error_events_30d: number;
  error_events_90d: number;
  days_since_last_error: number | null;
}

export interface ExperimentalPrediction {
  machine_id: number;
  prediction_time: string;
  p7: number;
  p14: number;
  p30: number;
  model_version: string;
  model_release: string;
  model_schema: 'unpooled-inclusive-v3';
  experimental: true;
  research_only: true;
  data_status: 'synthetic_azure_historical_demo';
  interval_semantics: string;
  note: string;
  request_id: string;
  evidence_profile: DescriptiveEvidence;
}

export interface RankedV3Machine { rank: number; machine_id: number; p7: number; p14: number; p30: number }
export interface ExperimentalRanking extends Omit<ExperimentalPrediction, 'machine_id' | 'p7' | 'p14' | 'p30' | 'evidence_profile'> {
  horizon_days: Horizon;
  scored_machine_count: 100;
  returned_machine_count: 10;
  machines: RankedV3Machine[];
}

function triplet(p: Record<string, unknown>): boolean {
  return [p.p7, p.p14, p.p30].every(x => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= 1)
    && (p.p7 as number) <= (p.p14 as number) && (p.p14 as number) <= (p.p30 as number);
}
function metadata(p: Record<string, unknown>): boolean {
  return p.experimental === true && p.research_only === true
    && p.data_status === 'synthetic_azure_historical_demo' && p.model_schema === 'unpooled-inclusive-v3'
    && p.model_release === 'v3-run_003-unpooled-hgb-sigmoid'
    && typeof p.model_version === 'string' && /^sha256:[a-f0-9]{64}$/.test(p.model_version)
    && ['interval_semantics', 'note', 'request_id'].every(key => typeof p[key] === 'string' && p[key].length > 0);
}
function evidence(p: unknown): boolean {
  if (!p || typeof p !== 'object') return false;
  const e = p as Record<string, unknown>;
  const nonnegative = (x: unknown) => typeof x === 'number' && Number.isFinite(x) && x >= 0;
  return ['M_prior_maintenance_events', 'F_prior_failure_events', 'error_events_prior', 'error_events_7d', 'error_events_30d', 'error_events_90d']
    .every(key => Number.isInteger(e[key]) && nonnegative(e[key]))
    && ['H_observed_maintenance_span_days', 'R_days_since_latest_maintenance', 'days_since_last_error']
      .every(key => e[key] === null || nonnegative(e[key]))
    && e.C_published_field_completeness_pct === 100;
}

function valid(value: unknown, body: FailureRiskRequest): value is ExperimentalPrediction {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const p = value as Record<string, unknown>;
  return triplet(p) && metadata(p) && evidence(p.evidence_profile)
    && p.machine_id === body.machine_id && typeof p.prediction_time === 'string'
    && p.prediction_time.replace(' ', 'T') === body.prediction_time;
}

export async function rankExperimental(predictionTime: string, horizon: Horizon, signal?: AbortSignal): Promise<ExperimentalRanking> {
  const value = await postJson('component3/experimental/top-risk-machines', { prediction_time: predictionTime, horizon_days: horizon }, signal);
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ApiError('Incompatible v3 ranking response.');
  const r = value as Record<string, unknown>;
  if (!metadata(r) || r.prediction_time !== predictionTime || r.horizon_days !== horizon
      || r.scored_machine_count !== 100 || r.returned_machine_count !== 10 || !Array.isArray(r.machines) || r.machines.length !== 10) {
    throw new ApiError('Incompatible v3 ranking response. No substitute ranking is displayed.');
  }
  let previous = Infinity; let previousId = 0; const seen = new Set<number>();
  for (const [index, row] of r.machines.entries()) {
    if (!row || typeof row !== 'object' || !triplet(row)) throw new ApiError('Invalid v3 ranked probabilities.');
    const score = row[`p${horizon}`];
    if (!Number.isInteger(row.machine_id) || row.machine_id < 1 || row.machine_id > 100 || seen.has(row.machine_id)
        || row.rank !== index + 1 || score > previous || (score === previous && row.machine_id <= previousId)) {
      throw new ApiError('Invalid v3 machine ordering.');
    }
    previous = score; previousId = row.machine_id; seen.add(row.machine_id);
  }
  return value as ExperimentalRanking;
}

export async function predictExperimental(body: FailureRiskRequest, signal?: AbortSignal): Promise<ExperimentalPrediction> {
  let result: unknown;
  try {
    result = await postJson('component3/experimental/predict-risk', body, signal);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      throw new ApiError('The experimental endpoint is disabled or unavailable. Check the backend experimental flag.', 404);
    }
    throw error;
  }
  if (!valid(result, body)) throw new ApiError('The experimental API returned incompatible or incoherent probabilities. No result has been displayed.');
  return result;
}
