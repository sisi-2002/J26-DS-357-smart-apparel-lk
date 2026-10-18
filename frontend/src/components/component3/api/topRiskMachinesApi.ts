import { ApiError, postJson } from '../../../shared/api/apiClient';
import { PREDICTION_TIMESTAMPS } from '../data/demoOptions';
import type { RankedMachine, TopRiskMachinesResponse } from '../types/component3.types';

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function rankedMachine(value: unknown): value is RankedMachine {
  return record(value) && typeof value.rank === 'number' && Number.isInteger(value.rank)
    && typeof value.machine_id === 'number' && Number.isInteger(value.machine_id)
    && value.machine_id >= 1 && value.machine_id <= 100
    && typeof value.predicted_failure_probability === 'number'
    && Number.isFinite(value.predicted_failure_probability)
    && value.predicted_failure_probability >= 0 && value.predicted_failure_probability <= 1;
}
function validRanking(value: unknown): value is TopRiskMachinesResponse {
  if (!record(value) || !Array.isArray(value.machines) || value.machines.length !== 10
    || !value.machines.every(rankedMachine)) return false;
  const rows = value.machines;
  return value.horizon_days === 7 && value.scored_machine_count === 100 && value.returned_machine_count === 10
    && value.research_only === true && value.data_status === 'synthetic_azure_historical_demo'
    && ['prediction_time', 'request_id', 'model_schema_version', 'model_version', 'note']
      .every(key => typeof value[key] === 'string' && value[key].length > 0)
    && new Set(rows.map(row => row.machine_id)).size === 10
    && rows.every((row, index) => row.rank === index + 1
      && (index === 0 || row.predicted_failure_probability <= rows[index - 1].predicted_failure_probability));
  // Equal serialized probabilities need not be ID-sorted: their raw model
  // probabilities may differ. Full-precision tie handling belongs to the backend.
}

export async function getTopRiskMachines(predictionTime: string, signal?: AbortSignal): Promise<TopRiskMachinesResponse> {
  if (!PREDICTION_TIMESTAMPS.some(time => time === predictionTime)) {
    throw new ApiError('Choose a supported historical snapshot before ranking.');
  }
  const response = await postJson('component3/top-risk-machines', { prediction_time: predictionTime }, signal);
  if (!validRanking(response)) throw new ApiError('The API returned an incompatible ranking. No ranking has been displayed.');
  if (response.prediction_time.replace(' ', 'T') !== predictionTime) {
    throw new ApiError('The API returned a ranking for a different snapshot. Please retry.');
  }
  return response;
}
