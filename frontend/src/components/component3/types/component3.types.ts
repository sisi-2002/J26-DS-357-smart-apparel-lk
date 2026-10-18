/** Mirrors the existing FastAPI Pydantic contract, including nullable evidence. */
export interface FailureRiskRequest {
  machine_id: number;
  prediction_time: string;
}

export interface EvidenceProfile {
  H_observed_maintenance_span_days: number | null;
  M_prior_maintenance_events: number;
  F_prior_failure_events: number;
  C_published_field_completeness_pct: 100;
  R_days_since_latest_maintenance: number | null;
}

export interface FailureRiskResponse extends FailureRiskRequest {
  horizon_days: 7;
  predicted_failure_probability: number;
  calibration: string;
  evidence_profile: EvidenceProfile;
  machine_type: string;
  research_only: true;
  note: string;
  request_id: string;
  model_schema_version: string;
  model_version: string;
  data_status: 'synthetic_azure_historical_demo';
}
