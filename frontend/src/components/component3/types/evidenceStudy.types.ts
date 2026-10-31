export type EvidenceSymbol = 'H' | 'M' | 'F' | 'R';
export interface EvidenceGroup {
  index: number; n_snapshots: number; n_machines: number;
  evidence_mean: number; evidence_min: number; evidence_max: number;
  mean_squared_prediction_error_brier: number; observed_failure_prevalence: number;
}
export interface StudyMetric { model: string; mae: number; rmse: number; r2: number }
export interface StudyCondition {
  id: string; label: string; short_label: string; lookback_days: number | null;
  snapshots: number; machines: number; dates: number; positive_cases: number; brier: number;
  evidence: Record<EvidenceSymbol, { rho: number; observations: number; groups: EvidenceGroup[] }>;
  evidence_cv: {
    mae_constant_mean_baseline: number; mae_evidence_ridge: number;
    rmse_constant_mean_baseline: number; rmse_evidence_ridge: number; r2_evidence_ridge: number;
  };
  incremental: StudyMetric[]; delta_mae_risk_minus_evidence: number;
}
export interface EvidenceStudy {
  schema_version: string; model_family: string; horizon_days: number; C_constant_pct: number;
  cohort: { first_snapshot: string; last_snapshot: string };
  conditions: StudyCondition[];
  methodology: {
    dataset: string; calibration_note: string; loss_definition: string; relationship_estimator: string;
    H_definition: string; M_definition: string; F_definition: string; C_definition: string; R_definition: string;
    scope_limitations: string[];
  };
  incremental_methodology: { risk_only_features: string[]; risk_plus_evidence_features: string[]; validity_warning: string };
  provenance: { path: string; sha256: string; bytes: number; rows?: number }[];
}
