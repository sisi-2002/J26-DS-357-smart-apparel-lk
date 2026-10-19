import snapshot from '../data/n2Study.snapshot.json';
import type { EvidenceStudy } from '../types/evidenceStudy.types';

export const bundledEvidenceStudy: unknown = snapshot;

/** The page uses the versioned bundle only. Never fetch a model or substitute results. */
export function readEvidenceStudy(value: unknown): EvidenceStudy {
  const fail = () => { throw new Error('The bundled N2 research snapshot is missing, invalid or inconsistent. Verify or regenerate it from the saved experiment outputs.'); };
  const object = (item: unknown): Record<string, unknown> => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return fail();
    return item as Record<string, unknown>;
  };
  const array = (item: unknown): unknown[] => Array.isArray(item) ? item : fail();
  const text = (item: unknown) => typeof item === 'string' && item.length > 0 ? item : fail();
  const number = (item: unknown, min = -Infinity, max = Infinity): number =>
    typeof item === 'number' && Number.isFinite(item) && item >= min && item <= max ? item : fail();
  const integer = (item: unknown, min = 0): number => Number.isInteger(number(item, min)) ? item as number : fail();
  const strings = (item: unknown) => { const rows = array(item); if (!rows.length) fail(); rows.forEach(text); };
  const close = (a: number, b: number) => { if (Math.abs(a - b) > 1e-9) fail(); };
  const root = object(value);
  if (root.schema_version !== 'n2-study-v1' || root.model_family !== 'original_uncalibrated_n2_pilot'
    || root.horizon_days !== 7 || root.C_constant_pct !== 100) fail();
  const cohort = object(root.cohort);
  if (!Number.isFinite(Date.parse(text(cohort.first_snapshot))) || !Number.isFinite(Date.parse(text(cohort.last_snapshot)))) fail();
  const conditions = array(root.conditions);
  if (conditions.length !== 3) fail();
  const ids = ['full', '180d', '90d'];
  const names = ['Full observed history', '180-day history', '90-day history'];
  const models = ['constant_mean_loss', 'risk_only', 'risk_plus_recency', 'risk_plus_HMF_machine', 'risk_plus_HMFR_machine'];
  const counts: number[][] = [];
  conditions.forEach((raw, index) => {
    const condition = object(raw);
    if (condition.id !== ids[index] || condition.label !== names[index] || condition.lookback_days !== [null, 180, 90][index]) fail();
    text(condition.short_label);
    const n = integer(condition.snapshots, 1); const machines = integer(condition.machines, 1);
    const positives = integer(condition.positive_cases); const dates = integer(condition.dates, 1);
    if (positives > n || n !== machines * dates) fail();
    counts.push([n, machines, dates, positives]);
    const brier = number(condition.brier, 0, 1);
    const evidence = object(condition.evidence);
    for (const symbol of ['H', 'M', 'F', 'R']) {
      const series = object(evidence[symbol]); number(series.rho, -1, 1);
      if (integer(series.observations) !== n) fail();
      const groups = array(series.groups);
      // The saved 90-day F series has four groups because quantile edges tie.
      if (groups.length !== (index === 2 && symbol === 'F' ? 4 : 5)) fail();
      let total = 0; let weightedLoss = 0;
      groups.forEach((rawGroup, groupIndex) => {
        const group = object(rawGroup);
        if (group.index !== groupIndex + 1) fail();
        const size = integer(group.n_snapshots, 1); total += size;
        if (integer(group.n_machines, 1) > machines) fail();
        const min = number(group.evidence_min, 0); const max = number(group.evidence_max, min);
        number(group.evidence_mean, min, max);
        weightedLoss += size * number(group.mean_squared_prediction_error_brier, 0, 1);
        number(group.observed_failure_prevalence, 0, 1);
      });
      if (total !== n) fail(); close(weightedLoss / total, brier);
    }
    const cv = object(condition.evidence_cv);
    ['mae_constant_mean_baseline', 'mae_evidence_ridge', 'rmse_constant_mean_baseline', 'rmse_evidence_ridge'].forEach(field => number(cv[field], 0, 1));
    number(cv.r2_evidence_ridge);
    const alternatives = array(condition.incremental);
    if (alternatives.length !== models.length) fail();
    alternatives.forEach((rawMetric, i) => {
      const metric = object(rawMetric); if (metric.model !== models[i]) fail();
      number(metric.mae, 0, 1); number(metric.rmse, 0, 1); number(metric.r2);
    });
    close(number(object(alternatives[1]).mae) - number(object(alternatives[4]).mae), number(condition.delta_mae_risk_minus_evidence));
  });
  if (counts.some(row => row.some((count, i) => count !== counts[0][i]))) fail();
  const method = object(root.methodology);
  ['dataset', 'calibration_note', 'loss_definition', 'relationship_estimator', 'H_definition', 'M_definition', 'F_definition', 'C_definition', 'R_definition'].forEach(field => text(method[field]));
  strings(method.scope_limitations);
  const incremental = object(root.incremental_methodology);
  strings(incremental.risk_only_features); strings(incremental.risk_plus_evidence_features); text(incremental.validity_warning);
  const sources = array(root.provenance);
  if (sources.length !== 12) fail();
  const paths = new Set<string>();
  sources.forEach(raw => {
    const source = object(raw); const path = text(source.path);
    if (!path.startsWith('research/component3/outputs_') || paths.has(path) || !/^[a-f0-9]{64}$/.test(text(source.sha256))) fail();
    paths.add(path); integer(source.bytes, 1); if (source.rows !== undefined) integer(source.rows, 1);
  });
  return value as EvidenceStudy;
}
