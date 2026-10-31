import type { EvidenceProfile } from '../types/component3.types';

export default function EvidenceSufficiencyPanel({ profile }: { profile: EvidenceProfile }) {
  const days = (value: number | null) => value === null ? 'No prior maintenance observed' : `${value.toLocaleString('en-GB', { maximumFractionDigits: 2 })} days`;
  const rows = [
    { key: 'H', label: 'Observed maintenance history span', value: days(profile.H_observed_maintenance_span_days),
      detail: 'Not guaranteed log coverage.' },
    { key: 'M', label: 'Previous recorded maintenance events', value: profile.M_prior_maintenance_events.toLocaleString('en-GB') },
    { key: 'F', label: 'Previous recorded failure events', value: profile.F_prior_failure_events.toLocaleString('en-GB') },
    { key: 'C', label: 'Completeness of published benchmark fields', value: `${profile.C_published_field_completeness_pct}%`,
      detail: 'Constant at 100% here; not factory data quality.' },
    { key: 'R', label: 'Days since last recorded maintenance event', value: days(profile.R_days_since_latest_maintenance),
      detail: 'Event recency, not database freshness.' },
  ];
  return <section className="panel evidence-panel" aria-labelledby="evidence-heading">
    <div className="evidence-heading"><h3 id="evidence-heading">Maintenance-evidence profile</h3>
      <span>H / M / F / C / R</span></div>
    <p className="panel-description">Recorded history for this machine at the selected snapshot.</p>
    <dl className="evidence-grid">{rows.map(row => <div className="metric-card" key={row.key}>
      <dt><span className="evidence-symbol" aria-hidden="true">{row.key}</span>{row.label}</dt>
      <dd data-testid={`evidence-${row.key}`}>{row.value}</dd>
      {row.detail && <dd className="metric-explanation">{row.detail}</dd>}
    </div>)}</dl>
  </section>;
}
