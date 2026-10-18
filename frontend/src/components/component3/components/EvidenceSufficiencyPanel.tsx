import type { EvidenceProfile } from '../types/component3.types';

export default function EvidenceSufficiencyPanel({ profile }: { profile: EvidenceProfile }) {
  const days = (value: number | null) => value === null ? 'No prior maintenance observed' : `${value.toLocaleString('en-GB', { maximumFractionDigits: 2 })} days`;
  const rows = [
    { key: 'H', label: 'Observed history span', value: days(profile.H_observed_maintenance_span_days),
      detail: 'Time since the earliest observed maintenance event. This does not establish logging coverage.' },
    { key: 'M', label: 'Prior maintenance events', value: profile.M_prior_maintenance_events.toLocaleString('en-GB'),
      detail: 'Recorded maintenance events strictly before the selected snapshot.' },
    { key: 'F', label: 'Prior failure events', value: profile.F_prior_failure_events.toLocaleString('en-GB'),
      detail: 'Recorded failures strictly before the selected snapshot.' },
    { key: 'C', label: 'Published-field completeness', value: `${profile.C_published_field_completeness_pct}%`,
      detail: 'Constant completeness of published Azure benchmark fields; not factory CMMS quality or prediction reliability.' },
    { key: 'R', label: 'Maintenance recency', value: days(profile.R_days_since_latest_maintenance),
      detail: 'Time since the most recent maintenance event, not database freshness.' },
  ];
  return <section className="panel evidence-panel" aria-labelledby="evidence-heading">
    <div className="evidence-heading"><h3 id="evidence-heading">Maintenance-evidence profile</h3>
      <span>H / M / F / C / R</span></div>
    <p className="panel-description">These characteristics describe available history. They do not measure an individual prediction’s reliability.</p>
    <dl className="evidence-grid">{rows.map(row => <div className="metric-card" key={row.key}>
      <dt><span className="evidence-symbol" aria-hidden="true">{row.key}</span>{row.label}</dt>
      <dd data-testid={`evidence-${row.key}`}>{row.value}</dd>
      <dd className="metric-explanation">{row.detail}</dd>
    </div>)}</dl>
  </section>;
}
