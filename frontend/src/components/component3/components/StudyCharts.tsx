import { useId } from 'react';
import type { EvidenceGroup } from '../types/evidenceStudy.types';

export function StudyBars({ title, rows, description }: {
  title: string; description: string; rows: { label: string; value: number; series?: 'evidence' | 'risk' }[];
}) {
  const id = useId();
  // Every comparison starts at zero and uses one explicitly labelled shared scale.
  const max = Math.max(0.01, Math.ceil(Math.max(...rows.map(row => row.value)) * 100) / 100);
  return <div className="study-bars" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
    <span className="visually-hidden" id={`${id}-title`}>{title}</span>
    <span className="visually-hidden" id={`${id}-desc`}>{description} Bar scale: 0 to {max.toFixed(2)}. Exact values are in the adjacent table.</span>
    {rows.map(row => <div className="study-bar-row" key={row.label}>
      <span>{row.label}</span><span className="study-bar-track" aria-hidden="true"><span
        className={`study-bar ${row.series || 'risk'}`} style={{ width: `${row.value / max * 100}%` }} /></span>
      <span>{row.value.toFixed(6)}</span>
    </div>)}
    <div className="study-bar-scale"><span>0 baseline</span><span>Shared maximum: {max.toFixed(2)}</span></div>
  </div>;
}

export function EvidenceLossPlot({ groups, xLabel }: { groups: EvidenceGroup[]; xLabel: string }) {
  const id = useId();
  const min = Math.min(...groups.map(group => group.evidence_min));
  const max = Math.max(...groups.map(group => group.evidence_max));
  const x = (value: number) => 65 + (value - min) / Math.max(max - min, 1) * 495;
  // All saved group means are <0.25. If a later reviewed snapshot exceeds it,
  // expand the axis rather than clipping or normalizing the observed losses.
  const yMax = Math.max(0.25, Math.ceil(Math.max(...groups.map(group => group.mean_squared_prediction_error_brier)) * 20) / 20);
  const plotY = (value: number) => 235 - value / yMax * 195;
  const tick = (value: number) => value.toLocaleString('en-GB', { maximumFractionDigits: 1 });
  return <>
    <p className="study-axis-note">Loss axis starts at 0. Evidence axis: {min}–{max}{min > 0 ? ' (starts above zero)' : ' (starts at zero)'}.</p>
    <div className="study-plot-scroll" tabIndex={0} role="region" aria-label="Evidence-loss chart, scroll horizontally on small screens">
    <svg className="study-loss-plot" viewBox="0 0 620 305" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
    <title id={`${id}-title`}>{xLabel} versus mean held-out squared loss</title>
    <desc id={`${id}-desc`}>{groups.length} observed quantile groups. Points show group means and horizontal lines show observed evidence ranges. Connecting lines guide reading, not a fitted curve or confidence interval. The full data table follows.</desc>
    {[0, 1, 2, 3, 4, 5].map(step => <g key={step}>
      <line x1="65" x2="560" y1={plotY(step * yMax / 5)} y2={plotY(step * yMax / 5)} className="study-gridline" />
      <text x="54" y={plotY(step * yMax / 5) + 4} textAnchor="end">{(step * yMax / 5).toFixed(2)}</text>
    </g>)}
    <text x="65" y="22">Mean squared loss</text>
    <polyline points={groups.map(group => `${x(group.evidence_mean)},${plotY(group.mean_squared_prediction_error_brier)}`).join(' ')} className="study-trend" />
    {groups.map(group => <g key={group.index}>
      <line x1={x(group.evidence_min)} x2={x(group.evidence_max)} y1={plotY(group.mean_squared_prediction_error_brier)} y2={plotY(group.mean_squared_prediction_error_brier)} className="study-range" />
      <circle cx={x(group.evidence_mean)} cy={plotY(group.mean_squared_prediction_error_brier)} r="6" className="study-point">
        <title>Group {group.index}: mean evidence {group.evidence_mean.toFixed(3)}, mean squared loss {group.mean_squared_prediction_error_brier.toFixed(6)}, {group.n_snapshots} cases</title>
      </circle>
    </g>)}
    {[0, 1, 2, 3, 4].map(step => {
      const value = min + (max - min) * step / 4;
      return <g key={step}>
        <line x1={x(value)} x2={x(value)} y1="235" y2="241" className="study-tick" />
        <text x={x(value)} y="263" textAnchor={step === 0 ? 'start' : step === 4 ? 'end' : 'middle'}>{tick(value)}</text>
      </g>;
    })}
    <text x="312" y="288" textAnchor="middle">{xLabel}</text>
  </svg></div>
    <p className="study-scroll-hint">On smaller screens, scroll the chart and numerical table horizontally to see all values.</p>
  </>;
}
