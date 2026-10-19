import { useState } from 'react';
import { bundledEvidenceStudy, readEvidenceStudy } from '../api/evidenceStudyData';
import { EvidenceLossPlot, StudyBars } from '../components/StudyCharts';
import SyntheticDataNotice from '../components/SyntheticDataNotice';
import type { EvidenceStudy, EvidenceSymbol } from '../types/evidenceStudy.types';
import '../evidence-study.css';

const evidenceLabels: Record<EvidenceSymbol, string> = {
  H: 'Observed history span (days)', M: 'Prior maintenance events',
  F: 'Prior failure events', R: 'Days since latest maintenance',
};
const modelLabels: Record<string, string> = {
  constant_mean_loss: 'Constant mean loss', risk_only: 'Risk only (p, p²)',
  risk_plus_recency: 'Risk + R', risk_plus_HMF_machine: 'Risk + H/M/F + machine type',
  risk_plus_HMFR_machine: 'Risk + H/M/F/R + machine type',
};
const fixed = (value: number) => value.toFixed(6);

export default function EvidenceStudyPage({ snapshot = bundledEvidenceStudy }: { snapshot?: unknown }) {
  const [conditionId, setConditionId] = useState('full');
  const [symbol, setSymbol] = useState<EvidenceSymbol>('H');
  let data: EvidenceStudy;
  try { data = readEvidenceStudy(snapshot); }
  catch (error) {
    return <div className="evidence-study"><h1>Maintenance Evidence Sufficiency Study</h1><SyntheticDataNotice />
      <div className="study-caution" role="alert"><h2>Research results unavailable</h2>
        <p>{error instanceof Error ? error.message : 'Unable to read the bundled research results.'}</p>
        <p>No substitute experimental results are displayed.</p></div></div>;
  }
  const selected = data.conditions.find(condition => condition.id === conditionId)!;
  const series = selected.evidence[symbol];
  const first = data.conditions[0];
  const lowestBrier = data.conditions.reduce((lowest, condition) => condition.brier < lowest.brier ? condition : lowest);
  const allRiskOnlyBetter = data.conditions.every(condition => condition.delta_mae_risk_minus_evidence < 0);
  const comparisonBars = data.conditions.flatMap(condition => [
    { label: `${condition.short_label} · risk only`, value: condition.incremental[1].mae, series: 'risk' as const },
    { label: `${condition.short_label} · + evidence`, value: condition.incremental[4].mae, series: 'evidence' as const },
  ]);
  return <div className="evidence-study">
    <header className="study-heading"><span className="study-eyebrow">COMPONENT 03 / FUNCTION 2 / N2</span>
      <h1>Maintenance Evidence Sufficiency Study</h1>
      <p>Offline experimental results · Original uncalibrated 7-day N2 pilot models</p></header>
    <SyntheticDataNotice />
    <section className="panel study-key-findings" aria-labelledby="key-findings-heading">
      <h2 id="key-findings-heading">Key Research Findings — Preliminary</h2>
      <div className="study-findings-grid">
        <article><h3>Finding 1 — History-window experiment</h3>
          <p>The {lowestBrier.lookback_days === null ? 'full-history' : `${lowestBrier.lookback_days}-day`} condition achieved the lowest
            observed Brier score: <strong>{fixed(lowestBrier.brier)}</strong>. Models were separately refitted;
            this does not show that shorter history is universally better.</p></article>
        <article><h3>Finding 2 — Continuous evidence relationships</h3>
          <p>H/M/F/R show descriptive associations with held-out prediction loss. These are observational and potentially confounded,
            with no causal conclusion or validated sufficiency threshold.</p></article>
        <article><h3>Finding 3 — Incremental evidence value</h3>
          <p>{allRiskOnlyBetter ? <><strong>Negative preliminary finding:</strong> Adding H/M/F/R and machine type slightly worsened
            loss-estimation MAE in all three history conditions.</> : 'Incremental MAE findings vary across history conditions; compare the saved results below.'}</p></article>
      </div>
    </section>
    <p className="study-intro">This page explores saved research results. Function 1 uses a separate calibrated model for historical
      machine predictions. Changing a study selector here does not request a prediction or rerun an experiment.</p>
    <div className="study-cohort" aria-label="Evaluation cohort">
      <div><strong>{first.snapshots.toLocaleString('en-GB')}</strong><span>paired cases per condition</span></div>
      <div><strong>{first.machines}</strong><span>benchmark machines</span></div>
      <div><strong>{first.dates}</strong><span>weekly historical snapshots</span></div>
      <div><strong>{first.positive_cases}</strong><span>positive machine-time cases</span></div>
    </div>
    <p className="study-period">Evaluation snapshots: {data.cohort.first_snapshot.replace(' 06:00:00', '')} to {data.cohort.last_snapshot.replace(' 06:00:00', '')},
      at 06:00 (timezone unspecified). These are repeated observations, not independent machines or counts of unique failure events.</p>

    <section className="panel study-section" aria-labelledby="truncation-heading">
      <span className="study-eyebrow">01 / CONTROLLED HISTORY TRUNCATION</span>
      <h2 id="truncation-heading">How did held-out prediction error change?</h2>
      <p>The same machine-time cases and future outcomes were evaluated with full, 180-day and 90-day prior histories.
        Each condition used a separately refitted failure classifier. Brier score is the mean squared probability error, (y − p)²; lower is better.</p>
      <div className="study-comparison">
        <StudyBars title="Held-out Brier score by history condition" description="Mean error of original uncalibrated failure predictions."
          rows={data.conditions.map(condition => ({ label: condition.short_label, value: condition.brier }))} />
        <div className="study-table-scroll" tabIndex={0} role="region" aria-label="History-truncation numerical results">
          <table><caption>Original failure predictions · 7-day Brier score</caption>
            <thead><tr><th scope="col">History</th><th scope="col">Cases</th><th scope="col">Brier ↓</th></tr></thead>
            <tbody>{data.conditions.map(condition => <tr key={condition.id}><th scope="row">{condition.short_label}</th>
              <td>{condition.snapshots.toLocaleString('en-GB')}</td><td>{fixed(condition.brier)}</td></tr>)}</tbody>
          </table></div>
      </div>
      <p className="study-caution">Shorter history is not universally better. Differences combine changed historical inputs and model refitting;
        they do not isolate a causal effect of removing records or establish a sufficiency threshold.</p>
    </section>

    <section className="panel study-section" aria-labelledby="continuous-heading">
      <span className="study-eyebrow">02 / CONTINUOUS EVIDENCE CHARACTERISTICS</span>
      <h2 id="continuous-heading">How were evidence characteristics associated with loss?</h2>
      <p>Explore continuous measurements through the saved quantile-group summaries. Each condition is paired with its own evidence and losses.
        These are descriptive associations, with no statistical significance or causal claim.</p>
      <div className="study-selectors">
        <div><label htmlFor="study-history">History condition</label>
          <select id="study-history" value={conditionId} onChange={event => setConditionId(event.target.value)}>
            {data.conditions.map(condition => <option key={condition.id} value={condition.id}>{condition.label}</option>)}</select></div>
        <div><label htmlFor="study-evidence">Evidence characteristic</label>
          <select id="study-evidence" value={symbol} onChange={event => setSymbol(event.target.value as EvidenceSymbol)}>
            {(Object.keys(evidenceLabels) as EvidenceSymbol[]).map(key => <option key={key} value={key}>{key} — {evidenceLabels[key]}</option>)}</select></div>
      </div>
      <p className="study-association" role="status">{selected.label} · {symbol} · {series.observations.toLocaleString('en-GB')} cases ·
        Descriptive Spearman ρ = {series.rho.toFixed(4)} · {series.groups.length} observed quantile groups</p>
      <EvidenceLossPlot groups={series.groups} xLabel={`${symbol}: ${evidenceLabels[symbol]}`} />
      <p className="study-chart-note">Points are group means; horizontal segments show observed evidence ranges, not uncertainty intervals.
        Connecting lines guide reading and are not a fitted continuous curve. Tied values stay together, so 90-day F has four groups.</p>
      <div className="study-table-scroll" tabIndex={0} role="region" aria-label="Evidence-loss group numerical results">
        <table><caption>{selected.label} · {symbol} quantile-group results</caption>
          <thead><tr><th scope="col">Group</th><th scope="col">Evidence range</th><th scope="col">Mean evidence</th>
            <th scope="col">Cases</th><th scope="col">Machines*</th><th scope="col">Mean squared loss</th><th scope="col">Failure prevalence</th></tr></thead>
          <tbody>{series.groups.map(group => <tr key={group.index}><th scope="row">{group.index}</th>
            <td>{group.evidence_min}–{group.evidence_max}</td><td>{group.evidence_mean.toFixed(3)}</td>
            <td>{group.n_snapshots}</td><td>{group.n_machines}</td><td>{fixed(group.mean_squared_prediction_error_brier)}</td>
            <td>{(group.observed_failure_prevalence * 100).toFixed(2)}%</td></tr>)}</tbody>
        </table></div>
      <p className="study-chart-note">*Distinct machines within each group; a machine may appear in multiple groups at different snapshots.
        Loss also varies with failure prevalence and case difficulty.</p>
      <dl className="study-evidence-notes">
        <div><dt>H · Observed span</dt><dd>Elapsed since earliest observed maintenance in the retained window; not verified logging coverage.</dd></div>
        <div><dt>R · Maintenance recency</dt><dd>Days since the latest recorded maintenance event; not database freshness or evidence of missing logs.</dd></div>
        <div><dt>C · Constant at {data.C_constant_pct}%</dt><dd>Published fields are complete in this benchmark. C is excluded from relationship fitting; its effect cannot be estimated here.</dd></div>
      </dl>
      <details className="study-disclosure"><summary>Evidence-only loss regression versus a constant-mean baseline</summary>
        <p>A separate five-fold machine-grouped Ridge analysis (α = 20) estimated squared prediction loss from H/M/F/R and machine type.
          It improved on the constant baseline, with modest R². This is separate from the incremental experiment below.</p>
        <div className="study-table-scroll" tabIndex={0} role="region" aria-label="Evidence-only cross-validation results">
          <table><caption>Second-stage evidence-only regression · out-of-fold results</caption><thead><tr><th scope="col">History</th>
            <th scope="col">Constant MAE</th><th scope="col">Evidence MAE</th><th scope="col">Evidence R²</th></tr></thead>
            <tbody>{data.conditions.map(condition => <tr key={condition.id}><th scope="row">{condition.short_label}</th>
              <td>{fixed(condition.evidence_cv.mae_constant_mean_baseline)}</td><td>{fixed(condition.evidence_cv.mae_evidence_ridge)}</td>
              <td>{fixed(condition.evidence_cv.r2_evidence_ridge)}</td></tr>)}</tbody></table></div>
      </details>
    </section>

    <section className="panel study-section" aria-labelledby="incremental-heading">
      <span className="study-eyebrow">03 / INCREMENTAL EVIDENCE VALUE</span>
      <h2 id="incremental-heading">Did evidence add information beyond the original risk?</h2>
      <p>This experiment tests whether H/M/F/R and machine type help estimate realized squared loss beyond the original predicted risk.</p>
      <p className="study-metric-explanation"><strong>Different outcomes:</strong> Brier above measures the original failure prediction’s error.
        Here, MAE measures error when estimating that realized squared loss in a second-stage model. MAE is not another failure probability or reliability percentage.</p>
      <p className="study-finding">{allRiskOnlyBetter
        ? 'Negative finding: risk-only slightly outperformed risk plus H/M/F/R and machine type in all three history conditions (lower MAE). No incremental MAE benefit was demonstrated.'
        : 'The incremental MAE findings differ across history conditions. Inspect every saved comparison below.'}</p>
      <div className="study-comparison">
        <StudyBars title="Risk-only versus risk-plus-evidence loss-estimation MAE" description="Second-stage out-of-fold MAE; lower is better."
          rows={comparisonBars} />
        <div className="study-table-scroll" tabIndex={0} role="region" aria-label="Risk versus evidence numerical results">
          <table><caption>Second-stage loss estimation · out-of-fold MAE</caption><thead><tr><th scope="col">History</th><th scope="col">Risk only</th>
            <th scope="col">+ H/M/F/R + type</th><th scope="col">Δ MAE*</th></tr></thead>
            <tbody>{data.conditions.map(condition => <tr key={condition.id}><th scope="row">{condition.short_label}</th>
              <td>{fixed(condition.incremental[1].mae)}</td><td>{fixed(condition.incremental[4].mae)}</td>
              <td>{fixed(condition.delta_mae_risk_minus_evidence)}</td></tr>)}</tbody></table></div>
      </div>
      <p className="study-chart-note">*Δ = risk-only MAE − full-evidence MAE. A negative value means risk-only had lower MAE in this experiment.
        Teal denotes risk-only and blue denotes evidence-augmented estimates; labels identify both without color.</p>
      <details className="study-disclosure"><summary>All five saved alternatives, including RMSE and R²</summary>
        <p>Risk-only uses p and p². The full comparison adds H/M/F/R and machine type. Both use the same fixed Ridge model (α = 5),
          fold-specific preprocessing and five-fold machine-grouped evaluation.</p>
        <div className="study-table-scroll" tabIndex={0} role="region" aria-label="All incremental model results">
          <table><caption>All predeclared incremental alternatives · five-fold machine-grouped results</caption>
            <thead><tr><th scope="col">History</th><th scope="col">Alternative</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">R²</th></tr></thead>
            <tbody>{data.conditions.flatMap(condition => condition.incremental.map(metric => <tr key={`${condition.id}-${metric.model}`}>
              <th scope="row">{condition.short_label}</th><td>{modelLabels[metric.model]}</td><td>{fixed(metric.mae)}</td>
              <td>{fixed(metric.rmse)}</td><td>{fixed(metric.r2)}</td></tr>))}</tbody></table></div>
      </details>
    </section>

    <details className="panel study-section study-methodology"><summary>Research methodology and provenance</summary>
      <h2>What these experiments establish</h2>
      <p>{data.methodology.calibration_note}</p><p>{data.incremental_methodology.validity_warning}</p>
      <ul>{data.methodology.scope_limitations.map(note => <li key={note}>{note}</li>)}</ul>
      <p>No validated reliability percentages, evidence-quality judgments, sufficiency cutoffs or maintenance recommendations are derived here.
        Event thinning, field masking, independent validation and real apparel-factory evaluation remain future work.</p>
      <h3>Evidence definitions</h3><dl className="study-definitions">{(['H', 'M', 'F', 'C', 'R'] as const).map(key =>
        <div key={key}><dt>{key}</dt><dd>{data.methodology[`${key}_definition`]}</dd></div>)}</dl>
      <h3>Snapshot provenance</h3>
      <p>Schema {data.schema_version}. The aggregate snapshot is bundled with the frontend source. No local research outputs,
        raw records, model files or backend calls are required to view it. Regeneration validates saved results without fitting models.</p>
      <ul className="study-sources">{data.provenance.map(source => <li key={source.path}><strong>{source.path}</strong>
        <span>{source.bytes.toLocaleString('en-GB')} bytes{source.rows !== undefined && ` · ${source.rows.toLocaleString('en-GB')} source rows`}</span>
        <code>SHA-256 {source.sha256}</code></li>)}</ul>
    </details>
  </div>;
}
