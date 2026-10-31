import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import PredictionControls from '../components/PredictionControls';
import TopRiskMachinesPanel from '../components/TopRiskMachinesPanel';
import EvidenceSufficiencyPanel from '../components/EvidenceSufficiencyPanel';
import { formatTimestamp, PREDICTION_TIMESTAMPS } from '../data/demoOptions';
import { getMachineDisplay } from '../data/machineDisplay';
import { useExperimentalPrediction } from './useExperimentalPrediction';
import { useExperimentalRanking } from './useExperimentalRanking';
import type { Horizon } from './experimentalApi';
import '../component3.css';
import './unified-risk.css';

export default function UnifiedRiskPage() {
  const [machineId, setMachineId] = useState(1);
  const [timestamp, setTimestamp] = useState<string>(PREDICTION_TIMESTAMPS[0]);
  const [horizon, setHorizon] = useState<Horizon>(7);
  const prediction = useExperimentalPrediction();
  const ranking = useExperimentalRanking();
  const { rank, reset: resetRanking } = ranking;
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => { void rank(timestamp, horizon); return resetRanking; }, [timestamp, horizon, rank, resetRanking]);

  function predict(mid: number) {
    const current = ranking.result;
    const row = current?.machines.find(machine => machine.machine_id === mid);
    void prediction.predict({ machine_id: mid, prediction_time: timestamp }, current ? {
      modelVersion: current.model_version, probabilities: row ? [row.p7, row.p14, row.p30] : undefined,
    } : undefined);
  }
  const result = prediction.result;
  const evidence = result?.evidence_profile;
  const machine = getMachineDisplay(machineId);
  const rankingView = ranking.result ? { prediction_time: ranking.result.prediction_time,
    machines: ranking.result.machines.map(row => ({ rank: row.rank, machine_id: row.machine_id,
      predicted_failure_probability: row[`p${horizon}`] })) } : null;
  const audit = result ?? ranking.result;

  return <div className="machine-risk-explorer unified-risk">
    <header className="page-heading"><div><span className="eyebrow">COMPONENT 03 / PREDICTIVE MAINTENANCE</span>
      <h1>Machine Failure-Risk Explorer</h1><p>Compare machines and explore cumulative failure probabilities across 7, 14 and 30 days.</p></div>
      <span className="unified-status">Research prototype · v3</span></header>
    <p className="prototype-note" aria-label="Prototype information">Synthetic Demo · Azure 2015 Data. Predictions are not validated maintenance alerts. Production lines are illustrative.</p>
    <PredictionControls multiHorizon machineId={machineId} timestamp={timestamp}
      predicting={prediction.loading} comparing={ranking.loading}
      onPredict={event => { event.preventDefault(); predict(machineId); }}
      onCompare={() => { void rank(timestamp, horizon); }}
      onMachineChange={mid => { prediction.reset(); setMachineId(mid); }}
      onDateChange={value => { if (value !== timestamp) { prediction.reset(); ranking.reset(); setTimestamp(value); } }} />
    <div className="explorer-grid">
      <div className="unified-comparison">
        <TopRiskMachinesPanel horizonDays={horizon} predictionTime={timestamp} selectedMachineId={machineId}
          controls={<div className="field horizon-control"><label htmlFor="ranking-horizon">Rank machines by</label>
          <select id="ranking-horizon" value={horizon} onChange={event => { ranking.reset(); setHorizon(Number(event.target.value) as Horizon); }}>
            {[7, 14, 30].map(days => <option key={days} value={days}>Next {days} days</option>)}
          </select><p className="field-hint">Same v3 model as all three prediction cards.</p></div>}
          loading={ranking.loading} error={ranking.error} result={rankingView}
          onSelect={row => {
            prediction.reset(); setMachineId(row.machine_id); predict(row.machine_id);
            if (window.matchMedia?.('(max-width: 1199px)').matches) {
              heading.current?.focus({ preventScroll: true }); heading.current?.scrollIntoView({ block: 'start', behavior: 'instant' });
            }
          }} />
      </div>
      <section className="panel selected-machine-panel" aria-labelledby="selected-prediction-heading" aria-busy={prediction.loading}>
        <div className="dashboard-panel-heading"><div><span className="small-label">02 / EXPLORE A MACHINE</span>
          <h2 id="selected-prediction-heading" tabIndex={-1} ref={heading}>Selected Machine Prediction</h2></div></div>
        <div className="unified-identity"><h3>{machine.name} <span>· {machine.line}</span></h3><p>{formatTimestamp(timestamp)} · Historical snapshot</p></div>
        {prediction.loading && <p className="ranking-status" role="status">Loading all three horizons from the saved v3 model…</p>}
        {prediction.error && <div className="error-message" role="alert"><strong>Prediction unavailable</strong><p>{prediction.error}</p>
          <p>No substitute probabilities are displayed. Check the v3 configuration or use the flag-controlled 7-day rollback.</p></div>}
        {!prediction.loading && !result && !prediction.error && <div className="dashboard-empty prediction-empty"><h3>Choose a machine to explore</h3>
          <p>Use <strong>Predict selected machine</strong> or choose <strong>View</strong> in the ranking. All three horizons appear together.</p>
          <p className="empty-footnote">Comparison does not automatically request a prediction.</p></div>}
        {result && <div aria-live="polite" className="prediction-result">
          <div className="unified-horizons">{([7, 14, 30] as const).map(days => <article key={days} aria-label={`${days}-day probability`}>
            <h3>Next {days} days</h3><p data-testid={`experimental-p${days}`}>{(result[`p${days}`] * 100).toFixed(2)}%</p><span>Probability of at least one failure</span>
          </article>)}</div>
          <p className="panel-description">Cumulative probabilities from one v3 conditional hazard system: P7 ≤ P14 ≤ P30.</p>
          {evidence && <>
            <EvidenceSufficiencyPanel profile={evidence} />
            <section className="unified-errors" aria-labelledby="error-history-heading"><h3 id="error-history-heading">Recorded error history</h3>
              <dl className="evidence-grid"><div className="metric-card"><dt>Previous recorded errors</dt><dd data-testid="evidence-errors">{evidence.error_events_prior}</dd>
                <dd className="metric-explanation">Past 7 / 30 / 90 days: <span data-testid="error-windows">{evidence.error_events_7d} / {evidence.error_events_30d} / {evidence.error_events_90d}</span></dd></div>
                <div className="metric-card"><dt>Days since last recorded error</dt><dd data-testid="evidence-error-recency">{evidence.days_since_last_error === null ? 'No prior error observed' : `${evidence.days_since_last_error.toLocaleString('en-GB', { maximumFractionDigits: 2 })} ${evidence.days_since_last_error === 1 ? 'day' : 'days'}`}</dd>
                  <dd className="metric-explanation">Recorded error events are not confirmed failures.</dd></div></dl>
            </section>
            <p className="unified-evidence-note">Descriptive history only. Evidence sufficiency for v3 has not yet been evaluated.</p>
          </>}
        </div>}
      </section>
    </div>
    <details className="panel research-details"><summary>Model &amp; Data Info<span>Methodology, provenance and research limitations</span></summary>
      <div className="research-details-body"><section><h3>One coherent v3 model system</h3>
        <p>All horizon cards and rankings use the saved unpooled HistGradientBoosting conditional hazard system with three frozen-model sigmoid calibrators. Ranking uses full-precision probabilities descending, with machine ID ascending for ties.</p>
        <p>Intervals are (0, 7], (7, 14] and (14, 30]. Historical features and descriptive evidence use only events strictly before the selected timestamp. Machine age/model are published benchmark metadata. Timezone is unspecified.</p></section>
        <section><h3>Evidence context and limitations</h3><p>H measures observed maintenance history span, not continuous log coverage. M/F count recorded maintenance and component-failure events. R is maintenance-event recency, not database freshness. C is constant at 100% for published benchmark fields, not factory data quality. Error history counts generic Azure error events, not validated apparel fault types.</p>
          <p>The <Link to="/component3/evidence-study">Maintenance Evidence Sufficiency study</Link> reports offline experiments with original uncalibrated LR models. Its findings are not validated evidence-loss relationships for v3. A future study must evaluate evidence sufficiency for v3 at each horizon.</p></section>
        <section><h3>Synthetic benchmark provenance</h3><p>Synthetic Microsoft Azure industrial-machine data, not real apparel-factory records. Lines are illustrative and never model features. The historical test period has been inspected repeatedly; results are exploratory. Thirty-day Top-10 ranking was weaker than the independent LR research reference. Zero ordering violations do not establish operational reliability.</p>
          <p>All 13 snapshots end their 30-day windows within the published benchmark observation period; complete machine uptime is not verified. No maintenance urgency, reliability score or repair recommendation is inferred.</p></section>
        {audit && <section><h3>Technical audit</h3><dl className="provenance"><dt>Model release</dt><dd>{audit.model_release}</dd><dt>Schema</dt><dd>{audit.model_schema}</dd>
          <dt>Artifact identity</dt><dd>{audit.model_version}</dd><dt>Prediction request ID</dt><dd>{result?.request_id ?? 'No individual prediction requested'}</dd>
          <dt>Ranking request ID</dt><dd>{ranking.result?.request_id ?? 'No ranking available'}</dd><dt>Event semantics</dt><dd>{audit.interval_semantics}</dd><dt>Model note</dt><dd>{audit.note}</dd></dl></section>}
      </div>
    </details>
  </div>;
}
