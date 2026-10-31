import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import ErrorMessage from '../../../shared/components/ErrorMessage';
import EvidenceSufficiencyPanel from '../components/EvidenceSufficiencyPanel';
import FailureRiskCard from '../components/FailureRiskCard';
import PredictionControls from '../components/PredictionControls';
import ResearchDetails from '../components/ResearchDetails';
import TopRiskMachinesPanel from '../components/TopRiskMachinesPanel';
import { MACHINE_IDS, PREDICTION_TIMESTAMPS } from '../data/demoOptions';
import { useFailureRisk } from '../hooks/useFailureRisk';
import { useTopRiskMachines } from '../hooks/useTopRiskMachines';
import type { RankedMachine } from '../types/component3.types';
import '../component3.css';

export default function FailureRiskPredictionPage() {
  const [machineId, setMachineId] = useState(1);
  const [timestamp, setTimestamp] = useState<string>(PREDICTION_TIMESTAMPS[0]);
  const [inputError, setInputError] = useState<string | null>(null);
  const prediction = useFailureRisk();
  const ranking = useTopRiskMachines();
  const { rank, reset: resetRanking } = ranking;
  const predictionHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    void rank(timestamp);
    return resetRanking;
  }, [timestamp, rank, resetRanking]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!MACHINE_IDS.includes(machineId) || !PREDICTION_TIMESTAMPS.some(time => time === timestamp)) {
      setInputError('Choose a supported benchmark machine and historical snapshot.');
      return;
    }
    setInputError(null);
    void prediction.predict({ machine_id: machineId, prediction_time: timestamp });
  }
  function selectionChanged() { prediction.reset(); setInputError(null); }
  function selectRankedMachine(machine: RankedMachine) {
    if (!ranking.result || ranking.result.prediction_time.replace(' ', 'T') !== timestamp) return;
    selectionChanged();
    setMachineId(machine.machine_id);
    void prediction.predict({ machine_id: machine.machine_id, prediction_time: timestamp }, {
      modelVersion: ranking.result.model_version, probability: machine.predicted_failure_probability,
    });
    if (window.matchMedia?.('(max-width: 1199px)').matches) {
      predictionHeading.current?.focus({ preventScroll: true });
      predictionHeading.current?.scrollIntoView({ block: 'start', behavior: 'instant' });
    }
  }

  return <div className="prediction-page machine-risk-explorer">
    <div className="page-heading"><div><span className="eyebrow">COMPONENT 03 / PREDICTIVE MAINTENANCE</span>
      <h1>Machine Failure-Risk Explorer</h1><p>Compare historical machine risk, then explore the evidence behind a prediction.</p></div>
      <div className="period-stamp"><strong>2015</strong><span>HISTORICAL BENCHMARK</span></div></div>
    <p className="prototype-note" aria-label="Prototype information">
      Prototype · Synthetic Azure 2015 data. Production lines are illustrative.
    </p>
    <PredictionControls machineId={machineId} timestamp={timestamp}
      predicting={prediction.loading} comparing={ranking.loading} onPredict={submit}
      onCompare={() => { void ranking.rank(timestamp); }}
      onMachineChange={value => { selectionChanged(); setMachineId(value); }}
      onDateChange={value => {
        if (value === timestamp) return;
        selectionChanged(); ranking.reset(); setTimestamp(value);
      }} />
    <div className="explorer-grid">
      <TopRiskMachinesPanel predictionTime={timestamp} selectedMachineId={machineId}
        loading={ranking.loading} error={ranking.error} result={ranking.result}
        onSelect={selectRankedMachine} />
      <section className="panel selected-machine-panel" aria-labelledby="selected-prediction-heading" aria-busy={prediction.loading}>
        <div className="dashboard-panel-heading"><div><span className="small-label">02 / EXPLORE A MACHINE</span>
          <h2 id="selected-prediction-heading" tabIndex={-1} ref={predictionHeading}>Selected Machine Prediction</h2></div></div>
        {(inputError || prediction.error) && <ErrorMessage message={inputError || prediction.error!} />}
        {prediction.loading && <div className="prediction-loading" role="status"><span className="spinner" aria-hidden="true" />
          <div><h3>Reading the historical snapshot</h3><p>Requesting Machine {String(machineId).padStart(3, '0')}’s prediction from the saved model…</p></div></div>}
        {!prediction.loading && !prediction.result && !prediction.error && !inputError && <div className="dashboard-empty prediction-empty">
          <h3>Choose a machine to explore</h3>
          <p>Use <strong>Predict selected machine</strong> above, or compare machines and choose <strong>View</strong> in the Top 10.</p>
          <p className="empty-footnote">Comparison does not automatically request a prediction.</p>
        </div>}
        {prediction.result && <div className="prediction-result" aria-live="polite">
          <FailureRiskCard result={prediction.result} />
          <EvidenceSufficiencyPanel profile={prediction.result.evidence_profile} />
        </div>}
      </section>
    </div>
    <ResearchDetails prediction={prediction.result} ranking={ranking.result} />
  </div>;
}
