import { useState } from 'react';
import type { FormEvent } from 'react';
import ErrorMessage from '../../../shared/components/ErrorMessage';
import EvidenceSufficiencyPanel from '../components/EvidenceSufficiencyPanel';
import FailureRiskCard from '../components/FailureRiskCard';
import MachineSelector from '../components/MachineSelector';
import PredictionDateSelector from '../components/PredictionDateSelector';
import SyntheticDataNotice from '../components/SyntheticDataNotice';
import { MACHINE_IDS, PREDICTION_TIMESTAMPS } from '../data/demoOptions';
import { useFailureRisk } from '../hooks/useFailureRisk';

export default function FailureRiskPredictionPage() {
  const [machineId, setMachineId] = useState(1);
  const [timestamp, setTimestamp] = useState<string>(PREDICTION_TIMESTAMPS[0]);
  const [inputError, setInputError] = useState<string | null>(null);
  const prediction = useFailureRisk();

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

  return <div className="prediction-page">
    <div className="page-heading"><div><span className="eyebrow">COMPONENT 03 / PREDICTIVE MAINTENANCE</span>
      <h1>Understand machine risk.</h1><p>Explore a historical prediction. See the maintenance evidence that supports it.</p></div>
      <div className="period-stamp"><strong>2015</strong><span>HISTORICAL BENCHMARK</span></div></div>
    <SyntheticDataNotice />
    <div className="prediction-grid">
      <div><section className="panel input-panel" aria-labelledby="input-heading">
        <div className="panel-heading"><div><span className="small-label">01 / SELECT A SNAPSHOT</span>
          <h2 id="input-heading">Prediction inputs</h2></div></div>
        <form onSubmit={submit}>
          <MachineSelector value={machineId} onChange={value => { selectionChanged(); setMachineId(value); }} />
          <PredictionDateSelector value={timestamp} onChange={value => { selectionChanged(); setTimestamp(value); }} />
          <div className="horizon-description"><span>Prediction horizon</span><strong>7 days <span aria-hidden="true">↗</span></strong></div>
          <button className="predict-button" type="submit" disabled={prediction.loading}>
            {prediction.loading ? <><span className="spinner" aria-hidden="true" /> Predicting…</> : <>Predict failure risk <span aria-hidden="true">→</span></>}
          </button>
          <p className="form-caption">Uses the saved, calibrated model. No retraining.</p>
        </form>
      </section><div className="method-note"><span className="small-label">HOW TO READ THIS</span>
        <p>Failure probability and maintenance evidence answer different questions. Review them together; neither is a maintenance instruction.</p></div></div>
      <div className="results-column" aria-busy={prediction.loading}>
        {(inputError || prediction.error) && <ErrorMessage message={inputError || prediction.error!} />}
        {prediction.loading && <section className="panel loading-state" role="status"><span className="spinner" aria-hidden="true" />
          <h2>Reading the historical snapshot</h2><p>Requesting a prediction from the saved model…</p></section>}
        {!prediction.loading && !prediction.result && <section className="panel empty-state">
          <div className="empty-emblem" aria-hidden="true"><span>07</span><small>DAYS</small></div>
          <span className="small-label">02 / EXPLORE THE RESULT</span><h2>Your historical snapshot starts here.</h2>
          <p>Select a machine and prediction date, then request its seven-day failure probability and maintenance-evidence profile.</p>
          <div className="empty-steps"><span>Choose a snapshot</span><span aria-hidden="true">→</span><span>Predict</span><span aria-hidden="true">→</span><span>Review evidence</span></div>
        </section>}
        {prediction.result && <div className="prediction-result" aria-live="polite">
          <FailureRiskCard result={prediction.result} />
          <EvidenceSufficiencyPanel profile={prediction.result.evidence_profile} />
          <section className="result-note"><strong>Keep this result in context</strong><p>{prediction.result.note}</p>
            <details><summary>Prediction provenance</summary><dl className="provenance">
              <dt>Calibration</dt><dd>{prediction.result.calibration}</dd>
              <dt>Model schema</dt><dd>{prediction.result.model_schema_version}</dd>
              <dt>Model artifact</dt><dd>{prediction.result.model_version}</dd>
              <dt>Request ID</dt><dd>{prediction.result.request_id}</dd>
              <dt>Data status</dt><dd>{prediction.result.data_status}</dd>
            </dl></details></section>
        </div>}
      </div>
    </div>
  </div>;
}
