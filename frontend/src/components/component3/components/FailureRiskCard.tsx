import { formatTimestamp } from '../data/demoOptions';
import { getMachineDisplay } from '../data/machineDisplay';
import type { FailureRiskResponse } from '../types/component3.types';

export default function FailureRiskCard({ result }: { result: FailureRiskResponse }) {
  const percentage = (result.predicted_failure_probability * 100).toFixed(2);
  const machine = getMachineDisplay(result.machine_id);
  return <section className="risk-card" aria-labelledby="risk-heading">
    <div className="risk-top"><div className="machine-identity">
      <strong data-testid="selected-machine-name">{machine.name}</strong>
      <span className="machine-line" data-testid="selected-machine-line">{machine.line}</span></div>
      <span>{formatTimestamp(result.prediction_time)}</span></div>
    <div className="probability-summary"><div><h3 id="risk-heading">7-day failure probability</h3>
      <p className="probability" data-testid="failure-probability">{percentage}<span>%</span></p></div>
      <p className="risk-explanation">Estimated chance of a recorded failure within the next seven days
        after this historical snapshot.</p></div>
  </section>;
}
