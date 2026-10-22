import { formatTimestamp } from '../data/demoOptions';
import type { FailureRiskResponse } from '../types/component3.types';

export default function FailureRiskCard({ result }: { result: FailureRiskResponse }) {
  const percentage = (result.predicted_failure_probability * 100).toFixed(2);
  return <section className="risk-card" aria-labelledby="risk-heading">
    <div className="risk-top"><strong>Machine {String(result.machine_id).padStart(3, '0')}</strong>
      <span>{formatTimestamp(result.prediction_time)}</span></div>
    <div className="probability-summary"><div><h3 id="risk-heading">7-day failure probability</h3>
      <p className="probability" data-testid="failure-probability">{percentage}<span>%</span></p></div>
      <p className="risk-explanation">Estimated chance of a recorded failure within the next seven days
        after this historical snapshot.</p></div>
  </section>;
}
