import { formatTimestamp } from '../data/demoOptions';
import type { FailureRiskResponse } from '../types/component3.types';

export default function FailureRiskCard({ result }: { result: FailureRiskResponse }) {
  const percentage = (result.predicted_failure_probability * 100).toFixed(2);
  return <section className="risk-card" aria-labelledby="risk-heading">
    <div className="risk-top"><span className="small-label">MODEL OUTPUT</span><span className="horizon-pill">7-DAY HORIZON</span></div>
    <h2 id="risk-heading">Predicted failure probability</h2>
    <p className="probability" data-testid="failure-probability">{percentage}<span>%</span></p>
    <p className="risk-explanation">Estimated chance of a recorded failure within the next seven days
      after this historical snapshot.</p>
    <div className="snapshot-summary"><span>Machine {String(result.machine_id).padStart(3, '0')}
      <small>Synthetic label: {result.machine_type}</small></span>
      <span>{formatTimestamp(result.prediction_time)}<small>Historical benchmark time</small></span></div>
    <p className="raw-probability">Returned probability: {result.predicted_failure_probability.toFixed(6)} ·
      A probability is not a reliability score.</p>
  </section>;
}
