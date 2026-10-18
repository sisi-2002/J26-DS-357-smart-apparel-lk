import { formatTimestamp } from '../data/demoOptions';
import type { FailureRiskResponse, TopRiskMachinesResponse } from '../types/component3.types';

interface Props {
  prediction: FailureRiskResponse | null;
  ranking: TopRiskMachinesResponse | null;
}

export default function ResearchDetails({ prediction, ranking }: Props) {
  return <details className="panel research-details">
    <summary>Research details <span>Dataset, calibration and model provenance</span></summary>
    <div className="research-details-body">
      {!prediction && !ranking && <p>Request a prediction or comparison to see its research notes and model provenance.</p>}
      {prediction && <section aria-labelledby="prediction-research-heading">
        <h3 id="prediction-research-heading">Individual prediction · Machine {String(prediction.machine_id).padStart(3, '0')}</h3>
        <p>{prediction.note}</p>
        <dl className="provenance">
          <dt>Snapshot</dt><dd>{formatTimestamp(prediction.prediction_time)}</dd>
          <dt>Returned probability</dt><dd>{prediction.predicted_failure_probability.toFixed(6)}</dd>
          <dt>Synthetic model label</dt><dd>{prediction.machine_type}</dd>
          <dt>Calibration</dt><dd>{prediction.calibration}</dd>
          <dt>Model schema</dt><dd>{prediction.model_schema_version}</dd>
          <dt>Model artifact</dt><dd>{prediction.model_version}</dd>
          <dt>Request ID</dt><dd>{prediction.request_id}</dd>
          <dt>Data status</dt><dd>{prediction.data_status}</dd>
        </dl>
      </section>}
      {ranking && <section aria-labelledby="ranking-research-heading">
        <h3 id="ranking-research-heading">Benchmark comparison · {formatTimestamp(ranking.prediction_time)}</h3>
        <p>{ranking.note}</p>
        <dl className="provenance">
          <dt>Machines scored</dt><dd>{ranking.scored_machine_count}</dd>
          <dt>Model schema</dt><dd>{ranking.model_schema_version}</dd>
          <dt>Model artifact</dt><dd>{ranking.model_version}</dd>
          <dt>Request ID</dt><dd>{ranking.request_id}</dd>
          <dt>Data status</dt><dd>{ranking.data_status}</dd>
        </dl>
      </section>}
    </div>
  </details>;
}
