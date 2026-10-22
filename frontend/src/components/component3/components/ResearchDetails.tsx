import { formatTimestamp } from '../data/demoOptions';
import type { FailureRiskResponse, TopRiskMachinesResponse } from '../types/component3.types';

interface Props {
  prediction: FailureRiskResponse | null;
  ranking: TopRiskMachinesResponse | null;
}

export default function ResearchDetails({ prediction, ranking }: Props) {
  return <details className="panel research-details">
    <summary>Model &amp; Data Info <span>Technical audit</span></summary>
    <div className="research-details-body">
      <section aria-labelledby="model-data-heading">
        <h3 id="model-data-heading">Model and dataset limitations</h3>
        <p>This historical demonstration uses the synthetic Microsoft Azure 2015 benchmark with 100 industrial machines,
          not real apparel-factory records. It is not live monitoring or validated maintenance advice.</p>
        <p>The saved 7-day Logistic Regression model uses chronological sigmoid calibration. Inference loads the saved
          model without retraining. A probability is not a reliability score; no validated alert thresholds or maintenance
          recommendations are provided.</p>
        <p>Ranking is relative to these 100 machines and preserves the API order based on unrounded model probabilities.
          Displaying two decimal places does not change the returned values. Bars use an absolute 0–100% scale;
          coral highlights positions 1–3, not danger or urgency.</p>
      </section>
      <section aria-labelledby="evidence-definitions-heading">
        <h3 id="evidence-definitions-heading">Maintenance-evidence definitions</h3>
        <p>All recorded events are strictly before the selected prediction timestamp. These characteristics describe
          available history; they do not establish individual prediction reliability.</p>
        <dl className="provenance">
          <dt>H · History span</dt><dd>Time since the earliest observed maintenance event, not guaranteed logging coverage.</dd>
          <dt>M · Maintenance</dt><dd>Number of previous recorded maintenance events.</dd>
          <dt>F · Failures</dt><dd>Number of previous recorded failure events.</dd>
          <dt>C · Completeness</dt><dd>Completeness of published benchmark fields, constant at 100% in this benchmark;
            not factory data quality or evidence of reliable predictions.</dd>
          <dt>R · Recency</dt><dd>Days since the last recorded maintenance event, not database freshness.</dd>
        </dl>
        <p>A missing H or R means no prior maintenance was observed; it is not treated as zero days.</p>
      </section>
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
