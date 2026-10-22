import { formatTimestamp } from '../data/demoOptions';
import type { RankedMachine, TopRiskMachinesResponse } from '../types/component3.types';

// Round display text only. Selection, bars and API ordering retain returned values.
const percentage = new Intl.NumberFormat('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

interface Props {
  predictionTime: string;
  selectedMachineId: number;
  loading: boolean;
  error: string | null;
  result: TopRiskMachinesResponse | null;
  onSelect: (machine: RankedMachine) => void;
}

export default function TopRiskMachinesPanel(props: Props) {
  return <section className="panel ranking-panel" aria-labelledby="ranking-heading" aria-busy={props.loading}>
    <div className="dashboard-panel-heading"><div><span className="small-label">01 / COMPARE MACHINES</span>
      <h2 id="ranking-heading">Highest Failure-Risk Machines <span className="ranking-horizon">(Next 7 Days)</span></h2></div>
      {props.result && <span className="count-tag">100 scored</span>}</div>
    <p className="ranking-description">Top 10 model probabilities among the 100 benchmark machines.</p>
    <p className="ranking-snapshot">{formatTimestamp(props.predictionTime)}</p>
    {props.loading && <p className="ranking-status" role="status">Scoring all 100 machines in one model batch…</p>}
    {props.error && <div className="error-message" role="alert"><strong>Ranking unavailable</strong><p>{props.error}</p></div>}
    {props.result && <>
      <p className="ranking-key">Coral marks relative positions 1–3 only, not validated danger or urgency levels.</p>
      <table className="ranking-table">
        <caption className="visually-hidden">Top 10 machine probabilities for {formatTimestamp(props.result.prediction_time)}</caption>
        <thead><tr><th scope="col">Rank</th><th scope="col">Machine</th><th scope="col">7-day probability<span className="probability-scale">0–100% scale</span></th><th scope="col">Details</th></tr></thead>
        <tbody>{props.result.machines.map(machine => <tr key={machine.machine_id}
          data-selected={machine.machine_id === props.selectedMachineId}>
          <td><span className="rank-position" data-leading={machine.rank <= 3}>{machine.rank}</span></td>
          <td>{String(machine.machine_id).padStart(3, '0')}
            {machine.machine_id === props.selectedMachineId && <span className="selection-label">Selected</span>}</td>
          <td title={`Returned probability: ${machine.predicted_failure_probability}`}>
            <span className="ranking-probability">{percentage.format(machine.predicted_failure_probability * 100)}%</span>
            <span className="probability-track" aria-hidden="true"><span className="probability-fill"
              style={{ width: `${machine.predicted_failure_probability * 100}%` }} /></span></td>
          <td><button type="button" onClick={() => props.onSelect(machine)}
            aria-label={`View prediction for machine ${machine.machine_id}`}
            aria-current={machine.machine_id === props.selectedMachineId ? 'true' : undefined}>View</button></td>
        </tr>)}</tbody>
      </table>
      <p className="ranking-caption">Percentages rounded to 2 decimals; bars use an absolute 0–100% scale. Choose <strong>View</strong> to inspect a machine’s evidence.</p>
    </>}
    {!props.loading && !props.result && !props.error && <div className="dashboard-empty ranking-empty">
      <p>Comparison loads automatically for the selected date. Use <strong>Refresh ranking</strong> to try again.</p>
      <p className="empty-footnote">No machine is automatically chosen or predicted.</p>
    </div>}
  </section>;
}
