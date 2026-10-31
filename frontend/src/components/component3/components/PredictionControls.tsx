import type { FormEvent } from 'react';
import MachineSelector from './MachineSelector';
import PredictionDateSelector from './PredictionDateSelector';

interface Props {
  multiHorizon?: boolean;
  machineId: number;
  timestamp: string;
  predicting: boolean;
  comparing: boolean;
  onMachineChange: (id: number) => void;
  onDateChange: (timestamp: string) => void;
  onPredict: (event: FormEvent) => void;
  onCompare: () => void;
}

export default function PredictionControls(props: Props) {
  return <section className="panel prediction-controls" aria-labelledby="controls-heading">
    <div className="controls-heading"><h2 id="controls-heading">Choose a historical snapshot</h2>
      <span className="fixed-horizon">{props.multiHorizon ? 'Cumulative horizons' : 'Fixed horizon'} <strong>{props.multiHorizon ? '7 / 14 / 30 days' : '7 days'}</strong></span></div>
    <form onSubmit={props.onPredict} className="controls-form">
      <PredictionDateSelector value={props.timestamp} onChange={props.onDateChange} />
      <MachineSelector value={props.machineId} onChange={props.onMachineChange} />
      <div className="control-action"><button className="predict-button" type="submit" disabled={props.predicting}>
        {props.predicting ? <><span className="spinner" aria-hidden="true" /> Predicting…</> : 'Predict selected machine'}
      </button></div>
      <div className="control-action"><button className="compare-button" type="button" onClick={props.onCompare} disabled={props.comparing}>
        {props.comparing ? <><span className="spinner" aria-hidden="true" /> Comparing 100 machines…</> : 'Refresh ranking'}
      </button></div>
    </form>
    <p className="controls-help">Comparison updates automatically with the date. Choose <strong>View</strong>, or predict any machine directly.</p>
  </section>;
}
