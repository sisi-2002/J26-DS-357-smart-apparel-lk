import type { FormEvent } from 'react';
import MachineSelector from './MachineSelector';
import PredictionDateSelector from './PredictionDateSelector';

interface Props {
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
      <span className="fixed-horizon">Fixed horizon <strong>7 days</strong></span></div>
    <form onSubmit={props.onPredict} className="controls-form">
      <PredictionDateSelector value={props.timestamp} onChange={props.onDateChange} />
      <MachineSelector value={props.machineId} onChange={props.onMachineChange} />
      <div className="control-action"><button className="predict-button" type="submit" disabled={props.predicting}>
        {props.predicting ? <><span className="spinner" aria-hidden="true" /> Predicting…</> : 'Predict selected machine'}
      </button></div>
      <div className="control-action"><button className="compare-button" type="button" onClick={props.onCompare} disabled={props.comparing}>
        {props.comparing ? <><span className="spinner" aria-hidden="true" /> Comparing 100 machines…</> : 'Compare all 100 machines'}
      </button></div>
    </form>
    <p className="controls-help">One date for both views. Predict a machine directly, or compare the benchmark and choose <strong>View</strong>.</p>
  </section>;
}
