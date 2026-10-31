import { formatTimestamp, PREDICTION_TIMESTAMPS } from '../data/demoOptions';

export default function PredictionDateSelector({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return <div className="field"><label htmlFor="prediction-time">Prediction snapshot</label>
    <select id="prediction-time" value={value} onChange={event => onChange(event.target.value)} aria-describedby="time-hint">
      {PREDICTION_TIMESTAMPS.map(time => <option key={time} value={time}>{formatTimestamp(time)}</option>)}
    </select><p className="field-hint" id="time-hint">13 historical snapshots · timezone unspecified</p></div>;
}
