import { MACHINE_IDS } from '../data/demoOptions';

export default function MachineSelector({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  return <div className="field"><label htmlFor="machine-id">Historical machine</label>
    <select id="machine-id" value={value} onChange={event => onChange(Number(event.target.value))}>
      {MACHINE_IDS.map(id => <option key={id} value={id}>Machine {String(id).padStart(3, '0')}</option>)}
    </select><p className="field-hint">100 synthetic benchmark machines</p></div>;
}
