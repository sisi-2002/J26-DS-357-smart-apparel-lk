import { MACHINE_IDS } from '../data/demoOptions';
import { getMachineDisplay } from '../data/machineDisplay';

export default function MachineSelector({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  return <div className="field"><label htmlFor="machine-id">Historical machine</label>
    <select id="machine-id" value={value} onChange={event => onChange(Number(event.target.value))}>
      {MACHINE_IDS.map(id => {
        const machine = getMachineDisplay(id);
        return <option key={id} value={id}>{machine.name} · {machine.line}</option>;
      })}
    </select><p className="field-hint">100 machines · 10 lines</p></div>;
}
