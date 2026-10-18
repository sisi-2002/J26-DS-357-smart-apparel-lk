import { useState } from "react";

const initialValues = {
  order_quantity: "10000",
  handover_days: "10",
  available_lines: "4",
  required_smv_total: "12.30",
  order_value: "2500000",
  style: "Polo",
  priority: "Normal",
};

export default function PredictionForm({ onSubmit, loading }) {
  const [values, setValues] = useState(initialValues);
  const [validationError, setValidationError] = useState("");

  function update(event) {
    setValues((current) => ({ ...current, [event.target.name]: event.target.value }));
  }

  function submit(event) {
    event.preventDefault();
    const numbers = ["order_quantity", "handover_days", "order_value"];
    if (numbers.some((name) => Number(values[name]) <= 0) ||
        parseSmv(values.required_smv_total) <= 0 ||
        Number(values.available_lines) < 1 || Number(values.available_lines) > 30) {
      setValidationError("Enter positive values. SMV seconds must be from 00 to 59, and available sewing lines must be from 1 to 30.");
      return;
    }
    setValidationError("");
    onSubmit({
      ...values,
      order_quantity: Number(values.order_quantity),
      handover_days: Number(values.handover_days),
      available_lines: Number(values.available_lines),
      required_smv_total: parseSmv(values.required_smv_total),
      order_value: Number(values.order_value),
    });
  }

  return (
    <form className="planning-form" onSubmit={submit} noValidate>
      {validationError && <div className="alert error compact-alert" role="alert"><span className="alert-icon">!</span>{validationError}</div>}
      <div className="input-group">
        <div className="group-title"><span className="group-icon">01</span><div><strong>Order context</strong><small>Volume, deadline and commercial value</small></div></div>
        <div className="form-grid">
          <Field label="Order quantity" name="order_quantity" value={values.order_quantity} onChange={update} min="1" hint="Number of garments in this order" />
          <Field label="Handover days" name="handover_days" value={values.handover_days} onChange={update} min="1" hint="Days available to complete the order" />
          <Field label="Order value (LKR)" name="order_value" value={values.order_value} onChange={update} min="0.01" step="0.01" hint="Total value of the order" />
          <SelectField label="Priority" name="priority" value={values.priority} onChange={update} options={["Low", "Normal", "High"]} />
        </div>
      </div>
      <div className="input-group">
        <div className="group-title"><span className="group-icon">02</span><div><strong>Production context</strong><small>Available sewing capacity and garment complexity</small></div></div>
        <div className="form-grid">
          <Field label="Available sewing lines" name="available_lines" value={values.available_lines} onChange={update} min="1" max="30" hint="How many lines can be assigned (1–30)" />
          <SmvField value={values.required_smv_total} onChange={update} />
          <SelectField label="Style" name="style" value={values.style} onChange={update} options={["Polo", "T-Shirt", "Hoodie", "Legging", "Jacket"]} />
        </div>
      </div>
      <div className="info"><span aria-hidden="true">i</span><div><strong>What the system calculates</strong>On-time likelihood, historical/model capacity ranges, intervention costs, and a required grade-based staffing plan.</div></div>
      <button className="primary-button" type="submit" disabled={loading}><span>{loading ? "Calculating assessment…" : "Calculate order feasibility"}</span><b aria-hidden="true">→</b></button>
    </form>
  );
}

function Field({ label, name, value, onChange, hint, ...props }) {
  return <div className="field"><label htmlFor={name}>{label}<span className="required-mark" aria-hidden="true">*</span></label><input id={name} name={name} type="number" value={value} onChange={onChange} required {...props} /><span className="hint">{hint}</span></div>;
}

function SmvField({ value, onChange }) {
  function updateSmv(event) {
    const nextValue = event.target.value.replace(/[^\d.]/g, "");
    const [, seconds = ""] = nextValue.split(".");
    const displayValue = nextValue.endsWith(".") || seconds.length < 2
      ? nextValue
      : normalizeSmv(nextValue);
    onChange({ target: { name: "required_smv_total", value: displayValue } });
  }

  function stepSmv(direction) {
    const current = parseSmv(value);
    const next = Math.max(0, current + direction / 60);
    onChange({ target: { name: "required_smv_total", value: formatSmv(next) } });
  }

  return <div className="field">
    <label htmlFor="required_smv_total">SMV (minutes per piece)<span className="required-mark" aria-hidden="true">*</span></label>
    <div className="smv-control">
      <input id="required_smv_total" name="required_smv_total" type="text" inputMode="decimal" pattern="\d+(\.\d{1,2})?" value={value} onChange={updateSmv} onBlur={() => value && onChange({ target: { name: "required_smv_total", value: normalizeSmv(value) } })} required aria-describedby="smv-hint" />
      <div className="smv-stepper" aria-label="Adjust SMV seconds">
        <button type="button" onClick={() => stepSmv(1)} aria-label="Increase SMV by one second">+</button>
        <button type="button" onClick={() => stepSmv(-1)} aria-label="Decrease SMV by one second">−</button>
      </div>
    </div>
    <span className="hint" id="smv-hint">Enter as minutes.seconds, for example 12.30. Seconds roll from 59 to 00.</span>
  </div>;
}

function SelectField({ label, name, value, onChange, options }) {
  return <div className="field"><label htmlFor={name}>{label}<span className="required-mark" aria-hidden="true">*</span></label><select id={name} name={name} value={value} onChange={onChange}>{options.map((option) => <option key={option}>{option}</option>)}</select></div>;
}

function parseSmv(value) {
  const [minutes = "0", seconds = "0"] = String(value).split(".");
  return Number(minutes) + Number(seconds.padEnd(2, "0")) / 60;
}

function normalizeSmv(value) {
  if (!value) return "";
  const [minutePart = "0", secondPart = ""] = value.split(".");
  const totalSeconds = Number(minutePart) * 60 + Number((secondPart || "0").slice(0, 2).padEnd(2, "0"));
  return formatSmv(totalSeconds / 60);
}

function formatSmv(decimalMinutes) {
  const totalSeconds = Math.max(0, Math.round(decimalMinutes * 60));
  return `${Math.floor(totalSeconds / 60)}.${String(totalSeconds % 60).padStart(2, "0")}`;
}
