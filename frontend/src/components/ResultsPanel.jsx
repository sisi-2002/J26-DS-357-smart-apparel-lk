import DataTable from "./DataTable";
import MetricCard from "./MetricCard";

const number = (value, digits = 0) => Number(value ?? 0).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
const money = (value) => `LKR ${number(value)}`;

export default function ResultsPanel({ result, onNewEstimate }) {
  const plan = result.production_plan;
  const recommended = plan.recommended;
  const n1 = result.n1_analysis;
  const n2 = result.n2_analysis;
  const status = result.decision.startsWith("CAPACITY FIT") ? "fit" : result.decision.startsWith("TIGHT") ? "warning" : "danger";

  return <main className="app-shell">
    <header className="app-header"><a className="brand" href="/" aria-label="Smart Apparel LK home"><span className="brand-symbol">SA</span><span><strong>Smart Apparel</strong><small>LK / Operations intelligence</small></span></a><div className="header-status"><span className="status-pulse" />Assessment complete <span className="header-divider" /> Component 04</div></header>
    <div className="results">
    <div className="results-header"><div><p className="eyebrow">Assessment result · Capacity decision</p><h1>Order production plan</h1><p className="sub">Order of {number(result.order_quantity)} pieces <span className="sub-separator">/</span> handover in {result.handover_days} days <span className="sub-separator">/</span> {result.style}</p></div><button className="secondary-button" onClick={onNewEstimate}><span aria-hidden="true">＋</span> New estimate</button></div>
    <nav className="result-nav" aria-label="Result sections"><a href="#overview">Overview</a><a href="#interventions">Interventions</a><a href="#capacity">Capacity</a><a href="#workforce">Workforce</a></nav>
    <section id="overview" className={`card summary-card ${status}`}><div className="summary-main"><span className="summary-label">On-time likelihood for entered capacity</span>{n1.available ? <strong className="probability">{number(n1.percent, 2)}<small>%</small></strong> : <p className="empty-state">{n1.message}</p>}<span className={`badge ${status}`}><i aria-hidden="true" />{result.decision}</span></div><div className="summary-context"><span className="summary-context-label">Model context</span><p>{plan.capacity_source}. The likelihood model uses {plan.available_lines} lines and {plan.factory_shifts_per_day} shifts.</p><div className="summary-line"><span>Order target</span><strong>{number(plan.daily_target)} pcs/day</strong></div></div></section>

    <Section eyebrow="Order snapshot" title="Calculated order and capacity details"><div className="metric-grid">
      <MetricCard label="Handover lead time" value={`${result.handover_days} days`} /><MetricCard label="SMV" value={`${number(result.estimated_smv, 2)} min / piece`} /><MetricCard label="Order value" value={money(result.estimated_order_value)} /><MetricCard label="Style" value={result.style} /><MetricCard label="Priority" value={result.priority} /><MetricCard label="Daily output target" value={`${number(plan.daily_target)} pcs / day`} /><MetricCard label="Required workload" value={`${number(result.estimated_workload)} minutes`} /><MetricCard label="Capacity coverage" value={`${number(result.capacity_ratio, 2)}×`} />
    </div><Note>The production model estimates aggregate output per line-shift. It does not identify workers or confirm current attendance.</Note></Section>

    <Section eyebrow="Risk signal" title="Delivery likelihood for selected line capacity"><div className="metric-grid">{n1.available ? <><MetricCard label="Predicted on-time likelihood" value={`${number(n1.percent, 2)}%`} tone={status} /><MetricCard label="Model" value={n1.model_name} /><MetricCard label="Interpretation" value={n1.decision} />{n1.validation_auc != null && <MetricCard label="Validation ROC-AUC" value={number(n1.validation_auc, 3)} />}</> : <p className="empty-state">{n1.message}</p>}</div>{n1.scope_note && <Note>{n1.scope_note}</Note>}</Section>

    <Section id="interventions" eyebrow="Decision support" title="Intervention options and cost outlook">{n2.available ? <><div className="metric-grid"><MetricCard label="Lowest estimated cost" value={`${n2.lowest_cost.label} · ${money(n2.lowest_cost.estimated_cost)}`} /><MetricCard label="Top historical selection" value={`${n2.top_selection.label} · ${number(n2.top_selection.selection_score * 100, 1)}%`} /><MetricCard label="Cost model" value={n2.cost_model_name} /></div><DataTable rows={n2.options} columns={[
      { key: "label", label: "Option" }, { key: "added_workers", label: "Workers", render: (r) => number(r.added_workers, 1) }, { key: "overtime_hours", label: "Overtime hours", render: (r) => number(r.overtime_hours, 1) }, { key: "added_machines", label: "Machines", render: (r) => number(r.added_machines, 1) }, { key: "outsourced_qty", label: "Outsourced pcs", render: (r) => number(r.outsourced_qty) }, { key: "estimated_cost", label: "Estimated cost", render: (r) => money(r.estimated_cost) }, { key: "selection_score", label: "Historical score", render: (r) => `${number(r.selection_score * 100, 1)}%` },
    ]} /><Note>{n2.scope_note}</Note></> : <p className="empty-state">{n2.message}</p>}</Section>

    <Section id="capacity" eyebrow="Production planning" title="Recommended line-shift capacity plan"><div className="metric-grid"><MetricCard label="Lines available" value={plan.available_lines} /><MetricCard label="Recommended line-shifts" value={recommended.line_shifts} /><MetricCard label="Expected output / day" value={`${number(recommended.daily_output)} pcs`} /><MetricCard label="Conservative output / day" value={`${number(recommended.daily_output_low)} pcs`} /><MetricCard label="Expected completion" value={`${recommended.estimated_days} days`} /><MetricCard label="Conservative completion" value={`${recommended.conservative_days} days`} /><MetricCard label="Required operators / day" value={recommended.required_operators} /><MetricCard label="Output range / line-shift" value={`${number(plan.rate_low)} · ${number(plan.rate_median)} · ${number(plan.rate_high)} pcs`} /></div><Note><strong>{recommended.name}</strong> ({recommended.basis}). {plan.staffing_basis} {plan.machine_note}</Note></Section>

    <Section title="Capacity with one or both shifts"><DataTable rows={plan.shift_scenarios} columns={[
      { key: "name", label: "Scenario" }, { key: "active_shifts", label: "Active shifts" }, { key: "lines", label: "Lines / shift" }, { key: "line_shifts", label: "Line-shifts / day" }, { key: "required_operators", label: "Operators" }, { key: "target_output", label: "Daily target", render: (r) => `${number(r.target_output)} pcs` }, { key: "expected_output", label: "Expected output", render: (r) => `${number(r.expected_output)} pcs` },
    ]} /><Note>Historical orders do not identify output by shift, so these are pooled capacity scenarios rather than separate measured shift rates.</Note></Section>

    <Section title="Required grade mix and daily schedule"><div className="metric-grid"><MetricCard label="Recommended active lines" value={recommended.lines} /><MetricCard label="Shifts used / day" value={recommended.shifts} /><MetricCard label="Operators / line-shift" value={recommended.operators_per_line_shift} /><MetricCard label="Daily target" value={`${number(plan.daily_target)} pcs`} /></div><h3>Shift staffing summary</h3><DataTable rows={recommended.shift_summary} columns={[{ key: "shift", label: "Shift" }, { key: "lines", label: "Lines", render: (r) => r.lines.join(", ") }, { key: "line_count", label: "Line-shifts" }, { key: "required_operators", label: "Operators" }, { key: "target_pieces_per_day", label: "Target", render: (r) => `${number(r.target_pieces_per_day)} pcs/day` }]} /><h3>Day-by-day production schedule</h3><DataTable rows={recommended.daily_schedule} columns={[{ key: "day", label: "Day" }, { key: "shift", label: "Shift" }, { key: "lines", label: "Lines", render: (r) => r.lines.join(", ") }, { key: "required_operators", label: "Operators" }, { key: "target_pieces", label: "Shift target", render: (r) => `${number(r.target_pieces)} pcs` }, { key: "cumulative_target", label: "Cumulative", render: (r) => `${number(r.cumulative_target)} pcs` }]} /></Section>

    <Section id="workforce" eyebrow="Staffing visibility" title="Workforce allocation"><div className="empty-state"><strong>Live allocation is not available in the current backend.</strong><p>{result.workforce.message}</p><p>The backend does provide a grade-based required staffing plan above. A worker recommendation view requires a new API route that loads <code>data/workers.csv</code>, <code>data/operations.csv</code>, machine availability, and the workforce model.</p></div></Section>
    <Section eyebrow="Audit trail" title="Why this result"><ul className="explain">{result.explanation.map((item) => <li key={item}>{item}</li>)}</ul></Section>
    </div>
  </main>;
}

function Section({ eyebrow, title, children, id }) { return <section id={id} className="card"><div className="section-heading"><div><p className="section-kicker">{eyebrow}</p><h2>{title}</h2></div></div>{children}</section>; }
function Note({ children }) { return <p className="note">{children}</p>; }
