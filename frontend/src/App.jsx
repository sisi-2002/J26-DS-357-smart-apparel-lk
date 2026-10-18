import { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import ResultsPanel from "./components/ResultsPanel";
import { predictOrder } from "./services/api";

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(values) {
    setLoading(true);
    setError("");
    try {
      setResult(await predictOrder(values));
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  if (result) return <ResultsPanel result={result} onNewEstimate={() => setResult(null)} />;
  return (
    <main className="app-shell">
      <AppHeader />
      <div className="shell">
        <section className="hero intro-card">
          <div className="hero-copy">
            <p className="eyebrow">Component 04 · Decision intelligence</p>
            <h1>Check an order before you accept it</h1>
            <p className="intro">Turn order details into a clear capacity and delivery decision before production is committed.</p>
          </div>
          <div className="hero-mark" aria-hidden="true"><span>SA</span><i /></div>
        </section>
        <section className="card form-card">
          <div className="section-heading"><div><p className="section-kicker">New assessment</p><h2>Order and capacity inputs</h2></div><span className="required-note"><b>*</b> Required fields</span></div>
          <p className="section-description">Provide the commercial and production context for the order. The existing Flask models will calculate delivery risk, capacity, staffing requirements, and intervention costs.</p>
          {error && <div className="alert error" role="alert"><span className="alert-icon">!</span><div><strong>We could not complete the assessment</strong><br />{error}</div></div>}
          <PredictionForm onSubmit={submit} loading={loading} />
        </section>
        <div className="trust-row"><span><span className="trust-dot" />Model-backed estimates</span><span>Historical capacity benchmarks</span><span>Two-shift factory plan</span></div>
        <p className="footer">Smart Apparel LK · Intelligent Order Acceptance and Workforce Allocation</p>
      </div>
    </main>
  );
}

function AppHeader() {
  return <header className="app-header"><a className="brand" href="/" aria-label="Smart Apparel LK home"><span className="brand-symbol">SA</span><span><strong>Smart Apparel</strong><small>LK / Operations intelligence</small></span></a><div className="header-status"><span className="status-pulse" />Planning workspace <span className="header-divider" /> Component 04</div></header>;
}
