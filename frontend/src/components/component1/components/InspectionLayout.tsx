import type { ReactNode } from 'react';
import { Link, NavLink } from 'react-router-dom';
import '../inspection.css';

export default function InspectionLayout({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return <div className="inspection-app">
    <a className="skip-link" href="#inspection-main">Skip to content</a>
    <header className="app-bar"><Link className="brand" to="/component1"><span className="brand-mark">SA</span><span>Smart Apparel<span className="brand-sub">LK / QUALITY INTELLIGENCE</span></span></Link><span className="badge">Component 01</span></header>
    <div className="dashboard-shell"><aside className="sidebar"><p className="eyebrow">FABRIC QUALITY</p><nav aria-label="Inspection navigation"><NavLink end to="/component1">New Inspection</NavLink><NavLink to="/component1/history">Inspection History</NavLink></nav></aside>
    <main id="inspection-main" className="workspace"><div className="page-heading"><p className="eyebrow">FABRIC DEFECT DETECTION</p><h1>{title}</h1><p>{description}</p></div>{children}<footer>Smart Apparel-LK · Inspection & explainability workspace</footer></main></div>
  </div>;
}
