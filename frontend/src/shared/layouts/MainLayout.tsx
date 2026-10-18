import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

export default function MainLayout({ children }: { children: ReactNode }) {
  return <div className="workspace">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark" aria-hidden="true">S<span>↗</span></span>
        <span>SMART APPAREL<span className="brand-country">LK / RESEARCH</span></span></div>
      <div className="nav-label">RESEARCH WORKSPACE</div>
      <nav aria-label="Main navigation"><NavLink to="/component3/predict-risk">
        <span aria-hidden="true">◈</span> Machine risk <span className="nav-number">03</span>
      </NavLink></nav>
      <div className="sidebar-note"><span className="small-label">COMPONENT 03</span>
        <p>Reliability-aware<br />predictive maintenance</p>
        <span className="sidebar-rule" />
        <p className="sidebar-caption">Explore historical risk.<br />Understand the evidence.</p>
      </div>
      <div className="sidebar-footer">J26-DS-357<span>Research prototype · v0.1</span></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><span>SMART APPAREL LK <span className="breadcrumb-divider">/</span> Component 3</span>
        <span className="research-tag"><span aria-hidden="true" /> Historical research</span></header>
      <main id="main-content">{children}</main>
      <footer className="page-footer"><span>SMART APPAREL LK</span><span>Synthetic data. Human interpretation.</span></footer>
    </div>
  </div>;
}
