import { Link, Navigate, NavLink, useMatch, useRoutes } from 'react-router-dom';
import component1Routes from '../components/component1/routes';
import { component3Routes } from '../components/component3/routes';
import MainLayout from '../shared/layouts/MainLayout';
import inspectionBaseStyles from '../index.css?inline';
import maintenanceStyles from './styles.css?inline';

export default function App() {
  const inspection = useMatch('/component1/*') !== null;
  const element = useRoutes([
    { path: '/', element: <Navigate to="/component3/predict-risk" replace /> },
    ...component1Routes,
    ...component3Routes,
    { path: '*', element: <section className="empty-state"><h1>Page not found</h1>
      <Link to="/component1">Return to defect inspection</Link>
      <Link to="/component3/predict-risk">Return to machine risk</Link></section> },
  ]);

  return <>
    {/* Each feature retains its existing theme without global selector collisions. */}
    <style>{inspection ? inspectionBaseStyles : maintenanceStyles}</style>
    <nav aria-label="Components" style={{ display: 'flex', flexWrap: 'wrap', gap: 12,
      padding: '8px 20px', background: '#142D4E', color: '#F5F7F8' }}>
      <NavLink to="/component1" style={{ color: 'inherit' }}>Defect inspection</NavLink>
      <NavLink to="/component3/predict-risk" style={{ color: 'inherit' }}>Predictive maintenance</NavLink>
    </nav>
    {inspection ? element : <MainLayout>{element}</MainLayout>}
  </>;
}
