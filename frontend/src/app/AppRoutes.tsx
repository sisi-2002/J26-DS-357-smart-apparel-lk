import { Link, Navigate, useRoutes } from 'react-router-dom';
import { component3Routes } from '../components/component3/routes';

export default function AppRoutes() {
  return useRoutes([
    { path: '/', element: <Navigate to="/component3/predict-risk" replace /> },
    ...component3Routes,
    { path: '*', element: <section className="empty-state"><h1>Page not found</h1>
      <Link to="/component3/predict-risk">Return to machine risk</Link></section> },
  ]);
}
