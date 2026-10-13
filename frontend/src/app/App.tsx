import { BrowserRouter, Navigate, useRoutes } from 'react-router-dom';
import component1Routes from '../components/component1/routes';

function Routes() {
  const element = useRoutes([
    ...component1Routes,
    { path: '*', element: <Navigate to="/component1" replace /> },
  ]);
  return element;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes />
    </BrowserRouter>
  );
}
