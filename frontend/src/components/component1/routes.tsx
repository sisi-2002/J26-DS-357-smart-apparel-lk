import type { RouteObject } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import HistoryPage from './pages/HistoryPage';

const component1Routes: RouteObject[] = [
  { path: '/component1', element: <UploadPage /> },
  { path: '/component1/history', element: <HistoryPage /> },
  { path: '/component1/inspections/:inspectionId', element: <ResultPage /> },
];

export default component1Routes;
