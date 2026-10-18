import type { RouteObject } from 'react-router-dom';
import FailureRiskPredictionPage from './pages/FailureRiskPredictionPage';

export const component3Routes: RouteObject[] = [
  { path: '/component3/predict-risk', element: <FailureRiskPredictionPage /> },
];
