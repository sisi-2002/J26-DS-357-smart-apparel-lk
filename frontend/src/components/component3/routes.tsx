import type { RouteObject } from 'react-router-dom';
import FailureRiskPredictionPage from './pages/FailureRiskPredictionPage';
import EvidenceStudyPage from './pages/EvidenceStudyPage';

export const component3Routes: RouteObject[] = [
  { path: '/component3/predict-risk', element: <FailureRiskPredictionPage /> },
  { path: '/component3/evidence-study', element: <EvidenceStudyPage /> },
];
