import type { RouteObject } from 'react-router-dom';
import FailureRiskPredictionPage from './pages/FailureRiskPredictionPage';
import EvidenceStudyPage from './pages/EvidenceStudyPage';
import ExperimentalRiskPage from './experimental/ExperimentalRiskPage';
import UnifiedRiskPage from './experimental/UnifiedRiskPage';
import { experimentalPreviewEnabled } from './experimental/featureFlag';

function ProductRiskPage() {
  return experimentalPreviewEnabled() ? <UnifiedRiskPage /> : <FailureRiskPredictionPage />;
}

export const component3Routes: RouteObject[] = [
  { path: '/component3/predict-risk', element: <ProductRiskPage /> },
  { path: '/component3/evidence-study', element: <EvidenceStudyPage /> },
  { path: '/component3/experimental-risk', element: <ExperimentalRiskPage /> },
];
