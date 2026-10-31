import { Navigate } from 'react-router-dom';

/** Preserve existing bookmarks while keeping one primary product workflow. */
export default function ExperimentalRiskPage() {
  return <Navigate to="/component3/predict-risk" replace />;
}
