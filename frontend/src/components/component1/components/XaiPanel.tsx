import { useEffect, useRef, useState } from 'react';
import { ApiError, artifactUrl } from '../../../shared/api/client';
import { generateExplanation, getExplanationsForDetection } from '../api/component1Api';
import XaiComparison from './XaiComparison';
import type { Detection, Explanation } from '../types/component1';

const OUTCOMES: Record<string, string> = {
  success: 'Explanation generated successfully.',
  degenerate: 'Heatmap is degenerate (all zeros or non-finite values).',
  unavailable: 'Detection could not be verified in the re-run.',
  ambiguous: 'Multiple candidates matched. The explanation may be ambiguous.',
  error: 'An error occurred during explanation generation.',
};
interface Props { inspectionId: string; detectionId: string; imageUrl?: string; imageWidth?: number; imageHeight?: number; detection?: Detection }
export default function XaiPanel(props: Props) {
  // Scope all async state to both identities, even when a caller does not provide a key.
  return <ExplanationViewer key={`${props.inspectionId}:${props.detectionId}`} {...props} />;
}
function ExplanationViewer({ inspectionId, detectionId, imageUrl, imageWidth, imageHeight, detection }: Props) {
  const [method, setMethod] = useState('grad-cam');
  const [saved, setSaved] = useState<Explanation[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const generation = useRef(0);
  useEffect(() => {
    let active = true;
    const initialGeneration = generation.current;
    getExplanationsForDetection(inspectionId, detectionId).then(items => {
      if (active && generation.current === initialGeneration) setSaved(items.filter(e => e.detection_id === detectionId));
    }).catch(err => {
      if (active && generation.current === initialGeneration && !(err instanceof ApiError && err.status === 404)) setError('Saved explanations could not be loaded. You can retry generation.');
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; ++generation.current; };
  }, [inspectionId, detectionId]);
  const explanation = saved.find(e => e.xai_method === method);
  const handleGenerate = async () => {
    if (loading || generating) return;
    const request = ++generation.current;
    setGenerating(true); setError('');
    try {
      const result = await generateExplanation(inspectionId, detectionId, method);
      if (request === generation.current) {
        if (result.detection_id !== detectionId || result.xai_method !== method) throw new Error('Explanation identity mismatch');
        setSaved(items => [...items.filter(e => e.xai_method !== method), result]);
      }
    } catch (err) {
      if (request === generation.current) setError(err instanceof ApiError ? err.detail : 'Failed to generate explanation. Please retry.');
    } finally { if (request === generation.current) setGenerating(false); }
  };
  return <div aria-busy={loading || generating}>
    <h3>Explanation (XAI)</h3><div className="toolbar"><label htmlFor="xai-method">Method</label><select id="xai-method" value={method} disabled={generating} onChange={e => { setMethod(e.target.value); setError(''); }}><option value="grad-cam">Grad-CAM</option><option value="grad-cam++">Grad-CAM++</option></select><button onClick={() => void handleGenerate()} disabled={loading || generating || (!!explanation?.overlay_url && explanation.outcome === 'success')}>{generating ? 'Generating…' : explanation?.overlay_url && explanation.outcome === 'success' ? 'Explanation ready' : 'Generate Explanation'}</button><a href="#fabric-view">Return to bounding boxes</a></div>
    {loading && <p role="status">Checking saved explanations…</p>}{generating && <p role="status">Generating explanation for the selected defect…</p>}{error && <p role="alert">{error}</p>}
    {explanation && <><p role="status">Status: <strong>{explanation.status}</strong>{explanation.outcome && ` / ${explanation.outcome}`}</p>{explanation.outcome && <p className="muted">{OUTCOMES[explanation.outcome] ?? explanation.outcome}</p>}{explanation.error_message && <p role="alert">{explanation.error_message}</p>}
      {(explanation.overlay_url || explanation.raw_heatmap_url) && <XaiComparison key={explanation.id} explanation={explanation} imageUrl={imageUrl} imageWidth={imageWidth} imageHeight={imageHeight} detection={detection} />}
      {explanation.raw_heatmap_url && <a href={artifactUrl(explanation.raw_heatmap_url)}>Download raw heatmap (.npy)</a>}
    </>}
    {!loading && !explanation && !generating && <p className="notice">No saved explanation for this method. Generate one for the selected defect.</p>}
    <p className="muted">Colors indicate relative model response. Strong colors do not establish explanation correctness or defect severity.</p>
  </div>;
}
