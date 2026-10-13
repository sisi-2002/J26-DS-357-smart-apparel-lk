import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ApiError } from '../../../shared/api/client';
import { getInspection } from '../api/component1Api';
import type { Inspection } from '../types/component1';
import BboxCanvas from '../components/BboxCanvas';
import XaiPanel from '../components/XaiPanel';
import InspectionLayout from '../components/InspectionLayout';

// Provisional QC presentation rule, not an experimentally validated operating
// threshold. Never use this value for inference, NMS or API request parameters.
export const LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.50;

export default function ResultPage() {
  const { inspectionId } = useParams<{ inspectionId: string }>();
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedDetId, setSelectedDetId] = useState<string | null>(null);

  useEffect(() => {
    if (!inspectionId) return;
    let cancelled = false;
    setLoading(true);
    setSelectedDetId(null);
    setError('');
    getInspection(inspectionId)
      .then((data) => {
        if (!cancelled) setInspection(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(
            err instanceof ApiError
              ? err.detail
              : 'Failed to load inspection.',
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [inspectionId]);

  const selected = inspection?.detections.find(d => d.id === selectedDetId) ?? null;
  const needsManualReview = inspection?.detections.some(
    detection => detection.confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD,
  ) ?? false;
  const detectionModel = inspection?.model.experiment_id === 'E02_optimized'
    ? 'E02 Optimized YOLOv8n'
    : inspection?.model.experiment_id || 'Model not recorded';

  return <InspectionLayout title="Inspection Results" description="Review detected regions and select a defect to explore its explanation.">
    {loading ? <p className="panel" role="status">Loading inspection…</p> : error ? <div className="panel"><p role="alert">{error}</p><Link to="/component1">Back to upload</Link></div> : inspection && <>
      <div className="split">
        <section className="panel"><div className="toolbar"><h2>Fabric image</h2><span className="badge">{inspection.image_width} × {inspection.image_height} px</span></div><p className="muted">Select a box or a defect in the list to inspect it.</p><div id="fabric-view" className="image-stage"><BboxCanvas imageUrl={inspection.image_url} imageWidth={inspection.image_width} imageHeight={inspection.image_height} detections={inspection.detections} selectedId={selectedDetId} onSelect={setSelectedDetId} /></div></section>
        <aside className="stack"><section className="panel"><h2>Inspection summary</h2><p>Total Defects Detected</p><div className="metric">{inspection.summary.total_defects}</div><p>{inspection.summary.total_defects} defect{inspection.summary.total_defects !== 1 && 's'} detected</p>
          {!inspection.detections.length && <p className="notice">No defects detected at the selected confidence threshold.</p>}
          {!!inspection.detections.length && <table><thead><tr><th>Detected Defect Class</th><th>Model Confidence (%)</th></tr></thead><tbody>{inspection.detections.map(d => <tr key={d.id} className={selectedDetId === d.id ? 'selected' : ''} onClick={() => setSelectedDetId(d.id)}><td><button className="detection-button" aria-pressed={selectedDetId === d.id} onClick={e => { e.stopPropagation(); setSelectedDetId(d.id); }}>{selectedDetId === d.id ? '✓ ' : ''}#{d.detection_order} <span>{d.class_name}</span></button></td><td>{(d.confidence * 100).toFixed(1)}%</td></tr>)}</tbody></table>}
          {needsManualReview && (
            <div className="notice" role="status">
              <strong>Manual Review Recommended</strong>
              <p>Low-confidence detection. Please verify the detected defect visually.</p>
            </div>
          )}
          <p className="muted">Model confidence is a model score, not a calibrated probability of correctness. All predictions require human QC verification, even when no review warning is shown.</p>
          <dl>
            <dt>Selected Confidence Threshold (%)</dt>
            <dd>{Math.round(inspection.model.confidence_threshold * 100)}%</dd>
            <dt>Detection Model</dt>
            <dd>{detectionModel}</dd>
            <dt>Inspection Date and Time</dt>
            <dd>{new Date(inspection.detected_at).toLocaleString()}</dd>
          </dl>
        </section><div className="toolbar"><Link className="button" to="/component1">New Inspection</Link><Link className="button secondary" to="/component1/history">History</Link></div></aside>
      </div>
      <section className="panel" style={{ marginTop: 24 }}><p className="eyebrow">VISUAL EXPLANATION</p>{selected ? <><h2>Detection #{selected.detection_order} — {selected.class_name}</h2><p className="muted">Selected defect · {(selected.confidence * 100).toFixed(1)}% confidence</p><XaiPanel key={`${inspection.id}:${selected.id}`} inspectionId={inspection.id} detectionId={selected.id} detection={selected} imageWidth={inspection.image_width} imageHeight={inspection.image_height} imageUrl={inspection.image_url} /></> : <><h2>Explore a detection</h2><p className="muted">{inspection.detections.length ? 'Select a defect above to view or generate its Grad-CAM explanation.' : 'Explanations are available for detected defects.'}</p></>}</section>
    </>}
  </InspectionLayout>;
}
