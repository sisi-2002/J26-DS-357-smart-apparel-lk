import { useEffect, useState } from 'react';
import { artifactUrl } from '../../../shared/api/client';
import type { Detection, Explanation, Bbox } from '../types/component1';
import { decodeHeatmap, defectCrop, heatmapImage } from '../utils/heatmap';

interface Props {
  explanation: Explanation;
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
  detection?: Detection;
}
type Mode = 'Original' | 'Overlay' | 'Heatmap Only' | 'Defect Zoom';

export default function XaiComparison({ explanation, imageUrl, imageWidth = 1, imageHeight = 1, detection }: Props) {
  const [mode, setMode] = useState<Mode>(explanation.overlay_url ? 'Overlay' : 'Original');
  const [opacity, setOpacity] = useState(1);
  const [raw, setRaw] = useState<string | null>(null);
  const [rawError, setRawError] = useState('');
  const [imageError, setImageError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const rawUrl = explanation.raw_heatmap_url;
  const loadRaw = mode === 'Heatmap Only' && !!rawUrl && !raw;
  useEffect(() => {
    if (!loadRaw || !rawUrl) return;
    const controller = new AbortController();
    fetch(artifactUrl(rawUrl), { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('The raw heatmap could not be loaded.');
      const buffer = await response.arrayBuffer();
      if (!controller.signal.aborted) setRaw(heatmapImage(decodeHeatmap(buffer, imageWidth, imageHeight)));
    }).catch(error => {
      if (!controller.signal.aborted) setRawError(error instanceof Error ? error.message : 'Unable to display raw heatmap.');
    });
    return () => controller.abort();
  }, [loadRaw, rawUrl, imageWidth, imageHeight, attempt]);

  const full: Bbox = { x_min: 0, y_min: 0, x_max: imageWidth, y_max: imageHeight };
  const crop = mode === 'Defect Zoom' && detection ? defectCrop(detection.bbox, imageWidth, imageHeight) : full;
  const cropWidth = crop.x_max - crop.x_min;
  const cropHeight = crop.y_max - crop.y_min;
  const showOverlay = mode === 'Overlay' || mode === 'Defect Zoom';
  const renderFrame = (original: boolean) => <div className="xai-frame" data-crop={`${crop.x_min},${crop.y_min},${crop.x_max},${crop.y_max}`} style={{ aspectRatio: `${cropWidth} / ${cropHeight}` }}>
    <div className="xai-image-plane" style={{ width: `${imageWidth / cropWidth * 100}%`, height: `${imageHeight / cropHeight * 100}%`, left: `${-crop.x_min / cropWidth * 100}%`, top: `${-crop.y_min / cropHeight * 100}%` }}>
      {imageUrl && <img src={artifactUrl(imageUrl)} alt={original ? 'Original fabric for comparison' : 'Original beneath explanation'} onError={() => setImageError('The original image could not be loaded.')} />}
      {!original && showOverlay && explanation.overlay_url && <img src={artifactUrl(explanation.overlay_url)} alt="XAI heatmap overlay" style={{ opacity }} onError={() => setImageError('The explanation image could not be loaded.')} />}
      {!original && mode === 'Heatmap Only' && raw && <img src={raw} alt="Raw heatmap visualization" />}
      {detection && <span role="img" aria-label={`Selected ${detection.class_name} bounding box`} className="xai-defect-box" style={{ left: `${detection.bbox.x_min / imageWidth * 100}%`, top: `${detection.bbox.y_min / imageHeight * 100}%`, width: `${(detection.bbox.x_max - detection.bbox.x_min) / imageWidth * 100}%`, height: `${(detection.bbox.y_max - detection.bbox.y_min) / imageHeight * 100}%` }} />}
    </div>
  </div>;

  return <section aria-label="Explanation comparison">
    <div className="toolbar" role="group" aria-label="Display mode">
      {(['Original', 'Overlay', 'Heatmap Only', 'Defect Zoom'] as const).map(value => <button key={value} className="secondary" aria-pressed={mode === value} disabled={(value === 'Heatmap Only' && !rawUrl) || (value === 'Defect Zoom' && (!detection || !explanation.overlay_url)) || (value === 'Overlay' && !explanation.overlay_url)} onClick={() => { setMode(value); setImageError(''); }}>{value}</button>)}
    </div>
    {!rawUrl && <p className="muted">Heatmap Only is unavailable: no genuine raw heatmap was saved. The composite cannot be separated into saliency data.</p>}
    {imageUrl && <><label htmlFor="overlay-opacity">Explanation comparison opacity: {Math.round(opacity * 100)}%</label><input id="overlay-opacity" type="range" min={0} max={1} step={0.05} value={opacity} disabled={!showOverlay} onChange={e => setOpacity(Number(e.target.value))} /><p className="muted">0% shows the original; 100% shows the saved composite unchanged. This is a comparison fade between complete images, not raw heatmap transparency.</p></>}
    {mode === 'Heatmap Only' && !raw && (rawError ? <p role="alert">{rawError} <button className="secondary" onClick={() => { setRawError(''); setAttempt(v => v + 1); }}>Retry raw heatmap</button></p> : <p role="status">Loading genuine raw heatmap...</p>)}
    {imageError && <p role="alert">{imageError} Switch modes or reload to retry.</p>}
    <div className="comparison">
      {imageUrl && <figure>{renderFrame(true)}<figcaption>{mode === 'Defect Zoom' ? 'Original - defect zoom' : 'Original image'}</figcaption></figure>}
      <figure>{mode === 'Heatmap Only' && !raw ? <div className="notice">Raw heatmap not displayed.</div> : renderFrame(false)}<figcaption>{mode === 'Original' ? 'Original image' : mode === 'Heatmap Only' ? 'Genuine raw heatmap - full-image scale' : `${explanation.xai_method === 'grad-cam' ? 'Grad-CAM' : 'Grad-CAM++'} ${mode === 'Defect Zoom' ? '- defect zoom' : 'overlay'}`}</figcaption></figure>
    </div>
    {detection && <p className="muted">Outlined region: selected {detection.class_name} defect ({(detection.confidence * 100).toFixed(1)}% model confidence).</p>}
    {mode === 'Defect Zoom' && <p className="muted">Both views use the same crop around the actual detection, with 15% context. The saved pixels are enlarged without rescaling the saliency range; activations outside the crop remain visible in Overlay mode.</p>}
    <div className="heatmap-legend" aria-label="Heatmap color legend"><strong>Relative activation - full-image scale (JET)</strong><div className="heatmap-ramp" /><div className="heatmap-ticks"><span>Low / blue (0)</span><span>Mid / green</span><span>High / red (1)</span></div><p className="muted">Colors show relative activation within this explanation, not defect probability or severity. The saved overlay is already blended 50/50 with the fabric; fabric colors and comparison opacity affect its appearance. Scales are not comparable across detections or methods.</p></div>
  </section>;
}
