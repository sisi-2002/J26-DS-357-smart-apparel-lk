import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ApiError } from '../../../shared/api/client';
import { submitInspection } from '../api/component1Api';

import InspectionLayout from '../components/InspectionLayout';

const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ACCEPTED = '.jpg,.jpeg,.png';

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.25);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const abortRef = useRef(false);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const inputRef = useRef<HTMLInputElement>(null);
  const [dimensions, setDimensions] = useState('');
  const [dragging, setDragging] = useState(false);
  const handleFile = (f: File | null) => {
    if (submitting || !f) return;
    setError(''); setFile(null); setPreview(null); setDimensions('');
    if (!['image/jpeg', 'image/png'].includes(f.type)) { setError('Unsupported format. Use JPEG or PNG.'); return; }
    if (f.size > MAX_SIZE_BYTES) { setError('File exceeds 10 MB limit.'); return; }
    if (!f.size) { setError('This file is empty. Choose a JPEG or PNG image.'); return; }
    setFile(f); setPreview(URL.createObjectURL(f));
  };

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!file || submitting) return;
      abortRef.current = false;
      setError('');
      setSubmitting(true);
      try {
        const result = await submitInspection(file, threshold);
        if (!abortRef.current) {
          navigate(`/component1/inspections/${result.id}`);
        }
      } catch (err) {
        if (!abortRef.current) {
          setError(
            err instanceof ApiError
              ? err.detail
              : 'Unexpected error. Please try again.',
          );
        }
      } finally {
        if (!abortRef.current) setSubmitting(false);
      }
    },
    [file, threshold, submitting, navigate],
  );

  useEffect(() => {
    return () => {
      abortRef.current = true;
    };
  }, []);

  return <InspectionLayout title="New Inspection" description="Inspect fabric, locate defects and explore the model’s visual explanations.">
    <form onSubmit={handleSubmit} className="split" aria-busy={submitting}>
      <section className="panel"><h2>01 / Add a fabric image</h2>
        <input ref={inputRef} className="sr-only" aria-label="Upload image" id="image-upload" type="file" accept={ACCEPTED} disabled={submitting} onChange={e => { handleFile(e.target.files?.[0] ?? null); e.target.value = ''; }} />
        <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0] ?? null); }}>
          {preview ? <img key={preview} className="preview" src={preview} alt="Preview" onLoad={e => setDimensions(`${e.currentTarget.naturalWidth} × ${e.currentTarget.naturalHeight} px`)} onError={() => { setFile(null); setPreview(null); setError('This image could not be decoded. Choose another JPEG or PNG.'); }} /> : <><div className="upload-symbol" aria-hidden="true">↑</div><h3>Drop your fabric image here</h3><p className="muted">Use a clear, well-lit image of the fabric surface.</p></>}
          <button type="button" className="secondary" disabled={submitting} onClick={() => inputRef.current?.click()}>{file ? 'Replace image' : 'Browse Files'}</button>
          <p className="muted">JPEG or PNG · Maximum 10 MB</p>
        </div>
        {file && <div className="file-meta"><h3>{file.name}</h3><p className="muted">{(file.size / 1024).toFixed(0)} KB {dimensions && `· ${dimensions}`}</p><button className="secondary" type="button" disabled={submitting} onClick={() => { setFile(null); setPreview(null); setDimensions(''); setError(''); }}>Remove image</button></div>}
      </section>
      <div className="stack"><section className="panel"><h2>02 / Inspection settings</h2><label htmlFor="threshold">Confidence threshold <strong className="badge">{Math.round(threshold * 100)}%</strong></label><input id="threshold" type="range" min={0} max={1} step={0.01} value={threshold} disabled={submitting} aria-describedby="threshold-help" onChange={e => setThreshold(Number(e.target.value))} /><p id="threshold-help" className="muted" title="Only detections at or above this confidence score are returned.">Lower values show more possible defects. Higher values keep only more confident predictions.</p>
        <details><summary>Advanced settings</summary><p className="muted">The confidence threshold is sent to the detector. Model, input resolution and overlap suppression use the backend configuration; recorded values appear with the result.</p></details>
        {error && <p role="alert">{error}</p>}<button type="submit" disabled={!file || submitting}>{submitting ? 'Inspecting…' : 'Run Inspection'}</button>{submitting && <p role="status">Processing your image. Please wait…</p>}
      </section><section className="panel"><h2>From image to insight</h2><p className="muted">1. Upload a fabric image<br />2. Review detected regions<br />3. Select a defect to generate an explanation</p><p className="muted">Model predictions support your quality review. Always verify the fabric visually.</p></section></div>
    </form>
  </InspectionLayout>;
}
