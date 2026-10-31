import '@testing-library/jest-dom/vitest';
/**
 * Focused UI tests for the Function 1 inspection flow:
 * upload -> detection viewer (bbox scaling) -> Grad-CAM/Grad-CAM++ panel.
 *
 * All backend calls are mocked at the component1Api module boundary —
 * no real network/backend/model involved.
 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import UploadPage from '../pages/UploadPage';
import HistoryPage from '../pages/HistoryPage';
import ResultPage from '../pages/ResultPage';
import BboxCanvas from '../components/BboxCanvas';
import XaiPanel from '../components/XaiPanel';
import * as api from '../api/component1Api';
import { ApiError } from '../../../shared/api/client';
import type { Detection, Explanation, Inspection } from '../types/component1';
import { RESIZE_OBSERVER_STUB_WIDTH } from '../../../test/setup';

vi.mock('../api/component1Api');
const mockApi = api as unknown as {
  [K in keyof typeof api]: ReturnType<typeof vi.fn>;
};

function baseDetection(overrides: Partial<Detection> = {}): Detection {
  return {
    id: 'det-1',
    detection_order: 0,
    class_id: 6,
    class_name: 'stain',
    confidence: 0.91,
    bbox: { x_min: 10, y_min: 20, x_max: 50, y_max: 80 },
    pre_nms_index: 142,
    ...overrides,
  };
}

function baseInspection(overrides: Partial<Inspection> = {}): Inspection {
  return {
    id: 'insp-1',
    image_width: 500,
    image_height: 500,
    image_url: '/api/v1/component1/artifacts/inspections/insp-1/original.jpg',
    thumbnail_url: null,
    detected_at: '2026-10-15T12:00:00Z',
    inference_time_ms: 42,
    summary: { total_defects: 1, classes_found: ['stain'] },
    model: {
      model_version: 'ultralytics-8.4.175',
      model_weights_hash: 'sha256:c578003c',
      device: 'cpu',
      confidence_threshold: 0.25,
      iou_threshold: 0.5,
      imgsz: 640,
      experiment_id: 'E02_optimized',
    },
    detections: [baseDetection()],
    complexity_tier: null,
    complexity_score: null,
    ...overrides,
  };
}

function baseExplanation(overrides: Partial<Explanation> = {}): Explanation {
  return {
    id: 'expl-1',
    detection_id: 'det-1',
    xai_method: 'grad-cam',
    status: 'completed',
    outcome: 'success',
    overlay_url: '/api/v1/component1/artifacts/inspections/insp-1/xai_expl-1.png',
    raw_heatmap_url: null,
    model_version: 'ultralytics-8.4.175',
    target_layer: 'model.model[21]',
    settings: null,
    error_message: null,
    inference_time_ms: 289,
    generated_at: '2026-10-15T12:01:00Z',
    created_at: '2026-10-15T12:01:00Z',
    ...overrides,
  };
}

function makeFile(name: string, type: string, sizeBytes = 1024): File {
  const buf = new Uint8Array(sizeBytes);
  return new File([buf], name, { type });
}

/**
 * userEvent.upload() respects the input's `accept` attribute and silently
 * filters out non-matching files before they ever reach onChange — which
 * is realistic for a real browser's file picker, but means it can't be
 * used to test the component's own defensive client-side format check
 * (a real user can still bypass `accept` via drag-and-drop or "All
 * Files"). Fire the native change event directly for that case.
 */
function uploadBypassingAccept(input: HTMLElement, file: File) {
  Object.defineProperty(input, 'files', { value: [file], configurable: true });
  fireEvent.change(input);
}

beforeEach(() => {
  vi.clearAllMocks();
});

// ── Upload ───────────────────────────────────────────────────────

describe('UploadPage', () => {
  it('rejects an unsupported file format client-side without calling the API', async () => {
    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>,
    );
    const input = screen.getByLabelText(/upload image/i);
    uploadBypassingAccept(input, makeFile('bad.gif', 'image/gif'));

    expect(await screen.findByRole('alert')).toHaveTextContent(/unsupported format/i);
    expect(mockApi.submitInspection).not.toHaveBeenCalled();
  });

  it('rejects an oversized file client-side without calling the API', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>,
    );
    const input = screen.getByLabelText(/upload image/i);
    const big = makeFile('big.jpg', 'image/jpeg', 11 * 1024 * 1024);
    await user.upload(input, big);

    expect(await screen.findByRole('alert')).toHaveTextContent(/exceeds 10 mb limit/i);
    expect(mockApi.submitInspection).not.toHaveBeenCalled();
  });

  it('shows a preview and accepts a valid JPEG', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>,
    );
    const input = screen.getByLabelText(/upload image/i);
    await user.upload(input, makeFile('good.jpg', 'image/jpeg'));

    expect(screen.getByAltText('Preview')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run inspection/i })).toBeEnabled();
  });

  it('submits the file and navigates to the result page on success', async () => {
    const user = userEvent.setup();
    mockApi.submitInspection.mockResolvedValue(baseInspection({ id: 'insp-42' }));

    render(
      <MemoryRouter initialEntries={['/component1']}>
        <Routes>
          <Route path="/component1" element={<UploadPage />} />
          <Route path="/component1/inspections/:inspectionId" element={<div>RESULT PAGE</div>} />
        </Routes>
      </MemoryRouter>,
    );

    await user.upload(screen.getByLabelText(/upload image/i), makeFile('good.jpg', 'image/jpeg'));
    await user.click(screen.getByRole('button', { name: /run inspection/i }));

    await waitFor(() => expect(screen.getByText('RESULT PAGE')).toBeInTheDocument());
    expect(mockApi.submitInspection).toHaveBeenCalledWith(expect.any(File), 0.25);
  });

  it('prevents duplicate submission while a request is in flight', async () => {
    const user = userEvent.setup();
    let resolveSubmit: (v: Inspection) => void = () => {};
    mockApi.submitInspection.mockImplementation(
      () => new Promise<Inspection>((resolve) => { resolveSubmit = resolve; }),
    );

    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>,
    );
    await user.upload(screen.getByLabelText(/upload image/i), makeFile('good.jpg', 'image/jpeg'));
    const button = screen.getByRole('button', { name: /run inspection/i });
    await user.click(button);
    expect(screen.getByRole('button', { name: /inspecting/i })).toBeDisabled();

    await user.click(screen.getByRole('button', { name: /inspecting/i }));
    expect(mockApi.submitInspection).toHaveBeenCalledTimes(1);

    resolveSubmit(baseInspection());
  });

  it('shows the backend error detail on API failure', async () => {
    const user = userEvent.setup();
    mockApi.submitInspection.mockRejectedValue(new ApiError(503, 'Model unavailable'));

    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>,
    );
    await user.upload(screen.getByLabelText(/upload image/i), makeFile('good.jpg', 'image/jpeg'));
    await user.click(screen.getByRole('button', { name: /run inspection/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Model unavailable');
  });
});

// ── Result page / detection viewer ────────────────────────────────

function renderResultPage(inspection: Inspection) {
  mockApi.getInspection.mockResolvedValue(inspection);
  return render(
    <MemoryRouter initialEntries={[`/component1/inspections/${inspection.id}`]}>
      <Routes>
        <Route path="/component1/inspections/:inspectionId" element={<ResultPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ResultPage', () => {
  it('shows a loading state before the inspection loads', () => {
    mockApi.getInspection.mockReturnValue(new Promise(() => {}));
    render(
      <MemoryRouter initialEntries={['/component1/inspections/insp-1']}>
        <Routes>
          <Route path="/component1/inspections/:inspectionId" element={<ResultPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText(/loading inspection/i)).toBeInTheDocument();
  });

  it('renders detections, confidence and model identity including the experiment id', async () => {
    const { container } = renderResultPage(baseInspection());
    await waitFor(() => expect(container.textContent).toMatch(/1\s*defects?\s*detected/i));
    expect(screen.getAllByText('stain').length).toBeGreaterThan(0);
    // "91.0%" legitimately appears twice (the bbox overlay label AND the
    // detection table cell) — assert on the table cell specifically
    // rather than an ambiguous getByText that depends on whether the
    // ResizeObserver-driven bbox overlay has rendered yet.
    const row = screen.getAllByRole('row')[1];
    expect(row).toHaveTextContent('91.0%');
    expect(screen.getByText('E02 Optimized YOLOv8n')).toBeInTheDocument();
  });

  it('does not claim an experiment identity when the backend omits it (historical inspection)', async () => {
    renderResultPage(
      baseInspection({
        model: {
          model_version: 'ultralytics-8.4.175',
          model_weights_hash: null,
          device: 'cpu',
          confidence_threshold: 0.25,
          iou_threshold: 0.5,
          imgsz: 640,
          experiment_id: null,
        },
      }),
    );
    await screen.findByText('Model not recorded');
    expect(screen.queryByText(/ultralytics/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/E02_optimized/)).not.toBeInTheDocument();
    expect(screen.queryByText(/experiment:/i)).not.toBeInTheDocument();
  });

  it('renders "no defect detected" for empty results without fabricating detections', async () => {
    renderResultPage(
      baseInspection({ detections: [], summary: { total_defects: 0, classes_found: [] } }),
    );
    expect(await screen.findByText(/No defects detected at the selected confidence threshold./i)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('renders multiple detections as distinct rows', async () => {
    renderResultPage(
      baseInspection({
        detections: [
          baseDetection({ id: 'det-1', class_name: 'stain', confidence: 0.9 }),
          baseDetection({ id: 'det-2', class_name: 'cut', confidence: 0.6, detection_order: 1 }),
        ],
        summary: { total_defects: 2, classes_found: ['cut', 'stain'] },
      }),
    );
    const rows = await screen.findAllByRole('row');
    // header + 2 detection rows
    expect(rows).toHaveLength(3);
  });

  it('shows the API error detail when loading the inspection fails', async () => {
    mockApi.getInspection.mockRejectedValue(new ApiError(404, 'Inspection not found'));
    render(
      <MemoryRouter initialEntries={['/component1/inspections/missing']}>
        <Routes>
          <Route path="/component1/inspections/:inspectionId" element={<ResultPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText('Inspection not found')).toBeInTheDocument();
  });

  it('selecting a detection reveals its detail panel and the XAI panel', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([]);
    const { container } = renderResultPage(baseInspection());

    await waitFor(() => expect(container.textContent).toMatch(/1\s*defects?\s*detected/i));
    const row = screen.getAllByRole('row')[1];
    await user.click(row);

    expect(await screen.findByText(/detection #0 — stain/i)).toBeInTheDocument();
    expect(screen.queryByText(/pre-nms index/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate explanation/i })).toBeInTheDocument();
  });
});

// ── Bounding box coordinate scaling ───────────────────────────────

describe('BboxCanvas', () => {
  it('scales bbox coordinates from original-image pixels to the rendered display size', () => {
    const imageWidth = 500;
    const detection = baseDetection({ bbox: { x_min: 100, y_min: 50, x_max: 300, y_max: 150 } });
    const { container } = render(
      <BboxCanvas
        imageUrl="/x.jpg"
        imageWidth={imageWidth}
        imageHeight={500}
        detections={[detection]}
        selectedId={null}
        onSelect={() => {}}
      />,
    );

    const scale = RESIZE_OBSERVER_STUB_WIDTH / imageWidth; // 900 / 500 = 1.8
    const box = container.querySelector('[title^="stain"]') as HTMLElement;
    expect(box).toBeTruthy();
    expect(box.style.left).toBe(`${100 * scale}px`);
    expect(box.style.top).toBe(`${50 * scale}px`);
    expect(box.style.width).toBe(`${(300 - 100) * scale}px`);
    expect(box.style.height).toBe(`${(150 - 50) * scale}px`);
  });

  it('handles a non-square image without distorting the scale factor per axis', () => {
    const detection = baseDetection({ bbox: { x_min: 0, y_min: 0, x_max: 200, y_max: 100 } });
    const { container } = render(
      <BboxCanvas
        imageUrl="/x.jpg"
        imageWidth={1000}
        imageHeight={400}
        detections={[detection]}
        selectedId={null}
        onSelect={() => {}}
      />,
    );
    const scaleX = RESIZE_OBSERVER_STUB_WIDTH / 1000; // 0.9
    const box = container.querySelector('[title^="stain"]') as HTMLElement;
    expect(box.style.width).toBe(`${200 * scaleX}px`);
  });

  it('marks the selected detection visually distinct from unselected ones', () => {
    const d1 = baseDetection({ id: 'a' });
    const d2 = baseDetection({ id: 'b', bbox: { x_min: 200, y_min: 200, x_max: 250, y_max: 250 } });
    const { container } = render(
      <BboxCanvas
        imageUrl="/x.jpg"
        imageWidth={500}
        imageHeight={500}
        detections={[d1, d2]}
        selectedId="a"
        onSelect={() => {}}
      />,
    );
    const boxes = container.querySelectorAll('[title^="stain"]');
    expect(boxes[0].getAttribute('style')).toContain('3px solid');
    expect(boxes[1].getAttribute('style')).toContain('2px solid');
  });
});

// ── Grad-CAM / Grad-CAM++ panel ────────────────────────────────────

describe('XaiPanel', () => {
  it('requests Grad-CAM for the given detection and displays the resulting heatmap', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([]);
    mockApi.generateExplanation.mockResolvedValue(baseExplanation({ xai_method: 'grad-cam' }));

    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
    await user.click(screen.getByRole('button', { name: /generate explanation/i }));

    expect(mockApi.generateExplanation).toHaveBeenCalledWith('insp-1', 'det-1', 'grad-cam');
    expect(await screen.findByAltText(/xai heatmap overlay/i)).toBeInTheDocument();
    expect(screen.getByText(/status:/i)).toHaveTextContent('completed');
  });

  it('requests Grad-CAM++ for the same detection when that method is selected', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([]);
    mockApi.generateExplanation.mockResolvedValue(baseExplanation({ xai_method: 'grad-cam++' }));

    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
    await user.selectOptions(screen.getByLabelText(/method/i), 'grad-cam++');
    await user.click(screen.getByRole('button', { name: /generate explanation/i }));

    expect(mockApi.generateExplanation).toHaveBeenCalledWith('insp-1', 'det-1', 'grad-cam++');
  });

  it('switching methods loads any already-saved explanation for that method', async () => {
    const savedPlusPlus = baseExplanation({ id: 'expl-pp', xai_method: 'grad-cam++' });
    mockApi.getExplanationsForDetection.mockResolvedValue([
      baseExplanation({ id: 'expl-cam', xai_method: 'grad-cam' }),
      savedPlusPlus,
    ]);

    const user = userEvent.setup();
    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
    await screen.findByAltText(/xai heatmap overlay/i); // grad-cam auto-loaded first

    await user.selectOptions(screen.getByLabelText(/method/i), 'grad-cam++');
    await waitFor(() => expect(mockApi.getExplanationsForDetection).toHaveBeenCalledTimes(1));
    // No new generate call needed — it reused the saved explanation.
    expect(mockApi.generateExplanation).not.toHaveBeenCalled();
  });

  it('never shows a stale explanation after switching to a different detection', async () => {
    mockApi.getExplanationsForDetection.mockResolvedValueOnce([
      baseExplanation({ detection_id: 'det-1' }),
    ]);
    const { rerender } = render(<XaiPanel key="det-1" inspectionId="insp-1" detectionId="det-1" />);
    await screen.findByAltText(/xai heatmap overlay/i);

    mockApi.getExplanationsForDetection.mockResolvedValueOnce([]); // det-2 has nothing yet
    rerender(<XaiPanel key="det-2" inspectionId="insp-1" detectionId="det-2" />);

    await waitFor(() => {
      expect(screen.queryByAltText(/xai heatmap overlay/i)).not.toBeInTheDocument();
    });
  });

  it('shows the error detail when explanation generation fails', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([]);
    mockApi.generateExplanation.mockRejectedValue(new ApiError(503, 'Model unavailable'));

    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
    await user.click(screen.getByRole('button', { name: /generate explanation/i }));

    expect(await screen.findByText('Model unavailable')).toBeInTheDocument();
  });

  it('shows a loading state while generation is in flight and disables re-submission', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([]);
    let resolveGenerate: (v: Explanation) => void = () => {};
    mockApi.generateExplanation.mockImplementation(
      () => new Promise<Explanation>((resolve) => { resolveGenerate = resolve; }),
    );

    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
    await user.click(screen.getByRole('button', { name: /generate explanation/i }));

    expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled();
    resolveGenerate(baseExplanation());
    await screen.findByAltText(/xai heatmap overlay/i);
  });

  it('handles a missing/unavailable explanation gracefully without an error banner', async () => {
    mockApi.getExplanationsForDetection.mockRejectedValue(new ApiError(404, 'not found'));
    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);

    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
    expect(screen.queryByText(/not found/i)).not.toBeInTheDocument();
  });

  it('renders a degenerate outcome message distinctly from a hard error', async () => {
    mockApi.getExplanationsForDetection.mockResolvedValue([
      baseExplanation({ outcome: 'degenerate', overlay_url: null }),
    ]);
    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
    expect(await screen.findByText(/heatmap is degenerate/i)).toBeInTheDocument();
  });
});


describe('Redesigned inspection interactions', () => {
  it('supports dropping, replacing and removing an image and displays dimensions', async () => {
    const user = userEvent.setup();
    const { container } = render(<MemoryRouter><UploadPage /></MemoryRouter>);
    fireEvent.drop(container.querySelector('.drop-zone')!, { dataTransfer: { files: [makeFile('first.png', 'image/png')] } });
    const preview = screen.getByAltText('Preview');
    Object.defineProperty(preview, 'naturalWidth', { value: 640 });
    Object.defineProperty(preview, 'naturalHeight', { value: 480 });
    fireEvent.load(preview);
    expect(screen.getByText(/640.*480 px/)).toBeInTheDocument();
    await user.upload(screen.getByLabelText(/upload image/i), makeFile('second.jpg', 'image/jpeg'));
    expect(screen.queryByText('first.png')).not.toBeInTheDocument();
    expect(screen.getByText('second.jpg')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Remove image' }));
    expect(screen.queryByAltText('Preview')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run Inspection' })).toBeDisabled();
  });
  it('clears an invalid replacement and rejects undecodable images', async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><UploadPage /></MemoryRouter>);
    const input = screen.getByLabelText(/upload image/i);
    await user.upload(input, makeFile('good.jpg', 'image/jpeg'));
    uploadBypassingAccept(input, makeFile('bad.gif', 'image/gif'));
    expect(screen.getByRole('button', { name: 'Run Inspection' })).toBeDisabled();
    uploadBypassingAccept(input, makeFile('broken.png', 'image/png'));
    fireEvent.error(screen.getByAltText('Preview'));
    expect(screen.getByRole('alert')).toHaveTextContent('could not be decoded');
  });
  it('submits the chosen percentage using the unchanged fractional contract', async () => {
    const user = userEvent.setup();
    mockApi.submitInspection.mockResolvedValue(baseInspection());
    render(<MemoryRouter><UploadPage /></MemoryRouter>);
    await user.upload(screen.getByLabelText(/upload image/i), makeFile('good.jpg', 'image/jpeg'));
    fireEvent.change(screen.getByLabelText(/confidence threshold/i), { target: { value: '0.65' } });
    expect(screen.getByText('65%')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Run Inspection' }));
    expect(mockApi.submitInspection).toHaveBeenCalledWith(expect.any(File), 0.65);
  });
  it('allows keyboard selection of a bounding box', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(<BboxCanvas imageUrl="/x.jpg" imageWidth={500} imageHeight={500} detections={[baseDetection()]} selectedId={null} onSelect={onSelect} />);
    screen.getByRole('button', { name: /select stain/i }).focus();
    await user.keyboard('{Enter}');
    expect(onSelect).toHaveBeenCalledWith('det-1');
  });
  it('shows the composite unchanged initially and supports original comparison', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([baseExplanation()]);
    render(<XaiPanel inspectionId="insp-1" detectionId="det-1" imageUrl="/original.jpg" />);
    expect(await screen.findByAltText('XAI heatmap overlay')).toHaveStyle({ opacity: '1' });
    fireEvent.change(screen.getByLabelText(/comparison opacity/i), { target: { value: '0.25' } });
    expect(screen.getByAltText('XAI heatmap overlay')).toHaveStyle({ opacity: '0.25' });
    await user.click(screen.getByRole('button', { name: 'Original' }));
    expect(screen.queryByAltText('XAI heatmap overlay')).not.toBeInTheDocument();
    expect(screen.getByAltText('Original beneath explanation')).toBeInTheDocument();
  });
  it('resets pagination with a filter in one request and links to inspection details', async () => {
    const user = userEvent.setup();
    mockApi.listInspections.mockResolvedValue({ total: 21, skip: 0, limit: 10, items: [{ id:'insp-1', detected_at:'2026-10-15T12:00:00Z', thumbnail_url:'/thumb.jpg', total_defects:1, defect_classes:['stain'], model_version:'recorded-model' }] });
    render(<MemoryRouter><HistoryPage /></MemoryRouter>);
    expect(await screen.findByRole('link', { name: /view inspection from/i })).toHaveAttribute('href', '/component1/inspections/insp-1');
    await user.click(screen.getByRole('button', { name: /next/i }));
    await waitFor(() => expect(mockApi.listInspections).toHaveBeenLastCalledWith(expect.objectContaining({ skip:10 })));
    const previous = mockApi.listInspections.mock.calls.length;
    await user.selectOptions(screen.getByLabelText('Class'), 'stain');
    await waitFor(() => expect(mockApi.listInspections).toHaveBeenCalledTimes(previous + 1));
    expect(mockApi.listInspections).toHaveBeenLastCalledWith(expect.objectContaining({ skip:0, class_name:'stain' }));
  });
  it('shows history empty and API failure states separately', async () => {
    mockApi.listInspections.mockResolvedValueOnce({ total:0, skip:0, limit:10, items:[] });
    const { unmount } = render(<MemoryRouter><HistoryPage /></MemoryRouter>);
    expect(await screen.findByText('No inspections found.')).toBeInTheDocument();
    unmount();
    mockApi.listInspections.mockRejectedValueOnce(new ApiError(503, 'History unavailable'));
    render(<MemoryRouter><HistoryPage /></MemoryRouter>);
    expect(await screen.findByRole('alert')).toHaveTextContent('History unavailable');
    expect(screen.queryByText('No inspections found.')).not.toBeInTheDocument();
  });
});


it('waits for saved explanations before allowing a generation request', () => {
  mockApi.getExplanationsForDetection.mockReturnValue(new Promise(() => {}));
  render(<XaiPanel inspectionId="insp-1" detectionId="det-1" />);
  expect(screen.getByRole('button', { name: /generate explanation/i })).toBeDisabled();
  expect(mockApi.generateExplanation).not.toHaveBeenCalled();
});


describe('QC inspection summary', () => {
  it.each([0.49, 0.4999, 0.50, 0.75])('applies the UI review rule to the original score %s', async confidence => {
    renderResultPage(baseInspection({ detections: [baseDetection({ confidence })] }));
    await screen.findByText('E02 Optimized YOLOv8n');
    if (confidence < 0.50) {
      expect(screen.getByText('Manual Review Recommended')).toBeInTheDocument();
      expect(screen.getByText('Low-confidence detection. Please verify the detected defect visually.')).toBeInTheDocument();
    } else {
      expect(screen.queryByText('Manual Review Recommended')).not.toBeInTheDocument();
    }
    expect(screen.getAllByRole('row')[1]).toHaveTextContent(`${(confidence * 100).toFixed(1)}%`);
    expect(screen.getByText(/not a calibrated probability of correctness/)).toBeInTheDocument();
    expect(screen.getByText(/All predictions require human QC verification/)).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(mockApi.submitInspection).not.toHaveBeenCalled();
  });

  it('warns if any displayed detection is low confidence and retains each score', async () => {
    renderResultPage(baseInspection({
      detections: [baseDetection({ confidence: 0.9 }), baseDetection({ id: 'det-2', detection_order: 1, class_name: 'cut', confidence: 0.3 })],
      summary: { total_defects: 2, classes_found: ['stain', 'cut'] },
    }));
    await screen.findByText('Manual Review Recommended');
    const rows = screen.getAllByRole('row');
    expect(rows[1]).toHaveTextContent('stain');
    expect(rows[1]).toHaveTextContent('90.0%');
    expect(rows[2]).toHaveTextContent('cut');
    expect(rows[2]).toHaveTextContent('30.0%');
  });

  it('shows only the no-detection message for an empty result', async () => {
    renderResultPage(baseInspection({ detections: [], summary: { total_defects: 0, classes_found: [] } }));
    await screen.findByText('No defects detected at the selected confidence threshold.');
    expect(screen.queryByText('Manual Review Recommended')).not.toBeInTheDocument();
  });

  it.each([null, undefined])('does not infer model identity from the software version when identity is %s', async experiment_id => {
    const inspection = baseInspection();
    inspection.model.experiment_id = experiment_id;
    renderResultPage(inspection);
    await screen.findByText('Model not recorded');
    expect(screen.queryByText('E02 Optimized YOLOv8n')).not.toBeInTheDocument();
  });

  it('does not relabel a different recorded experiment as E02', async () => {
    const inspection = baseInspection();
    inspection.model.experiment_id = 'Other_recorded_experiment';
    renderResultPage(inspection);
    await screen.findByText('Other_recorded_experiment');
    expect(screen.queryByText('E02 Optimized YOLOv8n')).not.toBeInTheDocument();
  });

  it('shows inspector labels and omits technical details even after selecting and loading XAI', async () => {
    const user = userEvent.setup();
    mockApi.getExplanationsForDetection.mockResolvedValue([baseExplanation()]);
    const { container } = renderResultPage(baseInspection());
    await screen.findByText('E02 Optimized YOLOv8n');
    for (const label of ['Total Defects Detected', 'Detected Defect Class', 'Model Confidence (%)', 'Selected Confidence Threshold (%)', 'Detection Model', 'Inspection Date and Time']) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    await user.click(screen.getByRole('button', { name: /#0 stain/i }));
    await screen.findByAltText('XAI heatmap overlay');
    expect(container.querySelector('details')).toBeNull();
    expect(container.textContent).not.toMatch(/technical details|inspection id|iou threshold|input size|device|weights hash|ultralytics|pre-nms|model version|insp-1|det-1|sha256:c578003c/i);
    expect(screen.getByLabelText('Method')).toBeInTheDocument();
  });
});


describe('Persisted inspection identity navigation', () => {
  it('fetches identity after upload and again on reload', async () => {
    const user = userEvent.setup();
    const inspection = baseInspection();
    mockApi.submitInspection.mockResolvedValue(inspection);
    mockApi.getInspection.mockResolvedValue(inspection);
    const { unmount } = render(<MemoryRouter initialEntries={['/component1']}><Routes>
      <Route path="/component1" element={<UploadPage />} />
      <Route path="/component1/inspections/:inspectionId" element={<ResultPage />} />
    </Routes></MemoryRouter>);
    await user.upload(screen.getByLabelText(/upload image/i), makeFile('fabric.png', 'image/png'));
    await user.click(screen.getByRole('button', { name: 'Run Inspection' }));
    await screen.findByText('E02 Optimized YOLOv8n');
    expect(mockApi.getInspection).toHaveBeenCalledWith(inspection.id);
    unmount();
    renderResultPage(inspection);
    await screen.findByText('E02 Optimized YOLOv8n');
    expect(mockApi.getInspection).toHaveBeenCalledTimes(2);
  });

  it('opens the saved identity from history rather than inferring it from the list version', async () => {
    const user = userEvent.setup();
    mockApi.listInspections.mockResolvedValue({ total:1, skip:0, limit:10, items:[{ id:'insp-1', detected_at:'2026-10-15T12:00:00Z', thumbnail_url:null, total_defects:1, defect_classes:['stain'], model_version:'software-version-only' }] });
    mockApi.getInspection.mockResolvedValue(baseInspection());
    render(<MemoryRouter initialEntries={['/component1/history']}><Routes>
      <Route path="/component1/history" element={<HistoryPage />} />
      <Route path="/component1/inspections/:inspectionId" element={<ResultPage />} />
    </Routes></MemoryRouter>);
    await user.click(await screen.findByRole('link', { name:/view inspection from/i }));
    await screen.findByText('E02 Optimized YOLOv8n');
    expect(mockApi.getInspection).toHaveBeenCalledWith('insp-1');
    expect(screen.queryByText('software-version-only')).not.toBeInTheDocument();
  });
});
