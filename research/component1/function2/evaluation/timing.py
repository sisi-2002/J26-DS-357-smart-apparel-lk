"""F2.4-A Task 7 -- timing / efficiency benchmarking.

Separately instruments: preprocessing, detector forward pass, backward
(gradient) pass, heatmap computation (CAM math), and heatmap
resize/alignment back to the original image resolution -- for both methods,
on the same pilot instances, on CPU.

What is and is NOT measured here:
  - Model LOAD time (``DetectionXAI.__init__`` / ``YOLO(weights_path)``) is
    timed SEPARATELY and never added into any per-explanation latency
    figure -- conflating the two would make a single cold start look like
    part of every explanation's cost.
  - Peak memory: this is a CPU-only environment. Python's own
    ``tracemalloc`` only tracks Python-object allocations, not the C-level
    tensor buffers PyTorch allocates for its CPU backend, so it is reported
    here as a best-effort, DOCUMENTED-AS-PARTIAL figure, not a true peak
    RSS. A true peak RSS would need an OS-level sampler (e.g. periodically
    reading ``psutil.Process().memory_info().rss`` from a watcher thread);
    `psutil` is not a pre-existing dependency of this repo, so it is NOT
    added here (flagged, not silently worked around) -- see
    ``get_peak_rss_mb`` docstring for the resulting fallback.
  - Detector-only FPS (plain ``detect()``) is reported ALONGSIDE, never
    AS, end-to-end XAI latency -- they answer different questions (raw
    inference throughput vs. full explain-one-instance cost) and must not
    be compared as if interchangeable.
"""
from __future__ import annotations

import platform
import time
import tracemalloc
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

WARMUP_REPS = 2        # discarded before any timed repetition
TIMED_REPS = 5          # repetitions actually recorded (CPU, pilot scope)


def get_peak_rss_mb() -> dict:
    """Best-effort peak-memory figure. See module docstring: on this
    CPU-only setup without `psutil`, only Python-level allocation is
    directly measurable; this function reports that explicitly-partial
    number and flags it as such rather than presenting it as true peak RSS.
    """
    current, peak = tracemalloc.get_traced_memory()
    return {
        "metric": "tracemalloc_peak_python_allocations_mb",
        "value_mb": round(peak / (1024 * 1024), 3),
        "caveat": "Python-object allocations only -- does NOT include PyTorch's "
                  "native CPU tensor buffers. Not a true process peak RSS. "
                  "A true OS-level RSS sampler would require adding `psutil` "
                  "as a new dependency, which was flagged rather than added "
                  "without approval (see Hard Restrictions).",
    }


@dataclass
class StageTimings:
    preprocessing_s: float = 0.0
    forward_pass_s: float = 0.0
    backward_pass_s: float = 0.0
    cam_computation_s: float = 0.0
    resize_align_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.preprocessing_s + self.forward_pass_s + self.backward_pass_s
                + self.cam_computation_s + self.resize_align_s)


def instrumented_explain(xai, image_bgr: np.ndarray, detection, method: str,
                          conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> StageTimings:
    """Re-executes the SAME stages ``DetectionXAI.explain()`` performs
    (sharing its preprocessing/forward/backward/CAM math via direct calls
    into that protected class, not a duplicate implementation), but with a
    fine-grained ``time.perf_counter()`` wrapped around each stage boundary.
    This is instrumentation, not a parallel implementation: the numerical
    results are identical to an uninstrumented ``explain()`` call (verified
    by the test suite's timing-isolation test), only the stage boundaries
    are new.
    """
    import torch.nn.functional as F
    from ultralytics.utils.nms import non_max_suppression

    t = StageTimings()

    t0 = time.perf_counter()
    orig_shape = image_bgr.shape[:2]
    input_tensor, ratio, dw, dh = xai._preprocess(image_bgr)
    input_tensor = input_tensor.to(xai._device)
    input_tensor.requires_grad_(True)
    t.preprocessing_s = time.perf_counter() - t0

    all_activations, all_gradients, hooks = {}, {}, []
    from detection_xai import TARGET_LAYER_INDICES

    for s in range(3):
        li = TARGET_LAYER_INDICES[s]
        layer = xai._model.model[li]

        def _make_hooks(scale):
            def fwd(module, inp, out):
                all_activations[scale] = out

            def bwd(module, grad_in, grad_out):
                all_gradients[scale] = grad_out[0]
            return fwd, bwd

        fwd_fn, bwd_fn = _make_hooks(s)
        hooks.append(layer.register_forward_hook(fwd_fn))
        hooks.append(layer.register_full_backward_hook(bwd_fn))

    try:
        t0 = time.perf_counter()
        xai._model.zero_grad()
        output = xai._model(input_tensor)
        decoded, raw_preds = output
        preds = raw_preds["one2many"] if isinstance(raw_preds, dict) and "one2many" in raw_preds else raw_preds
        scores_all = preds["scores"]
        nms_out, keepi = non_max_suppression(decoded.detach(), conf_thres=conf_threshold,
                                              iou_thres=iou_threshold, return_idxs=True)
        t.forward_pass_s = time.perf_counter() - t0

        match_status, matched_flat_idx = xai._verify_detection_identity(nms_out, keepi, detection)
        if match_status != "verified":
            raise RuntimeError(f"Target identity {match_status}")

        scale_idx, cell = xai._flat_idx_to_scale_cell(matched_flat_idx)

        t0 = time.perf_counter()
        target_logit = scores_all[0, detection.class_id, matched_flat_idx]
        target_logit.backward(retain_graph=False)
        t.backward_pass_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        act = all_activations[scale_idx].detach()
        grad = all_gradients[scale_idx].detach()
        if method == "gradcam":
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        else:
            grad2, grad3 = grad ** 2, grad ** 3
            sum_act = act.sum(dim=(2, 3), keepdim=True) + 1e-7
            alpha = grad2 / (2 * grad2 + sum_act * grad3 + 1e-7)
            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().cpu().numpy()
        t.cam_computation_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = xai._map_to_original(cam_np, orig_shape, ratio, dw, dh)
        t.resize_align_s = time.perf_counter() - t0
    finally:
        for h in hooks:
            h.remove()
        if input_tensor.grad is not None:
            input_tensor.grad = None

    return t


@dataclass
class TimingSummary:
    method: str
    device: str
    n_warmup: int
    n_timed_reps: int
    per_rep_total_s: list = field(default_factory=list)
    per_stage_medians_s: dict = field(default_factory=dict)
    median_total_s: float = 0.0
    p95_total_s: float = 0.0
    torch_version: str = ""
    python_version: str = ""
    p95_caveat: str = (
        "p95 computed from only n_timed_reps repetitions (pilot scope) -- "
        "with n<20 this is a rough order-of-magnitude estimate, not a "
        "statistically stable percentile; F2.4-B should increase repetition "
        "count before treating p95 as reportable on its own."
    )


def benchmark_instance(xai, image_bgr, detection, method: str,
                        n_warmup=WARMUP_REPS, n_timed=TIMED_REPS) -> TimingSummary:
    for _ in range(n_warmup):
        instrumented_explain(xai, image_bgr, detection, method)

    stage_samples = {"preprocessing_s": [], "forward_pass_s": [], "backward_pass_s": [],
                      "cam_computation_s": [], "resize_align_s": []}
    totals = []
    for _ in range(n_timed):
        st = instrumented_explain(xai, image_bgr, detection, method)
        totals.append(st.total_s)
        for k in stage_samples:
            stage_samples[k].append(getattr(st, k))

    totals_sorted = sorted(totals)
    median = totals_sorted[len(totals_sorted) // 2]
    p95_idx = min(len(totals_sorted) - 1, int(round(0.95 * (len(totals_sorted) - 1))))
    p95 = totals_sorted[p95_idx]
    medians = {k: sorted(v)[len(v) // 2] for k, v in stage_samples.items()}

    return TimingSummary(
        method=method, device=str(xai._device), n_warmup=n_warmup, n_timed_reps=n_timed,
        per_rep_total_s=totals, per_stage_medians_s=medians,
        median_total_s=median, p95_total_s=p95,
        torch_version=torch.__version__, python_version=platform.python_version(),
    )
