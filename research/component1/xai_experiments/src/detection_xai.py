"""Reusable detection inference and Grad-CAM explanation module for YOLOv8.

Target tensor semantics:
  - Detect.cv3 outputs raw CLASS LOGITS (pre-sigmoid) per grid cell.
  - The tensor `preds["scores"]` has shape (batch, num_classes, total_cells)
    where total_cells = 6400 + 1600 + 400 = 8400 for imgsz=640.
  - NMS confidence = sigmoid(class_logit). There is no separate objectness
    score in YOLOv8 — the class logit IS the detection score before sigmoid.
  - Grad-CAM backward target is the raw logit (not the sigmoid probability)
    at the matched cell. Using the logit avoids sigmoid saturation in gradients.

Preprocessing guarantee:
  - detect() and explain() share _preprocess() (640×640 letterbox, pad=114,
    center-aligned, INTER_LINEAR, BGR→RGB, float32/255).
  - detect() runs NMS with return_idxs=True, storing each survivor's flat
    pre-NMS cell index in Detection.pre_nms_index.
  - explain() reruns the identical forward pass (same tensor, same model,
    eval mode — only difference is autograd graph construction) and NMS,
    then verifies the detection's pre_nms_index is present in keepi with
    matching class and score within VERIFY_SCORE_ATOL.
  - VERIFY_SCORE_ATOL = 1e-4: justified because both passes receive the
    same input tensor in eval mode; autograd tracking does not change
    forward values. The tolerance covers only hardware float32
    non-determinism.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ["baekra", "color issues", "contamination", "cut",
               "gray stitch", "selvet", "stain"]
STRIDES = {0: 8, 1: 16, 2: 32}
GRID_SIZES = {0: 80, 1: 40, 2: 20}
CELLS_PER_SCALE = {0: 6400, 1: 1600, 2: 400}
TARGET_LAYER_INDICES = {0: 15, 1: 18, 2: 21}

VERIFY_SCORE_ATOL = 1e-4


@dataclass
class Detection:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    pre_nms_index: int = -1


@dataclass
class ExplanationResult:
    status: str  # "success", "degenerate", "error", "unavailable", "ambiguous"
    method: str
    detection_id: int
    class_id: int
    class_name: str
    target_layer: str
    target_scale: int
    target_cell_gy_gx: Optional[tuple[int, int]]
    target_logit: Optional[float]
    target_sigmoid: Optional[float]
    heatmap_raw: Optional[np.ndarray]
    heatmap_overlay: Optional[np.ndarray]
    is_finite: bool = True
    is_nonzero: bool = False
    heatmap_max: float = 0.0
    generation_time_s: float = 0.0
    error_message: str = ""
    model_version: str = ""
    settings: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    detections: list[Detection]
    image_shape: tuple[int, int]
    model_version: str
    error_message: str = ""


class DetectionXAI:
    """Model loading, detection inference, and per-detection explanation."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        import os
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        from ultralytics import YOLO
        self._yolo = YOLO(weights_path)
        self._model = self._yolo.model
        self._model.to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(True)

        self._device = device
        self._weights_path = weights_path

        import ultralytics
        self._model_version = f"ultralytics-{ultralytics.__version__}"

    def detect(self, image_bgr: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.5) -> InferenceResult:
        if image_bgr is None or image_bgr.size == 0:
            return InferenceResult([], (0, 0), self._model_version,
                                   error_message="Unreadable or empty image")
        h, w = image_bgr.shape[:2]
        input_tensor, ratio, dw, dh = self._preprocess(image_bgr)
        input_tensor = input_tensor.to(self._device)

        with torch.no_grad():
            output = self._model(input_tensor)

        if not isinstance(output, tuple):
            raise RuntimeError("Expected tuple output from model in eval mode")
        decoded, _ = output

        from ultralytics.utils.nms import non_max_suppression
        nms_out, keepi = non_max_suppression(
            decoded, conf_thres=conf_threshold,
            iou_thres=iou_threshold, return_idxs=True)

        survivors = nms_out[0]
        flat_indices = keepi[0]

        detections = []
        for i in range(len(survivors)):
            cls_id = int(survivors[i, 5].item())
            bbox_lb = survivors[i, :4].tolist()
            bbox_orig = self._bbox_from_letterbox(bbox_lb, ratio, dw, dh)
            bbox_orig = self._clip_bbox(bbox_orig, w, h)
            detections.append(Detection(
                detection_id=i,
                class_id=cls_id,
                class_name=(CLASS_NAMES[cls_id]
                            if cls_id < len(CLASS_NAMES)
                            else f"class_{cls_id}"),
                confidence=float(survivors[i, 4].item()),
                bbox_xyxy=[round(v, 1) for v in bbox_orig],
                pre_nms_index=int(flat_indices[i].item()),
            ))
        return InferenceResult(detections, (h, w), self._model_version)

    def explain(self, image_bgr: np.ndarray, detection: Detection,
                method: str = "gradcam",
                target_layer_scale: Optional[int] = None,
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.5) -> ExplanationResult:
        valid_methods = ("gradcam", "gradcampp")
        if method not in valid_methods:
            return ExplanationResult(
                status="error", method=method, detection_id=detection.detection_id,
                class_id=detection.class_id, class_name=detection.class_name,
                target_layer="", target_scale=-1, target_cell_gy_gx=None,
                target_logit=None, target_sigmoid=None,
                heatmap_raw=None, heatmap_overlay=None,
                error_message=f"Invalid method '{method}'. Valid: {valid_methods}",
                model_version=self._model_version)

        if image_bgr is None or image_bgr.size == 0:
            return self._error_result(detection, method, "Unreadable or empty image")

        t0 = time.time()
        orig_shape = image_bgr.shape[:2]
        input_tensor, ratio, dw, dh = self._preprocess(image_bgr)
        input_tensor = input_tensor.to(self._device)
        input_tensor.requires_grad_(True)

        all_activations = {}
        all_gradients = {}
        hooks = []

        for s in range(3):
            li = TARGET_LAYER_INDICES[s]
            layer = self._model.model[li]

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
            self._model.zero_grad()
            output = self._model(input_tensor)

            if isinstance(output, tuple):
                decoded, raw_preds = output
            else:
                raise RuntimeError("Expected tuple output from model in eval mode")

            preds = raw_preds
            if isinstance(preds, dict) and "one2many" in preds:
                preds = preds["one2many"]
            scores_all = preds["scores"]  # raw logits: (1, nc, 8400)

            from ultralytics.utils.nms import non_max_suppression
            nms_out, keepi = non_max_suppression(
                decoded.detach(), conf_thres=conf_threshold,
                iou_thres=iou_threshold, return_idxs=True)

            match_status, matched_flat_idx = self._verify_detection_identity(
                nms_out, keepi, detection)

            if match_status != "verified":
                elapsed = time.time() - t0
                return ExplanationResult(
                    status=match_status, method=method,
                    detection_id=detection.detection_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    target_layer="", target_scale=-1,
                    target_cell_gy_gx=None,
                    target_logit=None, target_sigmoid=None,
                    heatmap_raw=None, heatmap_overlay=None,
                    generation_time_s=round(elapsed, 3),
                    error_message=f"Target identity {match_status}",
                    model_version=self._model_version,
                    settings={"match_status": match_status,
                              "verify_score_atol": VERIFY_SCORE_ATOL})

            scale_idx, cell = self._flat_idx_to_scale_cell(matched_flat_idx)
            if target_layer_scale is not None:
                scale_idx = target_layer_scale

            layer_idx = TARGET_LAYER_INDICES[scale_idx]
            layer_name = f"model.model[{layer_idx}]"

            target_logit = scores_all[0, detection.class_id, matched_flat_idx]
            target_logit.backward(retain_graph=False)

            act = all_activations[scale_idx].detach()
            grad = all_gradients[scale_idx].detach()

            if method == "gradcam":
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            else:  # gradcampp
                grad2 = grad ** 2
                grad3 = grad ** 3
                sum_act = act.sum(dim=(2, 3), keepdim=True) + 1e-7
                alpha = grad2 / (2 * grad2 + sum_act * grad3 + 1e-7)
                weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * act).sum(dim=1, keepdim=True))

            cam_np = cam.squeeze().cpu().numpy()

        except Exception as e:
            return self._error_result(detection, method, f"Grad-CAM failed: {e}")
        finally:
            for h in hooks:
                h.remove()
            if input_tensor.grad is not None:
                input_tensor.grad = None

        cam_orig = self._map_to_original(cam_np, orig_shape, ratio, dw, dh)
        elapsed = time.time() - t0

        is_finite = bool(np.all(np.isfinite(cam_orig)))
        is_nonzero = bool(cam_orig.max() > 1e-8)

        if not is_finite:
            status = "error"
            error_msg = "Heatmap contains non-finite values"
        elif not is_nonzero:
            status = "degenerate"
            error_msg = "Heatmap is all zeros (degenerate)"
        else:
            status = "success"
            error_msg = ""

        cam_norm = self._normalize(cam_orig)
        overlay = self._overlay(image_bgr, cam_norm) if is_nonzero else None

        logit_val = float(target_logit.item())
        sigmoid_val = float(torch.sigmoid(target_logit).item())

        return ExplanationResult(
            status=status, method=method,
            detection_id=detection.detection_id,
            class_id=detection.class_id, class_name=detection.class_name,
            target_layer=layer_name, target_scale=scale_idx,
            target_cell_gy_gx=cell,
            target_logit=round(logit_val, 4),
            target_sigmoid=round(sigmoid_val, 4),
            heatmap_raw=cam_orig, heatmap_overlay=overlay,
            is_finite=is_finite, is_nonzero=is_nonzero,
            heatmap_max=round(float(cam_orig.max()), 6),
            generation_time_s=round(elapsed, 3),
            error_message=error_msg,
            model_version=self._model_version,
            settings={"imgsz": 640, "method": method, "target_layer": layer_name,
                       "scale": scale_idx, "pre_nms_index": matched_flat_idx,
                       "verify_score_atol": VERIFY_SCORE_ATOL},
        )

    @staticmethod
    def _verify_detection_identity(nms_out, keepi, detection,
                                    score_atol=VERIFY_SCORE_ATOL):
        """Verify a detection's pre_nms_index survived NMS with matching class/score.

        Returns:
            (status, flat_idx):
                "verified" + flat_idx when index found and class/score match.
                "unavailable" + -1 when index missing, class mismatch, or no
                    pre_nms_index set.
        """
        if detection.pre_nms_index < 0:
            return "unavailable", -1

        survivors = nms_out[0]
        flat_indices = keepi[0]

        if len(survivors) == 0:
            return "unavailable", -1

        idx_match = (flat_indices == detection.pre_nms_index)
        if not idx_match.any():
            return "unavailable", -1

        surv = survivors[idx_match][0]
        if int(surv[5].item()) != detection.class_id:
            return "unavailable", -1
        if abs(float(surv[4].item()) - detection.confidence) > score_atol:
            return "unavailable", -1

        return "verified", detection.pre_nms_index

    @staticmethod
    def _flat_idx_to_scale_cell(flat_idx):
        """Convert flat index (0..8399) to (scale_idx, (gy, gx))."""
        offset = 0
        for scale_idx in range(3):
            gs = GRID_SIZES[scale_idx]
            n = gs * gs
            if flat_idx < offset + n:
                local = flat_idx - offset
                return scale_idx, (local // gs, local % gs)
            offset += n
        raise ValueError(f"Flat index {flat_idx} out of range 0..8399")

    @staticmethod
    def _bbox_to_letterbox(bbox_xyxy, ratio, dw, dh):
        """Convert bbox from original image coords to letterboxed coords."""
        return [bbox_xyxy[0] * ratio + dw, bbox_xyxy[1] * ratio + dh,
                bbox_xyxy[2] * ratio + dw, bbox_xyxy[3] * ratio + dh]

    @staticmethod
    def _bbox_from_letterbox(bbox_lb, ratio, dw, dh):
        """Convert bbox from letterboxed coords to original image coords."""
        return [(bbox_lb[0] - dw) / ratio, (bbox_lb[1] - dh) / ratio,
                (bbox_lb[2] - dw) / ratio, (bbox_lb[3] - dh) / ratio]

    @staticmethod
    def _clip_bbox(bbox, img_w, img_h):
        """Clip bbox to image bounds."""
        return [max(0, min(bbox[0], img_w)),
                max(0, min(bbox[1], img_h)),
                max(0, min(bbox[2], img_w)),
                max(0, min(bbox[3], img_h))]

    @staticmethod
    def _box_iou(box1, boxes2):
        """IoU of one box (4,) against N boxes (N, 4). Returns (N,)."""
        x1 = torch.max(box1[0], boxes2[:, 0])
        y1 = torch.max(box1[1], boxes2[:, 1])
        x2 = torch.min(box1[2], boxes2[:, 2])
        y2 = torch.min(box1[3], boxes2[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        return inter / (a1 + a2 - inter + 1e-7)

    @staticmethod
    def _preprocess(img_bgr, imgsz=640):
        h, w = img_bgr.shape[:2]
        r = min(imgsz / h, imgsz / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (imgsz - new_unpad[0]) / 2
        dh = (imgsz - new_unpad[1]) / 2
        if (w, h) != new_unpad:
            resized = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = img_bgr
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor, r, dw, dh

    @staticmethod
    def _map_to_original(cam, orig_shape, ratio, dw, dh, imgsz=640):
        cam_resized = cv2.resize(cam, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        h_orig, w_orig = orig_shape[:2]
        x_start = int(round(dw))
        y_start = int(round(dh))
        new_w = int(round(w_orig * ratio))
        new_h = int(round(h_orig * ratio))
        x_end = min(x_start + new_w, imgsz)
        y_end = min(y_start + new_h, imgsz)
        cam_cropped = cam_resized[y_start:y_end, x_start:x_end]
        return cv2.resize(cam_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _normalize(cam):
        vmin, vmax = cam.min(), cam.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(cam)
        return (cam - vmin) / (vmax - vmin)

    @staticmethod
    def _overlay(img_bgr, cam_norm, alpha=0.5):
        heatmap = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    def _error_result(self, detection, method, msg):
        return ExplanationResult(
            status="error", method=method,
            detection_id=detection.detection_id,
            class_id=detection.class_id, class_name=detection.class_name,
            target_layer="", target_scale=-1, target_cell_gy_gx=None,
            target_logit=None, target_sigmoid=None,
            heatmap_raw=None, heatmap_overlay=None,
            error_message=msg, model_version=self._model_version)
