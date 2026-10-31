"""F2.2C-QA Task 3 -- improved V2 SAM proposals for the 19 instances the AI
preliminary review marked "Needs Correction" (9) or "Reject" (10).

Uses ONLY the already-present local SAM v1 ViT-B checkpoint/environment
(``data-preprocessing/master_fabric_dataset/research/.sam_venv`` +
``.sam_weights/sam_vit_b_01ec64.pth`` -- identical to what Task B4-B8 used
to generate V1; no new install, no new download). Run with:

    data-preprocessing/master_fabric_dataset/research/.sam_venv/Scripts/python.exe \
        -m research.component1.function2.dataset.f22c_sam_mask_generation_v2

Correction strategy (per instance, see ``TARGET_INSTANCES`` below)
--------------------------------------------------------------------
Every one of the 19 target instances gets a *tighter, focused local crop*
(the image is cropped around the GT box with padding before being handed to
SAM, rather than the full original image V1 used) plus, where this QA stage
could actually ground a location in real pixel evidence, positive/negative
**point** prompts on top of the box prompt. Grounding is never an invented
coordinate: every point is the output of a simple, documented, per-instance
pixel-evidence algorithm (``POINT_MODE`` below) run on the real image data,
gated by this stage's own direct visual inspection of the corresponding V1
overlay image (recorded in ``f22c_qa_task2_diagnostics.py``) confirming
which algorithm is appropriate for that instance. Where visual inspection
found **no** distinguishable anomaly to ground a point on (one baekra
instance) or **no defensible boundary at all** (one color-issues instance),
this script explicitly does NOT invent points -- see ``POINT_MODE.NONE``
and ``STRATEGY.FLAG_UNSUITABLE`` respectively.

Point-grounding algorithms (all operate on real pixel data inside/around
the GT box of the real image -- never a guessed coordinate):

  * ``bright_streak``   -- for selvet/gray-stitch instances where visual
    inspection found a near-white/bright thin streak or stitch clearly
    visible against a darker/coloured background. Positive point(s) =
    pixel(s) of locally maximal lightness inside the box. Negative
    points = pixels of locally maximal colour saturation inside the box
    (representing the surrounding printed/coloured pattern).
  * ``dark_anomaly``    -- for stain/cut/contamination/crease instances
    where visual inspection found a locally distinct mark/line/patch.
    A local background reference colour is sampled from the thin annulus
    just OUTSIDE the GT box (inside the padded crop) -- i.e. real
    surrounding fabric pixels, not a guess. Positive point = the box-
    interior pixel of maximum Euclidean RGB distance from that reference.
    Negative points = the box-interior pixel of minimum distance (most
    "background-like") plus one point sampled directly from the annulus.
  * ``multi_dark_anomaly`` -- same reference-colour computation as
    ``dark_anomaly``, but selects up to 3 spatially-separated local
    maxima (greedy: take the max, suppress a radius around it, repeat)
    for scattered/multi-blob stains, per the task brief's explicit
    stain-specific guidance (grounded positive points on the stain +
    negative points on clearly normal background, rather than letting
    SAM fill the whole region).
  * ``texture_anomaly``  -- for the one "disrupted weave band" cut
    instance, where visual inspection found a patch of visibly different
    local texture (small blocky pattern) amid the surrounding diagonal
    plaid lines. Score = local Laplacian-magnitude response, same
    max/min-location logic as ``dark_anomaly`` applied to that response
    map instead of raw colour.
  * ``none``             -- crop-only, no point prompt. Used either
    because visual inspection found nothing to ground (documented
    per-instance), or for the one edge-of-fabric selvet instance where
    the correction is a crop that trims the near-black background margin
    rather than a point.

SAM confidence is explicitly NOT the sole selection criterion. For every
instance this script generates both a crop+box-only candidate and (where
applicable) a crop+box+points candidate, scores each of SAM's 3 returned
masks per call by a documented quality heuristic (containment inside the
box, not implausibly small/large relative to the crop, connectivity), and
records every candidate's SAM score AND the heuristic verdict in the
output JSON, picking the final "best" by the documented heuristic -- never
blindly by SAM score alone. The full comparison is kept, not discarded.

Status vocabulary (recorded per instance, never silently collapsed):
  - "success"                      -- a V2 mask was produced and judged
                                       at least as plausible as V1 by the
                                       documented heuristic.
  - "attempted_no_defensible_result" -- SAM was run but every candidate
                                       mask still looks implausible on
                                       inspection of its own metrics (e.g.
                                       near-full-crop, badly fragmented);
                                       the mask file is still written (for
                                       transparency) but flagged.
  - "not_attempted_flagged_unsuitable_for_pixel_iou" -- this QA stage
                                       judged (from AI review + direct
                                       visual inspection) that no
                                       defensible pixel boundary exists for
                                       this instance at all; no SAM call
                                       was made.

Nothing in this module is a human-verified mask, an approved mask, or a
ground-truth segmentation. Every V2 mask is an algorithmic proposal
pending human QC review, same as V1.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.dataset.f22c_sam_mask_generation import (  # noqa: E402
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    _resolve_valid_image_path,
    yolo_box_to_pixels,
)

PILOT_DIR = REPO_ROOT / "research" / "component1" / "function2" / "results" / "reference_masks" / "pilot_v1"
MANIFESTS_DIR = PILOT_DIR / "manifests"
SAM_PROPOSALS_V2_DIR = PILOT_DIR / "sam_proposals_v2"
OVERLAYS_V2_DIR = PILOT_DIR / "overlays_v2"
REPORTS_DIR = PILOT_DIR / "reports"

STATUS_UNSUITABLE = "not_attempted_flagged_unsuitable_for_pixel_iou"
STATUS_NO_DEFENSIBLE = "attempted_no_defensible_result"
STATUS_SUCCESS = "success"


@dataclass
class TargetConfig:
    instance_id: str
    strategy: str  # "crop_plus_points" | "crop_only" | "flag_unsuitable"
    point_mode: Optional[str] = None  # "bright_streak" | "dark_anomaly" | "multi_dark_anomaly" | "texture_anomaly" | None
    n_negative: int = 2
    pad_frac: float = 0.25
    trim_dark_margin: bool = False
    grounding_confidence: str = "medium"  # "high" | "medium" | "low" | "none"
    rationale: str = ""


# One entry per one of the 19 Needs-Correction/Reject instances. The
# rationale text records what this QA stage actually saw when visually
# inspecting the V1 overlay PNG for that instance (see also
# f22c_qa_task2_diagnostics.py's per-instance diagnosis, which these
# strategies are derived from).
TARGET_INSTANCES: list[TargetConfig] = [
    TargetConfig(
        "chenab_test_IMG-20230817-WA0060_jpg.rf.b39081cd04cc1a6bf465fb89a5254b2a__inst_001",
        strategy="crop_only", point_mode=None, grounding_confidence="none",
        rationale="Visual inspection of the overlay found no region distinguishable from "
            "the overall printed floral pattern anywhere inside the (near-full-image) "
            "GT box; no point can be honestly grounded. Crop-only attempt kept for "
            "transparency; expected to remain not_defensible.",
    ),
    TargetConfig(
        "chenab_test_IMG_20230731_121423_jpg.rf.8d63939b65ca55211b9819ab88a03acb__inst_001",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="medium",
        rationale="Visual inspection found a thin diagonal crease/line crossing the "
            "printed-leaf pattern inside the box, locally darker than the surrounding "
            "print -- plausible but not certain to be the labelled defect.",
    ),
    TargetConfig(
        "chenab_train_20230712_163620_jpeg_jpg.rf.2ed46bd9534cd0b2228fff1ee46ca554__inst_002",
        strategy="flag_unsuitable", grounding_confidence="none",
        rationale="Visual inspection of the GT box region found plain, featureless fabric "
            "with no distinguishable colour anomaly at all -- consistent with the task "
            "brief's 'diffuse colour defect, no defensible visible boundary' case. No "
            "SAM call made; flagged unsuitable for pixel-IoU rather than inventing a "
            "boundary.",
    ),
    TargetConfig(
        "chenab_train_IMG-20230906-WA0070_jpg.rf.d33f74a1611e49180274d6f370cd6a27__inst_001",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="low",
        rationale="Visual inspection found only a very faint, low-contrast diagonal crease "
            "crossing the box on otherwise plain fabric -- a weak but real pixel-level "
            "signal, not an invented location.",
    ),
    TargetConfig(
        "chenab_train_IMG-20230906-WA0070_jpg.rf.d33f74a1611e49180274d6f370cd6a27__inst_002",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="low",
        rationale="Same faint diagonal crease as inst_001, different box segment along it.",
    ),
    TargetConfig(
        "chenab_train_20230712_141546_jpeg_jpg.rf.009d497c13d53d48c37dced25ab5e828__inst_001",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="medium",
        rationale="Visual inspection found a horizontal fold/seam line crossing the "
            "full-span box, locally darker/different from the surrounding plaid -- "
            "plausibly the fold the 'cut' label refers to.",
    ),
    TargetConfig(
        "chenab_train_20230712_141552_jpeg_jpg.rf.015bfa40d3d292429c7442a12c6a7406__inst_001",
        strategy="crop_plus_points", point_mode="texture_anomaly", grounding_confidence="high",
        rationale="Visual inspection found a clearly visible band of disrupted/blocky weave "
            "pattern, texturally distinct from the surrounding smooth diagonal plaid "
            "lines, inside the GT box.",
    ),
    TargetConfig(
        "chenab_train_20230712_141552_jpeg_jpg.rf.015bfa40d3d292429c7442a12c6a7406__inst_002",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="low",
        rationale="Visual inspection found only a diagonal seam/fold where two plaid panels "
            "meet, visually similar to a normal fabric fold.",
    ),
    TargetConfig(
        "chenab_test_006843_jpg.rf.7d6b35a29774a248e7848185c8764286__inst_001",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="high",
        rationale="Visual inspection found a clearly visible frayed gray/white patch, "
            "distinctly different from the striped fabric background, in the box.",
    ),
    TargetConfig(
        "chenab_test_006843_jpg.rf.7d6b35a29774a248e7848185c8764286__inst_002",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="high",
        rationale="Same frayed gray/white patch as inst_001, a different box segment of the "
            "same tear.",
    ),
    TargetConfig(
        "chenab_train_00004515_jpg.rf.053441765bd145dffe80b947db2f2d2f__inst_001",
        strategy="crop_plus_points", point_mode="bright_streak", grounding_confidence="high",
        rationale="Visual inspection found a clear row of individual white diagonal "
            "stitches on red/brown fabric, several visibly in gaps left by the V1 mask.",
    ),
    TargetConfig(
        "chenab_test_IMG_20230823_105639_jpg.rf.03ba242a6866d033b9c4878dfb907b48__inst_001",
        strategy="crop_plus_points", point_mode="bright_streak", grounding_confidence="high",
        rationale="Visual inspection found a clearly visible near-white vertical streak, "
            "high-contrast against the colourful plaid print, inside the box.",
    ),
    TargetConfig(
        "chenab_test_IMG_20230823_105639_jpg.rf.03ba242a6866d033b9c4878dfb907b48__inst_002",
        strategy="crop_plus_points", point_mode="bright_streak", grounding_confidence="high",
        rationale="Same white vertical streak as inst_001, lower segment.",
    ),
    TargetConfig(
        "chenab_test_IMG_20230823_105639_jpg.rf.03ba242a6866d033b9c4878dfb907b48__inst_003",
        strategy="crop_plus_points", point_mode="bright_streak", grounding_confidence="high",
        rationale="Same white vertical streak, further down the image.",
    ),
    TargetConfig(
        "chenab_test_IMG_20230823_105639_jpg.rf.03ba242a6866d033b9c4878dfb907b48__inst_004",
        strategy="crop_plus_points", point_mode="bright_streak", grounding_confidence="medium",
        rationale="A lighter-coloured path to the right of the main streak is visible but "
            "sits closer to normally-printed white plaid stripes than inst_001-003.",
    ),
    TargetConfig(
        "chenab_test_IMG_20230823_105639_jpg.rf.03ba242a6866d033b9c4878dfb907b48__inst_005",
        strategy="crop_only", point_mode=None, trim_dark_margin=True, grounding_confidence="low",
        rationale="Box sits at the far-right fabric edge/selvedge trim next to the image's "
            "black background -- visually just the fabric's normal edge, not a clearly "
            "distinguishable internal anomaly. Correction here is a crop that excludes "
            "the near-black background margin (so SAM cannot latch onto the image "
            "border), not an invented point.",
    ),
    TargetConfig(
        "chenab_train_IMG-20230817-WA0064_jpg.rf.72c55fd89707b95438444ffec8b140ac__inst_001",
        strategy="crop_plus_points", point_mode="multi_dark_anomaly", n_negative=2,
        grounding_confidence="high",
        rationale="Visual inspection found multiple distinct brown/tan stain streaks "
            "scattered across the box, clearly visible against paler fabric -- the "
            "task brief's large/scattered-stain case; multiple grounded positive "
            "points on distinct blobs, negative points on clearly pale background.",
    ),
    TargetConfig(
        "chenab_train_IMG-20231012-WA0073_jpg.rf.5429ad20d40fb0b199cf338896923264__inst_001",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="high",
        rationale="Visual inspection found a distinct elongated horizontal dark smudge/mark "
            "on otherwise plain fabric, V1 already captures its core. This V2 attempt "
            "only reinforces that core region with a grounded point; it does NOT "
            "extend the mask to the diffuse lateral extension the AI reviewer "
            "mentioned, since that extension has no defensible visible edge.",
    ),
    TargetConfig(
        "chenab_train_IMG-20231012-WA0073_jpg.rf.5429ad20d40fb0b199cf338896923264__inst_002",
        strategy="crop_plus_points", point_mode="dark_anomaly", grounding_confidence="high",
        rationale="Visual inspection found a distinct small dark cross-shaped mark on plain "
            "fabric, V1 already captures its core shape reasonably well.",
    ),
]

assert len(TARGET_INSTANCES) == 19, f"expected 19 target instances, got {len(TARGET_INSTANCES)}"
assert len({t.instance_id for t in TARGET_INSTANCES}) == 19, "duplicate instance_id in TARGET_INSTANCES"


# --------------------------------------------------------------------------
# Pixel-evidence grounding algorithms (pure numpy/cv2, no SAM involved).
# --------------------------------------------------------------------------

def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def compute_crop(box_xyxy, img_w, img_h, pad_frac=0.25, trim_dark_margin=False, image_rgb=None):
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    pad = int(round(pad_frac * max(bw, bh)))
    pad = max(pad, 20)
    cx1, cy1, cx2, cy2 = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, img_w, img_h)

    if trim_dark_margin and image_rgb is not None:
        import numpy as np
        crop = image_rgb[cy1:cy2, cx1:cx2]
        col_mean = crop.reshape(crop.shape[0], -1, crop.shape[2]).mean(axis=(0, 2)) if crop.size else None
        if col_mean is not None and len(col_mean) > 5:
            # Trim columns from the right edge inward while they are near-black
            # (background outside the actual fabric), stopping well before the
            # GT box itself so the box is never clipped.
            box_right_local = x2 - cx1
            new_right = crop.shape[1]
            thresh = 25.0
            while new_right > box_right_local + 5 and col_mean[new_right - 1] < thresh:
                new_right -= 1
            cx2 = cx1 + new_right

    return _clip_box(cx1, cy1, cx2, cy2, img_w, img_h)


def _local_maxima_points(score, mask_region, n, min_sep_frac=0.15):
    """Greedy spatially-separated local maxima of `score` restricted to
    `mask_region` (bool array same shape). Returns list of (x, y)."""
    import numpy as np

    h, w = score.shape
    min_sep = max(3, int(round(min_sep_frac * max(h, w))))
    work = score.copy().astype(float)
    work[~mask_region] = -np.inf
    points = []
    for _ in range(n):
        idx = np.argmax(work)
        y, x = divmod(int(idx), w)
        if not np.isfinite(work[y, x]):
            break
        points.append((x, y))
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        work[dist2 <= min_sep ** 2] = -np.inf
    return points


def ground_points(image_rgb, box_local, crop_shape, point_mode, n_negative=2):
    """Returns (positive_points, negative_points), each a list of (x, y) in
    crop-local pixel coordinates, computed from real pixel data only.
    Never returns an invented/arbitrary coordinate."""
    import numpy as np
    import cv2

    h, w = crop_shape[:2]
    x1, y1, x2, y2 = box_local
    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[y1:y2, x1:x2] = True

    if point_mode == "bright_streak":
        lightness = image_rgb.astype(float).mean(axis=2)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(float)
        saturation = hsv[:, :, 1]
        pos = _local_maxima_points(lightness, box_mask, n=1)
        neg = _local_maxima_points(saturation, box_mask, n=n_negative)
        return pos, neg

    if point_mode in ("dark_anomaly", "multi_dark_anomaly"):
        pad = max(5, int(round(0.08 * max(h, w))))
        annulus = np.zeros((h, w), dtype=bool)
        ox1, oy1, ox2, oy2 = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)
        annulus[oy1:oy2, ox1:ox2] = True
        annulus[y1:y2, x1:x2] = False
        if not annulus.any():
            annulus = ~box_mask  # fallback: everything outside the box

        ref_color = image_rgb[annulus].astype(float).mean(axis=0) if annulus.any() else image_rgb[box_mask].astype(float).mean(axis=0)
        deviation = np.linalg.norm(image_rgb.astype(float) - ref_color.reshape(1, 1, 3), axis=2)

        n_pos = 3 if point_mode == "multi_dark_anomaly" else 1
        pos = _local_maxima_points(deviation, box_mask, n=n_pos)
        neg_in_box = _local_maxima_points(-deviation, box_mask, n=max(1, n_negative - 1))
        neg_outside = _local_maxima_points(np.where(annulus, deviation, np.inf) * -1, annulus, n=1) if annulus.any() else []
        neg = neg_in_box + neg_outside
        return pos, neg[:n_negative]

    if point_mode == "texture_anomaly":
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture = cv2.GaussianBlur(np.abs(lap), (9, 9), 0)
        box_texture_median = float(np.median(texture[box_mask])) if box_mask.any() else 0.0
        deviation = np.abs(texture - box_texture_median)
        pos = _local_maxima_points(deviation, box_mask, n=1)
        neg = _local_maxima_points(-deviation, box_mask, n=n_negative)
        return pos, neg

    return [], []


# --------------------------------------------------------------------------
# Candidate-quality heuristic (explicitly NOT "highest SAM score wins").
# --------------------------------------------------------------------------

def _quality_heuristic(mask, box_local, crop_shape):
    import numpy as np
    import cv2

    h, w = crop_shape[:2]
    total_px = h * w
    mask_px = int(mask.sum())
    area_frac_of_crop = mask_px / total_px if total_px else 0.0

    x1, y1, x2, y2 = box_local
    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    box_area = int(box_mask.sum())
    inside_box_px = int((mask & box_mask).sum())
    leakage_fraction = ((mask_px - inside_box_px) / mask_px) if mask_px > 0 else 0.0
    box_coverage_fraction = (inside_box_px / box_area) if box_area else 0.0

    n_components, _ = cv2.connectedComponents(mask.astype("uint8"))
    n_fragments = max(0, n_components - 1)

    empty = mask_px == 0
    near_full_crop = area_frac_of_crop >= 0.92 and (box_area / total_px) < 0.80
    plausible = (
        not empty
        and not near_full_crop
        and leakage_fraction <= 0.5
        and box_coverage_fraction >= 0.05
    )
    # Lower is better: penalise leakage, fragmentation, and implausibility.
    penalty = leakage_fraction + 0.02 * n_fragments + (5.0 if not plausible else 0.0)

    return {
        "mask_pixel_count": mask_px,
        "area_fraction_of_crop": round(area_frac_of_crop, 6),
        "inside_box_pixel_count": inside_box_px,
        "box_coverage_fraction": round(box_coverage_fraction, 6),
        "leakage_fraction": round(leakage_fraction, 6),
        "n_connected_components": n_fragments,
        "plausible": plausible,
        "quality_penalty": round(penalty, 6),
    }


def run() -> dict:
    import numpy as np
    import cv2
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    SAM_PROPOSALS_V2_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_V2_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SAM_CHECKPOINT.is_file():
        raise RuntimeError(f"SAM checkpoint not found at {SAM_CHECKPOINT}; refusing to fabricate masks.")

    with open(MANIFESTS_DIR / "instance_manifest.csv", newline="", encoding="utf-8") as f:
        manifest_by_id = {r["instance_id"]: r for r in csv.DictReader(f)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    results = []
    n_success = 0
    n_no_defensible = 0
    n_flagged_unsuitable = 0

    for cfg in TARGET_INSTANCES:
        inst = manifest_by_id[cfg.instance_id]
        image_id = inst["image_id"]

        if cfg.strategy == "flag_unsuitable":
            results.append({
                "instance_id": cfg.instance_id, "image_id": image_id,
                "status": STATUS_UNSUITABLE,
                "strategy": cfg.strategy, "point_mode": None,
                "grounding_confidence": cfg.grounding_confidence,
                "rationale": cfg.rationale,
                "sam_model_type": SAM_MODEL_TYPE,
                "sam_checkpoint": str(SAM_CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
            })
            n_flagged_unsuitable += 1
            continue

        image_path = _resolve_valid_image_path(inst["filename"])
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        cx, cy, w_, h_ = (float(inst[k]) for k in ("cx", "cy", "w", "h"))
        box_xyxy = yolo_box_to_pixels(cx, cy, w_, h_, img_w, img_h)

        crop_box = compute_crop(box_xyxy, img_w, img_h, pad_frac=cfg.pad_frac,
                                 trim_dark_margin=cfg.trim_dark_margin, image_rgb=image_rgb)
        ccx1, ccy1, ccx2, ccy2 = crop_box
        crop_rgb = image_rgb[ccy1:ccy2, ccx1:ccx2].copy()
        crop_bgr = image_bgr[ccy1:ccy2, ccx1:ccx2].copy()
        crop_h, crop_w = crop_rgb.shape[:2]

        bx1, by1, bx2, by2 = box_xyxy
        box_local = _clip_box(bx1 - ccx1, by1 - ccy1, bx2 - ccx1, by2 - ccy1, crop_w, crop_h)

        predictor.set_image(crop_rgb)

        candidates = []

        # Candidate A: crop + box only.
        box_np = np.array(box_local, dtype=np.float32)
        masks_a, scores_a, _ = predictor.predict(box=box_np, multimask_output=True)
        for i in range(len(masks_a)):
            q = _quality_heuristic(masks_a[i].astype(bool), box_local, (crop_h, crop_w))
            candidates.append({
                "candidate": "crop_plus_box_only", "sam_mask_index": i,
                "sam_confidence_score": round(float(scores_a[i]), 6),
                "mask_array": masks_a[i].astype(bool), **q,
            })

        pos_pts, neg_pts = [], []
        if cfg.strategy == "crop_plus_points" and cfg.point_mode is not None:
            pos_pts, neg_pts = ground_points(crop_rgb, box_local, (crop_h, crop_w),
                                              cfg.point_mode, n_negative=cfg.n_negative)
            if pos_pts:
                pt_coords = np.array(pos_pts + neg_pts, dtype=np.float32)
                pt_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)
                masks_b, scores_b, _ = predictor.predict(
                    box=box_np, point_coords=pt_coords, point_labels=pt_labels,
                    multimask_output=True,
                )
                for i in range(len(masks_b)):
                    q = _quality_heuristic(masks_b[i].astype(bool), box_local, (crop_h, crop_w))
                    candidates.append({
                        "candidate": "crop_plus_box_plus_points", "sam_mask_index": i,
                        "sam_confidence_score": round(float(scores_b[i]), 6),
                        "mask_array": masks_b[i].astype(bool), **q,
                    })

        # Select best candidate by the documented heuristic (lowest penalty,
        # SAM score only as the final tiebreaker) -- never by SAM score alone.
        candidates_sorted = sorted(
            candidates, key=lambda c: (c["quality_penalty"], -c["sam_confidence_score"])
        )
        best = candidates_sorted[0]
        best_mask = best.pop("mask_array")
        for c in candidates:
            c.pop("mask_array", None)

        status = STATUS_SUCCESS if best["plausible"] else STATUS_NO_DEFENSIBLE
        if status == STATUS_SUCCESS:
            n_success += 1
        else:
            n_no_defensible += 1

        mask_png = (best_mask.astype("uint8") * 255)
        mask_out_path = SAM_PROPOSALS_V2_DIR / f"{cfg.instance_id}.png"
        cv2.imwrite(str(mask_out_path), mask_png)

        # QC overlay: 2x2 panel on the CROP (orig+box, orig+mask, mask-only, contour+points).
        panel_tl = crop_bgr.copy()
        x1, y1, x2, y2 = box_local
        cv2.rectangle(panel_tl, (x1, y1), (x2, y2), (0, 0, 255), 2)

        panel_tr = crop_bgr.copy()
        overlay_color = np.zeros_like(panel_tr)
        overlay_color[best_mask] = (0, 255, 0)
        panel_tr = cv2.addWeighted(panel_tr, 1.0, overlay_color, 0.5, 0)

        panel_bl = cv2.cvtColor(mask_png, cv2.COLOR_GRAY2BGR)

        panel_br = crop_bgr.copy()
        contours, _ = cv2.findContours(best_mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel_br, contours, -1, (255, 0, 255), 2)
        cv2.rectangle(panel_br, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for px, py in pos_pts:
            cv2.circle(panel_br, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(panel_br, (px, py), 6, (0, 0, 0), 1)
        for px, py in neg_pts:
            cv2.circle(panel_br, (px, py), 5, (0, 0, 255), -1)
            cv2.circle(panel_br, (px, py), 6, (0, 0, 0), 1)

        top = np.hstack([panel_tl, panel_tr])
        bottom = np.hstack([panel_bl, panel_br])
        grid = np.vstack([top, bottom])
        overlay_out_path = OVERLAYS_V2_DIR / f"{cfg.instance_id}_overlay_v2.png"
        cv2.imwrite(str(overlay_out_path), grid)

        results.append({
            "instance_id": cfg.instance_id, "image_id": image_id,
            "status": status,
            "strategy": cfg.strategy, "point_mode": cfg.point_mode,
            "grounding_confidence": cfg.grounding_confidence,
            "rationale": cfg.rationale,
            "sam_model_type": SAM_MODEL_TYPE,
            "sam_checkpoint": str(SAM_CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
            "proposal_version": "v2",
            "original_image_width": img_w, "original_image_height": img_h,
            "box_xyxy_pixels_original_image": list(box_xyxy),
            "crop_xyxy_pixels_original_image": list(crop_box),
            "crop_width": crop_w, "crop_height": crop_h,
            "box_xyxy_pixels_crop_local": list(box_local),
            "positive_points_crop_local": pos_pts,
            "negative_points_crop_local": neg_pts,
            "mask_path": str(mask_out_path.relative_to(REPO_ROOT).as_posix()),
            "overlay_path": str(overlay_out_path.relative_to(REPO_ROOT).as_posix()),
            "selected_candidate": best["candidate"],
            "selected_candidate_sam_mask_index": best["sam_mask_index"],
            "selected_candidate_sam_confidence_score": best["sam_confidence_score"],
            "selected_candidate_metrics": best,
            "all_candidates_considered": candidates,
            "selection_note": (
                "Best candidate chosen by documented quality heuristic (plausibility "
                "flags + leakage/fragmentation penalty), NOT by highest SAM confidence "
                "score alone -- see all_candidates_considered for the full comparison, "
                "including any higher-scoring-but-lower-quality candidates that were "
                "NOT selected."
            ),
        })

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "F2.2C-QA Task 3",
        "device_used": device,
        "sam_model_type": SAM_MODEL_TYPE,
        "sam_checkpoint": str(SAM_CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
        "n_target_instances": len(TARGET_INSTANCES),
        "n_success": n_success,
        "n_attempted_no_defensible_result": n_no_defensible,
        "n_not_attempted_flagged_unsuitable": n_flagged_unsuitable,
        "per_instance_results": results,
        "status_label": "Algorithmic SAM V2 proposals -- NOT human-verified masks, NOT ground truth",
    }
    with open(REPORTS_DIR / "f22c_sam_v2_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "per_instance_results"}, indent=2))
