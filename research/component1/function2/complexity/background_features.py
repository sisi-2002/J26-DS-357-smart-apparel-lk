"""F2.1B1 Task 2 — consistent 3-feature background-only extraction adapter.

Computes all three background-complexity descriptors --
luminance variation, (mask-aware) GLCM entropy, and Lab colour variation --
over the *exact same* valid background region, for a given image + GT boxes.

This module is new code under ``research/component1/function2/``. It:
  - reuses ``research/component1/function2/complexity/background_region.py``
    (F2.1A, unchanged) to turn GT YOLO boxes into a boolean background mask,
  - reuses the shared, frozen
    ``research/component1/complexity_calibration/src/complexity_features.py``
    module's ``compute_complexity_from_array`` for luminance_variation and
    colour_variation (these two ARE already correctly mask-restricted by
    that module -- see task3 report section 4), and
  - uses the NEW ``masked_glcm.compute_masked_glcm_entropy`` (Task 1 of this
    stage) for glcm_entropy, since the shared module's GLCM computation does
    not respect the mask at all.

Neither shared module is modified. No model predictions are used anywhere
in this file -- GT boxes only (same contract as background_region.py).

OpenCV Lab colour-space caveat (same as documented for the shared module in
``reports/task3_complexity_feature_verification.md``): ``cv2.cvtColor(...,
cv2.COLOR_BGR2LAB)`` returns **8-bit-scaled** OpenCV Lab, not canonical CIE
L*a*b* units. L is linearly rescaled from the canonical [0, 100] range to
[0, 255]; a*/b* are rescaled from the canonical ~[-127, 127] range to
[0, 255] centred at 128. All "luminance_variation" / "colour_variation"
values produced by this module (via the shared module) are therefore in
this native OpenCV 8-bit-mapped space, not literal CIE L*a*b* units. Any
cross-tool comparison (e.g. against skimage's ``rgb2lab``, which returns
canonical-unit floats) must account for this ~2.55x scale difference.

Neutrality
----------
This adapter performs no tuning of normalisation, weighting, or feature
combination with any downstream (XAI) outcome in mind. It is a faithful
measurement tool only; complexity grouping (Simple/Moderate/High) is
explicitly out of scope here (see reports/task7_panel_defense_masked_glcm.md).

Resize-boundary-contamination finding (investigated in this module's
docstring + reports/task2... see repo report task5 comparison report)
--------------------------------------------------------------------------
``compute_complexity_from_array`` (and this adapter, which mirrors its
resize behaviour for parity) resizes the RAW BGR image with
``cv2.INTER_AREA`` *before* any mask is applied. INTER_AREA downscaling is
an area/box average over a block of source pixels per destination pixel.
For a destination pixel that lands near a GT-box boundary, that source
block can straddle the boundary -- i.e. it can combine some genuine
background source pixels with some excluded defect-region source pixels
into a single resized output pixel's value. If the background mask is then
re-rasterised (nearest-neighbour) at the resized resolution and applied
AFTER this blending has already happened, a handful of pixels right at the
mask boundary that the mask calls "background" can, in fact, carry blended-
in defect-region information. This is a genuine, real contamination
pathway, not a theoretical one -- it is a direct consequence of computing
GLCM/variance statistics on a resized raster using an area-average filter.

Mitigation implemented here: after building the mask (at original
resolution) and nearest-neighbour-resizing it to the post-resize image
resolution (replicating the shared module's own mask-resize convention),
we **erode the valid (background=True) region by ``boundary_erosion_px``
pixels** (default 1) at the resized resolution, using a 3x3 structuring
element, whenever at least one GT box was excluded (i.e. there is an actual
exclusion boundary to protect against -- for images with zero excluded
boxes there is no such boundary and erosion is skipped to avoid
unnecessarily shrinking genuine background with no contamination risk).
This is a reasonable, documented, best-effort mitigation, not a proof of
zero contamination: for extreme downscale factors (very large originals)
the INTER_AREA box-filter support in *input*-pixel terms is wide, but it is
still exactly one output-pixel wide in *output*-pixel terms (each output
pixel samples one corresponding input block under an aspect-preserving
uniform grid resize), so a 1px erosion at the output resolution is the
geometrically-correct size for this mitigation under INTER_AREA specifically.
A different interpolation method (e.g. INTER_LINEAR/CUBIC, which has wider,
non-box support) could spread contamination over a wider radius; the
shared module's default is INTER_AREA, so this is documented as the
assumption this mitigation depends on, not silently generalised.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import cv2
import numpy as np

from research.component1.complexity_calibration.src.complexity_features import (
    DEFAULT_CONFIG,
    ProtocolConfig,
    compute_complexity_from_array,
)
from research.component1.function2.complexity.background_region import (
    DEFAULT_MAX_DEFECT_COVERAGE,
    YoloBox,
    compute_background_mask,
)
from research.component1.function2.complexity.masked_glcm import (
    CODE_VERSION as MASKED_GLCM_CODE_VERSION,
    DEFAULT_MIN_VALID_PAIRS,
    compute_masked_glcm_entropy,
)

DEFAULT_BOUNDARY_EROSION_PX = 1

_INTERP_MAP = {
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_NEAREST": cv2.INTER_NEAREST,
}


@dataclasses.dataclass
class BackgroundFeaturesResult:
    luminance_variation: Optional[float] = None
    glcm_entropy: Optional[float] = None
    colour_variation: Optional[float] = None

    status: str = "ok"
    quality_flags: list = dataclasses.field(default_factory=list)

    background_status: str = ""              # from background_region (original resolution)
    background_coverage_fraction: float = 0.0  # from background_region (original resolution)
    num_boxes_excluded: int = 0

    image_height: int = 0
    image_width: int = 0
    resized_height: int = 0
    resized_width: int = 0
    boundary_erosion_px: int = 0
    valid_pixel_fraction_before_erosion: Optional[float] = None
    valid_pixel_fraction_after_erosion: Optional[float] = None

    glcm_valid_pair_count: int = 0
    glcm_status: str = ""

    protocol: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _resize_preserve_aspect(img: np.ndarray, long_edge: int, interp: int) -> np.ndarray:
    """Replicates complexity_features._resize_preserve_aspect exactly
    (same formula: aspect-preserving resize only if max(h, w) > long_edge).
    Duplicated here (3 lines) rather than importing the shared module's
    underscore-prefixed private helper, to avoid depending on another
    module's non-public API surface; the shared module is not modified.
    """
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    scale = long_edge / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def compute_background_features(
    img_bgr: np.ndarray,
    yolo_boxes: Optional[Sequence[YoloBox]],
    is_negative: Optional[bool] = None,
    config: ProtocolConfig = DEFAULT_CONFIG,
    max_defect_coverage: float = DEFAULT_MAX_DEFECT_COVERAGE,
    min_valid_pairs: int = DEFAULT_MIN_VALID_PAIRS,
    boundary_erosion_px: int = DEFAULT_BOUNDARY_EROSION_PX,
) -> BackgroundFeaturesResult:
    """Compute all three background-complexity features over one consistent
    background mask.

    Parameters
    ----------
    img_bgr : in-memory BGR image array (OpenCV channel order), full
        original resolution, NOT pre-resized by the caller.
    yolo_boxes, is_negative, max_defect_coverage : passed straight through
        to ``background_region.compute_background_mask`` -- see that
        module's docstring for the unresolved/clean_negative/boxes_excluded
        state machine. GT boxes only, never model predictions.
    config : the shared module's protocol config (resize_long_edge,
        interpolation, GLCM parameters). Defaults to the shared
        ``DEFAULT_CONFIG`` -- do not override unless you have a specific,
        separately-justified reason (keeps this adapter's numbers
        comparable to the rest of the F2.1A/F2.1B1 pipeline).
    min_valid_pairs, boundary_erosion_px : see masked_glcm.py and this
        module's docstring (resize-boundary-contamination mitigation).

    Returns
    -------
    BackgroundFeaturesResult
    """
    quality_flags: list[str] = []

    if img_bgr is None or img_bgr.size == 0:
        return BackgroundFeaturesResult(
            status="error_empty_array", quality_flags=["empty_array"],
            protocol={"masked_glcm_code_version": MASKED_GLCM_CODE_VERSION},
        )

    h, w = img_bgr.shape[:2]
    if h < 8 or w < 8:
        return BackgroundFeaturesResult(
            status="error_too_small", quality_flags=["image_too_small"],
            image_height=h, image_width=w,
            protocol={"masked_glcm_code_version": MASKED_GLCM_CODE_VERSION},
        )

    # --- Step 1: background mask at ORIGINAL resolution (GT boxes are
    # normalised, resolution-independent) ---
    bg_result = compute_background_mask(
        image_height=h, image_width=w,
        yolo_boxes=yolo_boxes, is_negative=is_negative,
        max_defect_coverage=max_defect_coverage,
    )
    quality_flags.extend(f"bg:{wmsg}" for wmsg in bg_result.warnings)

    if bg_result.mask is None:
        # unresolved_missing_annotation or error_invalid_input: cannot
        # compute anything meaningful. Never silently assume clean.
        return BackgroundFeaturesResult(
            status=f"error_{bg_result.status}",
            quality_flags=quality_flags + ["background_mask_unavailable"],
            background_status=bg_result.status,
            background_coverage_fraction=bg_result.background_coverage_fraction,
            num_boxes_excluded=bg_result.num_boxes_excluded,
            image_height=h, image_width=w,
            protocol={"masked_glcm_code_version": MASKED_GLCM_CODE_VERSION},
        )

    # --- Step 2: resize image exactly as the shared module would ---
    interp = _INTERP_MAP.get(config.interpolation, cv2.INTER_AREA)
    img_resized = _resize_preserve_aspect(img_bgr, config.resize_long_edge, interp)
    rh, rw = img_resized.shape[:2]

    # --- Step 3: resize mask to match (nearest-neighbour -- same
    # convention the shared module itself uses for mask resizing; see
    # compute_complexity_from_array) ---
    mask_original = bg_result.mask
    if mask_original.shape[:2] != (rh, rw):
        mask_resized = cv2.resize(
            mask_original.astype(np.uint8), (rw, rh),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_resized = mask_original

    valid_frac_before = float(mask_resized.sum()) / float(mask_resized.size)

    # --- Step 4: resize-boundary-contamination mitigation (erosion) ---
    applied_erosion = 0
    mask_eroded = mask_resized
    if bg_result.num_boxes_excluded > 0 and boundary_erosion_px > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = mask_resized.astype(np.uint8)
        eroded_u8 = cv2.erode(mask_u8, kernel, iterations=boundary_erosion_px)
        mask_eroded = eroded_u8.astype(bool)
        applied_erosion = boundary_erosion_px
        quality_flags.append(
            f"boundary_erosion_applied_{applied_erosion}px: mitigates INTER_AREA "
            "resize blending across the GT-box exclusion boundary (see module docstring)"
        )

    valid_frac_after = float(mask_eroded.sum()) / float(mask_eroded.size)
    if applied_erosion and valid_frac_after < valid_frac_before:
        quality_flags.append(
            f"erosion_reduced_valid_fraction: {valid_frac_before:.4f} -> {valid_frac_after:.4f}"
        )

    # --- Step 5: luminance_variation / colour_variation via the shared,
    # already-correctly-masked module (pass the pre-resized image + the
    # already-matching-resolution eroded mask, so the shared function's own
    # internal resize/mask-resize steps are both no-ops) ---
    feat = compute_complexity_from_array(img_resized, config=config, mask=mask_eroded)
    quality_flags.extend(f"feat:{wmsg}" for wmsg in feat.quality_flags)

    # --- Step 6: mask-aware GLCM entropy (Task 1) on the SAME eroded mask,
    # same resized image, same gray conversion the shared module itself uses ---
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    glcm_result = compute_masked_glcm_entropy(
        gray_resized, mask_eroded, min_valid_pairs=min_valid_pairs,
    )
    quality_flags.extend(f"glcm:{wmsg}" for wmsg in glcm_result.quality_flags)

    overall_status = "ok"
    if feat.status != "ok":
        overall_status = f"feat_{feat.status}"
    if glcm_result.status not in ("valid", "insufficient_pairs"):
        # all_excluded / error_* on the GLCM side invalidates the combined
        # result's glcm_entropy specifically, but luminance/colour may
        # still be valid -- status records this rather than hiding it.
        overall_status = (
            overall_status + f"+glcm_{glcm_result.status}"
            if overall_status != "ok" else f"glcm_{glcm_result.status}"
        )
    elif glcm_result.status == "insufficient_pairs" and overall_status == "ok":
        overall_status = "ok_glcm_insufficient_pairs"

    protocol = {
        "masked_glcm_code_version": MASKED_GLCM_CODE_VERSION,
        "masked_glcm_protocol": glcm_result.protocol,
        "complexity_features_protocol_version": config.protocol_version,
        "boundary_erosion_px_requested": boundary_erosion_px,
        "boundary_erosion_px_applied": applied_erosion,
        "resize_interpolation": config.interpolation,
    }

    return BackgroundFeaturesResult(
        luminance_variation=feat.luminance_variation,
        glcm_entropy=glcm_result.entropy,
        colour_variation=feat.colour_variation,
        status=overall_status,
        quality_flags=quality_flags,
        background_status=bg_result.status,
        background_coverage_fraction=bg_result.background_coverage_fraction,
        num_boxes_excluded=bg_result.num_boxes_excluded,
        image_height=h, image_width=w,
        resized_height=rh, resized_width=rw,
        boundary_erosion_px=applied_erosion,
        valid_pixel_fraction_before_erosion=valid_frac_before,
        valid_pixel_fraction_after_erosion=valid_frac_after,
        glcm_valid_pair_count=glcm_result.valid_pair_count,
        glcm_status=glcm_result.status,
        protocol=protocol,
    )
