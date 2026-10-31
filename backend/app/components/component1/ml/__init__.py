"""Bridge to the research DetectionXAI module.

Imports from research/component1/xai_experiments/src/ so there is one
source of truth. The research venv must have ultralytics, torch, opencv
and numpy installed; this backend venv shares those same packages.
"""
import os
import sys

_RESEARCH_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",  # backend/app/components/component1/ml → repo root
    "research", "component1", "xai_experiments", "src",
))

if _RESEARCH_SRC not in sys.path:
    sys.path.insert(0, _RESEARCH_SRC)

from detection_xai import (  # noqa: E402, F401
    DetectionXAI,
    Detection,
    ExplanationResult,
    InferenceResult,
    CLASS_NAMES,
    VERIFY_SCORE_ATOL,
    TARGET_LAYER_INDICES,
    STRIDES,
    GRID_SIZES,
)
