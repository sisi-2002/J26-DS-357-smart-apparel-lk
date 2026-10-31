"""F2.1B2 Task 5 -- transparent composite complexity score (CANDIDATE ONLY,
NOT frozen, NOT expert-validated).

This module implements:
  - z-score normalisation of each of the 3 raw background-complexity
    features, using training-side (eligible-sample) statistics ONLY.
  - a simple, interpretable weighted-sum combination into one composite
    score (equal 1/3 weighting is the documented default candidate).
  - deterministic, pure functions: the same (raw_features, config) pair
    always produces the same output, with no hidden state, no RNG, and no
    dependency on anything other than the explicit arguments.

Nothing in this module tunes weights/thresholds against any Grad-CAM/XAI
outcome -- it only consumes the Task 2 training-side feature CSV and the
Task 1 eligibility decision already computed for each row.

Status discipline: every object this module produces that could be
confused with a finished calibration carries an explicit
``status="provisional"`` field (see ``CompositeScoreResult`` and
``build_calibration_config``). Nothing in this module ever sets
``status="validated"`` -- that word is reserved for a future, explicitly
expert-reviewed stage and is tested for in
``tests/test_calibration_scoring.py`` (provisional vs validated must never
be conflated).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

FEATURE_NAMES = ("luminance_variation", "glcm_entropy", "colour_variation")

DEFAULT_WEIGHTS = {
    "luminance_variation": 1.0 / 3.0,
    "glcm_entropy": 1.0 / 3.0,
    "colour_variation": 1.0 / 3.0,
}


@dataclasses.dataclass
class NormalizationParams:
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationParams":
        return cls(mean=float(d["mean"]), std=float(d["std"]))


def compute_normalization_params(values: list[float]) -> NormalizationParams:
    """Population mean/std (ddof=0, matching the shared module's own
    np.std convention) over a list of finite raw feature values from the
    ELIGIBLE training-side sample only. Raises ValueError on a degenerate
    (std==0 or empty) input rather than silently returning a param set
    that would divide by zero later -- an all-identical or empty training
    distribution is a data problem to surface, not paper over."""
    finite_values = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite_values) < 2:
        raise ValueError(
            f"compute_normalization_params requires >=2 finite values, got {len(finite_values)}"
        )
    n = len(finite_values)
    mean = sum(finite_values) / n
    var = sum((v - mean) ** 2 for v in finite_values) / n  # ddof=0
    std = math.sqrt(var)
    if std == 0.0:
        raise ValueError(
            "compute_normalization_params: zero standard deviation in training "
            "distribution -- z-score normalisation is undefined. This would "
            "indicate every eligible training sample has an identical feature "
            "value, which should be investigated, not silently divided through."
        )
    return NormalizationParams(mean=mean, std=std)


def zscore(value: Optional[float], params: NormalizationParams) -> Optional[float]:
    """Normalise one raw value. Returns None (never 0.0) if the input is
    missing/non-finite -- an invalid raw value must never be silently
    treated as 'average' (z=0)."""
    if value is None or not math.isfinite(value):
        return None
    return (value - params.mean) / params.std


@dataclasses.dataclass
class CompositeScoreResult:
    composite_score: Optional[float]
    z_scores: dict[str, Optional[float]]
    raw_features: dict[str, Optional[float]]
    weights: dict[str, float]
    status: str  # "provisional" always, for this module
    missing_features: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def compute_composite_score(
    raw_features: dict[str, Optional[float]],
    normalization: dict[str, NormalizationParams],
    weights: Optional[dict[str, float]] = None,
) -> CompositeScoreResult:
    """Deterministic composite score for one sample.

    Parameters
    ----------
    raw_features : dict with keys exactly FEATURE_NAMES, raw (un-normalised)
        feature values for one sample (None if a feature is invalid for
        this sample -- callers should normally only call this for samples
        the Task 1 eligibility policy already marked eligible, in which
        case all three will be present, but this function does not assume
        that and degrades explicitly instead of crashing).
    normalization : dict FEATURE_NAME -> NormalizationParams, computed ONCE
        from the eligible training distribution via
        ``compute_normalization_params`` and then reused unchanged for
        every sample (never recomputed per-sample -- that would make the
        score depend on which other samples happen to be scored alongside
        it, breaking determinism/comparability).
    weights : FEATURE_NAME -> weight, must sum to 1.0 (checked). Defaults
        to equal 1/3 weighting (``DEFAULT_WEIGHTS``), documented as a
        candidate default, not a validated optimum.

    Returns
    -------
    CompositeScoreResult. ``composite_score`` is None (never silently 0 or
    an average of the others) if ANY of the 3 features is missing/invalid
    for this sample -- a composite score built from only 2 of 3 features
    would silently misrepresent what it measures.
    """
    w = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
    missing = [name for name in FEATURE_NAMES if name not in w]
    if missing:
        raise ValueError(f"weights missing required feature keys: {missing}")
    weight_sum = sum(w.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

    z_scores: dict[str, Optional[float]] = {}
    missing_features: list[str] = []
    for name in FEATURE_NAMES:
        raw_val = raw_features.get(name)
        z = zscore(raw_val, normalization[name])
        z_scores[name] = z
        if z is None:
            missing_features.append(name)

    if missing_features:
        composite = None
    else:
        composite = sum(w[name] * z_scores[name] for name in FEATURE_NAMES)

    return CompositeScoreResult(
        composite_score=composite,
        z_scores=z_scores,
        raw_features={name: raw_features.get(name) for name in FEATURE_NAMES},
        weights=w,
        status="provisional",
        missing_features=missing_features,
    )


def assign_group(
    composite_score: Optional[float],
    simple_max: float,
    moderate_max: float,
    boundary_margin: float = 0.0,
) -> tuple[Optional[str], bool]:
    """Assign a Simple/Moderate/High group label from a composite score and
    two candidate cut points (``simple_max`` = upper edge of Simple,
    ``moderate_max`` = upper edge of Moderate). Returns (group, is_boundary_adjacent).

    ``group`` is None if ``composite_score`` is None (ineligible/ambiguous
    samples are never forced into a group -- they stay unlabeled).

    ``boundary_margin`` (same units as the composite score) controls the
    threshold-adjacent flag: a sample within ``boundary_margin`` of either
    cut point is flagged, regardless of which side of the line it falls on.
    A margin of 0.0 disables the flag (every sample reports False).
    """
    if composite_score is None:
        return None, False

    if composite_score <= simple_max:
        group = "Simple"
    elif composite_score <= moderate_max:
        group = "Moderate"
    else:
        group = "High"

    is_boundary_adjacent = boundary_margin > 0.0 and (
        abs(composite_score - simple_max) <= boundary_margin
        or abs(composite_score - moderate_max) <= boundary_margin
    )
    return group, is_boundary_adjacent


def tertile_thresholds(composite_scores: list[float]) -> tuple[float, float]:
    """Candidate Simple/Moderate/High cut points: the 33.33rd and 66.67th
    percentiles of the (eligible, finite) composite-score distribution
    ("tertiles"). This is explicitly ONE candidate scheme among several
    plausible ones (quantile-based), not an expert-validated optimum --
    see the Task 6 report for the full discussion and the bootstrap
    stability check.

    Uses linear interpolation between closest ranks (same convention as
    numpy.percentile's default 'linear' method), implemented locally with
    only the standard library so this module has no numpy dependency.
    """
    values = sorted(v for v in composite_scores if v is not None and math.isfinite(v))
    if len(values) < 3:
        raise ValueError("tertile_thresholds requires at least 3 finite composite scores")

    def _percentile(sorted_vals: list[float], pct: float) -> float:
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        rank = pct / 100.0 * (n - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return sorted_vals[int(rank)]
        frac = rank - lo
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac

    simple_max = _percentile(values, 100.0 / 3.0)
    moderate_max = _percentile(values, 200.0 / 3.0)
    return simple_max, moderate_max
