"""F2.1B2 Task 9 -- tests for complexity/scoring.py.

Covers: normalization-parameter correctness, numerical edge cases
(NaN/Inf/degenerate inputs), deterministic score calculation, group-
boundary edge behaviour, and calibration-config JSON round-trip (the
config dataclass serialisation used by the calibration pipeline).
"""

from __future__ import annotations

import json
import math

import pytest

from research.component1.function2.complexity.scoring import (
    DEFAULT_WEIGHTS,
    NormalizationParams,
    assign_group,
    compute_composite_score,
    compute_normalization_params,
    tertile_thresholds,
    zscore,
)


# ---------------------------------------------------------------------------
# Normalization parameters
# ---------------------------------------------------------------------------

def test_normalization_params_mean_std_correct():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    params = compute_normalization_params(values)
    assert params.mean == pytest.approx(3.0)
    # population std (ddof=0) of [1,2,3,4,5] = sqrt(2) ~= 1.41421356
    assert params.std == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_normalization_params_ignores_non_finite_inputs():
    values = [1.0, 2.0, float("nan"), 3.0, float("inf"), None, 4.0, 5.0]
    params = compute_normalization_params(values)
    assert params.mean == pytest.approx(3.0)


def test_normalization_params_raises_on_degenerate_zero_std():
    with pytest.raises(ValueError):
        compute_normalization_params([5.0, 5.0, 5.0])


def test_normalization_params_raises_on_too_few_values():
    with pytest.raises(ValueError):
        compute_normalization_params([5.0])


def test_normalization_params_json_round_trip():
    params = compute_normalization_params([1.0, 2.0, 3.0, 4.0])
    d = params.to_dict()
    serialized = json.dumps(d)
    reloaded = NormalizationParams.from_dict(json.loads(serialized))
    assert reloaded.mean == pytest.approx(params.mean)
    assert reloaded.std == pytest.approx(params.std)


# ---------------------------------------------------------------------------
# z-score
# ---------------------------------------------------------------------------

def test_zscore_basic():
    params = NormalizationParams(mean=10.0, std=2.0)
    assert zscore(12.0, params) == pytest.approx(1.0)
    assert zscore(10.0, params) == pytest.approx(0.0)
    assert zscore(6.0, params) == pytest.approx(-2.0)


def test_zscore_none_returns_none_not_zero():
    params = NormalizationParams(mean=10.0, std=2.0)
    assert zscore(None, params) is None


def test_zscore_nan_returns_none_not_zero():
    params = NormalizationParams(mean=10.0, std=2.0)
    assert zscore(float("nan"), params) is None


def test_zscore_inf_returns_none():
    params = NormalizationParams(mean=10.0, std=2.0)
    assert zscore(float("inf"), params) is None
    assert zscore(float("-inf"), params) is None


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def _norm():
    return {
        "luminance_variation": NormalizationParams(mean=20.0, std=5.0),
        "glcm_entropy": NormalizationParams(mean=7.0, std=1.0),
        "colour_variation": NormalizationParams(mean=4.0, std=2.0),
    }


def test_composite_score_equal_weights_matches_manual_computation():
    raw = {"luminance_variation": 25.0, "glcm_entropy": 8.0, "colour_variation": 6.0}
    result = compute_composite_score(raw, _norm())
    # z = (1.0, 1.0, 1.0) -> composite = 1.0 under equal weights
    assert result.composite_score == pytest.approx(1.0)
    assert result.status == "provisional"
    assert result.missing_features == []


def test_composite_score_missing_feature_yields_none_not_partial_average():
    raw = {"luminance_variation": 25.0, "glcm_entropy": None, "colour_variation": 6.0}
    result = compute_composite_score(raw, _norm())
    assert result.composite_score is None
    assert result.missing_features == ["glcm_entropy"]
    # The two valid z-scores are still reported, even though no composite
    # was produced from them.
    assert result.z_scores["luminance_variation"] is not None
    assert result.z_scores["colour_variation"] is not None


def test_composite_score_nan_feature_treated_as_missing():
    raw = {"luminance_variation": 25.0, "glcm_entropy": float("nan"), "colour_variation": 6.0}
    result = compute_composite_score(raw, _norm())
    assert result.composite_score is None


def test_composite_score_weights_must_sum_to_one():
    raw = {"luminance_variation": 25.0, "glcm_entropy": 8.0, "colour_variation": 6.0}
    bad_weights = {"luminance_variation": 0.5, "glcm_entropy": 0.5, "colour_variation": 0.5}
    with pytest.raises(ValueError):
        compute_composite_score(raw, _norm(), weights=bad_weights)


def test_composite_score_custom_weighting_changes_result():
    raw = {"luminance_variation": 30.0, "glcm_entropy": 7.0, "colour_variation": 4.0}
    result_equal = compute_composite_score(raw, _norm())
    weighted = {"luminance_variation": 0.8, "glcm_entropy": 0.1, "colour_variation": 0.1}
    result_weighted = compute_composite_score(raw, _norm(), weights=weighted)
    # luminance z = (30-20)/5 = 2.0, others are 0 -> equal-weight composite = 2/3,
    # heavily-weighted composite = 0.8*2.0 = 1.6
    assert result_equal.composite_score == pytest.approx(2.0 / 3.0)
    assert result_weighted.composite_score == pytest.approx(1.6)
    assert result_weighted.composite_score != result_equal.composite_score


def test_composite_score_deterministic_same_input_twice():
    raw = {"luminance_variation": 22.5, "glcm_entropy": 6.9, "colour_variation": 3.7}
    norm = _norm()
    r1 = compute_composite_score(raw, norm)
    r2 = compute_composite_score(raw, norm)
    assert r1.to_dict() == r2.to_dict()


def test_default_weights_are_equal_thirds_and_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    vals = list(DEFAULT_WEIGHTS.values())
    assert all(v == pytest.approx(vals[0]) for v in vals)


# ---------------------------------------------------------------------------
# Group assignment / thresholds
# ---------------------------------------------------------------------------

def test_assign_group_basic_tertiles():
    assert assign_group(-1.0, simple_max=-0.3, moderate_max=0.3) == ("Simple", False)
    assert assign_group(0.0, simple_max=-0.3, moderate_max=0.3) == ("Moderate", False)
    assert assign_group(1.0, simple_max=-0.3, moderate_max=0.3) == ("High", False)


def test_assign_group_none_score_never_forced_into_a_group():
    assert assign_group(None, simple_max=-0.3, moderate_max=0.3) == (None, False)


def test_assign_group_boundary_edge_behaviour_inclusive_upper_bound():
    # composite_score exactly equal to simple_max is Simple (<=), not Moderate.
    group, _ = assign_group(-0.3, simple_max=-0.3, moderate_max=0.3)
    assert group == "Simple"
    group, _ = assign_group(0.3, simple_max=-0.3, moderate_max=0.3)
    assert group == "Moderate"


def test_assign_group_boundary_margin_flags_adjacent_samples():
    # Within margin of simple_max cut point.
    group, flagged = assign_group(-0.32, simple_max=-0.3, moderate_max=0.3, boundary_margin=0.05)
    assert group == "Simple"
    assert flagged is True
    # Far from both cut points.
    group, flagged = assign_group(-0.9, simple_max=-0.3, moderate_max=0.3, boundary_margin=0.05)
    assert flagged is False


def test_assign_group_zero_margin_never_flags():
    group, flagged = assign_group(-0.3, simple_max=-0.3, moderate_max=0.3, boundary_margin=0.0)
    assert flagged is False


def test_tertile_thresholds_simple_known_distribution():
    values = list(range(1, 10))  # 1..9
    simple_max, moderate_max = tertile_thresholds([float(v) for v in values])
    assert simple_max < moderate_max
    # roughly splits into thirds
    assert 3.0 <= simple_max <= 4.0
    assert 6.0 <= moderate_max <= 7.0


def test_tertile_thresholds_ignores_non_finite():
    values = [1.0, 2.0, 3.0, float("nan"), 4.0, 5.0, 6.0, float("inf")]
    simple_max, moderate_max = tertile_thresholds(values)
    assert math.isfinite(simple_max)
    assert math.isfinite(moderate_max)


def test_tertile_thresholds_requires_minimum_samples():
    with pytest.raises(ValueError):
        tertile_thresholds([1.0, 2.0])
