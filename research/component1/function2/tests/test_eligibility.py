"""F2.1B2 Task 9 -- tests for complexity/eligibility.py.

Covers each of the 7 individual checks in isolation, combinations of
partial validity (e.g. valid GLCM but invalid luminance must still be
ineligible), and numerical edge cases (NaN/Inf rejected cleanly).
"""

from __future__ import annotations

import math

from research.component1.function2.complexity.eligibility import (
    CHECK_ORDER,
    REJECTION_REASON_LABELS,
    determine_eligibility,
)

VALID_KWARGS = dict(
    split_confirmed_train=True,
    background_status="boxes_excluded",
    luminance_variation=12.5,
    glcm_entropy=7.2,
    glcm_status="valid",
    glcm_valid_pair_count=10_000,
    colour_variation=3.1,
)


def test_all_valid_is_eligible():
    r = determine_eligibility(**VALID_KWARGS)
    assert r.eligible is True
    assert r.rejection_reason is None
    assert r.failed_checks == []
    assert set(r.checks.keys()) == set(CHECK_ORDER)
    assert all(r.checks.values())


def test_not_train_split_rejected():
    kwargs = dict(VALID_KWARGS, split_confirmed_train=False)
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.rejection_reason == "not_train_split"
    assert r.checks["split_confirmed_train"] is False


def test_unresolved_annotation_rejected():
    kwargs = dict(VALID_KWARGS, background_status="unresolved_missing_annotation")
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.rejection_reason == "annotation_unresolved"


def test_error_invalid_input_background_status_rejected():
    kwargs = dict(VALID_KWARGS, background_status="error_invalid_input")
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["annotation_resolved"] is False


def test_luminance_none_rejected():
    kwargs = dict(VALID_KWARGS, luminance_variation=None)
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["luminance_valid"] is False
    assert "luminance_invalid_or_nonfinite" in r.failed_checks


def test_luminance_nan_rejected():
    kwargs = dict(VALID_KWARGS, luminance_variation=float("nan"))
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["luminance_valid"] is False


def test_luminance_inf_rejected():
    kwargs = dict(VALID_KWARGS, luminance_variation=float("inf"))
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["luminance_valid"] is False


def test_colour_variation_nan_rejected():
    kwargs = dict(VALID_KWARGS, colour_variation=float("nan"))
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["colour_valid"] is False


def test_glcm_insufficient_pairs_status_rejected_even_if_entropy_numeric():
    """A reported entropy value does not make a sample eligible if the
    masked_glcm module itself flagged the pair count as unreliable."""
    kwargs = dict(VALID_KWARGS, glcm_status="insufficient_pairs", glcm_entropy=5.0,
                   glcm_valid_pair_count=200)
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["glcm_valid"] is False
    assert r.checks["glcm_pairs_ok"] is False


def test_glcm_all_excluded_rejected():
    kwargs = dict(VALID_KWARGS, glcm_status="all_excluded", glcm_entropy=None,
                   glcm_valid_pair_count=0)
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["glcm_valid"] is False


def test_glcm_entropy_nan_rejected_even_with_valid_status():
    """Defensive: even if glcm_status somehow said 'valid', a non-finite
    entropy value must still fail the check (never trust status alone)."""
    kwargs = dict(VALID_KWARGS, glcm_status="valid", glcm_entropy=float("nan"))
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["glcm_valid"] is False


def test_background_coverage_insufficient_rejected():
    kwargs = dict(VALID_KWARGS, background_status="insufficient_background")
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert r.checks["background_coverage_ok"] is False


def test_glcm_pair_count_below_threshold_rejected_independently():
    """Check 7 (glcm_pairs_ok) fires even if glcm_status were somehow
    inconsistent with the pair count -- the two checks are evaluated
    independently and never collapsed into one boolean."""
    kwargs = dict(VALID_KWARGS, glcm_valid_pair_count=100, min_valid_pairs=500)
    r = determine_eligibility(**kwargs)
    assert r.checks["glcm_pairs_ok"] is False
    # glcm_status is "valid" in VALID_KWARGS, so glcm_valid is independently True --
    # demonstrates the two checks are NOT merged (one can fail while the
    # other passes).
    assert r.checks["glcm_valid"] is True
    assert r.eligible is False


def test_valid_glcm_but_invalid_luminance_is_still_ineligible():
    """Explicit partial-validity combination required by the task brief:
    valid GLCM + invalid luminance must still be ineligible overall, and
    the per-check dict must show exactly which one failed."""
    kwargs = dict(VALID_KWARGS, luminance_variation=None)
    r = determine_eligibility(**kwargs)
    assert r.checks["glcm_valid"] is True
    assert r.checks["luminance_valid"] is False
    assert r.eligible is False
    assert r.rejection_reason == "luminance_invalid_or_nonfinite"


def test_multiple_simultaneous_failures_records_all_failed_checks():
    kwargs = dict(
        VALID_KWARGS,
        luminance_variation=None,
        colour_variation=float("nan"),
        background_status="insufficient_background",
    )
    r = determine_eligibility(**kwargs)
    assert r.eligible is False
    assert len(r.failed_checks) == 3
    assert "luminance_invalid_or_nonfinite" in r.failed_checks
    assert "colour_variation_invalid_or_nonfinite" in r.failed_checks
    assert "background_coverage_insufficient_provisional_70pct_rule" in r.failed_checks
    # Primary reason is the FIRST failing check in CHECK_ORDER priority
    # (luminance_valid comes before colour_valid and background_coverage_ok).
    assert r.rejection_reason == "luminance_invalid_or_nonfinite"


def test_rejection_reason_priority_matches_check_order():
    """The primary rejection_reason always corresponds to the first entry
    of CHECK_ORDER that is False -- verified exhaustively for every
    single-check failure."""
    for check_name in CHECK_ORDER:
        kwargs = dict(VALID_KWARGS)
        if check_name == "split_confirmed_train":
            kwargs["split_confirmed_train"] = False
        elif check_name == "annotation_resolved":
            kwargs["background_status"] = "unresolved_missing_annotation"
        elif check_name == "luminance_valid":
            kwargs["luminance_variation"] = None
        elif check_name == "glcm_valid":
            kwargs["glcm_status"] = "all_excluded"
            kwargs["glcm_entropy"] = None
        elif check_name == "colour_valid":
            kwargs["colour_variation"] = None
        elif check_name == "background_coverage_ok":
            kwargs["background_status"] = "insufficient_background"
        elif check_name == "glcm_pairs_ok":
            kwargs["glcm_valid_pair_count"] = 0
        r = determine_eligibility(**kwargs)
        assert r.rejection_reason == REJECTION_REASON_LABELS[check_name], check_name


def test_never_substitutes_zero_for_invalid_value():
    """Eligibility itself never fabricates a 0; this test guards the
    CONTRACT (raw_features passed through untouched) by checking that an
    explicitly-None luminance value is never silently interpreted as 0.0
    (which would have made the sample eligible)."""
    kwargs = dict(VALID_KWARGS, luminance_variation=None)
    r = determine_eligibility(**kwargs)
    assert r.checks["luminance_valid"] is False  # None != 0.0 behaviour


def test_policy_version_and_thresholds_recorded():
    r = determine_eligibility(**VALID_KWARGS)
    assert "PROVISIONAL" in r.policy_version
    assert r.max_defect_coverage_applied == 0.70
    assert r.min_valid_pairs_applied == 500


def test_custom_thresholds_are_applied_not_hardcoded():
    kwargs = dict(VALID_KWARGS, glcm_valid_pair_count=600, min_valid_pairs=1000)
    r = determine_eligibility(**kwargs)
    assert r.checks["glcm_pairs_ok"] is False
    assert r.min_valid_pairs_applied == 1000
