"""F2.1B2 Task 1 -- unified sample eligibility policy for 3-feature
complexity scoring.

This module does NOT compute any feature itself. It takes the already-
computed results from ``manifest_loader`` (split confirmation),
``background_region``/``background_features`` (background provenance +
the three raw feature values + their own per-feature validity signals) and
turns them into one explicit, auditable eligibility decision per sample.

Design principles (all load-bearing, see task brief):
  - Per-feature validity flags are kept distinct in the output. They are
    never collapsed into a single boolean that would hide which specific
    check failed.
  - An invalid value is never silently replaced with 0 or any other
    placeholder. ``Optional[float]`` fields stay ``None`` when invalid.
  - Every failed sample gets a specific ``rejection_reason`` (the first
    failing check, in the fixed priority order below) PLUS the full list
    of every failing check (``failed_checks``), so no information is
    thrown away just because one reason was reported as "the" reason.
  - The two numeric thresholds this policy depends on
    (``max_defect_coverage=0.70`` from F2.1A/background_region.py, and
    ``min_valid_pairs=500`` from F2.1B1/masked_glcm.py) are both reused
    as-is, not re-derived, and are explicitly labelled PROVISIONAL in
    ``POLICY_VERSION`` and in every ``EligibilityResult.policy_version``
    value. This module does not decide whether those thresholds are
    "right" -- it only applies them consistently and names them.

Eligibility requires ALL seven checks below to pass (task brief numbering):
  1. split_confirmed_train     -- confirmed TRAIN via the authoritative
                                   manifest (``assert_train_split``-style
                                   logic; the caller must have already run
                                   that check before opening any pixels --
                                   this module only records the boolean
                                   outcome, it does not re-open the
                                   manifest itself).
  2. annotation_resolved       -- background_status is not
                                   "unresolved_missing_annotation" (and not
                                   "error_invalid_input", which is the
                                   companion invalid-geometry case).
  3. luminance_valid           -- luminance_variation is not None and
                                   finite.
  4. glcm_valid                -- glcm_status == "valid" (i.e. NOT
                                   "insufficient_pairs" and NOT
                                   "all_excluded" and not any "error_*"
                                   status) AND glcm_entropy is not None
                                   and finite.
  5. colour_valid               -- colour_variation is not None and
                                   finite.
  6. background_coverage_ok    -- background_status != "insufficient_background"
                                   (i.e. the provisional 70% max-defect-
                                   coverage rule from background_region.py
                                   is satisfied).
  7. glcm_pairs_ok              -- glcm_valid_pair_count >= min_valid_pairs
                                   (kept as its OWN check, even though under
                                   the current masked_glcm.py implementation
                                   it is implied by glcm_status=="valid" --
                                   see module docstring note below -- so
                                   that a future change to either module
                                   cannot silently make one check a no-op
                                   without this policy noticing).

Note on checks 4 and 7: as of ``masked_glcm.py``'s current implementation,
``glcm_status`` becomes ``"insufficient_pairs"`` precisely when
``valid_pair_count < min_valid_pairs``, so checks 4 and 7 will always agree
in practice today. They are nonetheless kept as two separately-named,
separately-recorded checks (never merged into one boolean) per the task
brief's explicit instruction not to collapse per-feature validity into a
single hidden flag, and because that 1:1 relationship is an implementation
detail of masked_glcm.py, not a logical guarantee this policy module should
assume.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from research.component1.function2.complexity.background_region import (
    DEFAULT_MAX_DEFECT_COVERAGE,
)
from research.component1.function2.complexity.masked_glcm import (
    DEFAULT_MIN_VALID_PAIRS,
)

POLICY_VERSION = "eligibility_policy_v1_PROVISIONAL"

# Background-provenance statuses that mean "we cannot trust this image's
# annotation state at all" (never silently treated as clean background).
UNRESOLVED_ANNOTATION_STATUSES = frozenset({
    "unresolved_missing_annotation",
    "error_invalid_input",
})

# Background-provenance statuses that represent a resolved, well-formed
# background region (coverage check is evaluated separately -- a status
# can be "resolved" and still fail the coverage check, e.g.
# "insufficient_background" is itself a resolved-but-too-small outcome).
RESOLVED_BACKGROUND_STATUSES = frozenset({
    "boxes_excluded",
    "clean_negative",
    "clean_no_boxes_provided",
    "insufficient_background",
})

# Fixed priority order used to pick ONE primary ``rejection_reason`` when
# multiple checks fail simultaneously. This does not discard the other
# failures -- they remain in ``failed_checks`` -- it only gives a stable,
# documented tie-break for the single-string summary field.
CHECK_ORDER = [
    "split_confirmed_train",
    "annotation_resolved",
    "luminance_valid",
    "glcm_valid",
    "colour_valid",
    "background_coverage_ok",
    "glcm_pairs_ok",
]

REJECTION_REASON_LABELS = {
    "split_confirmed_train": "not_train_split",
    "annotation_resolved": "annotation_unresolved",
    "luminance_valid": "luminance_invalid_or_nonfinite",
    "glcm_valid": "glcm_entropy_invalid_or_nonfinite",
    "colour_valid": "colour_variation_invalid_or_nonfinite",
    "background_coverage_ok": "background_coverage_insufficient_provisional_70pct_rule",
    "glcm_pairs_ok": "glcm_pair_support_insufficient_provisional_min500",
}


@dataclasses.dataclass
class EligibilityResult:
    eligible: bool
    # None when eligible; otherwise the FIRST failing check (CHECK_ORDER
    # priority), as one of REJECTION_REASON_LABELS' values.
    rejection_reason: Optional[str]
    # ALL failing checks, in CHECK_ORDER order (empty list when eligible).
    # Never collapsed to just the primary reason.
    failed_checks: list[str]
    # Every one of the 7 named checks, always present, True/False only
    # (never omitted, never merged with another check).
    checks: dict[str, bool]
    policy_version: str = POLICY_VERSION
    # The two provisional numeric thresholds actually applied, recorded
    # per-result so a downstream reader never has to assume a global
    # default silently stayed the same.
    max_defect_coverage_applied: float = DEFAULT_MAX_DEFECT_COVERAGE
    min_valid_pairs_applied: int = DEFAULT_MIN_VALID_PAIRS

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def _is_finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def determine_eligibility(
    *,
    split_confirmed_train: bool,
    background_status: str,
    luminance_variation: Optional[float],
    glcm_entropy: Optional[float],
    glcm_status: str,
    glcm_valid_pair_count: int,
    colour_variation: Optional[float],
    max_defect_coverage: float = DEFAULT_MAX_DEFECT_COVERAGE,
    min_valid_pairs: int = DEFAULT_MIN_VALID_PAIRS,
) -> EligibilityResult:
    """Apply the unified 7-check eligibility policy to one sample.

    Parameters mirror the fields already produced by
    ``background_features.compute_background_features`` plus the
    split-confirmation boolean the caller is required to have already
    established via ``manifest_loader.assert_train_split`` before ever
    opening the image's pixels (this function trusts that boolean, it does
    not re-derive it, so that this module has no dependency on the dataset
    manifest files themselves and can be unit-tested in isolation).

    Returns
    -------
    EligibilityResult
    """
    checks: dict[str, bool] = {}

    checks["split_confirmed_train"] = bool(split_confirmed_train)
    checks["annotation_resolved"] = background_status not in UNRESOLVED_ANNOTATION_STATUSES
    checks["luminance_valid"] = _is_finite(luminance_variation)
    checks["glcm_valid"] = (glcm_status == "valid") and _is_finite(glcm_entropy)
    checks["colour_valid"] = _is_finite(colour_variation)
    checks["background_coverage_ok"] = background_status != "insufficient_background"
    checks["glcm_pairs_ok"] = glcm_valid_pair_count >= min_valid_pairs

    failed_checks = [name for name in CHECK_ORDER if not checks[name]]
    eligible = len(failed_checks) == 0
    primary_reason = (
        REJECTION_REASON_LABELS[failed_checks[0]] if failed_checks else None
    )

    return EligibilityResult(
        eligible=eligible,
        rejection_reason=primary_reason,
        failed_checks=[REJECTION_REASON_LABELS[c] for c in failed_checks],
        checks=checks,
        max_defect_coverage_applied=max_defect_coverage,
        min_valid_pairs_applied=min_valid_pairs,
    )
