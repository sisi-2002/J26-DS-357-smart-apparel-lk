"""Unit tests for complexity/inter_rater_agreement.py.

IMPORTANT: every rating sequence in this file is CLEARLY-LABELLED
SYNTHETIC/DUMMY DATA constructed purely to exercise the Cohen's-kappa
arithmetic. None of it is, or is derived from, any real expert-review
judgment -- F2.1B3 has only ONE QC reviewer available (see
reports/qc_expert_review_instructions.md), so no real second-rater data
exists anywhere in this repository. This function is not currently called
against any real review artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.component1.function2.complexity.inter_rater_agreement import (  # noqa: E402
    cohens_kappa,
    confusion_counts,
    weighted_kappa,
)

ORDERED_CATEGORIES = ["Simple", "Moderate", "High"]


def test_perfect_agreement_gives_kappa_one():
    dummy_rater_a = ["Simple", "Moderate", "High", "Simple", "Moderate", "High"]
    dummy_rater_b = ["Simple", "Moderate", "High", "Simple", "Moderate", "High"]
    assert cohens_kappa(dummy_rater_a, dummy_rater_b) == pytest.approx(1.0)


def test_systematic_disagreement_gives_kappa_at_or_below_zero():
    # dummy_rater_b always picks a DIFFERENT category than dummy_rater_a,
    # cycling -- worse than chance agreement given 3 roughly-equal categories.
    dummy_rater_a = ["Simple", "Moderate", "High"] * 4
    dummy_rater_b = ["Moderate", "High", "Simple"] * 4  # never matches rater_a
    kappa = cohens_kappa(dummy_rater_a, dummy_rater_b)
    assert kappa <= 0.0


def test_known_textbook_example_matches_hand_computed_value():
    # Hand-computable dummy example: 10 items, 8 agree, 2 disagree, with
    # balanced marginals -- lets us hand-verify p_o, p_e, kappa exactly.
    dummy_rater_a = ["Simple"] * 5 + ["High"] * 5
    dummy_rater_b = ["Simple"] * 4 + ["High"] * 1 + ["Simple"] * 1 + ["High"] * 4
    # p_o = 8/10 = 0.8
    # rater_a marginals: Simple 5/10=0.5, High 5/10=0.5
    # rater_b marginals: Simple 5/10=0.5, High 5/10=0.5
    # p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    # kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
    kappa = cohens_kappa(dummy_rater_a, dummy_rater_b)
    assert kappa == pytest.approx(0.6, abs=1e-9)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["Simple", "Moderate"], ["Simple"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_degenerate_single_category_raises_not_silently_fabricated():
    dummy_rater_a = ["Simple", "Simple", "Simple"]
    dummy_rater_b = ["Simple", "Simple", "Simple"]
    with pytest.raises(ValueError):
        cohens_kappa(dummy_rater_a, dummy_rater_b)


def test_confusion_counts_sums_to_n():
    dummy_rater_a = ["Simple", "Moderate", "High", "Simple"]
    dummy_rater_b = ["Simple", "High", "High", "Moderate"]
    table = confusion_counts(dummy_rater_a, dummy_rater_b)
    total = sum(count for row in table.values() for count in row.values())
    assert total == 4
    assert table["Simple"]["Simple"] == 1
    assert table["Moderate"]["High"] == 1
    assert table["High"]["High"] == 1
    assert table["Simple"]["Moderate"] == 1


# ---------------------------------------------------------------------------
# weighted_kappa (F2.1B4-R) -- synthetic/dummy data only, hand-computed by
# hand in the docstring-adjacent comments below so the arithmetic can be
# independently re-verified without running any code.
# ---------------------------------------------------------------------------
def test_weighted_kappa_perfect_agreement_gives_one():
    dummy_rater_a = ["Simple", "Moderate", "High", "Simple", "Moderate", "High"]
    dummy_rater_b = ["Simple", "Moderate", "High", "Simple", "Moderate", "High"]
    assert weighted_kappa(dummy_rater_a, dummy_rater_b, ORDERED_CATEGORIES, "linear") == pytest.approx(1.0)
    assert weighted_kappa(dummy_rater_a, dummy_rater_b, ORDERED_CATEGORIES, "quadratic") == pytest.approx(1.0)


def test_weighted_kappa_hand_computed_linear_example():
    # n=5 dummy items. Observed pairs (rater_a, rater_b):
    #   (Simple, Simple), (Simple, Moderate), (Moderate, Moderate),
    #   (Moderate, High), (High, High)
    # Hand computation (categories ordered Simple=0, Moderate=1, High=2,
    # linear weight w(i,j) = |i-j|/2):
    #   p_a = [0.4, 0.4, 0.2]  (2/5, 2/5, 1/5)
    #   p_b = [0.2, 0.4, 0.4]  (1/5, 2/5, 2/5)
    #   observed proportions: (0,0)=0.2 (0,1)=0.2 (1,1)=0.2 (1,2)=0.2 (2,2)=0.2
    #   sum(w*O) = 0*0.2 + 0.5*0.2 + 0*0.2 + 0.5*0.2 + 0*0.2 = 0.2
    #   full E[i][j] = p_a[i]*p_b[j]; sum(w*E) = 0.44 (hand-expanded over
    #   all 9 cells -- see module history / PR description for the full
    #   cell-by-cell expansion)
    #   kappa_w = 1 - 0.2/0.44 = 1 - 5/11 = 6/11
    dummy_rater_a = ["Simple", "Simple", "Moderate", "Moderate", "High"]
    dummy_rater_b = ["Simple", "Moderate", "Moderate", "High", "High"]
    kappa = weighted_kappa(dummy_rater_a, dummy_rater_b, ORDERED_CATEGORIES, "linear")
    assert kappa == pytest.approx(6.0 / 11.0, abs=1e-9)


def test_weighted_kappa_quadratic_penalises_far_disagreement_more_than_linear():
    # Same dummy data as above: quadratic weighting must give a DIFFERENT
    # (here, lower -- because the one far-apart pair gets disproportionately
    # penalised relative to its weighted-expectation share) kappa than linear,
    # confirming the two weight_type branches are not accidentally identical.
    dummy_rater_a = ["Simple", "Simple", "Moderate", "Moderate", "High"]
    dummy_rater_b = ["Simple", "Moderate", "Moderate", "High", "High"]
    kappa_linear = weighted_kappa(dummy_rater_a, dummy_rater_b, ORDERED_CATEGORIES, "linear")
    kappa_quadratic = weighted_kappa(dummy_rater_a, dummy_rater_b, ORDERED_CATEGORIES, "quadratic")
    assert kappa_linear != pytest.approx(kappa_quadratic)


def test_weighted_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        weighted_kappa(["Simple", "Moderate"], ["Simple"], ORDERED_CATEGORIES)


def test_weighted_kappa_empty_input_raises():
    with pytest.raises(ValueError):
        weighted_kappa([], [], ORDERED_CATEGORIES)


def test_weighted_kappa_unknown_label_raises():
    with pytest.raises(ValueError):
        weighted_kappa(["Simple", "Unknown"], ["Simple", "Moderate"], ORDERED_CATEGORIES)


def test_weighted_kappa_unknown_weight_type_raises():
    with pytest.raises(ValueError):
        weighted_kappa(["Simple"], ["Simple"], ORDERED_CATEGORIES, weight_type="cubic")


def test_weighted_kappa_single_category_raises():
    with pytest.raises(ValueError):
        weighted_kappa(["Simple"], ["Simple"], ["Simple"])
