"""F2.1B3 Task 6 -- Cohen's kappa inter-rater agreement helper.

Status (as of F2.1B3): only ONE QC expert reviewer was available for that
stage (see ``reports/qc_expert_review_instructions.md`` and the Task 6
section of the B3 panel-defense report) -- the ``cohens_kappa`` and
``confusion_counts`` functions below were, at that time, exercised only
against clearly-labelled synthetic/dummy data in this module's own unit
tests, never a real review artifact.

Update (F2.1B4-R): a genuine (AI-assisted, NOT human-expert) blind review
of 30 images now exists (``results/review/blind_judgments.csv``), and this
is the first stage that calls ``cohens_kappa``/``confusion_counts`` against
real reconciled data (see ``dataset/f21b4r_reconciliation.py``). This does
NOT change the status of a human QC expert review, which remains pending.

This module also provides ``weighted_kappa`` (added in F2.1B4-R), since
Simple/Moderate/High is an ordinal (not purely nominal) scale and an
ordinal-aware agreement statistic is appropriate alongside the unweighted
one -- see that function's docstring for the linear-vs-quadratic choice
and rationale.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters' categorical judgments over the same
    N items, in the same order.

    kappa = (p_o - p_e) / (1 - p_e)
      p_o = observed proportion agreement (fraction of items where both
            raters gave the same label)
      p_e = expected proportion agreement by chance, computed from each
            rater's OWN marginal label-frequency distribution:
            p_e = sum_over_categories( P(rater_a == c) * P(rater_b == c) )

    This is the standard (unweighted) Cohen's kappa -- appropriate for
    nominal categories (Simple/Moderate/High have a natural order, but
    unweighted kappa is the simpler, more conservative starting point;
    a weighted-kappa variant could be added later if ordinal distance
    between disagreements needs to be credited, but is NOT implemented
    here to keep this helper auditable and matching the textbook
    definition exactly).

    Parameters
    ----------
    rater_a, rater_b : same-length sequences of category labels (any
        hashable type, e.g. str), index-aligned (rater_a[i] and
        rater_b[i] are the two ratings for the SAME i-th item).

    Returns
    -------
    float kappa, in [-1, 1] typically (1 = perfect agreement, 0 = chance-
    level agreement, negative = worse than chance).

    Raises
    ------
    ValueError if the two sequences differ in length, are empty, or if
    expected agreement p_e == 1.0 (degenerate case: both raters used only
    ONE identical category for every item, so chance-agreement already
    equals 1 and kappa is mathematically undefined, not just numerically
    unstable -- this is surfaced loudly rather than silently returning
    NaN or a fabricated 1.0/0.0).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"rater_a and rater_b must be the same length, got {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one rated item")

    n_agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    p_o = n_agree / n

    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    categories = set(counts_a) | set(counts_b)
    p_e = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in categories)

    if p_e >= 1.0:
        raise ValueError(
            "cohens_kappa: expected chance-agreement p_e == 1.0 (both raters used only "
            "one identical category for every item) -- kappa is mathematically undefined "
            "(division by zero), not silently approximated."
        )

    return (p_o - p_e) / (1.0 - p_e)


def weighted_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    categories: Sequence[str],
    weight_type: str = "linear",
) -> float:
    """Weighted Cohen's kappa for two raters' ORDINAL categorical judgments,
    crediting partial agreement on near-miss disagreements (e.g. Simple vs.
    Moderate) rather than treating every disagreement as equally wrong
    (e.g. Simple vs. High).

    ``categories`` must be given in their natural ORDER (e.g.
    ``["Simple", "Moderate", "High"]``) -- the function uses each
    category's POSITION in this list as its ordinal index; it does not
    infer an order itself.

    Weight choice -- ``weight_type``:
      - ``"linear"`` (default): ``w(i, j) = |i - j| / (k - 1)``. Disagreement
        cost grows proportionally with ordinal distance.
      - ``"quadratic"``: ``w(i, j) = ((i - j) / (k - 1)) ** 2``. Disagreement
        cost grows with the SQUARE of ordinal distance.

    This project uses **linear** weighting by default (see
    ``dataset/f21b4r_reconciliation.py``) rather than quadratic, for an
    explicit, documented reason specific to this 3-category scale: with
    only k=3 ordered categories (Simple/Moderate/High), quadratic weighting
    makes a single Simple<->High disagreement count 4x as much as an
    adjacent Simple<->Moderate or Moderate<->High disagreement (w=1.0 vs.
    w=0.25), whereas linear weighting counts it only 2x as much (w=1.0 vs.
    w=0.5). On a small sample (n=30 in the first real use of this
    function), one or two Simple<->High flips under quadratic weighting
    would disproportionately dominate the statistic. Linear weighting is
    the more conservative, more easily hand-verified choice for a small-n
    descriptive analysis; quadratic remains available (pass
    ``weight_type="quadratic"``) for anyone who wants to compare.

    Formula (standard weighted kappa):
        O[i][j] = observed joint proportion rater_a==categories[i] AND
                   rater_b==categories[j]
        E[i][j] = p_a[i] * p_b[j]   (expected joint proportion under
                   independence, from each rater's own marginals)
        kappa_w = 1 - sum(w * O) / sum(w * E)

    Parameters
    ----------
    rater_a, rater_b : same-length, index-aligned category-label sequences
        (every value must be a member of ``categories``).
    categories : the full, ORDERED set of valid category labels.
    weight_type : ``"linear"`` or ``"quadratic"``.

    Returns
    -------
    float kappa_w (1.0 = perfect agreement; 0.0 = chance-level by the
    weighted-expectation definition; can be negative).

    Raises
    ------
    ValueError on mismatched lengths, empty input, fewer than 2 categories,
    an out-of-vocabulary label in either rater's sequence, an unknown
    ``weight_type``, or a degenerate zero weighted-expected-agreement
    denominator (mathematically undefined, surfaced loudly rather than
    silently returning NaN/0/1).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"rater_a and rater_b must be the same length, got {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("weighted_kappa requires at least one rated item")
    k = len(categories)
    if k < 2:
        raise ValueError(f"weighted_kappa requires at least 2 categories, got {k}")
    if weight_type not in ("linear", "quadratic"):
        raise ValueError(f"weight_type must be 'linear' or 'quadratic', got {weight_type!r}")

    index = {cat: i for i, cat in enumerate(categories)}
    for label, seq_name in ((rater_a, "rater_a"), (rater_b, "rater_b")):
        for v in label:
            if v not in index:
                raise ValueError(f"{seq_name} contains label {v!r} not present in categories={list(categories)}")

    def _w(i: int, j: int) -> float:
        dist = abs(i - j) / (k - 1)
        return dist if weight_type == "linear" else dist ** 2

    observed = [[0.0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        observed[index[a]][index[b]] += 1.0
    observed = [[c / n for c in row] for row in observed]

    p_a = [sum(observed[i][j] for j in range(k)) for i in range(k)]
    p_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    sum_w_o = sum(_w(i, j) * observed[i][j] for i in range(k) for j in range(k))
    sum_w_e = sum(_w(i, j) * p_a[i] * p_b[j] for i in range(k) for j in range(k))

    if sum_w_e == 0.0:
        raise ValueError(
            "weighted_kappa: weighted expected-agreement denominator is 0.0 -- "
            "mathematically undefined (this happens only in degenerate marginal "
            "distributions), not silently approximated."
        )

    return 1.0 - (sum_w_o / sum_w_e)


def confusion_counts(rater_a: Sequence[str], rater_b: Sequence[str]) -> dict[str, dict[str, int]]:
    """Return a {rater_a_label: {rater_b_label: count}} nested confusion
    table, for reporting alongside the kappa value (never used as a
    substitute for it)."""
    if len(rater_a) != len(rater_b):
        raise ValueError(f"rater_a and rater_b must be the same length, got {len(rater_a)} vs {len(rater_b)}")
    table: dict[str, dict[str, int]] = {}
    for a, b in zip(rater_a, rater_b):
        table.setdefault(a, {})
        table[a][b] = table[a].get(b, 0) + 1
    return table
