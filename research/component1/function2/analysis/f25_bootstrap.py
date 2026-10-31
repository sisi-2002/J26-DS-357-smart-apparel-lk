"""F2.5 Task 11 -- family-level block bootstrap utility.

Resampling unit: source_family_id (whole family resampled with replacement,
so every instance x method row belonging to a resampled family travels
together -- this preserves within-family/within-method correlation
structure instead of pretending instances are i.i.d.).

Seed: 357 (reused from every earlier Function-2 stage's deterministic seed,
per the task brief's explicit instruction, for consistency not novelty).
Resamples: 2000 per statistic. Justification for 2000, not some other
round number: with an effective family count in the 14-48 range in this
pool (see task2_unit_definitions_summary.json), 2000 resamples gives a
Monte-Carlo standard error on an estimated 2.5th/97.5th percentile of
roughly sqrt(0.025*0.975/2000) ~= 0.0035 in probability units -- precise
enough that the reported percentile CI bound does not meaningfully jitter
between repeated runs at the SAME seed (verified deterministic:
test_f25_analysis.py::test_bootstrap_reproducible_same_seed), while keeping
runtime trivial in pure NumPy for every (metric, method, stratum)
combination this stage computes. It is not tuned per result.

Sparsity rule (also Task 7/Task 8's declared cutoff): a stratum with fewer
than MIN_FAMILIES_FOR_CI (5) distinct families gets a DESCRIPTIVE summary
only (mean/median/min/max over the available family-level values) -- no
bootstrap CI is computed or reported for it, because a percentile interval
built from under 5 clusters is not a meaningful uncertainty statement, and
reporting one anyway would misrepresent precision that is not there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import SEED

N_RESAMPLES = 2000
MIN_FAMILIES_FOR_CI = 5


def bootstrap_family_mean(values_by_family: dict, seed: int = SEED, n_resamples: int = N_RESAMPLES) -> dict:
    """Percentile bootstrap CI for the family-first mean of a scalar metric.

    ``values_by_family``: {family_id: value}. Families with value None are
    dropped (that family contributed no valid observation for this metric)
    before resampling -- never imputed.
    """
    fam_ids = sorted(k for k, v in values_by_family.items() if v is not None)
    vals = np.array([values_by_family[k] for k in fam_ids], dtype=float)
    n_fam = len(fam_ids)

    result = {
        "resampling_unit": "source_family_id",
        "seed": seed,
        "n_resamples_requested": n_resamples,
        "n_effective_families": n_fam,
        "min_families_for_ci": MIN_FAMILIES_FOR_CI,
        "point_estimate_family_first_mean": float(np.mean(vals)) if n_fam else None,
    }
    if n_fam == 0:
        result.update({"ci_skipped": True, "skip_reason": "no_families_with_valid_data", "ci_lo": None, "ci_hi": None})
        return result
    if n_fam < MIN_FAMILIES_FOR_CI:
        result.update({
            "ci_skipped": True,
            "skip_reason": f"only {n_fam} families < sparsity cutoff {MIN_FAMILIES_FOR_CI}; descriptive only",
            "median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
            "ci_lo": None, "ci_hi": None,
        })
        return result

    rng = np.random.default_rng(seed)
    idx = np.arange(n_fam)
    boot_means = np.empty(n_resamples, dtype=float)
    resample_failures = 0
    for i in range(n_resamples):
        try:
            sample_idx = rng.choice(idx, size=n_fam, replace=True)
            boot_means[i] = vals[sample_idx].mean()
        except Exception:
            resample_failures += 1
            boot_means[i] = np.nan
    valid_boot = boot_means[~np.isnan(boot_means)]
    ci_lo, ci_hi = np.percentile(valid_boot, [2.5, 97.5]) if len(valid_boot) else (None, None)

    result.update({
        "ci_skipped": False,
        "ci_method": "percentile",
        "ci_lo": float(ci_lo) if ci_lo is not None else None,
        "ci_hi": float(ci_hi) if ci_hi is not None else None,
        "n_resample_failures": int(resample_failures),
        "bootstrap_mean_of_means": float(valid_boot.mean()) if len(valid_boot) else None,
    })
    return result


def bootstrap_family_paired_difference(diff_by_family: dict, seed: int = SEED, n_resamples: int = N_RESAMPLES) -> dict:
    """Same mechanics as bootstrap_family_mean, for a per-family paired
    (method A - method B) difference that has already been computed at the
    family level (see f25_task2_datasets.build_family_level)."""
    return bootstrap_family_mean(diff_by_family, seed=seed, n_resamples=n_resamples)


def reproducibility_self_check(values_by_family: dict, seed: int = SEED) -> bool:
    """Returns True iff two independent calls at the same seed produce an
    identical CI -- used by the regression test for deterministic bootstrap
    reproducibility (Task 15)."""
    r1 = bootstrap_family_mean(values_by_family, seed=seed)
    r2 = bootstrap_family_mean(values_by_family, seed=seed)
    return r1.get("ci_lo") == r2.get("ci_lo") and r1.get("ci_hi") == r2.get("ci_hi")
