"""F2.5 Task 11 companion -- consolidate every bootstrap CI actually computed
across Tasks 4/5/9 into one uncertainty manifest + the multiplicity
disclosure the pre-registered plan requires.

This script does not compute any NEW statistic -- it reads the already
-written Task 4/5/9 JSON tables (which embed their own
f25_bootstrap.bootstrap_family_mean/bootstrap_family_paired_difference
results) and flattens every CI found into one audit-friendly table, and
counts how many were actually computed vs. skipped for sparsity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.component1.function2.analysis.f25_common import F25_OUT, write_csv, write_json


def _load(name):
    with open(F25_OUT / "tables" / name, encoding="utf-8") as fh:
        return json.load(fh)


def _walk_bootstrap_dicts(obj, path=""):
    """Yield (path, dict) for every dict that looks like a bootstrap result
    (has the 'ci_skipped' key, the bootstrap_family_mean contract)."""
    if isinstance(obj, dict):
        if "ci_skipped" in obj and "n_effective_families" in obj:
            yield path, obj
        for k, v in obj.items():
            yield from _walk_bootstrap_dicts(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_bootstrap_dicts(v, f"{path}[{i}]")


def run():
    sources = {
        "method_comparison_overall.json": _load("method_comparison_overall.json"),
        "complexity_stratified_results.json": _load("complexity_stratified_results.json"),
        "stability_gap_investigation.json": _load("stability_gap_investigation.json"),
    }

    rows = []
    for source_name, obj in sources.items():
        for path, ci in _walk_bootstrap_dicts(obj):
            rows.append({
                "source_file": source_name,
                "path": path,
                "n_effective_families": ci.get("n_effective_families"),
                "point_estimate": ci.get("point_estimate_family_first_mean"),
                "ci_skipped": ci.get("ci_skipped"),
                "skip_reason": ci.get("skip_reason", ""),
                "ci_lo": ci.get("ci_lo"),
                "ci_hi": ci.get("ci_hi"),
                "seed": ci.get("seed"),
                "n_resamples_requested": ci.get("n_resamples_requested"),
            })

    n_computed = sum(1 for r in rows if not r["ci_skipped"])
    n_skipped = sum(1 for r in rows if r["ci_skipped"])

    write_csv(F25_OUT / "uncertainty" / "all_bootstrap_cis_flattened.csv", rows,
              fieldnames=["source_file", "path", "n_effective_families", "point_estimate",
                          "ci_skipped", "skip_reason", "ci_lo", "ci_hi", "seed", "n_resamples_requested"])

    disclosure = {
        "n_bootstrap_statistics_examined_total": len(rows),
        "n_cis_actually_computed_not_skipped": n_computed,
        "n_cis_skipped_for_sparsity": n_skipped,
        "statement": (
            f"{len(rows)} family-level bootstrap statistics were computed across Tasks 4/5/9 "
            f"(method x metric x stratum combinations); {n_computed} produced a percentile CI "
            f"(effective family count >= the declared cutoff), {n_skipped} were SKIPPED (reported "
            "descriptively only) because their stratum's effective family count fell below the "
            "declared sparsity cutoff. Every one is listed in all_bootstrap_cis_flattened.csv -- "
            "none is cherry-picked or omitted from this disclosure, per the pre-registered plan's "
            "multiplicity-disclosure requirement. None of these CIs is a hypothesis-test p-value; "
            "none is interpreted as causal evidence."
        ),
    }
    write_json(F25_OUT / "sensitivity" / "multiplicity_disclosure.json", disclosure)
    return disclosure


if __name__ == "__main__":
    d = run()
    print(d["statement"])
