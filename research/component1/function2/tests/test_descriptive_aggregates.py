"""F2.4-B Task 12 / Task 14 -- unit tests for
``evaluation/build_descriptive_aggregates.py``'s aggregation logic, using
synthetic hand-computable data (independent of the live full-run output)."""
from __future__ import annotations

from research.component1.function2.evaluation.build_descriptive_aggregates import (
    _mean, _to_bool, _to_float,
)


def test_to_float_parses_numeric_strings():
    assert _to_float("0.5") == 0.5
    assert _to_float("") is None
    assert _to_float(None) is None
    assert _to_float("None") is None


def test_to_bool_parses_true_false_strings():
    assert _to_bool("True") is True
    assert _to_bool("False") is False
    assert _to_bool("") is False


def test_mean_ignores_none():
    assert _mean([1.0, None, 3.0]) == 2.0
    assert _mean([None, None]) is None
    assert _mean([]) is None


def test_family_first_aggregation_weights_families_not_instances(tmp_path, monkeypatch):
    """A family with many instances must not dominate the family-first mean
    the way it would dominate a naive flat per-instance mean."""
    import csv
    import research.component1.function2.evaluation.build_descriptive_aggregates as mod

    out_root = tmp_path / "f24_full_v1"
    (out_root / "metrics").mkdir(parents=True)
    rows = []
    # family A: 5 instances all with stability similarity 0.0 (gradcam)
    for i in range(5):
        rows.append({"gt_instance_id": f"A__inst{i:03d}", "source_family_id": "family_A",
                     "complexity_group": "Simple", "defect_class": "stain",
                     "target_layer": "model.model[15]", "method": "gradcam",
                     "status": "valid", "similarity": "0.0",
                     "matched_iou_vs_original": "1.0", "perturbation": "brightness_plus5pct", "reason": ""})
    # family B: 1 instance with stability similarity 1.0 (gradcam)
    rows.append({"gt_instance_id": "B__inst000", "source_family_id": "family_B",
                 "complexity_group": "Simple", "defect_class": "stain",
                 "target_layer": "model.model[15]", "method": "gradcam",
                 "status": "valid", "similarity": "1.0",
                 "matched_iou_vs_original": "1.0", "perturbation": "brightness_plus5pct", "reason": ""})

    stab_path = out_root / "metrics" / "stability.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with open(stab_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    monkeypatch.setattr(mod, "OUT_ROOT", out_root)
    result = mod.build()

    # naive per-instance mean would be (5*0.0 + 1*1.0)/6 = 0.1667
    # family-first mean = mean(mean(family_A)=0.0, mean(family_B)=1.0) = 0.5
    ff = result["stability_similarity_family_first_v1"]["gradcam"]
    assert ff["n_families"] == 2
    assert abs(ff["mean_of_family_means"] - 0.5) < 1e-9

    naive = result["stability_similarity_by_method_v1"]["gradcam"]
    assert naive["n"] == 6
    assert abs(naive["mean"] - (5 * 0.0 + 1.0) / 6) < 1e-9
    # the two aggregation lenses must disagree here -- that's the point
    assert abs(ff["mean_of_family_means"] - naive["mean"]) > 0.3
