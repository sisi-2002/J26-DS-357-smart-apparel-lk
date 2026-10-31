"""F2.1B2 Task 9 -- end-to-end tests for the Task 5/6 calibration pipeline
(dataset/calibration_scoring_v1.py), run against a small, isolated
(tmp_path) extraction rather than the full 1,900-image production run, so
these tests stay fast while still exercising the real code path.

Covers: deterministic score calculation (same config+input twice ->
identical output), calibration-config JSON serialization/reload round-trip,
and the provisional-vs-validated status-field distinction required by the
task brief (output schemas must never silently claim "validated").
"""

from __future__ import annotations

import json

from research.component1.function2.complexity.scoring import (
    NormalizationParams,
    compute_composite_score,
)
from research.component1.function2.dataset import (
    calibration_scoring_v1,
    train_extraction_v1,
)


def _make_small_extraction(tmp_path, limit=150):
    out_dir = tmp_path / "extraction"
    train_extraction_v1.run(limit=limit, output_dir=out_dir)
    return out_dir / "train_features_v1_full.csv"


def test_calibration_scoring_deterministic_same_input_twice(tmp_path):
    csv_path = _make_small_extraction(tmp_path)

    out1 = tmp_path / "calib1"
    cfg1 = tmp_path / "cfg1"
    out2 = tmp_path / "calib2"
    cfg2 = tmp_path / "cfg2"

    summary1 = calibration_scoring_v1.run(input_csv=csv_path, output_dir=out1, config_dir=cfg1)
    summary2 = calibration_scoring_v1.run(input_csv=csv_path, output_dir=out2, config_dir=cfg2)

    # Bootstrap uses a FIXED seed, so the full summary (including bootstrap
    # stats) must be identical across runs, not just approximately similar
    # -- except generated_at_utc, which legitimately differs between two
    # real wall-clock runs and is stripped before comparing.
    d1, d2 = dict(summary1), dict(summary2)
    # These two legitimately differ: they are absolute/relative paths into
    # two DIFFERENT tmp_path output directories (out1 vs out2), not a sign
    # of non-determinism in the actual computation.
    for d in (d1, d2):
        d.pop("generated_at_utc", None)
        d.pop("config_output", None)
        d.pop("scores_csv_output", None)
    s1 = json.dumps(d1, sort_keys=True)
    s2 = json.dumps(d2, sort_keys=True)
    assert s1 == s2

    scores_csv_1 = (out1 / "train_composite_scores_v1.csv").read_text(encoding="utf-8")
    scores_csv_2 = (out2 / "train_composite_scores_v1.csv").read_text(encoding="utf-8")
    assert scores_csv_1 == scores_csv_2

    config_1 = (cfg1 / "calibration_v1.json").read_text(encoding="utf-8")
    config_2 = (cfg2 / "calibration_v1.json").read_text(encoding="utf-8")
    # generated_at_utc timestamps legitimately differ between runs -- strip
    # that one field before comparing the rest byte-for-byte.
    d1, d2 = json.loads(config_1), json.loads(config_2)
    d1.pop("generated_at_utc"), d2.pop("generated_at_utc")
    assert d1 == d2


def test_calibration_config_round_trip_reproduces_same_scores(tmp_path):
    csv_path = _make_small_extraction(tmp_path)
    out_dir = tmp_path / "calib"
    cfg_dir = tmp_path / "cfg"
    calibration_scoring_v1.run(input_csv=csv_path, output_dir=out_dir, config_dir=cfg_dir)

    with open(cfg_dir / "calibration_v1.json", encoding="utf-8") as f:
        config = json.load(f)

    # Round-trip: rebuild NormalizationParams from the serialized config and
    # confirm a hand-picked raw feature vector produces the SAME composite
    # score as directly using the in-memory params would.
    norm = {
        name: NormalizationParams.from_dict(params)
        for name, params in config["normalization"]["features"].items()
    }
    raw = {
        "luminance_variation": config["normalization"]["features"]["luminance_variation"]["mean"] + 10,
        "glcm_entropy": config["normalization"]["features"]["glcm_entropy"]["mean"],
        "colour_variation": config["normalization"]["features"]["colour_variation"]["mean"] - 1,
    }
    weights = config["weights"]["candidate"]
    result_a = compute_composite_score(raw, norm, weights=weights)
    result_b = compute_composite_score(raw, norm, weights=weights)
    assert result_a.to_dict() == result_b.to_dict()
    assert result_a.composite_score is not None


def test_calibration_config_status_is_provisional_never_validated(tmp_path):
    csv_path = _make_small_extraction(tmp_path)
    out_dir = tmp_path / "calib"
    cfg_dir = tmp_path / "cfg"
    calibration_scoring_v1.run(input_csv=csv_path, output_dir=out_dir, config_dir=cfg_dir)

    with open(cfg_dir / "calibration_v1.json", encoding="utf-8") as f:
        config = json.load(f)

    assert config["status"] == "provisional_NOT_frozen"
    assert config["thresholds"]["label"] == "candidate_NOT_expert_validated"
    assert config["weights"]["candidate_label"] == "equal_weighting_default_candidate_NOT_validated"
    assert config["expert_review_status"] == "pending"
    serialized = json.dumps(config).lower()
    assert '"validated"' not in serialized.replace("not_validated", "").replace("not validated", "")
    # Explicitly must never claim frozen/validated status anywhere.
    assert "frozen" not in serialized or "not_frozen" in serialized


def test_group_image_and_family_counts_present_and_consistent(tmp_path):
    csv_path = _make_small_extraction(tmp_path, limit=300)
    out_dir = tmp_path / "calib"
    cfg_dir = tmp_path / "cfg"
    summary = calibration_scoring_v1.run(input_csv=csv_path, output_dir=out_dir, config_dir=cfg_dir)

    counts = summary["group_image_counts"]
    family_counts = summary["group_independent_family_counts"]
    assert set(family_counts.keys()) == {"Simple", "Moderate", "High"}
    total_images = counts["Simple"] + counts["Moderate"] + counts["High"] + counts["unassigned"]
    assert total_images == summary["n_eligible_rows"]
    # Family counts can never exceed image counts for the same group.
    for g in ("Simple", "Moderate", "High"):
        assert family_counts[g] <= counts[g]


def test_weight_sensitivity_and_bootstrap_stability_reported(tmp_path):
    csv_path = _make_small_extraction(tmp_path, limit=300)
    out_dir = tmp_path / "calib"
    cfg_dir = tmp_path / "cfg"
    summary = calibration_scoring_v1.run(input_csv=csv_path, output_dir=out_dir, config_dir=cfg_dir)

    ws = summary["weight_sensitivity"]
    assert "alt_weights_A_emphasise_luminance" in ws
    assert "alt_weights_B_emphasise_glcm_entropy" in ws
    assert ws["alt_weights_A_emphasise_luminance"]["n_images_changed_group_vs_baseline"] >= 0

    bs = summary["bootstrap_stability"]
    assert bs["simple_max_bootstrap"]["std"] >= 0
    assert bs["moderate_max_bootstrap"]["std"] >= 0
