"""F2.6-QA -- focused validation tests for the v2 notebook and its real
kernel execution.

These tests do NOT touch E02, do not regenerate any XAI output, and do not
modify any protected F2.1-F2.6-v1 file. They only read:
  - the v1 and v2 notebook JSON files
  - the real kernel-executed v2 notebook produced by F2.6-QA
  - a handful of authoritative F2.4/F2.5 result files, to cross-check a
    sample of the executed notebook's printed numbers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

NB_DIR = REPO_ROOT / "research/component1/function2/notebooks"
V1 = NB_DIR / "pp1_xai_evaluation_final_v1.ipynb"
V2 = NB_DIR / "pp1_xai_evaluation_final_v2.ipynb"
EXECUTED = (
    REPO_ROOT
    / "research/component1/function2/results/pp1_research_package/f26_qa_v1"
    / "notebook_validation/pp1_xai_evaluation_final_v2_executed.ipynb"
)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _section_headers(nb: dict) -> list[tuple[int, str]]:
    out = []
    for i, c in enumerate(nb["cells"]):
        src = _cell_source(c)
        if c["cell_type"] == "markdown" and src.strip().startswith("## Section"):
            out.append((i, src.splitlines()[0]))
    return out


class TestV1Untouched:
    def test_v1_exists_and_parses(self):
        assert V1.is_file()
        nb = _load(V1)
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) == 129

    def test_v1_has_no_cell_ids(self):
        # Documents the starting condition this QA stage found and fixed in
        # v2 -- if this ever flips (v1 edited to add ids), that is a v1
        # mutation, which must never happen.
        nb = _load(V1)
        assert all("id" not in c for c in nb["cells"])

    def test_v1_still_contains_the_original_pointing_accuracy_wording(self):
        # Confirms v1 is untouched: the wording this QA stage flagged and
        # corrected in v2 must still be present, verbatim, in v1.
        nb = _load(V1)
        full = "\n".join(_cell_source(c) for c in nb["cells"])
        assert "thresholded top activation region" in full


class TestV2Structure:
    def test_v2_exists_and_parses(self):
        assert V2.is_file()
        nb = _load(V2)
        assert nb["nbformat"] == 4
        assert nb["nbformat_minor"] >= 5
        assert len(nb["cells"]) == 129

    def test_v2_all_cells_have_unique_valid_ids(self):
        nb = _load(V2)
        ids = [c.get("id") for c in nb["cells"]]
        assert all(isinstance(i, str) and i for i in ids), "every cell must have a non-empty id"
        assert len(set(ids)) == len(ids), "cell ids must be unique"

    def test_v2_section_3f_precedes_section_4(self):
        nb = _load(V2)
        headers = _section_headers(nb)
        idx_3f = next(i for i, h in headers if "Section 3F" in h)
        idx_4 = next(i for i, h in headers if h.strip().startswith("## Section 4"))
        assert idx_3f < idx_4

    def test_v2_no_stale_not_implemented_yet_in_sections_4_to_9(self):
        nb = _load(V2)
        headers = _section_headers(nb)
        start = next(i for i, h in headers if h.strip().startswith("## Section 4"))
        end = next(i for i, h in headers if h.strip().startswith("## Section 10"))
        for i in range(start, end):
            assert "Not Implemented Yet" not in _cell_source(nb["cells"][i])

    def test_v2_pointing_accuracy_wording_corrected(self):
        nb = _load(V2)
        full = "\n".join(_cell_source(c) for c in nb["cells"])
        assert "thresholded top activation region" not in full
        assert "argmax" in full.lower() or "max-activation" in full.lower()

    def test_v2_preserves_all_v1_code_cell_logic_except_documented_moves(self):
        # Every v1 code cell's source must still exist, verbatim, somewhere
        # in v2 (cells may have moved position for Section 3F, but no code
        # cell's logic was altered -- only markdown headers/prose changed).
        v1 = _load(V1)
        v2 = _load(V2)
        v1_code_sources = {_cell_source(c) for c in v1["cells"] if c["cell_type"] == "code"}
        v2_code_sources = {_cell_source(c) for c in v2["cells"] if c["cell_type"] == "code"}
        assert v1_code_sources == v2_code_sources


class TestV1EqualsV2ExceptDocumentedCells:
    def test_only_the_documented_markdown_cells_differ(self):
        """Every v2 cell is either (a) identical in source to some v1 cell,
        or (b) one of the specific markdown cells this QA stage documents
        as intentionally changed. No code cell source differs."""
        v1 = _load(V1)
        v2 = _load(V2)
        v1_sources = [_cell_source(c) for c in v1["cells"]]
        v2_sources = [_cell_source(c) for c in v2["cells"]]
        v1_multiset = {}
        for s in v1_sources:
            v1_multiset[s] = v1_multiset.get(s, 0) + 1
        unmatched_v2 = []
        for s in v2_sources:
            if v1_multiset.get(s, 0) > 0:
                v1_multiset[s] -= 1
            else:
                unmatched_v2.append(s)
        # Exactly the 9 markdown cells this stage intentionally rewrote
        # (cell 0 intro, cell 1 section 1.4, 6 stale section headers, 1
        # pointing-accuracy wording fix) should be "unmatched" against v1.
        assert len(unmatched_v2) == 9, (
            f"expected exactly 9 intentionally-changed cells, found {len(unmatched_v2)}"
        )


@pytest.mark.skipif(not EXECUTED.is_file(), reason="executed notebook not present in this environment")
class TestRealKernelExecution:
    def test_executed_notebook_all_49_code_cells_ran_with_no_errors(self):
        nb = _load(EXECUTED)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == 49
        exec_counts = [c.get("execution_count") for c in code_cells]
        assert exec_counts == list(range(1, 50)), "cells must run once, in order, from a fresh kernel"
        for c in code_cells:
            for out in c.get("outputs", []):
                assert out.get("output_type") != "error", f"cell errored: {out}"

    def test_executed_notebook_has_unique_valid_cell_ids(self):
        nb = _load(EXECUTED)
        ids = [c.get("id") for c in nb["cells"]]
        assert len(set(ids)) == len(ids)

    def test_executed_figures_are_non_trivial(self):
        nb = _load(EXECUTED)
        n_figs = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            for out in c.get("outputs", []):
                data = out.get("data", {})
                if "image/png" in data:
                    n_figs += 1
                    assert len(data["image/png"]) > 1000, "figure output looks empty/truncated"
        assert n_figs >= 10

    def _stream_text(self, nb: dict) -> str:
        parts = []
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            for out in c.get("outputs", []):
                if out.get("output_type") == "stream":
                    parts.append("".join(out.get("text", [])))
        return "\n".join(parts)

    def test_pointing_accuracy_numbers_match_authoritative_file(self):
        nb = _load(EXECUTED)
        text = self._stream_text(nb)
        mco_path = (
            REPO_ROOT
            / "research/component1/function2/results/complexity_xai_analysis/f25_pp1_v1"
            / "tables/method_comparison_overall.json"
        )
        with open(mco_path, encoding="utf-8") as f:
            mco = json.load(f)
        loc = mco["localization"]
        gc_rate = loc["gradcam"]["pointing_hit_rate"]["rate"]
        gcpp_rate = loc["gradcampp"]["pointing_hit_rate"]["rate"]
        assert f"{gc_rate:.3f}" in text
        assert f"{gcpp_rate:.3f}" in text

    def test_stability_numbers_match_authoritative_file(self):
        nb = _load(EXECUTED)
        text = self._stream_text(nb)
        mco_path = (
            REPO_ROOT
            / "research/component1/function2/results/complexity_xai_analysis/f25_pp1_v1"
            / "tables/method_comparison_overall.json"
        )
        with open(mco_path, encoding="utf-8") as f:
            mco = json.load(f)
        s1 = mco["stability_v1"]
        gc_v1 = s1["gradcam"]["mean_similarity_overall"]["mean"]
        gcpp_v1 = s1["gradcampp"]["mean_similarity_overall"]["mean"]
        assert f"{gc_v1:.4f}" in text
        assert f"{gcpp_v1:.4f}" in text

    def test_dataset_split_counts_match_authoritative_file(self):
        nb = _load(EXECUTED)
        text = self._stream_text(nb)
        # DS-01 split counts, independently re-verified in Task 1 of this
        # QA stage against f22a_task1_2_valid_audit_report.json.
        assert "1900" in text or "1,900" in text
        assert "345" in text
        assert "314" in text
