"""Regeneration checks against local saved outputs; normal frontend tests need none.

Run: python -B -m unittest discover -s frontend/scripts -p test_export_n2_study.py
"""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from export_n2_study import build_snapshot, DEFAULT_OUTPUT, DEFAULT_SOURCE, FILES


@unittest.skipUnless(all((DEFAULT_SOURCE / file).is_file() for file in FILES),
                     "Local saved experiments are needed only to test regeneration")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="n2-export-")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "sources"
        for file in FILES:
            target = self.source / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEFAULT_SOURCE / file, target)

    def alter(self, name, change):
        path = next(self.source / file for file in FILES if Path(file).name == name)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        headers = list(rows[0])
        change(rows)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader(); writer.writerows(rows)

    def test_snapshot_matches_and_runs_without_site_packages(self):
        actual = build_snapshot(DEFAULT_SOURCE)
        self.assertEqual(actual, json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8")))
        self.assertEqual(len(actual["conditions"][2]["evidence"]["F"]["groups"]), 4)
        command = [sys.executable, "-S", "-B", str(Path(__file__).with_name("export_n2_study.py")), "--check"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_source_fails_without_overwriting_output(self):
        (self.source / FILES[0]).unlink()
        output = Path(self.temp.name) / "snapshot.json"
        output.write_text("keep existing reviewed snapshot", encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("export_n2_study.py")),
                                 "--source-dir", str(self.source), "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("N2 export failed", result.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep existing reviewed snapshot")

    def test_loss_mismatch_rejected(self):
        self.alter("n2_paired_test_prediction_losses.csv", lambda rows: rows[0].update(sq_error="0.99"))
        with self.assertRaisesRegex(ValueError, "squared prediction loss"):
            build_snapshot(self.source)

    def test_quantile_mismatch_rejected(self):
        self.alter("n2_evidence_loss_quantile_summary.csv", lambda rows: rows[0].update(n_snapshots="1"))
        with self.assertRaisesRegex(ValueError, "quantile"):
            build_snapshot(self.source)

    def test_constant_completeness_violation_rejected(self):
        self.alter("n2_prediction_evidence_and_losses.csv", lambda rows: rows[0].update(C_field_completeness_pct="85"))
        with self.assertRaisesRegex(ValueError, "constant C"):
            build_snapshot(self.source)

    def test_machine_fold_crossover_rejected(self):
        def mutate(rows):
            row = rows[0]
            peer = next(other for other in rows[1:] if other["history_condition"] == row["history_condition"] and other["machineID"] == row["machineID"])
            peer["machine_group_fold"] = str(int(row["machine_group_fold"]) % 5 + 1)
        self.alter("n2_risk_vs_evidence_cv_predictions.csv", mutate)
        with self.assertRaisesRegex(ValueError, "crosses OOF folds"):
            build_snapshot(self.source)

    def test_incremental_metrics_mismatch_rejected(self):
        self.alter("n2_risk_vs_evidence_cv_metrics.csv", lambda rows: rows[0].update(mae="0.9"))
        with self.assertRaisesRegex(ValueError, "incremental mae"):
            build_snapshot(self.source)

    def test_check_fails_for_changed_snapshot(self):
        output = Path(self.temp.name) / "snapshot.json"
        output.write_text("{}\n", encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("export_n2_study.py")),
                                 "--source-dir", str(self.source), "--output", str(output), "--check"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Snapshot differs", result.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
