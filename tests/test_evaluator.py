"""
Unit tests for RecognitionEvaluator and EvaluationRecord.
"""

import tempfile
import unittest
from pathlib import Path

from slr.utils.evaluator import RecognitionEvaluator, EvaluationRecord


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_eval.csv"
        self.evaluator = RecognitionEvaluator(log_path=self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluation_record_correct(self):
        rec = EvaluationRecord(
            expected="A",
            predicted="A",
            confidence=0.88,
            handedness="Right",
            state="RECOGNIZED",
        )
        self.assertTrue(rec.is_correct)
        self.assertFalse(rec.is_uncertain)
        self.assertEqual(rec.expected, "A")
        self.assertEqual(rec.predicted, "A")
        self.assertEqual(rec.confidence, 0.88)

    def test_evaluation_record_incorrect(self):
        rec = EvaluationRecord(
            expected="A",
            predicted="B",
            confidence=0.75,
            handedness="Right",
            state="RECOGNIZED",
        )
        self.assertFalse(rec.is_correct)
        self.assertFalse(rec.is_uncertain)

    def test_evaluation_record_uncertain(self):
        rec = EvaluationRecord(
            expected="A",
            predicted=None,
            confidence=0.35,
            handedness="Left",
            state="UNCERTAIN",
        )
        self.assertFalse(rec.is_correct)
        self.assertTrue(rec.is_uncertain)
        self.assertEqual(rec.predicted, "NONE")

    def test_recognition_evaluator_logging_and_summary(self):
        # Record 4 trials: 2 correct, 1 uncertain, 1 incorrect
        self.evaluator.record_sample("A", "A", 0.90, "Right", "RECOGNIZED")
        self.evaluator.record_sample("B", "B", 0.85, "Left", "RECOGNIZED")
        self.evaluator.record_sample("C", None, 0.20, "Right", "UNCERTAIN")
        self.evaluator.record_sample("D", "E", 0.60, "Right", "RECOGNIZED")

        summary = self.evaluator.compute_summary()
        self.assertEqual(summary["total_samples"], 4)
        self.assertEqual(summary["correct_count"], 2)
        self.assertEqual(summary["uncertain_count"], 1)
        self.assertEqual(summary["incorrect_count"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.50, places=3)
        self.assertAlmostEqual(summary["uncertain_rate"], 0.25, places=3)

    def test_csv_export_and_append(self):
        self.assertFalse(self.csv_path.exists())

        rec1 = self.evaluator.record_sample("L", "L", 0.95, "Right", "RECOGNIZED", save_csv=True)
        self.assertTrue(self.csv_path.exists())

        rec2 = self.evaluator.record_sample("M", "M", 0.70, "Left", "RECOGNIZED", save_csv=True)

        # Read back lines
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Header + 2 rows = 3 lines
        self.assertEqual(len(lines), 3)
        self.assertIn("expected_letter", lines[0])
        self.assertIn("L,L,0.95,Right", lines[1])
        self.assertIn("M,M,0.7,Left", lines[2])


if __name__ == "__main__":
    unittest.main()
