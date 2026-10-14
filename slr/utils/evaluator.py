"""
Evaluation and benchmarking utility for ASL Fingerspelling Recognition.
Records evaluation trials (expected letter, predicted letter, confidence, handedness, correctness),
computes performance metrics, and exports results to CSV without storing video or images.
"""

import csv
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from slr.paths import PROJECT_ROOT


class EvaluationRecord:
    """Represents a single evaluated sign sample."""

    def __init__(
        self,
        expected: str,
        predicted: Optional[str],
        confidence: float,
        handedness: str,
        state: str,
        timestamp: Optional[str] = None,
    ):
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.expected = expected.upper().strip()
        self.predicted = predicted.upper().strip() if predicted else "NONE"
        self.confidence = float(confidence)
        self.handedness = handedness
        self.state = state
        self.is_correct = bool((self.predicted == self.expected) and (self.state == "RECOGNIZED"))
        self.is_uncertain = bool(
            (self.state == "UNCERTAIN") or (self.predicted in ["NONE", "UNCERTAIN", "NEUTRAL"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "expected_letter": self.expected,
            "predicted_letter": self.predicted,
            "confidence": round(self.confidence, 4),
            "handedness": self.handedness,
            "state": self.state,
            "is_correct": self.is_correct,
            "is_uncertain": self.is_uncertain,
        }


class RecognitionEvaluator:
    """
    Manages evaluation trial recording and CSV logging.
    """

    DEFAULT_CSV_PATH = PROJECT_ROOT / "eval_results.csv"

    def __init__(self, log_path: Optional[Union[str, Path]] = None):
        self.log_path = Path(log_path) if log_path else self.DEFAULT_CSV_PATH
        self.records: List[EvaluationRecord] = []

    def record_sample(
        self,
        expected: str,
        predicted: Optional[str],
        confidence: float,
        handedness: str,
        state: str,
        save_csv: bool = True,
    ) -> EvaluationRecord:
        """
        Records a single trial and optionally appends it to the CSV log.
        """
        record = EvaluationRecord(
            expected=expected,
            predicted=predicted,
            confidence=confidence,
            handedness=handedness,
            state=state,
        )
        self.records.append(record)

        if save_csv and self.log_path:
            self.append_to_csv(record, self.log_path)

        return record

    @staticmethod
    def append_to_csv(record: EvaluationRecord, filepath: Path) -> None:
        """Appends an evaluation record to a CSV file."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_exists = target.exists()

        fieldnames = [
            "timestamp",
            "expected_letter",
            "predicted_letter",
            "confidence",
            "handedness",
            "state",
            "is_correct",
            "is_uncertain",
        ]

        with open(target, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.to_dict())

    def compute_summary(self) -> Dict[str, Any]:
        """
        Computes summary evaluation metrics over the recorded trials.
        """
        if not self.records:
            return {
                "total_samples": 0,
                "correct_count": 0,
                "uncertain_count": 0,
                "incorrect_count": 0,
                "accuracy": 0.0,
                "uncertain_rate": 0.0,
                "average_confidence": 0.0,
            }

        total = len(self.records)
        correct = sum(1 for r in self.records if r.is_correct)
        uncertain = sum(1 for r in self.records if r.is_uncertain)
        incorrect = total - correct - uncertain
        avg_conf = sum(r.confidence for r in self.records) / total

        return {
            "total_samples": total,
            "correct_count": correct,
            "uncertain_count": uncertain,
            "incorrect_count": incorrect,
            "accuracy": round(correct / total, 4),
            "uncertain_rate": round(uncertain / total, 4),
            "average_confidence": round(avg_conf, 4),
        }

    def clear(self) -> None:
        """Clears in-memory records."""
        self.records.clear()
