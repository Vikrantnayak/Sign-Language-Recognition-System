"""
Unit tests for PredictionSmoother temporal consensus voting and state transitions.
"""

import unittest
from slr.utils.smoother import PredictionSmoother


class TestPredictionSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = PredictionSmoother(
            window_size=8,
            min_consensus_ratio=0.60,
            min_confidence_threshold=0.50,
            no_hand_patience=3,
        )

    def test_stable_prediction_on_consistent_input(self):
        for _ in range(6):
            pred, conf, state = self.smoother.update("A", 0.90, has_hand=True)

        self.assertEqual(pred, "A")
        self.assertEqual(state, "RECOGNIZED")
        self.assertGreaterEqual(conf, 0.80)

    def test_smoothing_filters_single_frame_flutter(self):
        # A, A, A, B (glitch), A, A
        sequence = ["A", "A", "A", "B", "A", "A"]
        last_pred = None
        last_state = None
        for letter in sequence:
            last_pred, _, last_state = self.smoother.update(letter, 0.85, has_hand=True)

        self.assertEqual(last_pred, "A", "Single glitch B should be smoothed out to A")
        self.assertEqual(last_state, "RECOGNIZED")

    def test_uncertain_state_on_ambiguous_flutter(self):
        # Evenly split between A and B
        sequence = ["A", "B", "A", "B", "A", "B"]
        last_pred = None
        last_state = None
        for letter in sequence:
            last_pred, _, last_state = self.smoother.update(letter, 0.60, has_hand=True)

        self.assertIsNone(last_pred, "Ambiguous alternating input should not emit a letter")
        self.assertEqual(last_state, "UNCERTAIN")

    def test_neutral_state_handling(self):
        # Resting / Neutral posture
        for _ in range(6):
            pred, _, state = self.smoother.update("Neutral", 0.70, has_hand=True)

        self.assertIsNone(pred)
        self.assertEqual(state, "NEUTRAL")

    def test_no_hand_patience_and_transition(self):
        # First stabilize on A
        for _ in range(6):
            self.smoother.update("A", 0.90, has_hand=True)

        # 1 frame no hand: should not immediately wipe out
        pred1, _, state1 = self.smoother.update(None, 0.0, has_hand=False)
        self.assertEqual(state1, "NO_HAND")

        # 2nd and 3rd frame no hand: should clear buffer
        self.smoother.update(None, 0.0, has_hand=False)
        pred3, _, state3 = self.smoother.update(None, 0.0, has_hand=False)
        self.assertIsNone(pred3)
        self.assertEqual(state3, "NO_HAND")
        self.assertEqual(len(self.smoother.history), 0, "Buffer should be empty after patience expires")

    def test_dynamic_gesture_immediate_priority(self):
        # Hand was signing A
        for _ in range(4):
            self.smoother.update("A", 0.80, has_hand=True)

        # Dynamic J completed
        pred, conf, state = self.smoother.update("J", 0.92, has_hand=True, is_dynamic=True)
        self.assertEqual(pred, "J", "Dynamic J should immediately override static consensus")
        self.assertEqual(state, "RECOGNIZED")
        self.assertEqual(conf, 0.92)

    def test_transition_motion_handling(self):
        # Even if raw prediction happens to guess 'W' during transition motion:
        pred, conf, state = self.smoother.update("W", 0.75, has_hand=True, is_transitioning=True)
        self.assertIsNone(pred, "Transition motion should suppress static letter display")
        self.assertEqual(state, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()

