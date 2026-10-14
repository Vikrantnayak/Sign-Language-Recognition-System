"""
Unit tests for handedness resolution, mapping, and landmark normalization conventions.
Verifies that MediaPipe handedness classifications are correctly mapped to anatomical labels
and that landmark normalization maintains strict single-mirroring parity between hands.
"""

import unittest
import numpy as np

from slr.utils.handedness import (
    resolve_handedness,
    get_handedness_and_normalization_flag,
)
from slr.utils.pre_process import pre_process_landmark
from slr.model.classifier import KeyPointClassifier
from slr.paths import load_labels, DEFAULT_KEYPOINT_PATH


class TestHandednessConventions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = KeyPointClassifier()
        cls.labels = load_labels()

    def test_resolve_handedness_mirrored_webcam(self):
        """
        Under the standard webcam convention (cv2.flip(img, 1)), the image is mirrored.
        MediaPipe Hands assumes a mirrored input for handedness output,
        so raw labels directly match the user's anatomical hand.
        """
        # User holds up Right hand in mirrored webcam -> MediaPipe reports 'Right'
        self.assertEqual(resolve_handedness("Right", is_mirrored=True), "Right")

        # User holds up Left hand in mirrored webcam -> MediaPipe reports 'Left'
        self.assertEqual(resolve_handedness("Left", is_mirrored=True), "Left")

    def test_resolve_handedness_unmirrored_photo(self):
        """
        For unmirrored images (e.g. rear-facing camera, static photos without cv2.flip):
        MediaPipe's output is inverted relative to the user's anatomical hand.
        """
        self.assertEqual(resolve_handedness("Right", is_mirrored=False), "Left")
        self.assertEqual(resolve_handedness("Left", is_mirrored=False), "Right")

    def test_get_handedness_and_normalization_flag(self):
        """
        Verifies that the normalization flag is_left_hand is set strictly for the
        user's anatomical Left hand under the mirrored webcam convention.
        """
        # Mirrored Right hand: anatomical is Right, is_left_hand flag is False
        hand_r, is_left_r = get_handedness_and_normalization_flag("Right", is_mirrored=True)
        self.assertEqual(hand_r, "Right")
        self.assertFalse(is_left_r, "Right hand must not be flagged for horizontal mirroring")

        # Mirrored Left hand: anatomical is Left, is_left_hand flag is True
        hand_l, is_left_l = get_handedness_and_normalization_flag("Left", is_mirrored=True)
        self.assertEqual(hand_l, "Left")
        self.assertTrue(is_left_l, "Left hand must be flagged for horizontal mirroring")

    def test_no_double_mirroring(self):
        """
        Verifies that normalization applies mirroring exactly once to the left hand
        and zero times to the right hand.
        """
        wrist = [320, 360]
        # Construct 21 distinct landmark instances
        right_landmarks = [[wrist[0], wrist[1]] for _ in range(21)]
        left_landmarks = [[wrist[0], wrist[1]] for _ in range(21)]

        # Right hand: thumb extended to the right (+dx = 60)
        right_landmarks[4] = [wrist[0] + 60, wrist[1] - 40]
        right_landmarks[8] = [wrist[0] + 5, wrist[1] - 100]

        # Left hand: thumb extended to the left (-dx = -60)
        left_landmarks[4] = [wrist[0] - 60, wrist[1] - 40]
        left_landmarks[8] = [wrist[0] - 5, wrist[1] - 100]

        feat_right = pre_process_landmark(right_landmarks, is_left_hand=False)
        feat_left = pre_process_landmark(left_landmarks, is_left_hand=True)

        # Thumb tip relative X is at index 4*2 = 8
        self.assertGreater(feat_right[8], 0.0, "Right hand thumb offset must remain positive")
        self.assertGreater(feat_left[8], 0.0, "Left hand thumb offset must be mirrored to positive")
        np.testing.assert_allclose(feat_right, feat_left, atol=1e-5)

    def test_mirrored_landmark_fixture_recognition_consistency(self):
        """
        Loads a ground-truth Class 10 ('L') sample from keypoint.csv.
        Reconstructs pixel landmarks for Right hand and mirrored Left hand.
        Verifies that both classify as 'L' when processed with their respective handedness flags.
        """
        # Read a verified Class 10 ('L') row from dataset
        with open(DEFAULT_KEYPOINT_PATH, "r", encoding="utf-8") as f:
            l_sample = None
            for line in f:
                parts = line.strip().split(",")
                if parts[0] == "10":
                    l_sample = [float(x) for x in parts[1:]]
                    break

        self.assertIsNotNone(l_sample, "Dataset must contain Class 10 sample")

        wrist = [320, 240]
        pts_r = []
        pts_l = []
        for i in range(21):
            dx = int(l_sample[i * 2] * 100)
            dy = int(l_sample[i * 2 + 1] * 100)
            pts_r.append([wrist[0] + dx, wrist[1] + dy])
            pts_l.append([wrist[0] - dx, wrist[1] + dy])  # Mirrored X

        # Predict Right Hand (is_left_hand=False)
        feat_r = pre_process_landmark(pts_r, is_left_hand=False)
        pred_r_id = self.classifier(feat_r)
        label_r = self.labels[pred_r_id] if pred_r_id < len(self.labels) else "Unknown"

        # Predict Left Hand (is_left_hand=True)
        feat_l = pre_process_landmark(pts_l, is_left_hand=True)
        pred_l_id = self.classifier(feat_l)
        label_l = self.labels[pred_l_id] if pred_l_id < len(self.labels) else "Unknown"

        self.assertEqual(label_r, "L", f"Right hand 'L' sample expected to be 'L', got {label_r}")
        self.assertEqual(label_l, "L", f"Left hand mirrored 'L' sample expected to be 'L', got {label_l}")
        self.assertEqual(pred_r_id, pred_l_id, "Both hands must yield the exact same class index (10)")


if __name__ == "__main__":
    unittest.main()
