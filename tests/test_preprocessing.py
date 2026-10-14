"""
Unit tests for landmark extraction, preprocessing, and hand normalization.
"""

import unittest
import numpy as np

from slr.paths import DEFAULT_KEYPOINT_PATH
from slr.utils.pre_process import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    calc_hand_scale,
    is_finger_extended,
    is_neutral_hand,
    calc_landmark_motion,
)


class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MockLandmarksObject:
    def __init__(self, points):
        self.landmark = [MockLandmark(p[0], p[1]) for p in points]


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # 21 synthetic landmarks (spread across hand)
        self.synthetic_points = []
        for i in range(21):
            self.synthetic_points.append((0.2 + (i % 5) * 0.1, 0.3 + (i // 5) * 0.1))
        self.mock_landmarks = MockLandmarksObject(self.synthetic_points)
        self.dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_calc_bounding_rect(self):
        brect = calc_bounding_rect(self.dummy_image, self.mock_landmarks)
        self.assertEqual(len(brect), 4)
        x1, y1, x2, y2 = brect
        self.assertLessEqual(x1, x2)
        self.assertLessEqual(y1, y2)
        self.assertGreaterEqual(x1, 0)
        self.assertLessEqual(x2, 640)

    def test_calc_landmark_list(self):
        landmark_list = calc_landmark_list(self.dummy_image, self.mock_landmarks)
        self.assertEqual(len(landmark_list), 21)
        for pt in landmark_list:
            self.assertEqual(len(pt), 2)
            self.assertTrue(0 <= pt[0] < 640)
            self.assertTrue(0 <= pt[1] < 480)

    def test_pre_process_landmark_normalization(self):
        landmark_list = calc_landmark_list(self.dummy_image, self.mock_landmarks)
        normalized = pre_process_landmark(landmark_list, is_left_hand=False)

        self.assertEqual(len(normalized), 42)
        # First two values correspond to wrist offset from wrist, so must be 0.0
        self.assertAlmostEqual(normalized[0], 0.0, places=6)
        self.assertAlmostEqual(normalized[1], 0.0, places=6)

        # Max absolute value must be 1.0 (since max_val was divided)
        max_abs = max(map(abs, normalized))
        self.assertAlmostEqual(max_abs, 1.0, places=6)

        # All values should be in range [-1.0, 1.0]
        for val in normalized:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)

    def test_left_right_hand_normalization(self):
        # Create a right hand
        right_hand_landmarks = [
            [300, 400],  # Wrist (0)
            [320, 380],  # Thumb 1
            [335, 360],  # Thumb 2
            [345, 340],  # Thumb 3
            [355, 320],  # Thumb 4
        ]
        # Append remaining 16 landmarks
        for i in range(5, 21):
            right_hand_landmarks.append([280 + (i % 4) * 15, 350 - (i // 4) * 20])

        # Mirror horizontally around the wrist X (wrist_x = 300)
        # If right hand has point (300 + dx, y), left hand has point (300 - dx, y)
        left_hand_landmarks = []
        for x, y in right_hand_landmarks:
            dx = x - 300
            left_hand_landmarks.append([300 - dx, y])

        # Preprocess right hand as right (is_left_hand=False)
        feat_right = pre_process_landmark(right_hand_landmarks, is_left_hand=False)

        # Preprocess left hand as left (is_left_hand=True)
        feat_left = pre_process_landmark(left_hand_landmarks, is_left_hand=True)

        self.assertEqual(len(feat_right), len(feat_left))
        # Both feature vectors must be identical within float precision
        np.testing.assert_allclose(
            feat_right,
            feat_left,
            atol=1e-5,
            err_msg="Left hand mirrored features must match right hand canonical features",
        )

    def test_regression_against_training_dataset_properties(self):
        """Verifies that preprocessing matches the mathematical properties of keypoint.csv."""
        with open(DEFAULT_KEYPOINT_PATH, "r", encoding="utf-8") as f:
            for _ in range(20):
                line = f.readline().strip()
                if not line:
                    break
                row = [float(x) for x in line.split(",")[1:]]
                self.assertEqual(len(row), 42)
                # Wrist offset is 0.0
                self.assertAlmostEqual(row[0], 0.0, places=5)
                self.assertAlmostEqual(row[1], 0.0, places=5)
                # Max absolute value in keypoint.csv is exactly 1.0
                max_abs = max(map(abs, row))
                self.assertAlmostEqual(max_abs, 1.0, places=5)

    def test_pre_process_landmark_zero_safety(self):
        # Edge case: all landmarks identical (e.g. at origin)
        zero_points = [(0.0, 0.0)] * 21
        zero_landmarks = MockLandmarksObject(zero_points)
        landmark_list = calc_landmark_list(self.dummy_image, zero_landmarks)

        try:
            normalized = pre_process_landmark(landmark_list)
            self.assertEqual(len(normalized), 42)
            self.assertEqual(normalized, [0.0] * 42)
        except ZeroDivisionError:
            self.fail("pre_process_landmark raised ZeroDivisionError on identical points")

    def test_hand_scale_and_finger_extension(self):
        landmarks = [[300, 400]] * 21
        # Set wrist at [300, 400] and middle MCP (9) at [300, 300] -> dist = 100
        landmarks[0] = [300, 400]
        landmarks[9] = [300, 300]
        scale = calc_hand_scale(landmarks)
        self.assertAlmostEqual(scale, 100.0, places=4)

        # Extended finger: index tip (8) at [300, 200] (dist=200), MCP (5) at [300, 320] (dist=80)
        landmarks[5] = [300, 320]
        landmarks[8] = [300, 200]
        self.assertTrue(is_finger_extended(landmarks, tip_idx=8, mcp_idx=5, ratio=1.2))

        # Curled finger: tip at [300, 350] (dist=50) vs MCP at [300, 320] (dist=80)
        landmarks[8] = [300, 350]
        self.assertFalse(is_finger_extended(landmarks, tip_idx=8, mcp_idx=5, ratio=1.2))

    def test_neutral_hand_detection(self):
        # 1. Open flat palm (all 5 fingers extended and spread)
        open_palm = [[320, 400]]  # 0: wrist
        open_palm += [[350, 370], [380, 350], [400, 330], [420, 310]]  # Thumb
        open_palm += [[300, 320], [290, 270], [285, 230], [280, 190]]  # Index
        open_palm += [[320, 310], [320, 260], [320, 210], [320, 170]]  # Middle
        open_palm += [[340, 320], [345, 270], [350, 230], [355, 190]]  # Ring
        open_palm += [[360, 330], [370, 290], [375, 260], [380, 220]]  # Pinky
        self.assertTrue(is_neutral_hand(open_palm), "Open flat palm should be identified as neutral posture")

        # 2. ASL 'B' (fingers closed together, thumb folded across palm)
        b_hand = [[320, 400]]
        b_hand += [[330, 380], [335, 360], [330, 340], [325, 330]]  # Thumb folded
        b_hand += [[310, 320], [308, 270], [306, 230], [305, 190]]  # Index
        b_hand += [[320, 310], [320, 260], [320, 210], [320, 170]]  # Middle
        b_hand += [[330, 320], [330, 270], [330, 230], [330, 190]]  # Ring
        b_hand += [[340, 330], [340, 290], [340, 260], [340, 220]]  # Pinky
        self.assertFalse(is_neutral_hand(b_hand), "ASL 'B' must not be misidentified as neutral posture")

        # 3. ASL 'F' (thumb and index touch)
        f_hand = [p.copy() for p in open_palm]
        f_hand[4] = [300, 250]  # Thumb touches index
        f_hand[8] = [300, 250]
        self.assertFalse(is_neutral_hand(f_hand), "ASL 'F' must not be misidentified as neutral posture")

        # 4. Fist
        fist = [[320, 400]]
        for i in range(1, 21):
            fist.append([320 + (i % 4) * 8, 380 - (i // 4) * 10])
        self.assertFalse(is_neutral_hand(fist), "Fist must not be identified as neutral posture")

        # 5. Robustness & edge cases: None, empty list, truncated list, zero coordinates
        self.assertFalse(is_neutral_hand(None), "None input must return False safely")
        self.assertFalse(is_neutral_hand([]), "Empty landmark list must return False safely")
        self.assertFalse(is_neutral_hand([[0, 0]] * 10), "Truncated landmark list (< 21) must return False safely")
        self.assertFalse(is_neutral_hand([[0, 0]] * 21), "All-zero collapsed landmarks must return False")
        self.assertFalse(is_neutral_hand([[100, 100]] * 21), "Identical coordinates must return False")

    def test_calc_landmark_motion(self):
        pts1 = [[300 + i * 2, 400 - i * 3] for i in range(21)]
        # No motion
        self.assertAlmostEqual(calc_landmark_motion(pts1, pts1, scale=100.0), 0.0, places=5)
        # None previous
        self.assertEqual(calc_landmark_motion(pts1, None, scale=100.0), 0.0)

        # Displaced by 20 pixels
        pts2 = [[p[0] + 20, p[1]] for p in pts1]
        motion = calc_landmark_motion(pts2, pts1, scale=100.0)
        self.assertAlmostEqual(motion, 0.20, places=4)


if __name__ == "__main__":
    unittest.main()
