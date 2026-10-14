"""
Unit tests for dynamic sequence-based recognition of ASL letters J and Z.
Tests handshape preconditions, trajectory validation, invalid motion rejection, and state machine.
"""

import unittest
import numpy as np

from slr.utils.dynamic_gesture import (
    is_j_handshape_precondition,
    is_z_handshape_precondition,
    verify_j_trajectory,
    verify_z_trajectory,
    DynamicGestureRecognizer,
)


def create_synthetic_hand(
    pinky_up: bool = False,
    index_up: bool = False,
    middle_up: bool = False,
    ring_up: bool = False,
    wrist_xy=(320, 360),
) -> list:
    """Helper creating a 21-point landmark list with specified finger extension states."""
    pts = [[wrist_xy[0], wrist_xy[1]]]  # Wrist (0)

    # Thumb (1-4)
    pts.extend([[wrist_xy[0] + 20, wrist_xy[1] - 20 * i] for i in range(1, 5)])

    # Index (5-8)
    ext_idx = 100 if index_up else 35
    pts.append([wrist_xy[0] - 20, wrist_xy[1] - 30])  # 5: MCP
    pts.append([wrist_xy[0] - 22, wrist_xy[1] - 50])  # 6: PIP
    pts.append([wrist_xy[0] - 23, wrist_xy[1] - 70])  # 7: DIP
    pts.append([wrist_xy[0] - 24, wrist_xy[1] - ext_idx])  # 8: TIP

    # Middle (9-12)
    ext_mid = 110 if middle_up else 35
    pts.append([wrist_xy[0] - 5, wrist_xy[1] - 35])  # 9: MCP
    pts.append([wrist_xy[0] - 5, wrist_xy[1] - 55])  # 10: PIP
    pts.append([wrist_xy[0] - 5, wrist_xy[1] - 75])  # 11: DIP
    pts.append([wrist_xy[0] - 5, wrist_xy[1] - ext_mid])  # 12: TIP

    # Ring (13-16)
    ext_rng = 100 if ring_up else 30
    pts.append([wrist_xy[0] + 12, wrist_xy[1] - 30])  # 13: MCP
    pts.append([wrist_xy[0] + 14, wrist_xy[1] - 50])  # 14: PIP
    pts.append([wrist_xy[0] + 15, wrist_xy[1] - 70])  # 15: DIP
    pts.append([wrist_xy[0] + 16, wrist_xy[1] - ext_rng])  # 16: TIP

    # Pinky (17-20)
    ext_pnk = 85 if pinky_up else 25
    pts.append([wrist_xy[0] + 28, wrist_xy[1] - 25])  # 17: MCP
    pts.append([wrist_xy[0] + 30, wrist_xy[1] - 45])  # 18: PIP
    pts.append([wrist_xy[0] + 32, wrist_xy[1] - 65])  # 19: DIP
    pts.append([wrist_xy[0] + 34, wrist_xy[1] - ext_pnk])  # 20: TIP

    return pts


class TestDynamicGestures(unittest.TestCase):
    def test_j_handshape_precondition(self):
        # I pose: only pinky up
        i_hand = create_synthetic_hand(pinky_up=True, index_up=False, middle_up=False, ring_up=False)
        self.assertTrue(is_j_handshape_precondition(i_hand))

        # Open palm (all fingers up): should fail J precondition
        open_hand = create_synthetic_hand(pinky_up=True, index_up=True, middle_up=True, ring_up=True)
        self.assertFalse(is_j_handshape_precondition(open_hand))

        # Fist (no fingers up): should fail
        fist = create_synthetic_hand(pinky_up=False, index_up=False, middle_up=False, ring_up=False)
        self.assertFalse(is_j_handshape_precondition(fist))

    def test_z_handshape_precondition(self):
        # 1 / D pose: only index up
        d_hand = create_synthetic_hand(index_up=True, pinky_up=False, middle_up=False, ring_up=False)
        self.assertTrue(is_z_handshape_precondition(d_hand))

        # Open palm: should fail Z precondition
        open_hand = create_synthetic_hand(index_up=True, middle_up=True, ring_up=True, pinky_up=True)
        self.assertFalse(is_z_handshape_precondition(open_hand))

        # I hand: should fail Z precondition
        i_hand = create_synthetic_hand(pinky_up=True, index_up=False)
        self.assertFalse(is_z_handshape_precondition(i_hand))

    def test_valid_j_trajectory(self):
        # Synthetic J: Drop down (increasing y), curve at bottom, hook up
        t = np.linspace(0, 1, 15)
        # y drops from 0 to 60, then rises to 35
        y = np.concatenate([np.linspace(0, 60, 10), np.linspace(60, 35, 5)])
        # x hooks slightly
        x = np.concatenate([np.linspace(0, 5, 10), np.linspace(5, 25, 5)])
        j_pts = list(zip(x, y))

        is_valid, conf = verify_j_trajectory(j_pts, hand_scale=100.0)
        self.assertTrue(is_valid, "Expected synthetic J trajectory to be valid")
        self.assertGreaterEqual(conf, 0.70)

    def test_invalid_j_pure_vertical_line(self):
        # Straight vertical drop without bottom curve and without upward hook
        y = np.linspace(0, 80, 15)
        x = np.zeros(15)
        pts = list(zip(x, y))

        is_valid, _ = verify_j_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Pure vertical drop without upward hook must be rejected")

    def test_invalid_j_horizontal_movement(self):
        # Horizontal waving motion
        x = np.linspace(0, 80, 15)
        y = np.zeros(15)
        pts = list(zip(x, y))

        is_valid, _ = verify_j_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Horizontal movement must not trigger J")

    def test_valid_z_trajectory(self):
        # Synthetic Z: Segment 1 (Right), Segment 2 (Down-Left), Segment 3 (Right)
        # Segment 1: x: 0 -> 40, y: 0 -> 5
        s1_x, s1_y = np.linspace(0, 40, 6), np.linspace(0, 5, 6)
        # Segment 2: x: 40 -> 5, y: 5 -> 45 (diagonal down-left)
        s2_x, s2_y = np.linspace(40, 5, 7), np.linspace(5, 45, 7)
        # Segment 3: x: 5 -> 35, y: 45 -> 48 (right)
        s3_x, s3_y = np.linspace(5, 35, 6), np.linspace(45, 48, 6)

        x = np.concatenate([s1_x, s2_x, s3_x])
        y = np.concatenate([s1_y, s2_y, s3_y])
        z_pts = list(zip(x, y))

        is_valid, conf = verify_z_trajectory(z_pts, hand_scale=100.0)
        self.assertTrue(is_valid, "Expected synthetic Z trajectory to be valid")
        self.assertGreaterEqual(conf, 0.75)

    def test_invalid_z_pure_horizontal_line(self):
        # Waving hand left-to-right
        x = np.linspace(0, 100, 18)
        y = np.zeros(18)
        pts = list(zip(x, y))

        is_valid, _ = verify_z_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Pure horizontal motion without zigzag must not trigger Z")

    def test_invalid_z_circular_motion(self):
        # Circle / arc
        theta = np.linspace(0, 2 * np.pi, 20)
        x = 30 * np.cos(theta)
        y = 30 * np.sin(theta)
        pts = list(zip(x, y))

        is_valid, _ = verify_z_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Circular motion must not trigger Z")

    def test_dynamic_recognizer_j_sequence(self):
        recognizer = DynamicGestureRecognizer(buffer_size=25, cooldown_frames=10)
        i_hand_base = create_synthetic_hand(pinky_up=True)

        recognized_gesture = None
        # Feed J trajectory frames
        for i in range(15):
            pts = [p.copy() for p in i_hand_base]
            # Move pinky tip (index 20) in a J path
            if i < 10:
                pts[20][1] += i * 5  # Down
                pts[20][0] += i * 1
            else:
                pts[20][1] += 50 - (i - 10) * 4  # Hook up
                pts[20][0] += 10 + (i - 10) * 3  # Hook right

            gesture, conf = recognizer.process(pts)
            if gesture == "J":
                recognized_gesture = gesture

        self.assertEqual(recognized_gesture, "J", "DynamicGestureRecognizer should recognize J")

    def test_dynamic_recognizer_rejects_z_with_wrong_handshape(self):
        recognizer = DynamicGestureRecognizer(buffer_size=25)
        # Open palm (wrong handshape for Z)
        open_hand = create_synthetic_hand(index_up=True, middle_up=True, ring_up=True, pinky_up=True)

        recognized = None
        for i in range(18):
            pts = [p.copy() for p in open_hand]
            # Move in a Z path
            pts[8][0] += (i % 6) * 10
            pts[8][1] += i * 4
            g, _ = recognizer.process(pts)
            if g is not None:
                recognized = g

        self.assertIsNone(recognized, "Z must be rejected when hand is open palm")


    def test_invalid_j_vertical_bob_without_hook_curve(self):
        # Drop down and rise straight up with zero lateral curvature (x=0)
        y = np.concatenate([np.linspace(0, 60, 8), np.linspace(60, 20, 7)])
        x = np.zeros(15)
        pts = list(zip(x, y))

        is_valid, _ = verify_j_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Pure vertical bob without lateral curvature must be rejected as J")

    def test_invalid_j_inverted_path(self):
        # Moving upward then curving downward (inverted J)
        y = np.concatenate([np.linspace(60, 0, 8), np.linspace(0, 30, 7)])
        x = np.concatenate([np.linspace(0, 5, 8), np.linspace(5, 25, 7)])
        pts = list(zip(x, y))

        is_valid, _ = verify_j_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Inverted J trajectory must be rejected")

    def test_invalid_j_incomplete_stroke(self):
        # Downward drop only, missing hook recovery
        y = np.linspace(0, 60, 15)
        x = np.linspace(0, 10, 15)
        pts = list(zip(x, y))

        is_valid, _ = verify_j_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Incomplete J (no upward recovery) must be rejected")

    def test_invalid_z_reversed_path(self):
        # Reversed Z: Left -> Up-Right -> Left
        s1_x, s1_y = np.linspace(40, 0, 6), np.linspace(48, 45, 6)
        s2_x, s2_y = np.linspace(0, 35, 7), np.linspace(45, 5, 7)
        s3_x, s3_y = np.linspace(35, 0, 6), np.linspace(5, 0, 6)
        pts = list(zip(np.concatenate([s1_x, s2_x, s3_x]), np.concatenate([s1_y, s2_y, s3_y])))

        is_valid, _ = verify_z_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Reversed Z trajectory must be rejected")

    def test_invalid_z_incomplete_stroke(self):
        # Incomplete Z: only strokes 1 and 2 (missing stroke 3)
        s1_x, s1_y = np.linspace(0, 40, 7), np.linspace(0, 5, 7)
        s2_x, s2_y = np.linspace(40, 5, 8), np.linspace(5, 45, 8)
        pts = list(zip(np.concatenate([s1_x, s2_x]), np.concatenate([s1_y, s2_y])))

        is_valid, _ = verify_z_trajectory(pts, hand_scale=100.0)
        self.assertFalse(is_valid, "Incomplete Z (missing final horizontal stroke) must be rejected")

    def test_dynamic_recognizer_holds_prediction_and_prevents_duplicate(self):
        recognizer = DynamicGestureRecognizer(buffer_size=25, cooldown_frames=10)
        i_hand_base = create_synthetic_hand(pinky_up=True)

        # Trigger J
        j_frames = []
        for i in range(15):
            pts = [p.copy() for p in i_hand_base]
            if i < 10:
                pts[20][1] += i * 5
                pts[20][0] += i * 1
            else:
                pts[20][1] += 50 - (i - 10) * 4
                pts[20][0] += 10 + (i - 10) * 3
            j_frames.append(pts)

        # Feed frames until J triggers
        triggered = False
        for pts in j_frames:
            g, _ = recognizer.process(pts)
            if g == "J":
                triggered = True
                break

        self.assertTrue(triggered, "Dynamic J should trigger on valid trajectory")

        # Verify it holds 'J' over subsequent frames without vanishing
        hold_count = 0
        static_hand = [p.copy() for p in i_hand_base]
        for _ in range(8):
            g, _ = recognizer.process(static_hand)
            if g == "J":
                hold_count += 1

        self.assertGreaterEqual(hold_count, 6, "Dynamic J must be held for multiple frames so user can see it")

    def test_dynamic_recognizer_is_motion_in_progress(self):
        recognizer = DynamicGestureRecognizer(buffer_size=25)
        i_hand = create_synthetic_hand(pinky_up=True)

        # At start, no motion
        recognizer.process(i_hand)
        self.assertFalse(recognizer.is_motion_in_progress())

        # Move pinky downward
        for i in range(5):
            pts = [p.copy() for p in i_hand]
            pts[20][1] += i * 8
            recognizer.process(pts)

        self.assertTrue(recognizer.is_motion_in_progress(), "Motion in progress should be detected while dipping")


if __name__ == "__main__":
    unittest.main()

