"""
Dynamic sequence-based gesture recognizer for ASL letters J and Z.
Replaces simplistic displacement heuristics with anatomical handshape preconditions
and trajectory shape verification.
"""

from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from slr.utils.pre_process import calc_hand_scale, is_finger_extended


def is_j_handshape_precondition(landmark_list: List) -> bool:
    """
    Checks if hand matches the 'I' handshape required for J:
    - Pinky extended
    - Index, middle, and ring curled into palm
    """
    if len(landmark_list) < 21:
        return False

    pinky_up = is_finger_extended(landmark_list, tip_idx=20, mcp_idx=17, ratio=1.2)
    index_down = not is_finger_extended(landmark_list, tip_idx=8, mcp_idx=5, ratio=1.25)
    middle_down = not is_finger_extended(landmark_list, tip_idx=12, mcp_idx=9, ratio=1.25)
    ring_down = not is_finger_extended(landmark_list, tip_idx=16, mcp_idx=13, ratio=1.25)

    return pinky_up and index_down and middle_down and ring_down


def is_z_handshape_precondition(landmark_list: List) -> bool:
    """
    Checks if hand matches the '1' / 'D' handshape required for Z:
    - Index finger extended
    - Middle, ring, and pinky curled into palm
    """
    if len(landmark_list) < 21:
        return False

    index_up = is_finger_extended(landmark_list, tip_idx=8, mcp_idx=5, ratio=1.2)
    middle_down = not is_finger_extended(landmark_list, tip_idx=12, mcp_idx=9, ratio=1.25)
    ring_down = not is_finger_extended(landmark_list, tip_idx=16, mcp_idx=13, ratio=1.25)
    pinky_down = not is_finger_extended(landmark_list, tip_idx=20, mcp_idx=17, ratio=1.25)

    return index_up and middle_down and ring_down and pinky_down


def verify_j_trajectory(points: List[Tuple[float, float]], hand_scale: float = 100.0) -> Tuple[bool, float]:
    """
    Validates whether tracking points form a true J-hook trajectory:
    1. Downward excursion (increasing y in screen coordinates).
    2. Deepest inflection point at the bottom curve (between 30% and 88% of trajectory duration).
    3. Upward recovery hook (decreasing y).
    4. Lateral horizontal hook curvature (must sweep laterally, rejecting pure 1D vertical bobs).

    :param points: List of (x, y) coordinates of pinky tip.
    :param hand_scale: Scale reference distance of the hand.
    :return: (is_valid, confidence)
    """
    if len(points) < 8:
        return False, 0.0

    pts = np.array(points, dtype=float)
    y_vals = pts[:, 1]
    x_vals = pts[:, 0]

    # Lowest point in screen coordinates has max y
    k = int(np.argmax(y_vals))
    n = len(points)

    # Inflection must happen between 30% and 88% of trajectory duration
    if k < int(0.3 * n) or k > int(0.88 * n):
        return False, 0.0

    downward_depth = y_vals[k] - y_vals[0]
    upward_hook = y_vals[k] - y_vals[-1]

    # Required thresholds scaled by hand size
    min_depth = max(0.12 * hand_scale, 15.0)
    min_hook = max(0.04 * hand_scale, 5.0)
    min_hook_x = max(0.05 * hand_scale, 6.0)

    if downward_depth < min_depth:
        return False, 0.0

    # Must have a clear upward hook recovery (at least 20% of the drop)
    if upward_hook < min_hook or upward_hook < (0.2 * downward_depth):
        return False, 0.0

    # Must have horizontal lateral sweep (distinguishes true J hook from pure 1D vertical bob)
    horizontal_span = float(np.ptp(x_vals))
    hook_lateral = abs(x_vals[-1] - x_vals[k])
    if horizontal_span < min_hook_x or hook_lateral < (0.4 * min_hook_x):
        return False, 0.0

    # Downward drop must be predominantly vertical (downward drop > 0.8 * lateral excursion)
    if downward_depth < (0.8 * horizontal_span):
        return False, 0.0

    # Confidence calculation based on hook clarity and curve definition
    hook_ratio = min(upward_hook / downward_depth, 1.0)
    lateral_ratio = min(hook_lateral / min_hook_x, 1.0)
    conf = float(np.clip(0.70 + 0.15 * hook_ratio + 0.10 * lateral_ratio, 0.70, 0.95))
    return True, conf


def verify_z_trajectory(points: List[Tuple[float, float]], hand_scale: float = 100.0) -> Tuple[bool, float]:
    """
    Validates whether tracking points form a 3-segment Z zigzag:
    - Segment 1: Stroke right (+X)
    - Segment 2: Diagonal stroke down and left (-X, +Y)
    - Segment 3: Stroke right (+X)

    :param points: List of (x, y) coordinates of index tip.
    :param hand_scale: Scale reference distance of the hand.
    :return: (is_valid, confidence)
    """
    if len(points) < 10:
        return False, 0.0

    pts = np.array(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    n = len(points)

    # Look for corner 1 (maximum X in first 65% of trajectory)
    # and corner 2 (minimum X after corner 1)
    k1 = int(np.argmax(x[: int(0.65 * n)]))
    if k1 < 2 or k1 >= n - 4:
        return False, 0.0

    k2_rel = int(np.argmin(x[k1:]))
    k2 = k1 + k2_rel
    if k2 <= k1 + 1 or k2 >= n - 2:
        return False, 0.0

    # Segment displacements
    dx1 = x[k1] - x[0]
    dy1 = y[k1] - y[0]

    dx2 = x[k2] - x[k1]
    dy2 = y[k2] - y[k1]

    dx3 = x[-1] - x[k2]
    dy3 = y[-1] - y[k2]

    min_stroke_x = max(0.08 * hand_scale, 10.0)
    min_diag_y = max(0.08 * hand_scale, 10.0)

    # Segment 1 must travel right (+X)
    if dx1 < min_stroke_x:
        return False, 0.0

    # Segment 2 must travel down-left (-X, +Y)
    if dx2 > -0.03 * hand_scale or dy2 < min_diag_y:
        return False, 0.0

    # Segment 3 must travel right again (+X)
    if dx3 < (0.5 * min_stroke_x):
        return False, 0.0

    # Net vertical progress must be downward
    total_dy = y[-1] - y[0]
    if total_dy < (0.5 * min_diag_y):
        return False, 0.0

    conf = float(np.clip(0.75 + 0.20 * (min(dx1, dx3) / min_stroke_x), 0.75, 0.95))
    return True, conf


class DynamicGestureRecognizer:
    """
    Stateful recognizer that monitors ongoing motion sequences for J and Z.
    Maintains landmark history buffers, applies handshape preconditions,
    and returns verified dynamic gestures with latching hold and cooldown.
    """

    def __init__(self, buffer_size: int = 25, cooldown_frames: int = 15):
        self.buffer_size = buffer_size
        self.cooldown_frames = cooldown_frames
        self.cooldown_counter = 0

        self.j_history: deque = deque(maxlen=buffer_size)
        self.z_history: deque = deque(maxlen=buffer_size)

        self.active_gesture: Optional[str] = None
        self.active_confidence: float = 0.0
        self.hold_frames: int = 0

    def reset(self):
        """Reset history buffers and state."""
        self.j_history.clear()
        self.z_history.clear()
        self.cooldown_counter = 0
        self.active_gesture = None
        self.active_confidence = 0.0
        self.hold_frames = 0

    def is_motion_in_progress(self) -> bool:
        """
        Returns True if a gesture tracking sequence has accumulated meaningful movement
        and is actively in progress.
        """
        if self.hold_frames > 0:
            return True
        if len(self.j_history) >= 4:
            pts = np.array(self.j_history, dtype=float)
            if np.ptp(pts[:, 1]) > 15.0:
                return True
        if len(self.z_history) >= 5:
            pts = np.array(self.z_history, dtype=float)
            if np.ptp(pts[:, 0]) > 15.0 or np.ptp(pts[:, 1]) > 15.0:
                return True
        return False

    def process(self, landmark_list: List, is_left_hand: bool = False) -> Tuple[Optional[str], float]:
        """
        Processes a single frame's landmarks.
        Returns (gesture_name, confidence) if a dynamic gesture is recognized/held, else (None, 0.0).
        """
        # 1. If currently holding a recognized gesture, maintain output across hold window
        if self.hold_frames > 0:
            self.hold_frames -= 1
            if self.hold_frames == 0:
                self.cooldown_counter = self.cooldown_frames
            return self.active_gesture, self.active_confidence

        # 2. If in cooldown period after hold, suppress duplicate triggers
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None, 0.0

        if not landmark_list or len(landmark_list) < 21:
            self.reset()
            return None, 0.0

        hand_scale = calc_hand_scale(landmark_list)

        # Coordinate extraction (mirror X if left hand for consistency)
        wrist = landmark_list[0]
        def get_rel_pt(idx):
            pt = landmark_list[idx]
            dx = pt[0] - wrist[0]
            if is_left_hand:
                dx = -dx
            dy = pt[1] - wrist[1]
            return (dx, dy)

        # 3. Check J preconditions & track pinky tip (Landmark 20)
        if is_j_handshape_precondition(landmark_list):
            pt_pinky = get_rel_pt(20)
            self.j_history.append(pt_pinky)
            is_valid_j, conf_j = verify_j_trajectory(list(self.j_history), hand_scale)
            if is_valid_j:
                self.active_gesture = "J"
                self.active_confidence = conf_j
                self.hold_frames = 10
                self.cooldown_counter = 0
                self.j_history.clear()
                self.z_history.clear()
                return "J", conf_j
        else:
            if len(self.j_history) > 0:
                self.j_history.popleft()

        # 4. Check Z preconditions & track index tip (Landmark 8)
        if is_z_handshape_precondition(landmark_list):
            pt_index = get_rel_pt(8)
            self.z_history.append(pt_index)
            is_valid_z, conf_z = verify_z_trajectory(list(self.z_history), hand_scale)
            if is_valid_z:
                self.active_gesture = "Z"
                self.active_confidence = conf_z
                self.hold_frames = 10
                self.cooldown_counter = 0
                self.j_history.clear()
                self.z_history.clear()
                return "Z", conf_z
        else:
            if len(self.z_history) > 0:
                self.z_history.popleft()

        return None, 0.0
