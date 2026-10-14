import copy
import itertools
from typing import List, Optional

import numpy as np
import cv2 as cv

__all__ = [
    "calc_bounding_rect",
    "calc_landmark_list",
    "pre_process_landmark",
    "calc_hand_scale",
    "is_finger_extended",
    "is_neutral_hand",
    "calc_landmark_motion",
]

def calc_bounding_rect(image, landmarks) -> List:
    image_width = image.shape[1]
    image_height = image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width -1)
        landmark_y = min(int(landmark.y * image_height), image_height -1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x+w, y+h]

def calc_landmark_list(image, landmarks) -> List:
    image_width = image.shape[1]
    image_height = image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list, is_left_hand: bool = False) -> List:
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        dx = landmark_point[0] - base_x
        dy = landmark_point[1] - base_y

        # Invert horizontal displacement for left hand to map onto right-hand model
        if is_left_hand:
            dx = -dx

        temp_landmark_list[index][0] = dx
        temp_landmark_list[index][1] = dy

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    if max_value == 0:
        return [0.0] * len(temp_landmark_list)

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def calc_hand_scale(landmark_list: Optional[List]) -> float:
    """Calculates characteristic hand scale (Euclidean distance between wrist 0 and middle MCP 9)."""
    if not landmark_list or len(landmark_list) < 10:
        return 1.0
    wrist = np.array(landmark_list[0], dtype=float)
    middle_mcp = np.array(landmark_list[9], dtype=float)
    dist = float(np.linalg.norm(middle_mcp - wrist))
    return max(dist, 1.0)


def is_finger_extended(landmark_list: Optional[List], tip_idx: int, mcp_idx: int, ratio: float = 1.25) -> bool:
    """Returns True if fingertip is extended further from wrist than its MCP joint by the given ratio."""
    if not landmark_list or len(landmark_list) <= max(tip_idx, mcp_idx):
        return False
    wrist = np.array(landmark_list[0], dtype=float)
    tip = np.array(landmark_list[tip_idx], dtype=float)
    mcp = np.array(landmark_list[mcp_idx], dtype=float)
    dist_tip = np.linalg.norm(tip - wrist)
    dist_mcp = np.linalg.norm(mcp - wrist)
    return dist_tip > (dist_mcp * ratio)


def is_neutral_hand(landmark_list: Optional[List]) -> bool:
    """
    Detects a relaxed open palm resting gesture:
    - All 5 fingers extended and spread apart
    - Distinguishes from ASL 'B' (where fingers are held tightly together and thumb is folded across palm)
    - Distinguishes from ASL 'F' and 'O' (where thumb touches fingertip)
    """
    if not landmark_list or len(landmark_list) < 21:
        return False

    scale = calc_hand_scale(landmark_list)
    wrist = np.array(landmark_list[0], dtype=float)
    thumb_tip = np.array(landmark_list[4], dtype=float)
    index_tip = np.array(landmark_list[8], dtype=float)
    middle_tip = np.array(landmark_list[12], dtype=float)
    ring_tip = np.array(landmark_list[16], dtype=float)
    pinky_tip = np.array(landmark_list[20], dtype=float)

    # In F and O, thumb touches index/fingers
    if np.linalg.norm(thumb_tip - index_tip) < 0.35 * scale:
        return False

    # Check extension of all 5 fingers
    ext_idx = is_finger_extended(landmark_list, tip_idx=8, mcp_idx=5, ratio=1.2)
    ext_mid = is_finger_extended(landmark_list, tip_idx=12, mcp_idx=9, ratio=1.2)
    ext_rng = is_finger_extended(landmark_list, tip_idx=16, mcp_idx=13, ratio=1.2)
    ext_pnk = is_finger_extended(landmark_list, tip_idx=20, mcp_idx=17, ratio=1.2)

    thumb_mcp = np.array(landmark_list[2], dtype=float)
    thumb_ext = np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_mcp - wrist) * 1.2

    if not (ext_idx and ext_mid and ext_rng and ext_pnk and thumb_ext):
        return False

    # Check that fingers are spread apart (distinguishes from 'B' where fingers 8, 12, 16, 20 are touching)
    spread_8_12 = np.linalg.norm(index_tip - middle_tip) > 0.20 * scale
    spread_8_20 = np.linalg.norm(index_tip - pinky_tip) > 0.70 * scale
    spread_thumb_index = np.linalg.norm(thumb_tip - index_tip) > 0.40 * scale

    return bool(spread_8_12 and spread_8_20 and spread_thumb_index)


def calc_landmark_motion(current_landmarks: Optional[List], previous_landmarks: Optional[List], scale: float = 100.0) -> float:
    """
    Calculates average normalized displacement between consecutive landmark frames.
    Returns motion velocity relative to hand scale.
    """
    if not current_landmarks or not previous_landmarks or len(previous_landmarks) < 21 or len(current_landmarks) < 21:
        return 0.0

    curr = np.array(current_landmarks, dtype=float)
    prev = np.array(previous_landmarks, dtype=float)
    disp = np.linalg.norm(curr - prev, axis=1)
    mean_disp = float(np.mean(disp))
    return mean_disp / max(scale, 1.0)



