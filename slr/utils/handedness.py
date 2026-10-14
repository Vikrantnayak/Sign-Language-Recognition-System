"""
Handedness resolution utilities for MediaPipe Hands.
Consistently interprets MediaPipe handedness labels based on frame mirroring conventions.
"""

from typing import Tuple


def resolve_handedness(raw_label: str, is_mirrored: bool = True) -> str:
    """
    Resolves the user's anatomical hand ('Right' or 'Left') from MediaPipe's classification.

    MediaPipe Hands Documentation:
    "Note that it determines handedness assuming the input image is mirrored,
     i.e., taken with a front-facing/selfie camera with images flipped horizontally.
     If that is not the case, use cv2.flip(image, 1) to flip the image first for
     a correct handedness output."

    Conventions:
    - If is_mirrored=True (standard webcam selfie view after cv2.flip(frame, 1)):
        raw_label == 'Right' -> User's anatomical Right hand
        raw_label == 'Left'  -> User's anatomical Left hand

    - If is_mirrored=False (unflipped image, rear camera, or unmirrored static photo):
        raw_label == 'Right' -> User's anatomical Left hand
        raw_label == 'Left'  -> User's anatomical Right hand

    :param raw_label: The string label from handedness.classification[0].label ('Right' or 'Left').
    :param is_mirrored: Whether the frame fed to MediaPipe was horizontally flipped.
    :return: 'Right' or 'Left' indicating the user's true anatomical hand.
    """
    clean_label = raw_label.strip().capitalize()

    if is_mirrored:
        return "Right" if clean_label == "Right" else "Left"
    else:
        # Invert when input frame is NOT mirrored
        return "Left" if clean_label == "Right" else "Right"


def get_handedness_and_normalization_flag(
    raw_label: str, is_mirrored: bool = True
) -> Tuple[str, bool]:
    """
    Returns both the anatomical hand name and the boolean flag for landmark normalization.

    :param raw_label: The raw label from MediaPipe ('Right' or 'Left').
    :param is_mirrored: Whether the frame fed to MediaPipe is horizontally flipped.
    :return: (anatomical_hand_str, is_left_hand_bool)
             is_left_hand_bool is True for the user's anatomical Left hand,
             signaling that horizontal offsets should be mirrored to match
             the right-hand canonical training space.
    """
    anatomical = resolve_handedness(raw_label, is_mirrored=is_mirrored)
    is_left_hand = (anatomical == "Left")
    return anatomical, is_left_hand
