"""
Temporal prediction smoothing and explicit state classification for ASL recognition.
Eliminates frame fluttering, enforces hysteresis, and handles clear states:
- RECOGNIZED: Confident, steady sign
- NEUTRAL: Hand present but resting or transitioning
- UNCERTAIN: Hand present but ambiguous classification or fluttering
- NO_HAND: No hand present in camera frame
"""

from collections import deque, Counter
from typing import Optional, Tuple, List, Dict, Any


class PredictionSmoother:
    """
    Rolling window consensus smoother for gesture predictions.
    Reduces single-frame prediction noise and produces reliable stable outputs.
    """

    def __init__(
        self,
        window_size: int = 8,
        min_consensus_ratio: float = 0.60,
        min_confidence_threshold: float = 0.40,
        no_hand_patience: int = 3,
    ):
        self.window_size = window_size
        self.min_consensus_ratio = min_consensus_ratio
        self.min_confidence_threshold = min_confidence_threshold
        self.no_hand_patience = no_hand_patience

        self.history: deque = deque(maxlen=window_size)
        self.no_hand_count: int = 0
        self.last_stable_prediction: Optional[str] = None
        self.last_stable_state: str = "NO_HAND"
        self.last_stable_confidence: float = 0.0

    def reset(self):
        """Reset internal history and state."""
        self.history.clear()
        self.no_hand_count = 0
        self.last_stable_prediction = None
        self.last_stable_state = "NO_HAND"
        self.last_stable_confidence = 0.0

    def update(
        self,
        raw_prediction: Optional[str],
        confidence: float,
        has_hand: bool,
        is_dynamic: bool = False,
        is_transitioning: bool = False,
    ) -> Tuple[Optional[str], float, str]:
        """
        Updates the smoother with the latest frame's detection and returns:
        (smoothed_letter, confidence, state)

        States:
        - 'RECOGNIZED': High-confidence stable letter
        - 'NEUTRAL': Hand present but resting/neutral
        - 'UNCERTAIN': Hand present but ambiguous or low confidence
        - 'NO_HAND': No hand in frame
        """
        if not has_hand:
            self.no_hand_count += 1
            if self.no_hand_count >= self.no_hand_patience:
                self.history.clear()
                self.last_stable_prediction = None
                self.last_stable_state = "NO_HAND"
                self.last_stable_confidence = 0.0
            return None, 0.0, "NO_HAND"

        self.no_hand_count = 0

        # If hand is in rapid transition between signs, suppress static recognition
        if is_transitioning and not is_dynamic:
            self.history.append(("Uncertain", confidence))
            self.last_stable_state = "UNCERTAIN"
            self.last_stable_confidence = confidence
            return None, confidence, "UNCERTAIN"

        # Dynamic gestures (J and Z) immediately bypass static window averaging
        if is_dynamic and raw_prediction in ["J", "Z"]:
            self.history.append((raw_prediction, confidence))
            self.last_stable_prediction = raw_prediction
            self.last_stable_state = "RECOGNIZED"
            self.last_stable_confidence = confidence
            return raw_prediction, confidence, "RECOGNIZED"

        # Record static frame result
        if not raw_prediction or raw_prediction in ["", "Uncertain", "Neutral"]:
            state_token = raw_prediction if raw_prediction else "Uncertain"
            self.history.append((state_token, confidence))
        else:
            self.history.append((raw_prediction, confidence))

        # Evaluate window consensus
        votes = [p for p, _ in self.history if p not in ["Uncertain", "Neutral"]]
        neutral_votes = [p for p, _ in self.history if p == "Neutral"]

        # If majority votes are explicitly Neutral
        if len(neutral_votes) >= int(self.min_consensus_ratio * len(self.history)):
            self.last_stable_prediction = None
            self.last_stable_state = "NEUTRAL"
            self.last_stable_confidence = confidence
            return None, confidence, "NEUTRAL"

        if not votes:
            self.last_stable_prediction = None
            self.last_stable_state = "UNCERTAIN"
            self.last_stable_confidence = confidence
            return None, confidence, "UNCERTAIN"

        counts = Counter(votes)
        top_sign, top_count = counts.most_common(1)[0]
        consensus = top_count / len(self.history)

        top_confs = [c for p, c in self.history if p == top_sign]
        avg_conf = sum(top_confs) / len(top_confs) if top_confs else 0.0

        if consensus >= self.min_consensus_ratio and avg_conf >= self.min_confidence_threshold:
            self.last_stable_prediction = top_sign
            self.last_stable_state = "RECOGNIZED"
            self.last_stable_confidence = avg_conf
            return top_sign, avg_conf, "RECOGNIZED"
        else:
            self.last_stable_state = "UNCERTAIN"
            self.last_stable_confidence = avg_conf
            return None, avg_conf, "UNCERTAIN"
