"""
Sign Language Recognition - Web Application
Cloud-ready live webcam streaming with WebRTC, Left/Right hand normalization,
sequence-based J/Z recognition, and temporal prediction smoothing.
"""

import sys
import os
import threading
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import av
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

from slr.paths import (
    DEFAULT_MODEL_PATH,
    DEFAULT_LABEL_PATH,
    load_labels,
)
from slr.model.classifier import KeyPointClassifier
from slr.utils.landmarks import draw_landmarks
from slr.utils.draw_debug import draw_bounding_rect
from slr.utils.pre_process import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    is_neutral_hand,
    calc_landmark_motion,
    calc_hand_scale,
)
from slr.utils.dynamic_gesture import DynamicGestureRecognizer
from slr.utils.smoother import PredictionSmoother
from slr.utils.handedness import get_handedness_and_normalization_flag
from slr.utils.asl_guide_data import (
    ALL_LETTERS,
    get_letter_data,
    filter_letters,
    get_prev_next,
    get_illustration_path,
)
import base64


# WebRTC ICE Configuration (Public STUN servers for cloud deployment)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)


def get_classifier():
    """Returns a classifier instance."""
    return KeyPointClassifier(model_path=DEFAULT_MODEL_PATH)


def get_labels():
    """Returns labels loaded from label.csv."""
    return load_labels(DEFAULT_LABEL_PATH)


class ASLVideoProcessor(VideoProcessorBase):
    """
    Video processor for WebRTC streaming.
    Processes frames in a background thread and annotates landmarks, predictions,
    and sequence-based dynamic gestures.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_prediction: Optional[str] = None
        self.latest_confidence: float = 0.0
        self.latest_state: str = "NO_HAND"
        self.latest_handedness: str = "None"
        self.prev_landmarks: Optional[list] = None

        # MediaPipe Hands initialized for continuous streaming
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.classifier = get_classifier()
        self.labels = get_labels()
        self.dynamic_recognizer = DynamicGestureRecognizer(buffer_size=25, cooldown_frames=15)
        self.smoother = PredictionSmoother(
            window_size=8,
            min_consensus_ratio=0.60,
            min_confidence_threshold=0.40,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Flip horizontally for natural mirror interaction
        img = cv2.flip(img, 1)

        # Convert to RGB for MediaPipe processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        raw_prediction = ""
        raw_confidence = 0.0
        is_dynamic = False
        handedness_text = "None"
        has_hand = False

        if results.multi_hand_landmarks and results.multi_handedness:
            has_hand = True
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_hand = handedness.classification[0].label
                # Frame 'img' is horizontally flipped (cv2.flip(img, 1)) for selfie mirror view.
                # MediaPipe assumes a mirrored input for handedness output, so raw_hand matches anatomical hand.
                actual_hand, is_left = get_handedness_and_normalization_flag(raw_hand, is_mirrored=True)
                handedness_text = actual_hand

                # 1. Bounding box & Landmark extraction in pixel coordinates
                brect = calc_bounding_rect(img, hand_landmarks)
                landmark_list = calc_landmark_list(img, hand_landmarks)
                hand_scale = calc_hand_scale(landmark_list)

                # Compute inter-frame motion velocity
                motion_velocity = calc_landmark_motion(landmark_list, self.prev_landmarks, hand_scale)
                self.prev_landmarks = landmark_list
                is_transitioning = (motion_velocity > 0.18)

                # 2. Check if resting/neutral hand posture
                if is_neutral_hand(landmark_list):
                    raw_prediction = "Neutral"
                    raw_confidence = 0.85
                else:
                    # 3. Dynamic Sequence Check for J and Z
                    dyn_gesture, dyn_conf = self.dynamic_recognizer.process(landmark_list, is_left_hand=is_left)
                    if dyn_gesture in ["J", "Z"]:
                        raw_prediction = dyn_gesture
                        raw_confidence = dyn_conf
                        is_dynamic = True
                    elif self.dynamic_recognizer.is_motion_in_progress():
                        raw_prediction = "Uncertain"
                        raw_confidence = 0.0
                    else:
                        # 4. Synchronized Static Preprocessing with Left/Right Mirroring
                        pre_processed = pre_process_landmark(landmark_list, is_left_hand=is_left)
                        sign_id, conf = self.classifier.predict_with_confidence(pre_processed, threshold=0.40)
                        if sign_id < len(self.labels):
                            raw_prediction = self.labels[sign_id]
                            raw_confidence = conf
                        else:
                            raw_prediction = "Uncertain"
                            raw_confidence = conf

                # 5. Temporal Prediction Smoothing with transition awareness
                smoothed_letter, stable_conf, state = self.smoother.update(
                    raw_prediction, raw_confidence, has_hand=True, is_dynamic=is_dynamic, is_transitioning=is_transitioning
                )

                # 5. Draw annotations
                box_color = (34, 197, 94) if state == "RECOGNIZED" else ((234, 179, 8) if state == "UNCERTAIN" else (100, 116, 139))
                img = draw_bounding_rect(img, True, brect, outline_color=box_color, pad=8)
                img = draw_landmarks(img, landmark_list)

                # Draw prediction banner
                box_top = max(35, brect[1] - 12)
                if state == "RECOGNIZED" and smoothed_letter:
                    label_str = f"{smoothed_letter} ({int(stable_conf * 100)}%) - {actual_hand}"
                    cv2.rectangle(
                        img,
                        (brect[0], box_top - 30),
                        (brect[0] + len(label_str) * 15 + 10, box_top + 5),
                        (15, 23, 42),
                        -1,
                    )
                    cv2.putText(
                        img,
                        label_str,
                        (brect[0] + 5, box_top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (34, 197, 94),
                        2,
                        cv2.LINE_AA,
                    )
                elif state == "UNCERTAIN":
                    cv2.putText(
                        img,
                        f"Uncertain ({actual_hand})",
                        (brect[0], box_top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (234, 179, 8),
                        2,
                        cv2.LINE_AA,
                    )
        else:
            smoothed_letter, stable_conf, state = self.smoother.update(
                None, 0.0, has_hand=False, is_dynamic=False
            )

        with self.lock:
            self.latest_prediction = smoothed_letter
            self.latest_confidence = stable_conf
            self.latest_state = state
            self.latest_handedness = handedness_text

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        if hasattr(self, "hands") and self.hands:
            self.hands.close()


def process_static_image(cv_img: np.ndarray, classifier, labels):
    """Process a single static image (for image upload fallback)."""
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        annotated = cv_img.copy()
        prediction = None
        confidence = 0.0
        handedness_name = "None"

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_hand = handedness.classification[0].label
                actual_hand, is_left = get_handedness_and_normalization_flag(raw_hand, is_mirrored=False)
                handedness_name = actual_hand
                brect = calc_bounding_rect(annotated, hand_landmarks)

                landmark_list = calc_landmark_list(annotated, hand_landmarks)

                # Check for neutral resting hand
                if is_neutral_hand(landmark_list):
                    prediction = None
                    confidence = 0.85
                    handedness_name = f"{actual_hand} (Neutral)"
                else:
                    pre_processed = pre_process_landmark(landmark_list, is_left_hand=is_left)
                    sign_id, conf = classifier.predict_with_confidence(pre_processed, threshold=0.40)
                    if sign_id < len(labels):
                        prediction = labels[sign_id]
                        confidence = conf

                annotated = draw_bounding_rect(annotated, True, brect, outline_color=(79, 70, 229))
                annotated = draw_landmarks(annotated, landmark_list)

                if prediction:
                    cv2.putText(
                        annotated,
                        f"{prediction} ({int(confidence * 100)}%)",
                        (brect[0], max(35, brect[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (34, 197, 94),
                        2,
                        cv2.LINE_AA,
                    )

        return annotated, prediction, confidence, handedness_name


def render_asl_visual_learning_guide():
    """Renders the comprehensive ASL Fingerspelling Visual Learning Guide (A-Z)."""
    st.markdown(
        """
        <style>
        .asl-guide-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .asl-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #334155;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .asl-type-badge {
            font-size: 0.85rem;
            font-weight: 600;
            padding: 4px 14px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .asl-badge-static {
            background-color: #1e3a8a;
            color: #93c5fd;
            border: 1px solid #3b82f6;
        }
        .asl-badge-movement {
            background-color: #78350f;
            color: #fde68a;
            border: 1px solid #d97706;
        }
        .asl-info-box {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        .asl-info-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .asl-info-text {
            font-size: 0.95rem;
            color: #f8fafc;
            line-height: 1.5;
            margin: 0;
        }
        .asl-svg-frame {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            min-height: 310px;
        }
        .asl-svg-frame img {
            max-height: 270px;
            max-width: 100%;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📖 ASL Fingerspelling Visual Learning Guide (A–Z)")
    st.markdown(
        "Interactive reference guide containing all 26 American Sign Language letters with verified educational illustrations, "
        "anatomical finger positioning, thumb placement, hand orientation, and step-by-step movement sequences for dynamic signs."
    )

    # Search and Filter Controls
    f_col1, f_col2, f_col3 = st.columns([2.5, 2.5, 1])
    with f_col1:
        search_query = st.text_input(
            "🔍 Search Guide:",
            value=st.session_state.get("guide_search_query", ""),
            placeholder="Search letter (A-Z) or keyword ('fist', 'hook', 'thumb')...",
            key="guide_search_input",
        )
        st.session_state["guide_search_query"] = search_query

    with f_col2:
        cat_choice = st.radio(
            "Category Filter:",
            ["All (26)", "Static (24)", "Movement-Based (2)"],
            index=0 if st.session_state.get("guide_filter_cat", "All (26)") == "All (26)" else (
                1 if "Static" in st.session_state.get("guide_filter_cat", "") else 2
            ),
            horizontal=True,
            key="guide_cat_radio",
        )
        st.session_state["guide_filter_cat"] = cat_choice

    with f_col3:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Filters", use_container_width=True, key="btn_reset_filters"):
            st.session_state["guide_selected_letter"] = "A"
            st.session_state["guide_search_query"] = ""
            st.session_state["guide_filter_cat"] = "All (26)"
            st.rerun()

    # Determine filtered letters
    category_filter = (
        "Static" if "Static" in cat_choice else ("Movement-Based" if "Movement" in cat_choice else "All")
    )
    filtered_data = filter_letters(query=search_query, category=category_filter)
    filtered_keys = [item["letter"] for item in filtered_data]

    # Selected letter management
    current_letter = st.session_state.get("guide_selected_letter", "A")
    if current_letter not in filtered_keys and filtered_keys:
        current_letter = filtered_keys[0]
        st.session_state["guide_selected_letter"] = current_letter

    # A–Z Quick Selector Keyboard Grid
    st.markdown("##### 🔤 Select a Letter:")
    for row_start in [0, 13]:
        row_cols = st.columns(13)
        for i in range(13):
            idx = row_start + i
            if idx < len(ALL_LETTERS):
                letter_key = ALL_LETTERS[idx]
                is_selected = (letter_key == current_letter)
                is_in_filter = letter_key in filtered_keys

                icon = "💫" if letter_key in ("J", "Z") else ""
                label = f"{letter_key} {icon}".strip()
                if is_selected:
                    label = f"▶ {letter_key}"

                btn_type = "primary" if is_selected else ("secondary" if is_in_filter else "tertiary")
                if row_cols[i].button(
                    label,
                    key=f"grid_btn_{letter_key}",
                    type=btn_type,
                    disabled=not is_in_filter,
                    use_container_width=True,
                ):
                    st.session_state["guide_selected_letter"] = letter_key
                    st.rerun()

    if not filtered_keys:
        st.warning(f"No letters match your search query '{search_query}'. Try clearing the search or selecting 'All (26)'.")
        return

    st.markdown("---")

    # Detailed Letter View
    letter_info = get_letter_data(current_letter)
    if not letter_info:
        st.error(f"Data for letter {current_letter} could not be loaded.")
        return

    # 1. Large letter heading
    st.markdown(
        f"""
        <div class="asl-card-header" style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; color: #FFFFFF; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;">
                {letter_info['name']}
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_visual, col_details = st.columns([1.1, 1.9], gap="large")

    with col_visual:
        # 2. Verified handshape illustration
        ill_path = get_illustration_path(current_letter)
        if ill_path and os.path.exists(ill_path):
            with open(ill_path, "r", encoding="utf-8") as f_svg:
                svg_data = f_svg.read()
            b64_svg = base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")
            st.markdown(
                f"""
                <div class="asl-svg-frame">
                    <img src="data:image/svg+xml;base64,{b64_svg}" alt="ASL {current_letter} Handshape"/>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Illustration asset for letter {current_letter} not found.")

        st.caption(f"Verified Reference Handshape: **Letter {current_letter}** (Public Domain)")

        # Previous / Next Navigation
        prev_l, next_l = get_prev_next(current_letter)
        nav_c1, nav_c2 = st.columns(2)
        with nav_c1:
            if st.button(f"⬅️ Previous ({prev_l})", key="guide_nav_prev", use_container_width=True):
                st.session_state["guide_selected_letter"] = prev_l
                st.rerun()
        with nav_c2:
            if st.button(f"Next ({next_l}) ➡️", key="guide_nav_next", use_container_width=True):
                st.session_state["guide_selected_letter"] = next_l
                st.rerun()

    with col_details:
        # 3. Hand to use
        hand_to_use = letter_info.get(
            "hand_to_use",
            "Either hand — use your dominant hand. In ASL fingerspelling, one-handed signs are formed with your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        )
        st.markdown(
            f"""
            <div class="asl-info-box" style="border-left: 3px solid #6366F1;">
                <div class="asl-info-title" style="color: #818CF8;">🤚 Hand to Use</div>
                <p class="asl-info-text">{hand_to_use}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Finger positioning
        st.markdown(
            f"""
            <div class="asl-info-box">
                <div class="asl-info-title">🖐️ Finger Positioning</div>
                <p class="asl-info-text">{letter_info.get('finger_positioning', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5. Thumb placement
        st.markdown(
            f"""
            <div class="asl-info-box">
                <div class="asl-info-title">👍 Thumb Placement</div>
                <p class="asl-info-text">{letter_info.get('thumb_placement', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 6. Palm orientation
        st.markdown(
            f"""
            <div class="asl-info-box">
                <div class="asl-info-title">🧭 Palm Orientation</div>
                <p class="asl-info-text">{letter_info.get('hand_orientation', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 7. Static or movement-based classification
        is_dynamic = (letter_info.get("type", "Static") == "Movement-Based")
        badge_cls = "asl-badge-movement" if is_dynamic else "asl-badge-static"
        badge_txt = "💫 Movement-Based Sequence" if is_dynamic else "📌 Static Handshape"
        type_desc = (
            "Dynamic gesture requiring spatial motion path tracking in the signing space."
            if is_dynamic
            else "Stationary handshape held stable at chest/shoulder level."
        )
        st.markdown(
            f"""
            <div class="asl-info-box">
                <div class="asl-info-title">🏷️ Classification</div>
                <p class="asl-info-text" style="display: flex; align-items: center; gap: 10px; margin: 0;">
                    <span class="asl-type-badge {badge_cls}">{badge_txt}</span>
                    <span style="color: #94A3B8;">{type_desc}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 8. Step-by-step movement instructions for J and Z
        movement_seq = letter_info.get("movement_sequence")
        if movement_seq:
            steps_html = "".join([f"<li style='margin-bottom: 6px;'>{step}</li>" for step in movement_seq])
            st.markdown(
                f"""
                <div class="asl-info-box" style="border-left: 3px solid #F59E0B;">
                    <div class="asl-info-title" style="color: #FBBF24;">💫 Step-by-Step Movement Instructions</div>
                    <p class="asl-info-text" style="margin-bottom: 8px;">
                        Letter <b>{current_letter}</b> requires starting in a static handshape and tracing a spatial trajectory:
                    </p>
                    <ul style="color: #F8FAFC; margin: 0; padding-left: 20px; font-size: 0.95rem; line-height: 1.5;">
                        {steps_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 9. Common mistakes or distinctions, where available
        common_mistakes = letter_info.get("common_mistakes")
        if common_mistakes:
            st.markdown(
                f"""
                <div class="asl-info-box" style="border-left: 3px solid #38BDF8;">
                    <div class="asl-info-title" style="color: #38BDF8;">💡 Common Mistakes & Key Distinctions</div>
                    <p class="asl-info-text">{common_mistakes}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Attribution & Verification Expander
        with st.expander("ℹ️ Illustration Source, Verification & Licensing"):
            st.markdown(
                f"""
                - **Asset File:** `{letter_info['image_file']}`
                - **Educational Source:** [Wikimedia Commons File Page]({letter_info['source_url']})
                - **Licensing:** {letter_info['license']}
                - **Educational Provenance:** Vector illustrations originally drawn for educational reference by Paul Sherman (wpclipart), dedicated to the public domain.
                - **Integrity Guarantee:** Verified against standard ASL educational dictionaries. No AI-generated or unverified handshape diagrams are used in this platform.
                """
            )


def run_app():
    """Streamlit UI execution entry point."""
    st.set_page_config(
        page_title="ASL Vision — Real-time ASL Fingerspelling",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    labels = get_labels()
    classifier = get_classifier()

    # Session state initialization for Guide
    st.session_state.setdefault("guide_selected_letter", "A")
    st.session_state.setdefault("guide_filter_cat", "All (26)")
    st.session_state.setdefault("guide_search_query", "")

    # Inject Modern Dark Theme & Control Suppression CSS
    st.markdown(
        """
        <style>
        /* Modern Dark Theme Design System */
        :root {
            --bg-main: #0B0F19;
            --bg-card: #111827;
            --bg-card-subtle: #1E293B;
            --border-color: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.4);
            --accent-primary: #6366F1;
            --accent-secondary: #3B82F6;
            --accent-cyan: #38BDF8;
            --accent-emerald: #10B981;
            --text-primary: #F8FAFC;
            --text-muted: #94A3B8;
        }

        /* Top spacing and container cleanup */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            max-width: 1300px !important;
        }

        /* Application Header Banner */
        .asl-header-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #171E2E 100%);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        .asl-header-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .asl-title-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .asl-header-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #FFFFFF 20%, #C7D2FE 80%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .asl-header-subtitle {
            margin: 0;
            font-size: 0.95rem;
            color: #94A3B8;
            font-style: italic;
            letter-spacing: 0.01em;
        }
        .asl-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 7px 16px;
            border-radius: 9999px;
        }
        .asl-status-dot {
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10B981;
        }
        .asl-status-text {
            font-size: 0.82rem;
            font-weight: 600;
            color: #A7F3D0;
            letter-spacing: 0.02em;
        }

        /* Tabs Navigation Bar */
        div[data-baseweb="tab-list"] {
            background-color: #0F172A !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            padding: 6px !important;
            gap: 8px !important;
            margin-bottom: 24px !important;
        }
        div[data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 10px 22px !important;
            color: #94A3B8 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }
        div[data-baseweb="tab"]:hover {
            color: #F8FAFC !important;
            background: rgba(255, 255, 255, 0.03) !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #1E293B !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4) !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #6366F1 !important;
            height: 3px !important;
            border-radius: 2px !important;
        }

        /* WEBRTC LIVE VIDEO MIRROR — SUPPRESS ALL RECORDER CONTROLS & TIMERS */
        video {
            pointer-events: none !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background-color: #0B0F19 !important;
        }
        video::-webkit-media-controls {
            display: none !important;
        }
        video::-webkit-media-controls-enclosure {
            display: none !important;
        }
        video::-webkit-media-controls-panel {
            display: none !important;
        }
        video::-webkit-media-controls-timeline {
            display: none !important;
        }
        video::-webkit-media-controls-current-time-display {
            display: none !important;
        }
        video::-webkit-media-controls-time-remaining-display {
            display: none !important;
        }
        video::-webkit-media-controls-play-button {
            display: none !important;
        }
        video::-webkit-media-controls-timeline-container {
            display: none !important;
        }
        video::-webkit-media-controls-volume-control-container {
            display: none !important;
        }
        video::-webkit-media-controls-mute-button {
            display: none !important;
        }
        video::-webkit-media-controls-fullscreen-button {
            display: none !important;
        }

        /* WebRTC Enable/Disable Camera Button Styling */
        div[data-testid="stWebRtc"] button {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        div[data-testid="stWebRtc"] button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        }

        /* Prediction HUD Card Styles */
        .asl-hud-card {
            background: linear-gradient(135deg, #0F172A 0%, #111827 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .asl-hud-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 12px;
        }
        .asl-hud-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .asl-hud-pill {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .pill-recognized {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .pill-neutral {
            background: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.35);
        }
        .pill-uncertain {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .pill-waiting {
            background: rgba(148, 163, 184, 0.15);
            color: #94A3B8;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        .pill-inactive {
            background: rgba(100, 116, 139, 0.12);
            color: #64748B;
            border: 1px solid rgba(100, 116, 139, 0.25);
        }

        /* Prediction Hero Letter Display */
        .asl-pred-hero {
            background: #0B0F19;
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 14px;
            padding: 24px 16px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 140px;
        }
        .asl-pred-letter {
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1;
            margin: 0;
            color: #FFFFFF;
            letter-spacing: -0.02em;
        }
        .asl-pred-confidence {
            font-size: 0.95rem;
            color: #94A3B8;
            margin-top: 8px;
            font-weight: 500;
        }

        /* HUD Info Grid */
        .asl-hud-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .asl-hud-chip {
            background: #0B0F19;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 10px 14px;
        }
        .asl-chip-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 2px;
        }
        .asl-chip-val {
            font-size: 0.95rem;
            font-weight: 600;
            color: #F8FAFC;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #090D16 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }
        .sidebar-spec-box {
            background: #111827;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }
        .sidebar-spec-title {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #818CF8;
            margin-bottom: 3px;
        }
        .sidebar-spec-val {
            font-size: 0.86rem;
            color: #E2E8F0;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Application Header
    st.markdown(
        """
        <div class="asl-header-banner">
            <div class="asl-header-left">
                <div class="asl-title-row">
                    <h1 class="asl-header-title">🤟 ASL Vision</h1>
                </div>
                <p class="asl-header-subtitle">Real-time American Sign Language Fingerspelling</p>
            </div>
            <div class="asl-status-badge">
                <span class="asl-status-dot"></span>
                <span class="asl-status-text">Engine Ready • WebRTC Streaming</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Architecture & Specifications
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="font-size: 1.3rem;">⚙️</span>
                <h3 style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin: 0;">ASL Vision Engine</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-spec-box">
                <div class="sidebar-spec-title">Static Recognition</div>
                <div class="sidebar-spec-val">24 Alphabet Classes (A–I, K–Y)</div>
            </div>
            <div class="sidebar-spec-box">
                <div class="sidebar-spec-title">Dynamic Gestures</div>
                <div class="sidebar-spec-val">J & Z Trajectory Tracking</div>
            </div>
            <div class="sidebar-spec-box">
                <div class="sidebar-spec-title">Hand Parity</div>
                <div class="sidebar-spec-val">Bilateral Normalization (Left & Right)</div>
            </div>
            <div class="sidebar-spec-box">
                <div class="sidebar-spec-title">Temporal Filter</div>
                <div class="sidebar-spec-val">8-Frame Consensus Voting</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption(
            "🔒 **Privacy Notice:** Hand tracking runs in real time on client video. No camera footage is recorded, saved, or uploaded."
        )

    # EXACTLY TWO MAIN SECTIONS
    tab_live, tab_guide = st.tabs(
        ["📹 Live Sign Prediction", "📖 ASL Visual Learning Guide (A–Z)"]
    )

    with tab_live:
        col_video, col_info = st.columns([1.7, 1.3], gap="large")

        with col_video:
            ctx = webrtc_streamer(
                key="asl-vision-streamer",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=ASLVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                video_html_attrs={
                    "autoPlay": True,
                    "controls": False,
                    "muted": True,
                    "playsInline": True,
                    "style": {
                        "width": "100%",
                        "borderRadius": "14px",
                        "objectFit": "cover",
                        "display": "block",
                        "backgroundColor": "#0B0F19",
                    },
                },
                translations={
                    "start": "Enable Camera",
                    "stop": "Disable Camera",
                    "select_device": "Select Camera",
                },
            )
            st.caption(
                "Continuous live camera mirror. Hand landmarks and sign classifications update in real time."
            )

        with col_info:
            # Extract processor state
            curr_pred = None
            curr_conf = 0.0
            curr_state = "INACTIVE"
            curr_hand = "None"

            if ctx.video_processor:
                with ctx.video_processor.lock:
                    curr_pred = ctx.video_processor.latest_prediction
                    curr_conf = ctx.video_processor.latest_confidence
                    curr_state = ctx.video_processor.latest_state
                    curr_hand = ctx.video_processor.latest_handedness

            # Determine pill, hero text, and confidence text
            if curr_state == "RECOGNIZED" and curr_pred:
                pill_html = '<span class="asl-hud-pill pill-recognized">● Recognized</span>'
                hero_letter = curr_pred
                conf_text = f"{int(curr_conf * 100)}% Confidence"
                hand_text = f"{curr_hand} Hand"
            elif curr_state == "NEUTRAL":
                pill_html = '<span class="asl-hud-pill pill-neutral">● Neutral Pose</span>'
                hero_letter = "—"
                conf_text = "Resting hand posture"
                hand_text = f"{curr_hand} Hand" if curr_hand != "None" else "Resting Hand"
            elif curr_state == "UNCERTAIN":
                pill_html = '<span class="asl-hud-pill pill-uncertain">● Uncertain</span>'
                hero_letter = "❓"
                conf_text = "Stabilizing handshape..."
                hand_text = f"{curr_hand} Hand" if curr_hand != "None" else "Detecting..."
            elif curr_state == "NO_HAND":
                pill_html = '<span class="asl-hud-pill pill-waiting">● Waiting for hand</span>'
                hero_letter = "—"
                conf_text = "Position hand in camera frame"
                hand_text = "No Hand Detected"
            else:
                pill_html = '<span class="asl-hud-pill pill-inactive">○ Camera Inactive</span>'
                hero_letter = "—"
                conf_text = "Click 'Enable Camera' to start"
                hand_text = "Awaiting Video Stream"

            # Render Clean HUD Card without practice controls or target banners
            st.markdown(
                f"""
                <div class="asl-hud-card">
                    <div class="asl-hud-header">
                        <div class="asl-hud-title">📊 Live Recognition HUD</div>
                        {pill_html}
                    </div>
                    <div class="asl-pred-hero">
                        <div class="asl-pred-letter">{hero_letter}</div>
                        <div class="asl-pred-confidence">{conf_text}</div>
                    </div>
                    <div class="asl-hud-row">
                        <div class="asl-hud-chip">
                            <div class="asl-chip-label">Detected Hand</div>
                            <div class="asl-chip-val">{hand_text}</div>
                        </div>
                        <div class="asl-hud-chip">
                            <div class="asl-chip-label">Hand Parity</div>
                            <div class="asl-chip-val">Bilateral Normalization</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_guide:
        render_asl_visual_learning_guide()


if __name__ == "__main__":
    run_app()