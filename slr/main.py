print("INFO: Initializing System")
import copy
import csv
import os
import datetime
from pathlib import Path

try:
    import pyautogui
except ImportError:
    pyautogui = None

import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv

from slr.paths import (
    DEFAULT_KEYPOINT_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_BACKGROUND_PATH,
    PROJECT_ROOT,
    load_labels,
)
from slr.model.classifier import KeyPointClassifier
from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks
from slr.utils.draw_debug import get_result_image
from slr.utils.draw_debug import get_fps_log_image
from slr.utils.draw_debug import draw_bounding_rect
from slr.utils.draw_debug import draw_hand_label
from slr.utils.draw_debug import show_fps_log
from slr.utils.draw_debug import show_result
from slr.utils.pre_process import calc_bounding_rect
from slr.utils.pre_process import calc_landmark_list
from slr.utils.pre_process import pre_process_landmark
from slr.utils.logging import log_keypoints
from slr.utils.logging import get_dict_form_list
from slr.utils.logging import get_mode

def main():
    load_dotenv()
    args = get_args()

    counter_obj = get_dict_form_list(str(DEFAULT_KEYPOINT_PATH))

    CAP_DEVICE = args.device
    CAP_WIDTH = args.width
    CAP_HEIGHT = args.height

    USE_STATIC_IMAGE_MODE = args.use_static_image_mode
    MAX_NUM_HANDS = args.max_num_hands
    MIN_DETECTION_CONFIDENCE = args.min_detection_confidence
    MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence

    USE_BRECT = args.use_brect
    MODE = args.mode
    DEBUG = int(os.environ.get("DEBUG", "0")) == 1

    print("INFO: System initialization Successful")
    print("INFO: Opening Camera")

    cap = cv.VideoCapture(CAP_DEVICE)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    if DEFAULT_BACKGROUND_PATH.exists():
        background_image = cv.imread(str(DEFAULT_BACKGROUND_PATH))
    else:
        import numpy as np
        background_image = np.zeros((720, 1080, 3), dtype=np.uint8)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    keypoint_classifier = KeyPointClassifier()
    keypoint_classifier_labels = load_labels(DEFAULT_LABEL_PATH)

    cv_fps = CvFpsCalc(buffer_len=10)
    print("INFO: System is up & running")
    
    while True:
        fps = cv_fps.get()

        key = cv.waitKey(1)
        if key == 27:
            print("INFO: Exiting...")
            break
        elif key == 57:
            name = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            ss_dir = PROJECT_ROOT / "ss"
            ss_dir.mkdir(parents=True, exist_ok=True)
            if pyautogui is not None:
                myScreenshot = pyautogui.screenshot()
                myScreenshot.save(str(ss_dir / f"{name}.png"))
                print(f"INFO: Screenshot saved to {ss_dir / f'{name}.png'}")
            else:
                print("WARNING: pyautogui not installed. Screenshot skipped.")


        success, image = cap.read()
        if not success:
            continue
        
        image = cv.resize(image, (CAP_WIDTH, CAP_HEIGHT))
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if DEBUG:
            MODE = get_mode(key, MODE)
            fps_log_image = show_fps_log(fps_log_image, fps)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                use_brect = True
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                if MODE == 0:
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 25:
                        hand_sign_text = ""
                    else:
                        hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    result_image = show_result(result_image, handedness, hand_sign_text)

                elif MODE == 1:
                    log_keypoints(key, pre_processed_landmark_list, counter_obj, data_limit=1000)

                debug_image = draw_bounding_rect(debug_image, use_brect, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        background_image[170:170 + 480, 50:50 + 640] = debug_image
        background_image[240:240 + 127, 731:731 + 299] = result_image
        background_image[678:678 + 30, 118:118 + 640] = fps_log_image

        cv.imshow("Sign Language Recognition", background_image)

    cap.release()
    cv.destroyAllWindows()
    print("INFO: Bye")

if __name__ == "__main__":
    main()
