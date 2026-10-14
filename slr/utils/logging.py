import csv
import cv2 as cv
import json
import numpy as np


def log_keypoints(key, landmark_list, counter_obj, data_limit=1000):
    """
    Log keypoints to CSV file for training data collection
    
    :param key: Keyboard key (letter)
    :param landmark_list: Preprocessed landmark list
    :param counter_obj: Counter object for tracking data collection
    :param data_limit: How many rows needed for each sign
    :return: None
    """
    from slr.paths import DEFAULT_COUNTER_PATH, DEFAULT_KEYPOINT_PATH, resolve_path
    counter_file = str(DEFAULT_COUNTER_PATH)

    _dict = {}
    csv_path = str(DEFAULT_KEYPOINT_PATH)
    index = -1

    counter_obj = counter_obj


    if key == 106 or key == 74:
        return

    if 65 <= key <= 89 or 97 <= key <= 121:

        if 65 <= key <= 90:
            index = key - 65

            if key > 74:
                index -= 1

        elif 97 <= key <= 122:
            index = key - 97

            if key > 106:
                index -= 1

        if str(index) in counter_obj.keys():
            counter_obj[str(index)] += 1
        else:
            counter_obj[str(index)] = 1
            
        if counter_obj[str(index)] > data_limit:
            print(f"Dataset limit reached for {chr(key).upper()} [{counter_obj[str(index)]-1}/{data_limit}]")
            return
        else:
            with open(counter_file, "w") as cf:
                counter_obj_writable = json.dumps(counter_obj, indent=4)
                cf.write(counter_obj_writable)

            print(f"{chr(key).upper()} => {counter_obj[str(index)]}/{data_limit}")

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([index, *landmark_list])

    return


def get_dict_form_list(file=None):
    from slr.paths import DEFAULT_KEYPOINT_PATH, resolve_path
    target = resolve_path(file) if file else DEFAULT_KEYPOINT_PATH
    if not target.exists():
        return {}

    _list = []
    with open(target, encoding="utf-8") as f:
        for row in f:
            clean = row.strip()
            if clean:
                _list.append(clean.split(",")[0])


    if len(_list)==0:
        return {}

    obj = {}
    set_list = set(_list)

    for i in set_list:
        obj[str(i)] = _list.count(i)

    return obj


def _get_alphabet_index(key):
    """
    Get alphabet index from keyboard key
    
    :param key: Keyboard key (letter)
    :return: Index of alphabet
    """
    cap_ascii_list = [
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90
    ]
    sm_ascii_list = [
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122
    ]

    index = 0
    return index


def get_mode(key, _mode):
    """
    Get program mode based on pressed key
    
    :param key: Pressed key
    :param mode: Current mode of program
    :return: Updated mode
    """
    mode = _mode
    if key == 48:
        mode = 0
    elif key == 49:
        mode = 1

    return mode