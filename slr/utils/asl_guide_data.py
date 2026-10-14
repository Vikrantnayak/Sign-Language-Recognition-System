"""
ASL Fingerspelling Reference Guide Data (A-Z).
Contains structured educational metadata, handshape descriptions, thumb placement,
hand orientation, dynamic movement sequences for J and Z, common distinctions,
and verified Public Domain reference illustration assets.

Illustrations Source:
  Wikimedia Commons (https://commons.wikimedia.org/wiki/Category:Sign_language_icons_-_letters)
  Vector illustrations originally created by wpclipart.
  License: Public Domain Dedication (CC0 / PD-self).
"""

from typing import Dict, List, Optional, Tuple, Any
from slr.paths import resolve_path, ILLUSTRATIONS_DIR


# Complete reference dictionary for all 26 letters of the ASL fingerspelling alphabet
ASL_ALPHABET_DATA: Dict[str, Dict[str, Any]] = {
    "A": {
        "letter": "A",
        "name": "Letter A",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. In ASL fingerspelling, one-handed signs are formed with your primary signing hand (right hand for right-handed signers, left hand for left-handed signers). Maintain your chosen hand consistently throughout signing.",
        "finger_positioning": "Fingers (index, middle, ring, pinky) are curled tightly into the palm forming a compact fist.",
        "thumb_placement": "Thumb rests straight and upright against the outside edge of the index finger. Thumb is NOT wrapped across the front of the fingers.",
        "hand_orientation": "Palm faces directly forward toward the viewer/camera at shoulder height.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'S' (thumb across front) and 'T' (thumb tucked between index and middle). For 'A', the thumb stays strictly alongside the index finger.",
        "image_file": "A.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_A.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "B": {
        "letter": "B",
        "name": "Letter B",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. One-handed signs are formed with your primary signing hand (right for right-handed signers, left for left-handed signers). Keep the wrist straight and hand comfortable at shoulder level.",
        "finger_positioning": "Four fingers (index, middle, ring, pinky) are held straight upright, touching closely side-by-side without gaps.",
        "thumb_placement": "Thumb is folded horizontally across the palm, tucked flat against the base of the fingers.",
        "hand_orientation": "Palm faces directly forward toward the viewer.",
        "common_mistakes": "Keep all four fingers strictly vertical and closed together. Do not allow fingers to spread apart like the number '4'.",
        "movement_sequence": None,
        "image_file": "B.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_B.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "C": {
        "letter": "C",
        "name": "Letter C",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Either hand can form the 'C' curve using your primary signing hand. The opening of the 'C' curves toward the opposite side of your signing hand so the profile is clear to the viewer.",
        "finger_positioning": "All four fingers curve together uniformly in an arch, forming the top curve of a letter 'C'.",
        "thumb_placement": "Thumb curves outward and upward from the bottom, mirroring the fingers to form an open 'C' shape with space between fingertips and thumb.",
        "hand_orientation": "Hand is slightly turned so the viewer can clearly see the open profile of the 'C' curve.",
        "movement_sequence": None,
        "common_mistakes": "Do not let the fingertips touch the thumb (that would form an 'O'). The space between fingers and thumb must remain open.",
        "image_file": "C.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_C.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "D": {
        "letter": "D",
        "name": "Letter D",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form the upright index finger and thumb circle using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index finger is extended vertically pointing straight up. Middle, ring, and pinky curl inward.",
        "thumb_placement": "Thumb curves upward and its tip touches the tips of the curled middle, ring, and pinky fingers, creating a round circle loop at the base.",
        "hand_orientation": "Palm faces forward or slightly angled so the circle loop formed by thumb and fingers is visible.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'F' and '1'. In 'D', only the index finger points up and the other three fingers form the circle with the thumb. In 'F', the index touches the thumb and three fingers are up.",
        "image_file": "D.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_D.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "E": {
        "letter": "E",
        "name": "Letter E",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form the curled fingers resting over the thumb with your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "All four fingers are tightly curled downward so their fingertips rest against the top edge or pad of the thumb.",
        "thumb_placement": "Thumb is folded horizontally underneath the curled fingertips, supporting them across the bottom edge of the handshape.",
        "hand_orientation": "Palm faces directly forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Fingertips should rest gently on the thumb without extending forward like claws. Keep knuckles level and palm facing front.",
        "image_file": "E.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_E.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "F": {
        "letter": "F",
        "name": "Letter F",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form the circular index-thumb loop with three upright fingers using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index finger curves down to touch the thumb tip, forming a circular loop. Middle, ring, and pinky fingers extend straight upright and spread slightly apart.",
        "thumb_placement": "Thumb tip touches the index fingertip to form the circular loop.",
        "hand_orientation": "Palm faces forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Do not confuse with 'D' (where index points up and three fingers touch thumb) or '9' (number sign). In 'F', three fingers are held upright while index touches thumb.",
        "image_file": "F.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_F.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "G": {
        "letter": "G",
        "name": "Letter G",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Orient the index finger and thumb horizontally across your chest pointing toward your non-dominant side (pointing left if right-handed, pointing right if left-handed).",
        "finger_positioning": "Index finger extends horizontally pointing sideways across the body. Middle, ring, and pinky fingers are curled tightly into the palm.",
        "thumb_placement": "Thumb extends horizontally parallel to the index finger with a small gap (approx. 1 inch) between index and thumb, resembling a small caliper or pinch.",
        "hand_orientation": "Hand is oriented sideways with the thumb on top and index on bottom (or vice versa), palm facing inward toward the signer's chest.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'H' (which extends both index and middle fingers) and 'L' (which forms an upright 90-degree angle with palm forward). 'G' points horizontally sideways with palm facing the signer.",
        "image_file": "G.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_G.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "H": {
        "letter": "H",
        "name": "Letter H",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Orient the extended index and middle fingers horizontally pointing toward your non-dominant side (pointing left if right-handed, pointing right if left-handed).",
        "finger_positioning": "Index and middle fingers are extended horizontally side-by-side, touching closely together and pointing sideways across the body. Ring and pinky are curled into the palm.",
        "thumb_placement": "Thumb is tucked across the ring and pinky fingers, resting flat underneath the extended index and middle fingers.",
        "hand_orientation": "Hand is turned sideways with the palm facing inward toward the signer's body; extended fingers point horizontally toward the non-dominant side.",
        "movement_sequence": None,
        "common_mistakes": "Keep index and middle fingers touching firmly together pointing horizontally. If you point with only one finger, it becomes 'G'.",
        "image_file": "H.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_H.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "I": {
        "letter": "I",
        "name": "Letter I",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Extend the pinky upright using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Pinky finger is extended straight upright. Index, middle, and ring fingers are curled tightly into the palm.",
        "thumb_placement": "Thumb folds across the curled index, middle, and ring fingers, locking them into a fist.",
        "hand_orientation": "Palm faces directly forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Ensure only the pinky is extended and fully straight. Keep the other three fingers curled and held down by the thumb.",
        "image_file": "I.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_I.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "J": {
        "letter": "J",
        "name": "Letter J",
        "type": "Movement-Based",
        "hand_to_use": "Either hand — use your dominant hand. Start in the 'I' handshape at shoulder level with your dominant hand. Move downward and trace the 'J' curve inward toward your body's centerline (hooking toward the left if using your right hand; hooking toward the right if using your left hand).",
        "finger_positioning": "Begins in the 'I' handshape: pinky finger extended straight upright, other fingers curled into a fist held by thumb.",
        "thumb_placement": "Thumb folds across the curled index, middle, and ring fingers, maintaining the 'I' handshape throughout the movement.",
        "hand_orientation": "Starts with palm facing forward; as the movement progresses, the wrist rotates inward so the palm finishes facing partially toward the signer.",
        "movement_sequence": [
            "Step 1 (Start Handshape): Form the static 'I' handshape with pinky upright and palm facing forward at chest/shoulder level.",
            "Step 2 (Downward Path): Move hand smoothly downward in the signing space approximately 4 to 6 inches.",
            "Step 3 (Curved Hook): Curve the pinky tip smoothly inward toward your midline, tracing the bottom hook of the letter 'J'.",
            "Step 4 (Completion): Finish with a slight upward scoop as the wrist rotates inward, ending with pinky pointing upward and palm facing toward the signer."
        ],
        "common_mistakes": "Do not just wiggle the pinky. The entire hand must trace a distinct 2D 'J' curve in the air with clear downward displacement followed by a lateral inward hook.",
        "image_file": "J.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_J.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "K": {
        "letter": "K",
        "name": "Letter K",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form the upright index and forward middle finger with thumb between them using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index finger points straight up. Middle finger extends forward and slightly tilted downward at approx. 45 degrees. Ring and pinky are curled into the palm.",
        "thumb_placement": "Thumb stands straight upright, with its tip resting squarely between the index and middle fingers, pressing against the middle finger knuckle/joint.",
        "hand_orientation": "Palm faces forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'V' (where thumb folds across curled fingers) and 'P' (which has the exact same handshape as 'K' but points downward). For 'K', hand is held upright with thumb wedged between index and middle.",
        "image_file": "K.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_K.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "L": {
        "letter": "L",
        "name": "Letter L",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form the perpendicular right angle between index finger and thumb with your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index finger is extended vertically pointing straight up. Middle, ring, and pinky are curled tightly into the palm.",
        "thumb_placement": "Thumb extends horizontally outward at a sharp 90-degree right angle to the index finger, forming an unmistakable letter 'L'.",
        "hand_orientation": "Palm faces directly forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Ensure the angle between thumb and index finger is crisp and perpendicular (90 degrees). Do not let thumb droop.",
        "image_file": "L.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_L.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "M": {
        "letter": "M",
        "name": "Letter M",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Fold three fingers over the thumb using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Three fingers (index, middle, ring) are folded forward over the thumb. Pinky finger is curled into the palm behind them.",
        "thumb_placement": "Thumb is tucked underneath the three fingers (index, middle, ring) so its tip protrudes slightly between the ring and pinky fingers.",
        "hand_orientation": "Palm faces forward toward the viewer, with the three mounds of the folded fingers prominently displayed.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'N' (two fingers over thumb) and 'T' (one finger over thumb). Remember: 'M' has THREE finger mounds resting over the thumb.",
        "image_file": "M.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_M.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "N": {
        "letter": "N",
        "name": "Letter N",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Fold two fingers over the thumb using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Two fingers (index and middle) are folded forward over the thumb. Ring and pinky fingers are curled into the palm.",
        "thumb_placement": "Thumb is tucked underneath the index and middle fingers so its tip protrudes between the middle and ring fingers.",
        "hand_orientation": "Palm faces forward toward the viewer, clearly displaying the two finger mounds over the thumb.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'M' (three fingers over thumb). For 'N', only TWO fingers (index and middle) fold over the thumb.",
        "image_file": "N.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_N.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "O": {
        "letter": "O",
        "name": "Letter O",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form the complete closed fingertip-to-thumb circle using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "All four fingers curve forward together in a smooth arc to meet the thumb.",
        "thumb_placement": "Thumb curves upward and outward so the thumb tip touches the tips of all four fingers, forming a complete closed 'O' circle.",
        "hand_orientation": "Palm faces forward or slightly angled so the open circular silhouette is visible to the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'C' (which is open with gap between fingers and thumb). In 'O', all fingertips firmly touch the thumb tip to close the ring.",
        "image_file": "O.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_O.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "P": {
        "letter": "P",
        "name": "Letter P",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Tilt the 'K' handshape downward from your dominant wrist with index pointing forward and middle finger pointing directly toward the ground.",
        "finger_positioning": "Uses the same finger configuration as 'K': index finger extended, middle finger pointing forward at an angle, ring and pinky curled.",
        "thumb_placement": "Thumb tip rests against the joint/knuckle of the middle finger, just like in 'K'.",
        "hand_orientation": "Hand is rotated downward at the wrist by 90 degrees so the middle finger points straight down toward the ground.",
        "movement_sequence": None,
        "common_mistakes": "'P' is simply an upside-down 'K'. Ensure the middle finger points downward and the index finger extends forward horizontally.",
        "image_file": "P.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_P.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "Q": {
        "letter": "Q",
        "name": "Letter Q",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Tilt the 'G' handshape downward from your dominant wrist with index finger and thumb pointing downward toward the floor.",
        "finger_positioning": "Uses the same finger configuration as 'G': index finger and thumb extended parallel; middle, ring, pinky curled into fist.",
        "thumb_placement": "Thumb extends parallel to index finger with approx. 1 inch gap between them.",
        "hand_orientation": "Hand is rotated downward so both index finger and thumb point down toward the floor, with the back of the hand facing forward/sideways.",
        "movement_sequence": None,
        "common_mistakes": "'Q' is the downward-pointing counterpart of 'G'. Do not point sideways; point both index and thumb downward toward the floor.",
        "image_file": "Q.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_Q.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "R": {
        "letter": "R",
        "name": "Letter R",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Cross the middle finger over the index finger using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index and middle fingers are extended upright and crossed over each other (middle finger crosses in front of the index finger). Ring and pinky are curled into palm.",
        "thumb_placement": "Thumb folds across the curled ring and pinky fingers, keeping them securely against the palm.",
        "hand_orientation": "Palm faces directly forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Ensure middle finger crosses in front of index finger like 'good luck' crossed fingers. Keep both fingers extended straight upward.",
        "image_file": "R.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_R.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "S": {
        "letter": "S",
        "name": "Letter S",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Form a fist with thumb locked across the front of the knuckles using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "All four fingers curl tightly into the palm forming a firm fist.",
        "thumb_placement": "Thumb folds across the front of the curled fingers, resting horizontally over the index and middle finger knuckles.",
        "hand_orientation": "Palm faces forward toward the viewer, showing the thumb locked across the front of the knuckles.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'A' (thumb rests along the outside edge of index) and 'T' (thumb tucked between index and middle). In 'S', the thumb is placed directly over the front of the curled fingers.",
        "image_file": "S.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_S.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "T": {
        "letter": "T",
        "name": "Letter T",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Tuck the thumb tip between index and middle fingers using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Fingers curl into a fist with the index finger folded over the thumb.",
        "thumb_placement": "Thumb is inserted directly between the index and middle fingers, with the thumb tip protruding upward between them.",
        "hand_orientation": "Palm faces forward toward the viewer, showing the thumb tip peeking between the first and second knuckles.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'A' (thumb beside index) and 'S' (thumb across front). For 'T', think: ONE finger (index) folded over the thumb.",
        "image_file": "T.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_T.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "U": {
        "letter": "U",
        "name": "Letter U",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Extend index and middle fingers held tightly together side-by-side using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index and middle fingers are extended straight upright and held tightly together side-by-side with no space between them. Ring and pinky are curled into palm.",
        "thumb_placement": "Thumb folds horizontally across the curled ring and pinky fingers, holding them against the palm.",
        "hand_orientation": "Palm faces forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'V' (fingers spread apart in a V) and 'R' (fingers crossed). For 'U', keep index and middle fingers firmly pressed together.",
        "image_file": "U.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_U.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "V": {
        "letter": "V",
        "name": "Letter V",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Extend index and middle fingers spread in a 'V' shape using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index and middle fingers are extended upright and spread apart into a distinct 'V' (peace sign) shape. Ring and pinky are curled into palm.",
        "thumb_placement": "Thumb folds across the curled ring and pinky fingers, holding them down against the palm.",
        "hand_orientation": "Palm faces forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Distinguish from 'U' (fingers touching) and 'K' (thumb placed between fingers). In 'V', the fingers must be visibly separated and the thumb stays folded over ring/pinky.",
        "image_file": "V.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_V.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "W": {
        "letter": "W",
        "name": "Letter W",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Extend index, middle, and ring fingers spread apart upright using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index, middle, and ring fingers are extended straight upright and spread apart, forming the shape of a 'W'. Pinky finger is curled into the palm.",
        "thumb_placement": "Thumb folds inward to touch and hold the curled pinky finger against the palm.",
        "hand_orientation": "Palm faces forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Do not confuse with the number '6' sign or 'B'. Ensure three distinct fingers (index, middle, ring) are spread apart upright while pinky and thumb touch.",
        "image_file": "W.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_W.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "X": {
        "letter": "X",
        "name": "Letter X",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Bend the index finger into a hooked knuckle using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Index finger is bent at the middle knuckle into a curved hook or claw shape. Middle, ring, and pinky are curled into a tight fist.",
        "thumb_placement": "Thumb folds across the curled fingers, resting against the middle finger.",
        "hand_orientation": "Palm faces forward or slightly angled so the curved hook of the index finger is clearly visible.",
        "movement_sequence": None,
        "common_mistakes": "Ensure the index finger is bent like a pirate's hook, not pointing straight up. Other three fingers must be curled completely.",
        "image_file": "X.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_X.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "Y": {
        "letter": "Y",
        "name": "Letter Y",
        "type": "Static",
        "hand_to_use": "Either hand — use your dominant hand. Extend thumb and pinky outward with central fingers folded using your primary signing hand (right hand for right-handed signers, left hand for left-handed signers).",
        "finger_positioning": "Thumb and pinky finger are fully extended outward in opposite directions ('hang loose' or phone shape). Index, middle, and ring fingers are curled tightly into the palm.",
        "thumb_placement": "Thumb extends outward laterally from the side of the hand, mirroring the outward reach of the pinky.",
        "hand_orientation": "Palm faces forward toward the viewer.",
        "movement_sequence": None,
        "common_mistakes": "Keep index, middle, and ring fingers curled firmly into the palm. Do not let them loosen or pop out.",
        "image_file": "Y.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_Y.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
    "Z": {
        "letter": "Z",
        "name": "Letter Z",
        "type": "Movement-Based",
        "hand_to_use": "Either hand — use your dominant hand. With the index finger of your dominant hand, trace the three-stroke 'Z' across the signing space (from non-dominant side across to dominant side, diagonal down, then across) from the signer's perspective so the letter reads cleanly to the viewer.",
        "finger_positioning": "Begins with the index finger extended and pointing forward/upward (like a pointing finger). Middle, ring, and pinky fingers are curled into a fist held by thumb.",
        "thumb_placement": "Thumb folds across the curled middle, ring, and pinky fingers, keeping the base of the fist secure throughout the drawing motion.",
        "hand_orientation": "Hand faces forward with index finger pointing forward into the signing space.",
        "movement_sequence": [
            "Step 1 (Start Handshape): Form a pointing handshape with index finger extended straight forward/upward at shoulder height, remaining fingers in a fist.",
            "Step 2 (Top Stroke): Move index fingertip horizontally from left to right (approx. 4-6 inches) across the front plane.",
            "Step 3 (Diagonal Stroke): Draw diagonally downward and to the left back toward the starting lateral position.",
            "Step 4 (Bottom Stroke): Draw horizontally from left to right again, completing the classic three-stroke 'Z' pattern in the air."
        ],
        "common_mistakes": "Draw the letter from the signer's perspective so the viewer reads a normal 'Z'. Keep the movement crisp and deliberate across all three strokes.",
        "image_file": "Z.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sign_language_Z.svg",
        "license": "Public Domain (wpclipart / Wikimedia Commons)",
    },
}


ALL_LETTERS: List[str] = list(ASL_ALPHABET_DATA.keys())


def get_letter_data(letter: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve structured reference data for a specific ASL letter (A-Z).
    Case-insensitive. Returns None if letter not found.
    """
    if not letter:
        return None
    return ASL_ALPHABET_DATA.get(letter.upper().strip())


def get_all_letters() -> List[Dict[str, Any]]:
    """Return all 26 letter reference entries in alphabetical order."""
    return [ASL_ALPHABET_DATA[k] for k in ALL_LETTERS]


def get_all_letter_keys() -> List[str]:
    """Return ordered list of letters ['A', 'B', ..., 'Z']."""
    return list(ALL_LETTERS)


def filter_letters(
    query: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter letters by text search query and/or category.
    category options: 'All', 'Static', 'Movement-Based' (or case-insensitive variations).
    """
    results = []
    clean_query = query.strip().lower() if query else None
    clean_category = category.strip().lower() if category else "all"

    for letter_key in ALL_LETTERS:
        entry = ASL_ALPHABET_DATA[letter_key]

        # Category filter
        if clean_category in ("static", "movement-based", "dynamic"):
            target_cat = "movement-based" if clean_category in ("movement-based", "dynamic") else "static"
            if entry["type"].lower() != target_cat:
                continue

        # Text query filter
        if clean_query:
            if len(clean_query) == 1:
                # Single-letter query: match exact letter
                if entry["letter"].lower() != clean_query:
                    continue
            else:
                # Multi-character query: match letter, name, positioning, thumb, or mistakes
                match_letter = entry["letter"].lower() == clean_query
                match_name = clean_query in entry["name"].lower()
                match_pos = clean_query in entry["finger_positioning"].lower()
                match_thumb = clean_query in entry["thumb_placement"].lower()
                match_mistakes = clean_query in entry["common_mistakes"].lower()
                match_hand = clean_query in entry.get("hand_to_use", "").lower()
                if not (match_letter or match_name or match_pos or match_thumb or match_mistakes or match_hand):
                    continue

        results.append(entry)

    return results


def get_prev_next(letter: str) -> Tuple[str, str]:
    """
    Return previous and next letter keys with wraparound (A -> Z, Z -> A).
    """
    clean = letter.upper().strip() if letter else "A"
    if clean not in ALL_LETTERS:
        clean = "A"
    idx = ALL_LETTERS.index(clean)
    prev_idx = (idx - 1) % len(ALL_LETTERS)
    next_idx = (idx + 1) % len(ALL_LETTERS)
    return ALL_LETTERS[prev_idx], ALL_LETTERS[next_idx]


def get_illustration_path(letter: str) -> Optional[str]:
    """
    Return resolved filesystem path to the letter's illustration SVG, or None if not found.
    """
    data = get_letter_data(letter)
    if not data or "image_file" not in data:
        return None
    p = resolve_path(data["image_file"])
    return str(p) if p.exists() else None
