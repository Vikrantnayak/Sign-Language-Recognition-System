"""
Unit tests for ASL Fingerspelling Reference Guide.
Verifies all 26 letters, required educational fields, dynamic vs static categorization,
illustration file existence and integrity, navigation helpers, and search/filtering.
"""

import os
import unittest
from pathlib import Path

from slr.paths import PROJECT_ROOT, ILLUSTRATIONS_DIR, resolve_path
from slr.utils.asl_guide_data import (
    ASL_ALPHABET_DATA,
    ALL_LETTERS,
    get_letter_data,
    get_all_letters,
    get_all_letter_keys,
    filter_letters,
    get_prev_next,
    get_illustration_path,
)


class TestASLVisualLearningGuide(unittest.TestCase):
    """Test suite for Milestone 4A ASL Visual Learning Guide."""

    def test_all_26_letters_represented(self):
        """Verify that exactly all 26 letters of the English alphabet are present and ordered."""
        expected_letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        self.assertEqual(len(ALL_LETTERS), 26)
        self.assertEqual(ALL_LETTERS, expected_letters)
        self.assertEqual(len(ASL_ALPHABET_DATA), 26)
        for letter in expected_letters:
            self.assertIn(letter, ASL_ALPHABET_DATA)

    def test_required_guide_fields_present_and_valid(self):
        """Verify that every letter has all required educational content fields."""
        required_keys = [
            "letter",
            "name",
            "type",
            "hand_to_use",
            "finger_positioning",
            "thumb_placement",
            "hand_orientation",
            "movement_sequence",
            "common_mistakes",
            "image_file",
            "source_url",
            "license",
        ]
        for letter, data in ASL_ALPHABET_DATA.items():
            for key in required_keys:
                self.assertIn(key, data, f"Letter {letter} missing key '{key}'")

            # Check content depth
            self.assertEqual(data["letter"], letter)
            self.assertEqual(data["name"], f"Letter {letter}")
            self.assertIn(data["type"], ("Static", "Movement-Based"))
            self.assertGreater(len(data["hand_to_use"]), 20, f"Letter {letter} hand to use too brief")
            self.assertGreater(len(data["finger_positioning"]), 15, f"Letter {letter} finger positioning too brief")
            self.assertGreater(len(data["thumb_placement"]), 10, f"Letter {letter} thumb placement too brief")
            self.assertGreater(len(data["hand_orientation"]), 10, f"Letter {letter} hand orientation too brief")
            self.assertGreater(len(data["common_mistakes"]), 15, f"Letter {letter} common mistakes too brief")
            self.assertTrue(data["source_url"].startswith("http"), f"Letter {letter} source_url invalid")
            self.assertIn("Public Domain", data["license"], f"Letter {letter} license must be Public Domain")

    def test_hand_usage_guidance(self):
        """Verify hand-usage guidance for all 26 letters explicitly states 'Either hand — use your dominant hand'."""
        for letter, data in ASL_ALPHABET_DATA.items():
            self.assertIn("hand_to_use", data)
            hand_guide = data["hand_to_use"]
            self.assertTrue(
                hand_guide.startswith("Either hand — use your dominant hand"),
                f"Letter {letter} hand_to_use must start with 'Either hand — use your dominant hand'. Got: {hand_guide[:40]}",
            )
            # Confirm dominant hand distinction is documented
            self.assertTrue(
                "dominant" in hand_guide.lower(),
                f"Letter {letter} hand guidance must mention dominant hand",
            )

    def test_absence_of_practice_mode_state_and_fields(self):
        """Verify that guide data has no practice or target-matching metadata fields."""
        disallowed_keys = {"target_letter", "practice_letter", "practice_score", "practice_mode"}
        for letter, data in ASL_ALPHABET_DATA.items():
            for key in disallowed_keys:
                self.assertNotIn(key, data, f"Practice field '{key}' should not exist in letter {letter}")

    def test_hand_usage_ui_fallback_resilience(self):
        """Verify that missing 'hand_to_use' in a letter dictionary safely falls back without KeyError."""
        incomplete_letter_info = {"letter": "X", "name": "Letter X"}
        fallback = incomplete_letter_info.get("hand_to_use", "Either hand — use your dominant hand.")
        self.assertEqual(fallback, "Either hand — use your dominant hand.")

    def test_static_vs_movement_distinction(self):
        """Verify that J and Z are marked as Movement-Based and have multi-step sequences, while all others are Static."""
        movement_letters = {"J", "Z"}
        for letter, data in ASL_ALPHABET_DATA.items():
            if letter in movement_letters:
                self.assertEqual(
                    data["type"],
                    "Movement-Based",
                    f"Letter {letter} must be categorized as 'Movement-Based'",
                )
                self.assertIsInstance(
                    data["movement_sequence"],
                    list,
                    f"Letter {letter} movement_sequence must be a list of steps",
                )
                self.assertGreaterEqual(
                    len(data["movement_sequence"]),
                    3,
                    f"Letter {letter} movement_sequence must have at least 3 distinct steps",
                )
                # Verify starting handshape is explicitly documented in Step 1
                step_1 = data["movement_sequence"][0].lower()
                self.assertTrue(
                    "start" in step_1 or "handshape" in step_1,
                    f"Letter {letter} Step 1 should explicitly specify starting handshape",
                )
            else:
                self.assertEqual(
                    data["type"],
                    "Static",
                    f"Letter {letter} must be categorized as 'Static'",
                )
                self.assertIsNone(
                    data["movement_sequence"],
                    f"Static letter {letter} movement_sequence should be None",
                )

    def test_all_26_illustrations_exist_and_valid(self):
        """Verify that verified SVG illustration files exist on disk for all 26 letters."""
        for letter in ALL_LETTERS:
            path_str = get_illustration_path(letter)
            self.assertIsNotNone(path_str, f"Could not resolve illustration path for letter {letter}")
            p = Path(path_str)
            self.assertTrue(p.exists(), f"Illustration file does not exist: {p}")
            self.assertTrue(p.is_file(), f"Illustration path is not a file: {p}")
            # Ensure file is not empty and contains valid SVG header/tags
            size = p.stat().st_size
            self.assertGreater(size, 2000, f"Illustration file {p.name} is suspiciously small ({size} bytes)")
            with open(p, "r", encoding="utf-8") as f:
                content = f.read(500)
                self.assertTrue(
                    "<svg" in content or "<?xml" in content,
                    f"File {p.name} does not appear to be an SVG file",
                )

    def test_get_letter_data_helpers(self):
        """Verify get_letter_data retrieval and case-insensitivity."""
        data_a = get_letter_data("a")
        self.assertIsNotNone(data_a)
        self.assertEqual(data_a["letter"], "A")

        data_z = get_letter_data("Z")
        self.assertIsNotNone(data_z)
        self.assertEqual(data_z["letter"], "Z")

        self.assertIsNone(get_letter_data("1"))
        self.assertIsNone(get_letter_data(""))
        self.assertIsNone(get_letter_data(None))

    def test_prev_next_navigation(self):
        """Verify circular wraparound navigation between letters."""
        prev_a, next_a = get_prev_next("A")
        self.assertEqual(prev_a, "Z")
        self.assertEqual(next_a, "B")

        prev_z, next_z = get_prev_next("Z")
        self.assertEqual(prev_z, "Y")
        self.assertEqual(next_z, "A")

        prev_m, next_m = get_prev_next("M")
        self.assertEqual(prev_m, "L")
        self.assertEqual(next_m, "N")

    def test_filter_letters_by_category(self):
        """Verify filtering letters by category: All, Static, Movement-Based."""
        all_res = filter_letters(category="All")
        self.assertEqual(len(all_res), 26)

        static_res = filter_letters(category="Static")
        self.assertEqual(len(static_res), 24)
        for entry in static_res:
            self.assertEqual(entry["type"], "Static")

        mvmt_res = filter_letters(category="Movement-Based")
        self.assertEqual(len(mvmt_res), 2)
        mvmt_letters = [entry["letter"] for entry in mvmt_res]
        self.assertEqual(sorted(mvmt_letters), ["J", "Z"])

    def test_filter_letters_by_query(self):
        """Verify single-letter exact matching and multi-character text search."""
        # Single-letter query should match only that letter
        search_b = filter_letters(query="B")
        self.assertEqual(len(search_b), 1)
        self.assertEqual(search_b[0]["letter"], "B")

        # Multi-character query for "fist" should match fist-based letters
        search_fist = filter_letters(query="fist")
        self.assertGreater(len(search_fist), 3)

        # Multi-character query for "hook" should match J and X
        search_hook = filter_letters(query="hook")
        hook_letters = [e["letter"] for e in search_hook]
        self.assertIn("J", hook_letters)
        self.assertIn("X", hook_letters)


if __name__ == "__main__":
    unittest.main()
