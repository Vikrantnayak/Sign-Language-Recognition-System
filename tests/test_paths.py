"""
Unit tests for path resolution and asset integrity.
"""

import os
import unittest
from pathlib import Path

from slr.paths import (
    PROJECT_ROOT,
    SLR_DIR,
    MODEL_DIR,
    RESOURCES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_HDF5_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_KEYPOINT_PATH,
    DEFAULT_COUNTER_PATH,
    DEFAULT_BACKGROUND_PATH,
    resolve_path,
    load_labels,
)


class TestPaths(unittest.TestCase):
    def test_core_directories_exist(self):
        self.assertTrue(PROJECT_ROOT.exists(), f"PROJECT_ROOT missing: {PROJECT_ROOT}")
        self.assertTrue(SLR_DIR.exists(), f"SLR_DIR missing: {SLR_DIR}")
        self.assertTrue(MODEL_DIR.exists(), f"MODEL_DIR missing: {MODEL_DIR}")
        self.assertTrue(RESOURCES_DIR.exists(), f"RESOURCES_DIR missing: {RESOURCES_DIR}")

    def test_default_assets_exist(self):
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model file missing: {DEFAULT_MODEL_PATH}")
        self.assertTrue(DEFAULT_HDF5_PATH.exists(), f"HDF5 file missing: {DEFAULT_HDF5_PATH}")
        self.assertTrue(DEFAULT_LABEL_PATH.exists(), f"Label file missing: {DEFAULT_LABEL_PATH}")
        self.assertTrue(DEFAULT_KEYPOINT_PATH.exists(), f"Keypoint dataset missing: {DEFAULT_KEYPOINT_PATH}")
        self.assertTrue(DEFAULT_COUNTER_PATH.exists(), f"Counter file missing: {DEFAULT_COUNTER_PATH}")
        self.assertTrue(DEFAULT_BACKGROUND_PATH.exists(), f"Background image missing: {DEFAULT_BACKGROUND_PATH}")

    def test_load_labels(self):
        labels = load_labels()
        self.assertEqual(len(labels), 24, f"Expected 24 labels, got {len(labels)}")
        self.assertEqual(labels[0], "A")
        self.assertEqual(labels[-1], "Y")
        self.assertNotIn("J", labels, "Letter J should not be in the static label set")
        self.assertNotIn("Z", labels, "Letter Z should not be in the static label set")

    def test_resolve_path_cwd_independence(self):
        original_cwd = os.getcwd()
        try:
            # Change directory to parent or root to simulate running from elsewhere
            target_cwd = PROJECT_ROOT.parent
            os.chdir(target_cwd)

            # From target_cwd, resolve_path should still locate model and label
            resolved_model = resolve_path("slr/model/slr_model.tflite")
            self.assertTrue(resolved_model.exists())
            self.assertEqual(resolved_model, DEFAULT_MODEL_PATH)

            resolved_label = resolve_path("slr/model/label.csv")
            self.assertTrue(resolved_label.exists())
            self.assertEqual(resolved_label, DEFAULT_LABEL_PATH)

            # Test None returns default model
            self.assertEqual(resolve_path(None), DEFAULT_MODEL_PATH)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
