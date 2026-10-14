"""
Unit tests for KeyPointClassifier (camera-independent inference).
"""

import os
import unittest
import numpy as np

from slr.model.classifier import KeyPointClassifier
from slr.paths import DEFAULT_KEYPOINT_PATH, PROJECT_ROOT


class TestKeyPointClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = KeyPointClassifier()

    def test_classifier_initialization_default(self):
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.interpreter)

    def test_classifier_initialization_cwd_independent(self):
        orig_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT.parent)
            clf = KeyPointClassifier()
            self.assertIsNotNone(clf)
        finally:
            os.chdir(orig_cwd)

    def test_tensor_io_shapes(self):
        input_details = self.classifier.input_details
        output_details = self.classifier.output_details

        self.assertEqual(len(input_details), 1)
        self.assertEqual(len(output_details), 1)

        input_shape = list(input_details[0]["shape"])
        output_shape = list(output_details[0]["shape"])

        self.assertEqual(input_shape, [1, 42], f"Expected input [1, 42], got {input_shape}")
        self.assertEqual(output_shape, [1, 24], f"Expected output [1, 24], got {output_shape}")

    def test_inference_returns_valid_class_or_sentinel(self):
        # Model output must always be in [0, 23] or sentinel 25
        zero_vector = [0.0] * 42
        pred = self.classifier(zero_vector)
        self.assertIn(pred, list(range(24)) + [25])

    def test_inference_sentinel_on_low_confidence_vector(self):
        rng = np.random.default_rng(42)
        # Vector index 1 in seed 42 produces confidence < 0.5
        _ = rng.uniform(-1, 1, 42)
        low_conf_vector = rng.uniform(-1, 1, 42).tolist()
        pred = self.classifier(low_conf_vector)
        self.assertEqual(pred, 25, f"Expected sentinel 25 for low-confidence vector, got {pred}")

    def test_inference_on_real_keypoint_dataset_sample(self):
        self.assertTrue(DEFAULT_KEYPOINT_PATH.exists())
        with open(DEFAULT_KEYPOINT_PATH, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        parts = first_line.split(",")
        label_id = int(parts[0])
        keypoints = [float(x) for x in parts[1:]]

        self.assertEqual(len(keypoints), 42)
        pred = self.classifier(keypoints)

        self.assertEqual(
            pred,
            label_id,
            f"Classifier expected to predict {label_id} for training vector, got {pred}",
        )

    def test_predict_with_confidence_custom_threshold(self):
        # Find an 'M' (Class 11) sample in keypoint.csv
        m_sample = None
        with open(DEFAULT_KEYPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if parts[0] == "11":
                    m_sample = [float(x) for x in parts[1:]]
                    break

        self.assertIsNotNone(m_sample)
        # With standard threshold 0.40, M is successfully recognized
        sign_id, conf = self.classifier.predict_with_confidence(m_sample, threshold=0.40)
        self.assertEqual(sign_id, 11, f"Expected Class 11 ('M'), got {sign_id}")
        self.assertGreaterEqual(conf, 0.40)

        # With excessively high threshold 0.99, returns sentinel 25
        sign_id_high, _ = self.classifier.predict_with_confidence(m_sample, threshold=0.99)
        self.assertEqual(sign_id_high, 25)

    def test_static_classes_mapping_alignment(self):
        # Verify 24 classes in output tensor matches 24 labels in label.csv
        from slr.paths import load_labels
        labels = load_labels()
        self.assertEqual(len(labels), 24)
        output_dim = self.classifier.output_details[0]["shape"][1]
        self.assertEqual(output_dim, len(labels))
        # Verify J and Z are NOT in the static labels
        self.assertNotIn("J", labels)
        self.assertNotIn("Z", labels)


if __name__ == "__main__":
    unittest.main()

