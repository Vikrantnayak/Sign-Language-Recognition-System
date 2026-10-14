"""
Unit tests for camera-independent video and image processing pipeline.
Verifies WebRTC frame processor and static image upload handler without requiring a physical camera.
"""

import unittest
import numpy as np
import av

from streamlit_app import ASLVideoProcessor, process_static_image
from slr.model.classifier import KeyPointClassifier
from slr.paths import load_labels


class TestVideoPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = KeyPointClassifier()
        cls.labels = load_labels()

    def test_webrtc_video_processor_blank_frame(self):
        processor = ASLVideoProcessor()

        # Create a synthetic black frame (640x480 BGR)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        in_frame = av.VideoFrame.from_ndarray(dummy_img, format="bgr24")

        # Process frame
        out_frame = processor.recv(in_frame)

        self.assertIsInstance(out_frame, av.VideoFrame)
        out_arr = out_frame.to_ndarray(format="bgr24")
        self.assertEqual(out_arr.shape, (480, 640, 3))

        # Check status after processing a blank frame
        with processor.lock:
            self.assertIsNone(processor.latest_prediction)
            self.assertEqual(processor.latest_state, "NO_HAND")
            self.assertEqual(processor.latest_handedness, "None")

    def test_webrtc_video_processor_noisy_frame(self):
        processor = ASLVideoProcessor()

        # Random noise frame (simulating camera noise)
        noisy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        in_frame = av.VideoFrame.from_ndarray(noisy_img, format="bgr24")

        out_frame = processor.recv(in_frame)
        self.assertIsInstance(out_frame, av.VideoFrame)
        out_arr = out_frame.to_ndarray(format="bgr24")
        self.assertEqual(out_arr.shape, (480, 640, 3))

    def test_static_image_processing_blank(self):
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, pred, conf, hand = process_static_image(blank_img, self.classifier, self.labels)

        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertIsNone(pred)
        self.assertEqual(conf, 0.0)
        self.assertEqual(hand, "None")

    def test_static_image_upload_processing(self):
        # Verify static image upload processor handles non-blank images safely
        synthetic_img = np.full((480, 640, 3), 128, dtype=np.uint8)
        annotated, pred, conf, hand = process_static_image(synthetic_img, self.classifier, self.labels)

        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertIsInstance(annotated, np.ndarray)
        self.assertIsInstance(conf, float)
        self.assertIsInstance(hand, str)


if __name__ == "__main__":
    unittest.main()
