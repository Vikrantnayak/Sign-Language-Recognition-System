"""
Unit tests verifying deployment configuration and dependency integrity.
"""

import unittest
from pathlib import Path

from slr.paths import PROJECT_ROOT


class TestConfigAndDependencies(unittest.TestCase):
    def test_streamlit_config_exists_and_valid(self):
        config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
        self.assertTrue(config_path.exists(), f"Missing config.toml at {config_path}")

        content = config_path.read_text(encoding="utf-8")
        self.assertIn("[server]", content)
        self.assertIn("headless = true", content)
        self.assertIn("enableCORS = false", content)

    def test_requirements_cloud_ready(self):
        req_path = PROJECT_ROOT / "requirements.txt"
        self.assertTrue(req_path.exists(), f"Missing requirements.txt at {req_path}")

        reqs = req_path.read_text(encoding="utf-8").lower()
        self.assertIn("streamlit-webrtc", reqs, "requirements.txt must include streamlit-webrtc for cloud camera")
        self.assertIn("av", reqs, "requirements.txt must include av")
        self.assertNotIn("pyautogui", reqs, "pyautogui must be removed from web cloud requirements.txt")

    def test_desktop_requirements_exist(self):
        desktop_req = PROJECT_ROOT / "requirements-desktop.txt"
        self.assertTrue(desktop_req.exists(), f"Missing requirements-desktop.txt at {desktop_req}")
        content = desktop_req.read_text(encoding="utf-8").lower()
        self.assertIn("pyautogui", content, "Desktop requirements should retain pyautogui")

    def test_packages_txt_valid(self):
        pkg_path = PROJECT_ROOT / "packages.txt"
        self.assertTrue(pkg_path.exists(), f"Missing packages.txt at {pkg_path}")
        content = pkg_path.read_text(encoding="utf-8")
        self.assertIn("libgl1", content)
        self.assertIn("libglib2.0-0", content)


if __name__ == "__main__":
    unittest.main()
