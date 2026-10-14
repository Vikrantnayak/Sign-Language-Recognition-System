"""
Unit tests verifying desktop application entry point integrity.
"""

import sys
import unittest
from pathlib import Path

from slr.paths import PROJECT_ROOT


class TestDesktopApp(unittest.TestCase):
    def test_app_importable(self):
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        import app
        self.assertTrue(hasattr(app, "main"), "app.py should expose main")
        self.assertTrue(callable(app.main), "app.main should be callable")

    def test_slr_main_importable(self):
        from slr import main
        self.assertTrue(callable(main), "slr.main should be callable")


if __name__ == "__main__":
    unittest.main()
