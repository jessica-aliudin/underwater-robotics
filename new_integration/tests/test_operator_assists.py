import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from mission2_docking_assist import DockingAdvisor
from mission3_line_assist import analyse_black_line


class TestDockingAdvisor(unittest.TestCase):
    def setUp(self):
        self.advisor = DockingAdvisor()

    def test_no_target_and_wrong_target(self):
        self.assertIn("not visible", self.advisor.advise(None, 7).message)
        self.assertIn("Wrong chamber", self.advisor.advise(3, 7, 0.0).message)

    def test_left_right_and_centred(self):
        self.assertIn("left", self.advisor.advise(7, 7, -0.4).message)
        self.assertIn("right", self.advisor.advise(7, 7, 0.4).message)
        guidance = self.advisor.advise(7, 7, 0.02, area_fraction=0.25)
        self.assertTrue(guidance.aligned)
        self.assertTrue(guidance.close)


class TestLineAssist(unittest.TestCase):
    def test_centred_black_line(self):
        frame = np.full((480, 640, 3), 220, np.uint8)
        cv2.rectangle(frame, (290, 170), (350, 479), (0, 0, 0), -1)
        observation, overlay = analyse_black_line(frame)
        self.assertTrue(observation.found)
        self.assertAlmostEqual(observation.horizontal_error, 0.0, delta=0.05)
        self.assertEqual(overlay.shape, frame.shape)

    def test_missing_line(self):
        frame = np.full((480, 640, 3), 220, np.uint8)
        observation, _ = analyse_black_line(frame)
        self.assertFalse(observation.found)

    def test_line_on_right(self):
        frame = np.full((480, 640, 3), 220, np.uint8)
        cv2.rectangle(frame, (500, 170), (550, 479), (0, 0, 0), -1)
        observation, _ = analyse_black_line(frame)
        self.assertGreater(observation.horizontal_error, 0.15)
        self.assertIn("right", observation.guidance)


if __name__ == "__main__":
    unittest.main()

