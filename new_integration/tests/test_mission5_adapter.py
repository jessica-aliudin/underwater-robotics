import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from mission5_adapter import Mission5Adapter


class NoopDetector:
    def detect(self, _frame):
        return {"red": None, "yellow": None, "blue": None}


class TestMission5Adapter(unittest.TestCase):
    def test_starts_paused(self):
        adapter = Mission5Adapter("R-B-Y", detector=NoopDetector())
        snapshot = adapter.update_detections({}, now=10.0)
        self.assertFalse(snapshot.running)
        self.assertEqual((snapshot.surge, snapshot.yaw, snapshot.heave), (0.0, 0.0, 0.0))

    def test_start_uses_existing_controller_without_editing_it(self):
        adapter = Mission5Adapter("R-B-Y", detector=NoopDetector())
        adapter.start()
        snapshot = adapter.update_detections({}, now=10.0)
        self.assertTrue(snapshot.running)
        self.assertEqual(snapshot.target, "red")
        self.assertEqual(snapshot.state, "SEARCH")
        self.assertNotEqual(snapshot.yaw, 0.0)

    def test_invalid_order_is_rejected_by_existing_parser(self):
        with self.assertRaises(ValueError):
            Mission5Adapter("R-R-Y", detector=NoopDetector())


if __name__ == "__main__":
    unittest.main()

