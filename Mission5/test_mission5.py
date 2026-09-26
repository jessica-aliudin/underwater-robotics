"""Tests for Mission 5. Run:  python -m unittest -v test_mission5

- order parsing
- detector on synthetic underwater images (blue-green cast, noise, decoys)
- controller state machine with fake detections
- closed-loop simulation: a 2D pool with 3 poles, rendered to camera images,
  detector + controller drive a simulated ROV; checks the poles get touched
  in the referee's order for every possible order.
"""

import itertools
import math
import unittest

import cv2
import numpy as np

from mission5 import (
    DEFAULT_CONFIG,
    Command,
    ControlParams,
    Detection,
    MissionController,
    PoleDetector,
    parse_order,
)

W, H = 640, 480
# How poles might look through murky blue-green water (BGR)
POLE_BGR = {"red": (45, 40, 165), "yellow": (40, 190, 215), "blue": (175, 75, 25)}
WATER_BGR = (140, 115, 45)


def water_background(seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:] = WATER_BGR
    img *= np.linspace(1.1, 0.75, H)[:, None, None]  # darker towards the bottom
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_noise(img, seed=1):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)


def detector():
    return PoleDetector(json_copy(DEFAULT_CONFIG))  # ignore any local colors.json


def json_copy(d):
    import json
    return json.loads(json.dumps(d))


class TestParseOrder(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_order("R-B-Y"), ["red", "blue", "yellow"])
        self.assertEqual(parse_order("y-r-b"), ["yellow", "red", "blue"])
        self.assertEqual(parse_order(" B Y R "), ["blue", "yellow", "red"])
        self.assertEqual(parse_order("RYB"), ["red", "yellow", "blue"])

    def test_invalid(self):
        for bad in ["R-R-Y", "R-B", "R-B-Y-G", "", "R-G-B"]:
            with self.assertRaises(ValueError, msg=bad):
                parse_order(bad)


class TestDetector(unittest.TestCase):
    def test_three_poles(self):
        img = water_background()
        xs = {"red": 100, "yellow": 300, "blue": 500}
        for c, x in xs.items():
            cv2.rectangle(img, (x, 80), (x + 24, 400), POLE_BGR[c], -1)
        det = detector().detect(add_noise(img))
        for c, x in xs.items():
            d = det[c]
            self.assertIsNotNone(d, c)
            self.assertAlmostEqual(d.center[0], x + 12, delta=6, msg=c)
            self.assertAlmostEqual(d.height_frac, 320 / H, delta=0.05, msg=c)

    def test_empty_water_gives_nothing(self):
        det = detector().detect(water_background(seed=3))
        self.assertEqual(det, {"red": None, "yellow": None, "blue": None})

    def test_ignores_small_and_wide_blobs(self):
        img = water_background()
        cv2.circle(img, (200, 200), 5, POLE_BGR["red"], -1)            # speck
        cv2.rectangle(img, (300, 300), (500, 330), POLE_BGR["red"], -1)  # horizontal bar
        self.assertIsNone(detector().detect(add_noise(img))["red"])

    def test_close_up_pole_cut_off_by_frame(self):
        # right in front of the pole: fills the full height and is wide
        img = water_background()
        cv2.rectangle(img, (260, 0), (380, H - 1), POLE_BGR["yellow"], -1)
        d = detector().detect(add_noise(img))["yellow"]
        self.assertIsNotNone(d)
        self.assertGreater(d.width_frac, 0.15)

    def test_pole_with_ball_and_band(self):
        # pole split by a white band with a ball on top: still one detection
        img = water_background()
        cv2.rectangle(img, (300, 100), (320, 420), POLE_BGR["blue"], -1)
        cv2.rectangle(img, (300, 250), (320, 258), (230, 230, 230), -1)
        cv2.circle(img, (310, 95), 18, POLE_BGR["blue"], -1)
        d = detector().detect(add_noise(img))["blue"]
        self.assertIsNotNone(d)
        self.assertGreater(d.height_frac, 0.6)


def fake_det(color, err_x=0.0, width_frac=0.03):
    return Detection(color=color, bbox=(0, 0, 1, 1), center=(0, 0), area=1, err_x=err_x,
                     err_y=0.0, height_frac=0.5, width_frac=width_frac, area_frac=0.01, score=1)


class TestController(unittest.TestCase):
    def test_full_cycle_for_one_pole(self):
        ctrl = MissionController(["blue", "red", "yellow"])
        t = 0.0
        # nothing visible -> search (yaw, no surge)
        cmd = ctrl.update({}, t)
        self.assertEqual(ctrl.state, "SEARCH")
        self.assertNotEqual(cmd.yaw, 0)
        self.assertEqual(cmd.surge, 0)
        # a *red* pole is ignored, we want blue first
        for _ in range(5):
            t += 0.1
            ctrl.update({"red": fake_det("red")}, t)
        self.assertEqual(ctrl.state, "SEARCH")
        # blue pole off to the right -> align, yaw right
        for _ in range(4):
            t += 0.1
            cmd = ctrl.update({"blue": fake_det("blue", err_x=0.5)}, t)
        self.assertEqual(ctrl.state, "ALIGN")
        self.assertGreater(cmd.yaw, 0)
        self.assertEqual(cmd.surge, 0)
        # centred -> approach
        t += 0.1
        cmd = ctrl.update({"blue": fake_det("blue", err_x=0.05)}, t)
        self.assertEqual(ctrl.state, "APPROACH")
        self.assertGreater(cmd.surge, 0)
        # close -> hit, then back off, then wait for confirmation
        t += 0.1
        cmd = ctrl.update({"blue": fake_det("blue", width_frac=0.2)}, t)
        self.assertEqual(ctrl.state, "HIT")
        t += ctrl.p.hit_time + 0.01
        ctrl.update({}, t)
        self.assertEqual(ctrl.state, "BACK_OFF")
        cmd = ctrl.update({}, t + 0.1)
        self.assertLess(cmd.surge, 0)
        t += ctrl.p.backoff_time + 0.2
        cmd = ctrl.update({}, t)
        self.assertEqual(ctrl.state, "CONFIRM")
        self.assertEqual((cmd.surge, cmd.yaw), (0, 0))
        ctrl.confirm(True, t)
        self.assertEqual(ctrl.target, "red")
        self.assertEqual(ctrl.results, {"blue": "hit"})

    def test_retry_and_skip(self):
        ctrl = MissionController(["red", "yellow", "blue"])
        ctrl._enter(ctrl.CONFIRM, 0)
        ctrl.confirm(False, 1)
        self.assertEqual((ctrl.target, ctrl.state), ("red", "SEARCH"))
        ctrl.skip(2)
        ctrl.skip(3)
        ctrl.skip(4)
        self.assertEqual(ctrl.state, "DONE")
        self.assertEqual(ctrl.update({}, 5), Command())

    def test_reposition_after_full_spin(self):
        ctrl = MissionController(["red", "yellow", "blue"])
        ctrl.update({}, 0.0)
        ctrl.update({}, ctrl.p.search_turn_time + 0.1)
        self.assertEqual(ctrl.state, "REPOSITION")
        cmd = ctrl.update({}, ctrl.p.search_turn_time + 0.5)
        self.assertGreater(cmd.surge, 0)
        ctrl.update({}, ctrl.p.search_turn_time + ctrl.p.reposition_time + 0.2)
        self.assertEqual(ctrl.state, "SEARCH")

    def test_real_clock_and_pause(self):
        # the app passes time.monotonic() (large numbers), not 0-based time
        ctrl = MissionController(["red", "yellow", "blue"])
        ctrl.update({}, 50000.0)
        self.assertEqual(ctrl.state, "SEARCH")
        # paused for a long time, then resumed: must not count as searching
        ctrl.resume()
        ctrl.update({}, 50000.0 + 10 * ctrl.p.search_turn_time)
        self.assertEqual(ctrl.state, "SEARCH")

    def test_lost_target_returns_to_search(self):
        ctrl = MissionController(["red", "yellow", "blue"])
        for i in range(4):
            ctrl.update({"red": fake_det("red", err_x=0.05)}, i * 0.1)
        self.assertIn(ctrl.state, ("ALIGN", "APPROACH"))
        ctrl.update({}, 0.5)            # short dropout: hold
        self.assertNotEqual(ctrl.state, "SEARCH")
        ctrl.update({}, 0.5 + ctrl.p.lost_timeout + 0.1)
        self.assertEqual(ctrl.state, "SEARCH")


# --------------------------------------------------------------------------
# closed-loop simulation
# --------------------------------------------------------------------------
HFOV = math.radians(70)
FOCAL = (W / 2) / math.tan(HFOV / 2)
POLE_DIAM, POLE_HEIGHT = 0.06, 1.0
MAX_SPEED, MAX_YAW_RATE = 0.5, math.radians(60)   # at command = 1.0
TOUCH_DIST = 0.15                                  # ROV nose to pole centre


class PoolSim:
    def __init__(self, poles, x=0.0, y=0.0, heading=0.0, seed=0):
        self.poles = poles  # color -> (x, y) metres
        self.x, self.y, self.th = x, y, heading
        self.bg = water_background(seed)
        self.rng = np.random.default_rng(seed)
        self.touched = []

    def render(self):
        img = self.bg.copy()
        c, s = math.cos(self.th), math.sin(self.th)
        items = []
        for color, (px, py) in self.poles.items():
            dx, dy = px - self.x, py - self.y
            fwd = dx * c + dy * s
            right = dx * s - dy * c
            if fwd < 0.05:
                continue
            u = W / 2 + FOCAL * right / fwd
            half_w = max(1.0, FOCAL * POLE_DIAM / fwd / 2)
            half_h = FOCAL * POLE_HEIGHT / fwd / 2
            items.append((fwd, color, u, half_w, half_h))
        for fwd, color, u, hw, hh in sorted(items, reverse=True):  # far first
            x0, x1 = int(round(u - hw)), int(round(u + hw))
            y0, y1 = int(round(H / 2 - hh)), int(round(H / 2 + hh))
            if x1 < 0 or x0 >= W:
                continue
            cv2.rectangle(img, (max(x0, -1), max(y0, -1)), (min(x1, W), min(y1, H)), POLE_BGR[color], -1)
        noise = self.rng.normal(0, 5, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def step(self, cmd, dt):
        self.th -= cmd.yaw * MAX_YAW_RATE * dt          # +yaw = turn right
        self.x += math.cos(self.th) * cmd.surge * MAX_SPEED * dt
        self.y += math.sin(self.th) * cmd.surge * MAX_SPEED * dt
        for color, (px, py) in self.poles.items():
            if math.hypot(px - self.x, py - self.y) < TOUCH_DIST and color not in self.touched:
                self.touched.append(color)


def run_sim(order, poles, heading=0.0, dt=0.1, max_time=240.0, seed=0):
    sim = PoolSim(poles, heading=heading, seed=seed)
    det = detector()
    ctrl = MissionController(order, ControlParams(), auto_confirm=True)
    t = 0.0
    while t < max_time and ctrl.state != ctrl.DONE:
        cmd = ctrl.update(det.detect(sim.render()), t)
        sim.step(cmd, dt)
        t += dt
    return sim, ctrl, t


class TestClosedLoopSim(unittest.TestCase):
    POLES = {"red": (3.0, 1.0), "yellow": (2.5, -1.5), "blue": (0.5, 3.0)}

    def test_all_orders(self):
        for i, order in enumerate(itertools.permutations(["red", "yellow", "blue"])):
            with self.subTest(order=order):
                sim, ctrl, t = run_sim(list(order), self.POLES, heading=i * 1.0, seed=i)
                self.assertEqual(ctrl.state, ctrl.DONE, f"timed out after {t:.0f}s")
                self.assertEqual(sim.touched, list(order),
                                 f"touched {sim.touched} in {t:.0f}s")


if __name__ == "__main__":
    unittest.main()
