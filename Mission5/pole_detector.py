"""Colour pole detection for Mission 5 (red / yellow / blue poles).

Pipeline per frame:
  1. optional gray-world white balance (removes the blue/green underwater cast)
  2. blur + convert to HSV
  3. threshold each colour (red wraps around hue 0, so it has two ranges)
  4. morphology to remove speckle and join broken pole segments
  5. pick the best contour per colour, preferring tall/vertical blobs

HSV ranges live in colors.json (create/edit it with hsv_tuner.py). If the file
is missing, the defaults below are used.
"""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

COLORS = ("red", "yellow", "blue")

# BGR colours used only for drawing
DRAW_BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "blue": (255, 80, 0)}

# OpenCV HSV: H 0-179, S 0-255, V 0-255
DEFAULT_CONFIG = {
    "white_balance": True,
    "blur_ksize": 5,
    "min_area_frac": 0.001,    # ignore blobs smaller than 0.1% of the frame
    "max_area_frac": 0.90,     # ignore "blobs" that are basically the whole frame
    "min_aspect": 1.3,         # height / width; poles are tall and thin
    "ranges": {
        "red": [[[0, 110, 60], [8, 255, 255]], [[165, 110, 60], [179, 255, 255]]],
        "yellow": [[[15, 100, 90], [40, 255, 255]]],
        "blue": [[[100, 130, 50], [128, 255, 255]]],
    },
}

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colors.json")


@dataclass
class Detection:
    color: str
    bbox: tuple          # x, y, w, h in pixels
    center: tuple        # cx, cy in pixels
    area: float          # contour area in pixels
    err_x: float         # horizontal offset from image centre, -1 (left) .. +1 (right)
    err_y: float         # vertical offset from image centre, -1 (top) .. +1 (bottom)
    height_frac: float   # bbox height / frame height
    width_frac: float    # bbox width / frame width (keeps growing right up to contact)
    area_frac: float     # contour area / frame area
    score: float


def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        ranges = user.pop("ranges", {})
        cfg.update(user)
        cfg["ranges"].update(ranges)
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def white_balance_gains(bgr):
    """Per-channel gains that make the typical pixel gray. Uses the median rather
    than the mean so a big pole filling part of the view doesn't skew it."""
    sample = bgr[::4, ::4].reshape(-1, 3)
    med = np.median(sample, axis=0).astype(np.float32)
    gain = med.mean() / np.maximum(med, 1.0)
    return np.clip(gain, 0.5, 3.0)  # don't blow up a channel that's nearly empty


def gray_world_white_balance(bgr, gain=None):
    """Remove the blue/green underwater cast (red is absorbed by water)."""
    if gain is None:
        gain = white_balance_gains(bgr)
    lut = np.clip(np.arange(256, dtype=np.float32)[:, None] * gain[None, :], 0, 255)
    return cv2.LUT(bgr, lut.astype(np.uint8).reshape(1, 256, 3))


class PoleDetector:
    def __init__(self, config=None):
        self.cfg = config if config is not None else load_config()
        self._wb_gain = None  # smoothed over frames so it doesn't jump around
        self._open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # small: far poles are thin
        # tall kernel joins pole segments split by tape, the ball, reflections...
        self._close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))

    def preprocess(self, bgr):
        if self.cfg.get("white_balance", True):
            g = white_balance_gains(bgr)
            self._wb_gain = g if self._wb_gain is None else 0.9 * self._wb_gain + 0.1 * g
            bgr = gray_world_white_balance(bgr, self._wb_gain)
        k = int(self.cfg.get("blur_ksize", 5))
        if k >= 3:
            bgr = cv2.GaussianBlur(bgr, (k | 1, k | 1), 0)
        return bgr

    def mask(self, hsv, color):
        m = None
        for lo, hi in self.cfg["ranges"][color]:
            part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            m = part if m is None else cv2.bitwise_or(m, part)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._open_k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._close_k)
        return m

    def detect(self, bgr, return_masks=False):
        """Return {color: Detection or None} (and {color: mask} if requested)."""
        H, W = bgr.shape[:2]
        frame_area = float(H * W)
        hsv = cv2.cvtColor(self.preprocess(bgr), cv2.COLOR_BGR2HSV)

        results, masks = {}, {}
        for color in COLORS:
            m = self.mask(hsv, color)
            masks[color] = m
            results[color] = self._best_blob(m, color, W, H, frame_area)
        return (results, masks) if return_masks else results

    def _best_blob(self, mask, color, W, H, frame_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            area_frac = area / frame_area
            if area_frac < self.cfg["min_area_frac"] or area_frac > self.cfg["max_area_frac"]:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = h / max(w, 1)
            # Up close the pole runs off the top/bottom of the frame, so its visible
            # piece can look short and wide; don't reject those on shape.
            touches_edge = y <= 2 or y + h >= H - 2
            if aspect < self.cfg["min_aspect"] and not touches_edge:
                continue
            solidity = area / max(w * h, 1)  # 1.0 = perfectly rectangular blob
            score = area * (0.5 + solidity) * min(aspect, 6.0)
            if best is None or score > best.score:
                cx, cy = x + w / 2.0, y + h / 2.0
                best = Detection(
                    color=color,
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                    area=area,
                    err_x=(cx - W / 2.0) / (W / 2.0),
                    err_y=(cy - H / 2.0) / (H / 2.0),
                    height_frac=h / float(H),
                    width_frac=w / float(W),
                    area_frac=area_frac,
                    score=score,
                )
        return best


def draw_detections(frame, detections, target=None):
    for color, det in detections.items():
        if det is None:
            continue
        x, y, w, h = det.bbox
        thick = 4 if color == target else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), DRAW_BGR[color], thick)
        label = f"{color} w={det.width_frac:.2f}"
        cv2.putText(frame, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, DRAW_BGR[color], 2 if color == target else 1)
    return frame
