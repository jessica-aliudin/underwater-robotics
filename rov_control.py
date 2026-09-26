#!/usr/bin/env python3
"""
rov_control.py - topside control program for the CityUHK UR Fall Training 2026 ROV.

One window with:
  * live camera feed with AprilTag and colour-detection overlays
  * gamepad or keyboard driving, with arming, speed steps and a hook toggle
  * thruster output bars
  * Mission 1:   send the Wi-Fi details to the ESP32 and show the text it retrieves
  * Mission 2.1: remembers every AprilTag seen, picks the largest/smallest ID,
                 and keeps the clearest photo of each tag
  * Gallery (G): photos of the LOWEST and HIGHEST tag IDs side by side, plus
                 thumbnails of every tag captured. Photos are also saved as JPGs.
  * Mission 5:   enter the pole order (e.g. R-B-Y) and send it to the ROV

Install (Python 3.9+):
    pip install pygame opencv-python pyserial numpy
    (OpenCV 4.7 or newer is needed for the AprilTag detector.)

Run:
    python rov_control.py --camera demo --dry-run   no hardware at all: fake tags, nothing sent
    python rov_control.py --dry-run                 laptop webcam, commands shown but not sent
    python rov_control.py --port COM3 --camera 1    real ROV on COM3, USB camera number 1
    python rov_control.py --camera pool_run.mp4     replay a recorded video
    python rov_control.py --list-ports              list serial ports and exit

Photos are saved in ./captures/<date_time>/ :
    tag_007.jpg, tag_023.jpg, ...   clearest photo of each tag ID
    lowest.jpg, highest.jpg         copies of the lowest and highest ID photos
    photo_<time>.jpg                manual snapshots (P key)

Serial protocol (text lines, 115200 baud). Your ESP32 firmware must match this:
  laptop -> ROV
    T,<armed>,<hook>,<t1>,...,<tN>     about 30 times per second
                                       armed: 0/1   hook: 0 = closed, 1 = open
                                       thrusters: -100..100 percent, 0 = stop
    WIFI\t<ssid>\t<password>\t<url>    Mission 1 settings (tab-separated)
    FETCH                              ESP32 joins the Wi-Fi, does an HTTP GET, replies MSG
    ORDER\t<R-B-Y>                     Mission 5 pole order
  ROV -> laptop
    MSG\t<text>                        the text the ESP32 retrieved (Mission 1)
    LOG\t<text>                        status messages, shown in the log panel
  FAILSAFE (firmware side): if no T line arrives for 500 ms, stop every thruster.
"""

import argparse
import math
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pygame
except ImportError as exc:
    sys.exit(f"Missing package '{exc.name}'. Install with: pip install pygame opencv-python pyserial numpy")

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


# =============================================================================
# CONFIGURATION - edit this section to match your ROV
# =============================================================================

# Thruster layout. Each thruster lists how much it pushes for each motion:
#   surge = forward (+) / back (-)     sway = right (+) / left (-)
#   heave = up (+) / down (-)          yaw  = turn right (+) / turn left (-)
# Set "reverse": True for any thruster that spins the wrong way on the bench.
# The order here is the order of the numbers in the T line sent to the ESP32.
THRUSTERS = [
    {"name": "L",  "surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw":  1.0, "reverse": False},
    {"name": "R",  "surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": -1.0, "reverse": False},
    {"name": "V1", "surge": 0.0, "sway": 0.0, "heave": 1.0, "yaw":  0.0, "reverse": False},
    {"name": "V2", "surge": 0.0, "sway": 0.0, "heave": 1.0, "yaw":  0.0, "reverse": False},
]
# Example for 4 angled horizontal thrusters + 2 vertical. The signs depend on
# how each thruster is angled, so check every one on the bench before diving.
# THRUSTERS = [
#     {"name": "FL", "surge": 1, "sway":  1, "heave": 0, "yaw":  1, "reverse": False},
#     {"name": "FR", "surge": 1, "sway": -1, "heave": 0, "yaw": -1, "reverse": False},
#     {"name": "BL", "surge": 1, "sway": -1, "heave": 0, "yaw":  1, "reverse": False},
#     {"name": "BR", "surge": 1, "sway":  1, "heave": 0, "yaw": -1, "reverse": False},
#     {"name": "V1", "surge": 0, "sway":  0, "heave": 1, "yaw":  0, "reverse": False},
#     {"name": "V2", "surge": 0, "sway":  0, "heave": 1, "yaw":  0, "reverse": False},
# ]

# Gamepad mapping. Axis and button numbers differ between controllers and
# operating systems: the panel shows the live axis numbers, and pressing an
# unassigned button prints its number in the log, so you can fill these in.
AXIS = {"surge": 1, "yaw": 0, "sway": 2, "heave": 3}      # Xbox pad on Windows
AXIS_INVERT = {"surge": True, "yaw": False, "sway": False, "heave": True}
BUTTON = {"arm": 7, "disarm": 1, "hook": 2, "photo": 3, "speed_down": 4, "speed_up": 5}
DEADZONE = 0.10     # stick movement ignored near the centre
EXPO = 0.5          # 0 = linear, 1 = very gentle near centre (finer control for docking)

SPEED_STEPS = [0.25, 0.50, 0.75, 1.00]
DEFAULT_SPEED_INDEX = 1

BAUD = 115200
SEND_HZ = 30

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
TAG_CONFIRM_FRAMES = 3      # a tag must be seen in this many frames before it counts
TAG_BETTER_FACTOR = 1.15    # replace a tag's photo when it appears 15% bigger (closer, clearer)
CAPTURE_ROOT = Path("captures")

# Colour detection for Mission 5, in OpenCV HSV (hue 0-180). Pool lighting and
# water shift colours, so tune these with pool footage. A blue pool floor can
# trigger "B"; raise the blue saturation minimum if that happens.
COLOUR_RANGES = {
    "R": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "Y": [((20, 120, 70), (35, 255, 255))],
    "B": [((100, 150, 60), (130, 255, 255))],
}
MIN_COLOUR_AREA = 600       # pixels; ignore smaller blobs


# =============================================================================
# Window layout and colours
# =============================================================================

WIN_W, WIN_H = 1280, 720
VIEW = pygame.Rect(10, 46, 853, 640)     # camera or gallery
PANEL = pygame.Rect(873, 46, 397, 640)   # status panel

BG = (24, 26, 30)
PANEL_BG = (34, 37, 43)
LINE = (60, 65, 75)
TEXT = (230, 232, 236)
MUTED = (150, 156, 166)
ACCENT = (100, 175, 255)
GOOD = (90, 205, 125)
WARN = (240, 190, 70)
BAD = (240, 85, 85)

COLOUR_NAMES = {"R": "red", "Y": "yellow", "B": "blue"}
COLOUR_BGR = {"R": (60, 60, 255), "Y": (0, 220, 255), "B": (255, 140, 30)}
TAG_FAMILIES = {
    "16h5": "DICT_APRILTAG_16h5",
    "25h9": "DICT_APRILTAG_25h9",
    "36h10": "DICT_APRILTAG_36h10",
    "36h11": "DICT_APRILTAG_36h11",
}
AXES_ORDER = ("surge", "sway", "heave", "yaw")

HELP_LEFT = [
    ("Keyboard driving", None),
    ("W / S", "forward / back"),
    ("A / D", "turn left / right"),
    ("Q / E", "strafe left / right"),
    ("R / F", "up / down"),
    ("1 2 3 4", "speed 25 / 50 / 75 / 100 %"),
    ("Space", "arm/disarm (centre sticks)"),
    ("X", "disarm immediately"),
    ("H", "open or close the hook"),
    ("Gamepad (default mapping)", None),
    ("Left stick", "forward/back and turn"),
    ("Right stick", "strafe and up/down"),
    ("Start / B", "arm toggle / disarm"),
    ("X / Y", "hook / photo"),
    ("LB / RB", "speed down / up"),
]
HELP_RIGHT = [
    ("Missions", None),
    ("F2", "Mission 1: enter Wi-Fi, fetch"),
    ("F4", "Mission 1: fetch again"),
    ("M", "Mission 2.1: largest or smallest"),
    ("T", "AprilTag detection on / off"),
    ("G", "gallery: lowest/highest tags"),
    ("F5 twice", "clear captured tags"),
    ("F3", "Mission 5: pole order"),
    ("C", "colour detection on / off"),
    ("Other", None),
    ("P", "save a photo of the camera view"),
    ("F1 / Esc", "close this help"),
    ("Ctrl+Q", "quit (disarms first)"),
]


# =============================================================================
# Small helpers
# =============================================================================

def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def shape_axis(value):
    """Apply the deadzone and expo curve to a raw stick value (-1..1)."""
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    value = sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)
    return clamp((1.0 - EXPO) * value + EXPO * value ** 3)


def mix(cmd, speed):
    """Turn surge/sway/heave/yaw commands (-1..1) into thruster percentages (-100..100)."""
    raw = [sum(t[axis] * cmd[axis] for axis in AXES_ORDER) for t in THRUSTERS]
    # Horizontal and vertical thrusters are scaled separately, so full forward
    # plus full turn does not also weaken the vertical thrusters.
    scaled = [0.0] * len(THRUSTERS)
    for vertical in (False, True):
        group = [i for i, t in enumerate(THRUSTERS) if (t["heave"] != 0) == vertical]
        peak = max([abs(raw[i]) for i in group] + [1.0])
        for i in group:
            scaled[i] = raw[i] / peak
    result = []
    for t, value in zip(THRUSTERS, scaled):
        value *= speed
        if t["reverse"]:
            value = -value
        result.append(int(round(clamp(value) * 100)))
    return result


def fit_size(width, height, box_w, box_h):
    """Largest size that fits in the box without changing the aspect ratio."""
    scale = min(box_w / width, box_h / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def cv_to_surface(image, size):
    """OpenCV BGR image -> pygame surface of the given size."""
    if (image.shape[1], image.shape[0]) != tuple(size):
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), tuple(size), "RGB").convert()


def fmt_ids(ids):
    return ", ".join(str(i) for i in ids) if ids else "-"


def validate_not_empty(text):
    return None if text.strip() else "This cannot be empty"


def validate_url(text):
    t = text.strip()
    if not t.lower().startswith(("http://", "https://")) or len(t) <= len("http://") or " " in t:
        return "Enter the full address from the referee, e.g. http://192.168.1.10/"
    return None


def validate_order(text):
    parts = [p.strip().upper() for p in text.split("-")]
    if sorted(parts) != ["B", "R", "Y"]:
        return "Use R, Y and B once each, joined by hyphens, e.g. R-B-Y"
    return None


def aruco_dictionary(family):
    if not hasattr(cv2, "aruco"):
        sys.exit("This OpenCV has no AprilTag support. Run: pip install --upgrade opencv-python")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_FAMILIES[family]))


def generate_marker(dictionary, tag_id, size):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, size)
    return cv2.aruco.drawMarker(dictionary, tag_id, size)


# =============================================================================
# Vision
# =============================================================================

class TagDetector:
    """Finds AprilTags with OpenCV's built-in detector."""

    def __init__(self, family):
        self.dictionary = aruco_dictionary(family)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
            self._detect = detector.detectMarkers
        else:  # OpenCV 4.6 and older (needs opencv-contrib-python)
            params = cv2.aruco.DetectorParameters_create()
            self._detect = lambda gray: cv2.aruco.detectMarkers(gray, self.dictionary, parameters=params)

    def detect(self, frame):
        """Returns a list of (tag_id, corner_points, area_in_pixels)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        if ids is None:
            # Murky water lowers contrast; try again with contrast enhancement.
            corners, ids, _ = self._detect(self.clahe.apply(gray))
        found = []
        if ids is not None:
            for quad, tag_id in zip(corners, ids.flatten()):
                pts = quad.reshape(4, 2).astype(np.float32)
                found.append((int(tag_id), pts, float(cv2.contourArea(pts))))
        return found


class TagTracker:
    """Remembers every AprilTag seen and keeps the clearest photo of each one."""

    def __init__(self, root):
        self.root = root
        self.version = 0
        self.reset()

    def reset(self):
        self.capture_dir = self.root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.hits = {}        # tag id -> number of frames it has been seen in
        self.best = {}        # tag id -> {"image", "area", "time"}
        self.pending = set()  # tag ids whose photo still needs writing to disk
        self.last_save = {}
        self.version += 1

    @property
    def ids(self):
        return sorted(self.best)

    def lowest(self):
        return min(self.best) if self.best else None

    def highest(self):
        return max(self.best) if self.best else None

    def target(self, mode):
        return self.highest() if mode == "largest" else self.lowest()

    def update(self, detections, frame):
        """Feed one frame's detections. Returns tag ids captured for the first time."""
        new_ids = []
        for tag_id, pts, area in detections:
            self.hits[tag_id] = self.hits.get(tag_id, 0) + 1
            if self.hits[tag_id] < TAG_CONFIRM_FRAMES:
                continue
            best = self.best.get(tag_id)
            if best is not None and area <= best["area"] * TAG_BETTER_FACTOR:
                continue
            if best is None:
                new_ids.append(tag_id)
            self.best[tag_id] = {"image": self._annotate(frame, tag_id, pts), "area": area,
                                 "time": datetime.now().strftime("%H:%M:%S")}
            self.pending.add(tag_id)
            self.version += 1
        return new_ids

    @staticmethod
    def _annotate(frame, tag_id, pts):
        image = frame.copy()
        corners = pts.astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 230, 255), 4, cv2.LINE_AA)
        x, y = int(corners[:, 0].min()), int(corners[:, 1].min())
        label = f"ID {tag_id}"
        cv2.putText(image, label, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, label, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2, cv2.LINE_AA)
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, h - 32), (w, h), (0, 0, 0), -1)
        caption = f"AprilTag ID {tag_id}   captured {datetime.now():%Y-%m-%d %H:%M:%S}"
        cv2.putText(image, caption, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def flush(self, force=False):
        """Write new or improved photos to disk, at most once per second per tag."""
        now = time.time()
        wrote = False
        for tag_id in sorted(self.pending):
            if not force and now - self.last_save.get(tag_id, 0.0) < 1.0:
                continue
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.capture_dir / f"tag_{tag_id:03d}.jpg"), self.best[tag_id]["image"])
            self.last_save[tag_id] = now
            self.pending.discard(tag_id)
            wrote = True
        if wrote:
            cv2.imwrite(str(self.capture_dir / "lowest.jpg"), self.best[self.lowest()]["image"])
            cv2.imwrite(str(self.capture_dir / "highest.jpg"), self.best[self.highest()]["image"])


def detect_colours(frame):
    """Returns a list of (colour letter, (x, y, w, h), area) for red/yellow/blue blobs."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    found = []
    for name, ranges in COLOUR_RANGES.items():
        mask = None
        for low, high in ranges:
            part = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= MIN_COLOUR_AREA:
                found.append((name, cv2.boundingRect(contour), area))
    return found


def draw_tag_overlays(image, detections, target_id):
    for tag_id, pts, _ in detections:
        corners = pts.astype(np.int32)
        is_target = tag_id == target_id
        colour = (0, 215, 255) if is_target else (90, 220, 90)
        cv2.polylines(image, [corners], True, colour, 3, cv2.LINE_AA)
        label = f"ID {tag_id}" + ("  TARGET" if is_target else "")
        x, y = int(corners[:, 0].min()), int(corners[:, 1].min())
        pos = (x, max(22, y - 8))
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)


def draw_colour_overlays(image, found):
    for name, (x, y, w, h), _ in found:
        colour = COLOUR_BGR[name]
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(image, COLOUR_NAMES[name], (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2, cv2.LINE_AA)


# =============================================================================
# Video sources
# =============================================================================

class CameraSource:
    """Reads a camera or video file in a background thread so the controls never wait for it."""

    def __init__(self, source, width, height):
        self.source, self.width, self.height = source, width, height
        self.status = "starting"
        self.fps = 0.0
        self.frame_id = 0
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source)
        return cap

    def _run(self):
        is_file = isinstance(self.source, str)
        cap, delay, failures = None, 0.0, 0
        count, t0 = 0, time.time()
        while self._running:
            if cap is None:
                cap = self._open()
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    self.status = f"{self.source} not found"
                    time.sleep(2.0)
                    continue
                self.status = "ok"
                delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if is_file and failures < 3:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the video
                    continue
                self.status = "no picture, reconnecting"
                cap.release()
                cap, failures = None, 0
                time.sleep(1.0)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self.frame_id += 1
            count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self.fps, count, t0 = count / elapsed, 0, time.time()
            if delay:
                time.sleep(delay)
        if cap is not None:
            cap.release()

    def latest(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False


class DemoSource:
    """Fake scene for testing with no camera: AprilTags drift past one at a time,
    getting closer and further away, plus red, yellow and blue balls."""

    def __init__(self, width, height, family, tag_ids=(7, 23, 12)):
        self.width, self.height = width, height
        self.status = "demo"
        self.fps = 30.0
        self._start = time.time()
        dictionary = aruco_dictionary(family)
        self._tags = []
        for tag_id in tag_ids:
            marker = generate_marker(dictionary, tag_id, 150)
            marker = cv2.copyMakeBorder(marker, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
            self._tags.append(cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR))

    @property
    def frame_id(self):
        return int((time.time() - self._start) * 30)

    def latest(self):
        t = time.time() - self._start
        frame = np.full((self.height, self.width, 3), (115, 105, 85), np.uint8)
        tag = self._tags[int(t // 4) % len(self._tags)]
        phase = (t % 4.0) / 4.0
        scale = 0.7 + 0.5 * math.sin(math.pi * phase)
        tag = cv2.resize(tag, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        th, tw = tag.shape[:2]
        x = int(10 + phase * (self.width - tw - 20))
        y = max(0, (self.height - th) // 2 - 50)
        frame[y:y + th, x:x + tw] = tag
        for i, bgr in enumerate(((40, 40, 220), (40, 210, 230), (210, 90, 30))):
            cv2.circle(frame, (int(self.width * (0.2 + 0.3 * i)), self.height - 60), 32, bgr, -1)
        return cv2.add(frame, np.random.randint(0, 12, frame.shape, np.uint8))

    def stop(self):
        pass


# =============================================================================
# Serial link to the ESP32
# =============================================================================

class SerialLink:
    """Serial connection to the ESP32 with automatic reconnect and a background reader."""

    PORT_HINTS = ("cp210", "ch34", "wch", "usb serial", "usb-serial", "uart", "esp32", "silicon labs", "jtag")

    def __init__(self, port, baud, dry_run, log, on_connect=None, on_lost=None):
        self.requested_port = port
        self.baud = baud
        self.dry_run = dry_run
        self.log = log
        self.on_connect = on_connect
        self.on_lost = on_lost
        self.ser = None
        self._lock = threading.Lock()
        self._incoming = queue.Queue()
        self._reader_failed = False
        self._next_attempt = 0.0
        self._last_error = None
        self._running = True
        if not dry_run:
            if serial is None:
                sys.exit("pyserial is not installed. Run: pip install pyserial   (or use --dry-run)")
            threading.Thread(target=self._reader, daemon=True).start()

    @property
    def ready(self):
        return self.dry_run or self.ser is not None

    @property
    def status(self):
        if self.dry_run:
            return "dry run"
        if self.ser is not None:
            return f"{self.ser.port} connected"
        return "ROV not connected"

    def _find_port(self):
        if self.requested_port:
            return self.requested_port
        for p in serial.tools.list_ports.comports():
            text = f"{p.description} {p.manufacturer or ''} {p.hwid}".lower()
            if any(hint in text for hint in self.PORT_HINTS):
                return p.device
        return None

    def _report(self, message):
        if message != self._last_error:
            self.log(message)
            self._last_error = message

    def maintain(self):
        """Call every frame: reconnects when needed and reports a lost connection."""
        if self.dry_run:
            return
        if self._reader_failed:
            self._reader_failed = False
            self._drop("read error")
        if self.ser is not None or time.time() < self._next_attempt:
            return
        self._next_attempt = time.time() + 2.0
        port = self._find_port()
        if port is None:
            self._report("No ESP32 serial port found. Plug it in, or use --port (see --list-ports).")
            return
        try:
            s = serial.Serial()
            s.port, s.baudrate = port, self.baud
            s.timeout, s.write_timeout = 0.1, 0.1
            s.dtr = False   # stops many ESP32 boards from rebooting when the port opens
            s.rts = False
            s.open()
        except (serial.SerialException, OSError) as exc:
            self._report(f"Could not open {port}: {exc}")
            return
        with self._lock:
            self.ser = s
        self._last_error = None
        self.log(f"Serial connected on {port}")
        if self.on_connect:
            self.on_connect()

    def _drop(self, reason):
        with self._lock:
            s, self.ser = self.ser, None
        if s is None:
            return
        try:
            s.close()
        except Exception:
            pass
        self.log(f"Serial connection lost ({reason})")
        if self.on_lost:
            self.on_lost()

    def send(self, line):
        if self.dry_run:
            return True
        error = None
        with self._lock:
            if self.ser is None:
                return False
            try:
                self.ser.write((line + "\n").encode("utf-8"))
            except (serial.SerialException, OSError) as exc:
                error = exc
        if error is None:
            return True
        self._drop(str(error))
        return False

    def _reader(self):
        while self._running:
            s = self.ser
            if s is None:
                time.sleep(0.1)
                continue
            try:
                raw = s.readline()
            except Exception:
                if self.ser is s:
                    self._reader_failed = True
                time.sleep(0.2)
                continue
            if raw:
                self._incoming.put(raw.decode("utf-8", errors="replace").strip())

    def poll(self):
        lines = []
        while True:
            try:
                lines.append(self._incoming.get_nowait())
            except queue.Empty:
                return lines

    def close(self):
        self._running = False
        with self._lock:
            s, self.ser = self.ser, None
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


# =============================================================================
# Text entry box (Wi-Fi details, pole order)
# =============================================================================

class Prompt:
    def __init__(self, title, fields, on_done):
        """fields: list of (label, default text, validator or None)."""
        self.title = title
        self.fields = fields
        self.on_done = on_done
        self.index = 0
        self.values = []
        self.text = fields[0][1]
        self.error = ""

    @property
    def label(self):
        return self.fields[self.index][0]

    def key(self, event):
        """Handle a key press. Returns True when the box should close."""
        if event.key == pygame.K_ESCAPE:
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            validator = self.fields[self.index][2]
            error = validator(self.text) if validator else None
            if error:
                self.error = error
                return False
            self.values.append(self.text)
            self.index += 1
            if self.index == len(self.fields):
                self.on_done(self.values)
                return True
            self.text, self.error = self.fields[self.index][1], ""
            return False
        if event.key == pygame.K_BACKSPACE:
            self.text, self.error = self.text[:-1], ""
        elif event.key == pygame.K_DELETE:
            self.text, self.error = "", ""
        elif event.unicode and event.unicode.isprintable():
            self.text += event.unicode
            self.error = ""
        return False


# =============================================================================
# Main application
# =============================================================================

class App:
    def __init__(self, args):
        pygame.display.init()   # only what we need: no audio, so no sound-card errors
        pygame.font.init()
        pygame.joystick.init()
        pygame.display.set_caption("ROV control")
        try:
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED | pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        mono = "consolas,menlo,dejavusansmono,couriernew"
        self.font = pygame.font.SysFont(mono, 15)
        self.font_bold = pygame.font.SysFont(mono, 15, bold=True)
        self.font_small = pygame.font.SysFont(mono, 13)
        self.font_big = pygame.font.SysFont(mono, 22, bold=True)
        self.clock = pygame.time.Clock()

        self.log_lines = deque(maxlen=60)
        self.running = True
        self._closed = False
        self.armed = False
        self.hook_open = False
        self.speed_index = DEFAULT_SPEED_INDEX
        self.mode = args.mode
        self.family = args.family
        self.tags_on = True
        self.colours_on = False
        self.gallery = False
        self.show_help = False
        self.prompt = None
        self.reset_pressed_at = 0.0
        self.wifi = None
        self.pole_order = ""
        self.wifi_message = ""
        self.joystick = None
        self.cmd = {axis: 0.0 for axis in AXES_ORDER}
        self.thrust = [0] * len(THRUSTERS)
        self.in_view = []
        self.colours_in_view = []
        self.frame_surface = None
        self.last_display = None
        self.last_frame_id = -1
        self.last_sent = ""
        self.next_send = 0.0
        self._cache = {}
        self._cache_version = -1

        self.detector = TagDetector(args.family)
        self.tracker = TagTracker(CAPTURE_ROOT)
        if args.camera == "demo":
            self.camera = DemoSource(CAMERA_WIDTH, CAMERA_HEIGHT, args.family)
        else:
            source = int(args.camera) if args.camera.isdigit() else args.camera
            self.camera = CameraSource(source, CAMERA_WIDTH, CAMERA_HEIGHT)
        self.serial = SerialLink(args.port, args.baud, args.dry_run, self.log,
                                 on_connect=self.on_serial_connect, on_lost=self.on_serial_lost)
        self.log("Ready. Press F1 for help.")
        if args.dry_run:
            self.log("Dry run: commands are shown but not sent.")

    # ---------------------------------------------------------------- logging
    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"{stamp} {text}")
        print(f"[{stamp}] {text}")

    # ---------------------------------------------------------------- main loop
    def run(self):
        while self.running:
            self.step()

    def step(self):
        for event in pygame.event.get():
            self.handle_event(event)
        self.serial.maintain()
        self.handle_serial_lines()
        self.update_camera()
        self.cmd = self.read_commands()
        speed = SPEED_STEPS[self.speed_index]
        self.thrust = mix(self.cmd, speed) if self.armed else [0] * len(THRUSTERS)
        self.send_command()
        self.tracker.flush()
        self.draw()
        pygame.display.flip()
        self.clock.tick(60)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.armed = False
        stop = "T,0,0," + ",".join("0" for _ in THRUSTERS)
        for _ in range(3):
            self.serial.send(stop)
            time.sleep(0.03)
        self.serial.close()
        self.camera.stop()
        self.tracker.flush(force=True)
        pygame.quit()

    # ---------------------------------------------------------------- serial
    def send_text(self, line):
        """Send a one-off command (not the regular T line)."""
        if self.serial.dry_run:
            self.log("Dry run, not sent: " + line.replace("\t", " | "))
        elif not self.serial.send(line):
            self.log("Not sent: ROV not connected")

    def send_command(self):
        now = time.time()
        if now < self.next_send:
            return
        self.next_send = now + 1.0 / SEND_HZ
        values = ",".join(str(v) for v in self.thrust)
        self.last_sent = f"T,{int(self.armed)},{int(self.hook_open)},{values}"
        self.serial.send(self.last_sent)

    def on_serial_connect(self):
        # The ESP32 may have rebooted, so send the mission settings again.
        if self.wifi:
            self.serial.send("WIFI\t" + "\t".join(self.wifi))
        if self.pole_order:
            self.serial.send("ORDER\t" + self.pole_order)

    def on_serial_lost(self):
        self.disarm("serial connection lost")

    def handle_serial_lines(self):
        for line in self.serial.poll():
            if line.startswith("MSG\t"):
                self.wifi_message = line[4:]
                self.log("Wi-Fi message received from the ROV")
            elif line.startswith("LOG\t"):
                self.log("ROV: " + line[4:])
            elif line:
                self.log("ROV: " + line)

    # ---------------------------------------------------------------- camera
    def update_camera(self):
        frame_id = self.camera.frame_id
        if frame_id == self.last_frame_id:
            return
        frame = self.camera.latest()
        if frame is None:
            return
        self.last_frame_id = frame_id
        display = frame.copy()
        if self.tags_on:
            detections = self.detector.detect(frame)
            self.in_view = sorted({d[0] for d in detections})
            for tag_id in self.tracker.update(detections, frame):
                self.log(f"New AprilTag captured: ID {tag_id}")
            draw_tag_overlays(display, detections, self.tracker.target(self.mode))
        else:
            self.in_view = []
        if self.colours_on:
            found = detect_colours(frame)
            self.colours_in_view = sorted({f[0] for f in found}, key="RYB".index)
            draw_colour_overlays(display, found)
        else:
            self.colours_in_view = []
        self.last_display = display
        size = fit_size(display.shape[1], display.shape[0], VIEW.width, VIEW.height)
        self.frame_surface = cv_to_surface(display, size)

    def save_photo(self):
        if self.last_display is None:
            self.log("No camera picture to save yet")
            return
        self.tracker.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.tracker.capture_dir / f"photo_{datetime.now():%H-%M-%S}.jpg"
        cv2.imwrite(str(path), self.last_display)
        self.log(f"Saved {path}")

    # ---------------------------------------------------------------- input
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.JOYDEVICEADDED:
            if self.joystick is None:
                self.joystick = pygame.joystick.Joystick(event.device_index)
                self.joystick.init()
                self.log(f"Controller connected: {self.joystick.get_name()}")
        elif event.type == pygame.JOYDEVICEREMOVED:
            if self.joystick is not None and event.instance_id == self.joystick.get_instance_id():
                self.joystick = None
                self.disarm("controller unplugged")
                self.log("Controller disconnected")
        elif event.type == pygame.JOYBUTTONDOWN:
            if self.joystick is not None and event.instance_id == self.joystick.get_instance_id():
                self.on_button(event.button)
        elif event.type == pygame.KEYDOWN:
            if self.prompt is not None:
                if self.prompt.key(event):
                    self.prompt = None
            else:
                self.on_key(event)

    def on_button(self, button):
        if button == BUTTON["arm"]:
            self.toggle_arm()
        elif button == BUTTON["disarm"]:
            self.disarm("gamepad")
        elif button == BUTTON["hook"]:
            self.toggle_hook()
        elif button == BUTTON["photo"]:
            self.save_photo()
        elif button == BUTTON["speed_down"]:
            self.set_speed(self.speed_index - 1)
        elif button == BUTTON["speed_up"]:
            self.set_speed(self.speed_index + 1)
        else:
            self.log(f"Gamepad button {button} is not assigned")

    def on_key(self, event):
        k = event.key
        speed_keys = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}
        if k == pygame.K_q and event.mod & pygame.KMOD_CTRL:
            self.running = False
        elif k == pygame.K_ESCAPE:
            self.show_help = False
            self.gallery = False
        elif k == pygame.K_F1:
            self.show_help = not self.show_help
        elif k == pygame.K_SPACE:
            self.toggle_arm()
        elif k == pygame.K_x:
            self.disarm("X key")
        elif k == pygame.K_h:
            self.toggle_hook()
        elif k in speed_keys:
            self.set_speed(speed_keys[k])
        elif k == pygame.K_g:
            self.gallery = not self.gallery
        elif k == pygame.K_m:
            self.mode = "smallest" if self.mode == "largest" else "largest"
            self.log(f"Mission 2.1 mode: {self.mode}")
        elif k == pygame.K_t:
            self.tags_on = not self.tags_on
            self.log(f"AprilTag detection {'on' if self.tags_on else 'off'}")
        elif k == pygame.K_c:
            self.colours_on = not self.colours_on
            self.log(f"Colour detection {'on' if self.colours_on else 'off'}")
        elif k == pygame.K_p:
            self.save_photo()
        elif k == pygame.K_F2:
            self.open_wifi_prompt()
        elif k == pygame.K_F3:
            self.prompt = Prompt("Mission 5 pole order",
                                 [("Order from the referee, e.g. R-B-Y", self.pole_order, validate_order)],
                                 self.apply_order)
        elif k == pygame.K_F4:
            if self.wifi is None:
                self.log("Enter the Wi-Fi details first (F2)")
            else:
                self.send_text("FETCH")
        elif k == pygame.K_F5:
            now = time.time()
            if now - self.reset_pressed_at < 2.0:
                self.tracker.reset()
                self.reset_pressed_at = 0.0
                self.log("Captured tags cleared. Earlier photos stay on disk.")
            else:
                self.reset_pressed_at = now
                self.log("Press F5 again within 2 s to clear captured tags")

    def read_commands(self):
        cmd = {axis: 0.0 for axis in AXES_ORDER}
        if self.joystick is not None:
            count = self.joystick.get_numaxes()
            for name, axis in AXIS.items():
                if axis < count:
                    value = self.joystick.get_axis(axis)
                    if AXIS_INVERT.get(name):
                        value = -value
                    cmd[name] += shape_axis(value)
        if self.prompt is None:
            keys = pygame.key.get_pressed()
            cmd["surge"] += keys[pygame.K_w] - keys[pygame.K_s]
            cmd["yaw"] += keys[pygame.K_d] - keys[pygame.K_a]
            cmd["sway"] += keys[pygame.K_e] - keys[pygame.K_q]
            cmd["heave"] += keys[pygame.K_r] - keys[pygame.K_f]
        return {name: clamp(value) for name, value in cmd.items()}

    def toggle_arm(self):
        if self.armed:
            self.disarm("operator")
            return
        if not self.serial.ready:
            self.log("Cannot arm: ROV not connected (use --dry-run to practise without it)")
            return
        if any(abs(v) > 0 for v in self.cmd.values()):
            self.log("Cannot arm: centre the sticks and release the drive keys first")
            return
        self.armed = True
        self.log("ARMED")

    def disarm(self, reason):
        if self.armed:
            self.armed = False
            self.log(f"Disarmed ({reason})")

    def toggle_hook(self):
        self.hook_open = not self.hook_open
        self.log(f"Hook {'open' if self.hook_open else 'closed'}")

    def set_speed(self, index):
        self.speed_index = max(0, min(len(SPEED_STEPS) - 1, index))
        self.log(f"Speed {int(SPEED_STEPS[self.speed_index] * 100)}%")

    def open_wifi_prompt(self):
        ssid, password, url = self.wifi or ("UR_Field", "12345678", "http://")
        fields = [
            ("Wi-Fi name (SSID)", ssid, validate_not_empty),
            ("Wi-Fi password", password, None),
            ("Server address from the referee", url, validate_url),
        ]
        self.prompt = Prompt("Mission 1 Wi-Fi details", fields, self.apply_wifi)

    def apply_wifi(self, values):
        ssid, password, url = values
        self.wifi = (ssid, password, url.strip())
        self.wifi_message = ""
        self.log(f"Sending Wi-Fi details for {ssid} and asking the ROV to fetch")
        self.send_text("WIFI\t" + "\t".join(self.wifi))
        self.send_text("FETCH")

    def apply_order(self, values):
        self.pole_order = "-".join(p.strip().upper() for p in values[0].split("-"))
        self.log(f"Pole order set to {self.pole_order}")
        self.send_text("ORDER\t" + self.pole_order)

    # ---------------------------------------------------------------- drawing
    def text(self, s, x, y, colour=TEXT, font=None, max_width=None):
        font = font or self.font
        if max_width is not None and font.size(s)[0] > max_width:
            while s and font.size(s + "...")[0] > max_width:
                s = s[:-1]
            s += "..."
        surf = font.render(s, True, colour)
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def text_centered(self, s, rect, colour=TEXT, font=None):
        surf = (font or self.font).render(s, True, colour)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def wrap(self, s, width, font):
        lines, line = [], ""
        for word in s.split(" "):
            trial = word if not line else f"{line} {word}"
            if font.size(trial)[0] <= width:
                line = trial
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    def pill(self, x, y, label, colour):
        surf = self.font_bold.render(label, True, (20, 22, 26))
        rect = pygame.Rect(x, y, surf.get_width() + 16, 26)
        pygame.draw.rect(self.screen, colour, rect, border_radius=6)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        return rect.right + 8

    def fitted(self, key, image, box_size):
        """Cached, aspect-correct surface for gallery images."""
        if self._cache_version != self.tracker.version:
            self._cache.clear()
            self._cache_version = self.tracker.version
        cache_key = (key, box_size)
        if cache_key not in self._cache:
            size = fit_size(image.shape[1], image.shape[0], *box_size)
            self._cache[cache_key] = cv_to_surface(image, size)
        return self._cache[cache_key]

    def draw(self):
        self.screen.fill(BG)
        self.draw_top_bar()
        if self.gallery:
            self.draw_gallery()
        else:
            self.draw_camera()
        self.draw_panel()
        self.draw_bottom_bar()
        if self.prompt is not None:
            self.draw_prompt()
        if self.show_help:
            self.draw_help()

    def draw_top_bar(self):
        self.text("ROV control", 12, 11, TEXT, self.font_big)
        x = 180
        if self.joystick is not None:
            x = self.pill(x, 10, self.joystick.get_name()[:24], GOOD)
        else:
            x = self.pill(x, 10, "keyboard only", MUTED)
        if self.serial.dry_run:
            colour = WARN
        else:
            colour = GOOD if self.serial.ready else BAD
        x = self.pill(x, 10, self.serial.status, colour)
        camera_ok = self.frame_surface is not None and self.camera.status in ("ok", "demo")
        label = f"camera {self.camera.status}"
        if len(label) > 34:
            label = label[:31] + "..."
        self.pill(x, 10, label, GOOD if camera_ok else BAD)
        label = "ARMED" if self.armed else "DISARMED"
        surf = self.font_big.render(label, True, (20, 22, 26))
        rect = pygame.Rect(0, 6, surf.get_width() + 28, 34)
        rect.right = WIN_W - 10
        pygame.draw.rect(self.screen, BAD if self.armed else (95, 100, 112), rect, border_radius=6)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw_camera(self):
        pygame.draw.rect(self.screen, (0, 0, 0), VIEW)
        if self.frame_surface is None:
            self.text_centered(f"Waiting for camera: {self.camera.status}", VIEW, MUTED)
            return
        self.screen.blit(self.frame_surface, self.frame_surface.get_rect(center=VIEW.center))
        info = (f"{self.camera.fps:.0f} fps   tags {'on' if self.tags_on else 'off'}   "
                f"colours {'on' if self.colours_on else 'off'}   G: gallery")
        surf = self.font_small.render(info, True, TEXT)
        back = pygame.Surface((surf.get_width() + 12, surf.get_height() + 6), pygame.SRCALPHA)
        back.fill((0, 0, 0, 150))
        self.screen.blit(back, (VIEW.x + 6, VIEW.y + 6))
        self.screen.blit(surf, (VIEW.x + 12, VIEW.y + 9))

    def draw_gallery(self):
        area = VIEW
        pygame.draw.rect(self.screen, (14, 16, 20), area)
        target = self.tracker.target(self.mode)
        low, high = self.tracker.lowest(), self.tracker.highest()
        header = f"Captured AprilTags   assigned: {self.mode}   target: {target if target is not None else '-'}"
        self.text(header, area.x + 10, area.y + 8, TEXT, self.font_bold)
        if self.armed:
            self.text("ARMED - press G for full camera", area.right - 290, area.y + 8, BAD, self.font_bold)

        card_w, img_h = 410, 308
        for i, (label, tag_id) in enumerate((("Lowest ID", low), ("Highest ID", high))):
            x = area.x + 10 + i * (card_w + 13)
            is_target = tag_id is not None and tag_id == target
            caption = f"{label}: {tag_id if tag_id is not None else '-'}"
            if tag_id is not None:
                caption += f"   seen {self.tracker.best[tag_id]['time']}"
            if is_target:
                caption += "   TARGET"
            self.text(caption, x, area.y + 36, GOOD if is_target else TEXT, self.font_bold, max_width=card_w)
            box = pygame.Rect(x, area.y + 58, card_w, img_h)
            pygame.draw.rect(self.screen, (0, 0, 0), box)
            if tag_id is None:
                self.text_centered("No tags captured yet", box, MUTED)
            else:
                surf = self.fitted(("card", tag_id), self.tracker.best[tag_id]["image"], box.size)
                self.screen.blit(surf, surf.get_rect(center=box.center))
            pygame.draw.rect(self.screen, GOOD if is_target else LINE, box, 4 if is_target else 1)

        y = area.y + 58 + img_h + 12
        ids = self.tracker.ids
        self.text(f"All captured tags ({len(ids)}): {fmt_ids(ids)}", area.x + 10, y, TEXT, max_width=540)

        live = pygame.Rect(0, 0, 272, 204)
        live.bottomright = (area.right - 10, area.bottom - 10)
        thumb_w, thumb_h = 128, 96
        cols = max(1, (live.x - 10 - (area.x + 10)) // (thumb_w + 8))
        for n, tag_id in enumerate(ids[: cols * 2]):
            r = pygame.Rect(area.x + 10 + (n % cols) * (thumb_w + 8),
                            y + 24 + (n // cols) * (thumb_h + 8), thumb_w, thumb_h)
            pygame.draw.rect(self.screen, (0, 0, 0), r)
            surf = self.fitted(("thumb", tag_id), self.tracker.best[tag_id]["image"], r.size)
            self.screen.blit(surf, surf.get_rect(center=r.center))
            pygame.draw.rect(self.screen, GOOD if tag_id == target else LINE, r, 2 if tag_id == target else 1)
            tag_label = self.font_small.render(f"ID {tag_id}", True, TEXT)
            pygame.draw.rect(self.screen, (0, 0, 0), (r.x + 2, r.y + 2, tag_label.get_width() + 8, 18))
            self.screen.blit(tag_label, (r.x + 6, r.y + 4))
        self.text(f"Photos saved in {self.tracker.capture_dir}", area.x + 10, area.bottom - 22,
                  MUTED, self.font_small, max_width=live.x - area.x - 20)

        pygame.draw.rect(self.screen, (0, 0, 0), live)
        if self.frame_surface is not None:
            size = fit_size(self.frame_surface.get_width(), self.frame_surface.get_height(), *live.size)
            small = pygame.transform.smoothscale(self.frame_surface, size)
            self.screen.blit(small, small.get_rect(center=live.center))
        pygame.draw.rect(self.screen, LINE, live, 1)
        self.text("Live", live.x + 6, live.y + 4, TEXT, self.font_small)

    def section(self, x, y, w, title):
        self.text(title, x, y, ACCENT, self.font_bold)
        pygame.draw.line(self.screen, LINE, (x, y + 20), (x + w, y + 20))
        return y + 25

    def draw_panel(self):
        p = PANEL
        pygame.draw.rect(self.screen, PANEL_BG, p, border_radius=8)
        x, w = p.x + 12, p.width - 24
        y = self.section(x, p.y + 8, w, "Drive")
        speed = int(SPEED_STEPS[self.speed_index] * 100)
        self.text(f"Speed {speed}%    Hook {'OPEN' if self.hook_open else 'closed'}", x, y)
        y += 19
        if self.joystick is not None:
            axes = [self.joystick.get_axis(i) for i in range(min(6, self.joystick.get_numaxes()))]
            for start in range(0, len(axes), 3):
                part = "  ".join(f"{i}:{axes[i]:+.2f}" for i in range(start, min(start + 3, len(axes))))
                self.text(("Axes  " if start == 0 else "      ") + part, x, y, MUTED, max_width=w)
                y += 19
        else:
            self.text("WASD drive, Q/E strafe, R/F up/down", x, y, MUTED, max_width=w)
            y += 19

        y = self.section(x, y + 6, w, "Thrusters")
        bar_x, bar_w = x + 40, w - 40 - 50
        mid = bar_x + bar_w // 2
        for t, value in zip(THRUSTERS, self.thrust):
            self.text(t["name"], x, y)
            pygame.draw.rect(self.screen, (55, 60, 70), (bar_x, y + 5, bar_w, 8), border_radius=4)
            length = int(abs(value) / 100 * (bar_w / 2))
            if length:
                start = mid if value > 0 else mid - length
                pygame.draw.rect(self.screen, ACCENT, (start, y + 5, length, 8), border_radius=4)
            pygame.draw.line(self.screen, MUTED, (mid, y + 2), (mid, y + 16))
            self.text(f"{value:+4d}", bar_x + bar_w + 8, y)
            y += 19

        y = self.section(x, y + 6, w, "AprilTags (Mission 2.1)")
        target = self.tracker.target(self.mode)
        self.text(f"Mode {self.mode}   {self.family}   detect {'on' if self.tags_on else 'OFF'}",
                  x, y, max_width=w)
        y += 19
        self.text(f"In view:  {fmt_ids(self.in_view)}", x, y, max_width=w)
        y += 19
        self.text(f"Captured: {fmt_ids(self.tracker.ids)}", x, y, max_width=w)
        y += 19
        self.text(f"Target:   {target if target is not None else '-'}", x, y,
                  GOOD if target is not None else MUTED, self.font_bold)
        y += 19

        y = self.section(x, y + 6, w, "Wi-Fi message (Mission 1)")
        if self.wifi_message:
            for line in self.wrap(self.wifi_message, w, self.font)[:3]:
                self.text(line, x, y, TEXT, max_width=w)
                y += 19
        else:
            hint = "Waiting for the ROV (F4 fetches again)" if self.wifi else "Press F2 to enter the Wi-Fi details"
            self.text(hint, x, y, MUTED, max_width=w)
            y += 19

        y = self.section(x, y + 6, w, "Poles (Mission 5)")
        self.text(f"Order: {self.pole_order or '- (press F3)'}", x, y, max_width=w)
        y += 19
        colours = " ".join(COLOUR_NAMES[c] for c in self.colours_in_view) or "-"
        if not self.colours_on:
            colours = "detection off (C)"
        self.text(f"Colours in view: {colours}", x, y, max_width=w)
        y += 19

        y = self.section(x, y + 6, w, "Log")
        rows = max(0, (p.bottom - 8 - y) // 17)
        if rows:
            for line in list(self.log_lines)[-rows:]:
                self.text(line, x, y, MUTED, self.font_small, max_width=w)
                y += 17

    def draw_bottom_bar(self):
        y = WIN_H - 24
        hints = ("F1 help   Space arm/disarm   X disarm   G gallery   P photo   "
                 "F2 Wi-Fi   F3 pole order   Ctrl+Q quit")
        self.text(hints, 12, y, MUTED, self.font_small)
        right = f"UI {self.clock.get_fps():.0f} fps"
        if self.serial.dry_run:
            right = f"would send: {self.last_sent}    {right}"
        surf = self.font_small.render(right, True, MUTED)
        self.screen.blit(surf, (WIN_W - 12 - surf.get_width(), y))

    def dim_screen(self):
        shade = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        self.screen.blit(shade, (0, 0))

    def draw_prompt(self):
        self.dim_screen()
        pr = self.prompt
        box = pygame.Rect(0, 0, 680, 210)
        box.center = (WIN_W // 2, WIN_H // 2)
        pygame.draw.rect(self.screen, PANEL_BG, box, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT, box, 1, border_radius=10)
        self.text(pr.title, box.x + 20, box.y + 16, TEXT, self.font_big)
        self.text(f"{pr.label}   ({pr.index + 1} of {len(pr.fields)})", box.x + 20, box.y + 58, MUTED)
        field = pygame.Rect(box.x + 20, box.y + 84, box.width - 40, 36)
        pygame.draw.rect(self.screen, (18, 20, 24), field, border_radius=6)
        pygame.draw.rect(self.screen, LINE, field, 1, border_radius=6)
        cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
        shown = pr.text
        while shown and self.font.size(shown + "|")[0] > field.width - 20:
            shown = shown[1:]   # keep the end of long text visible
        self.text(shown + cursor, field.x + 10, field.y + 9)
        if pr.error:
            self.text(pr.error, box.x + 20, box.y + 132, BAD, max_width=box.width - 40)
        self.text("Enter: confirm    Esc: cancel    Delete: clear", box.x + 20, box.bottom - 30,
                  MUTED, self.font_small)

    def draw_help(self):
        self.dim_screen()
        box = pygame.Rect(0, 0, 940, 470)
        box.center = (WIN_W // 2, WIN_H // 2)
        pygame.draw.rect(self.screen, PANEL_BG, box, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT, box, 1, border_radius=10)
        self.text("Controls", box.x + 24, box.y + 16, TEXT, self.font_big)
        for col, rows in enumerate((HELP_LEFT, HELP_RIGHT)):
            x = box.x + 24 + col * 460
            y = box.y + 56
            for key, desc in rows:
                if desc is None:
                    y += 4
                    self.text(key, x, y, ACCENT, self.font_bold)
                else:
                    self.text(key, x, y, TEXT, self.font_bold)
                    self.text(desc, x + 130, y, MUTED, max_width=310)
                y += 25


# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ROV topside control program")
    parser.add_argument("--camera", default="0",
                        help="camera number (0, 1, ...), a video file path, or 'demo' for a fake scene")
    parser.add_argument("--port", default=None,
                        help="serial port, e.g. COM3 or /dev/ttyUSB0 (found automatically if left out)")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--dry-run", action="store_true",
                        help="run without a ROV: commands are shown on screen but not sent")
    parser.add_argument("--family", choices=sorted(TAG_FAMILIES), default="36h11",
                        help="AprilTag family (ask the organisers which one is used)")
    parser.add_argument("--mode", choices=("largest", "smallest"), default="largest",
                        help="your assigned Mission 2.1 rule")
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_ports:
        if serial is None:
            sys.exit("pyserial is not installed. Run: pip install pyserial")
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            print(f"{p.device:15} {p.description}")
        if not ports:
            print("No serial ports found.")
        return
    app = App(args)
    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
