"""Mission 5: find the coloured poles and hit them in the referee's order.

Run one of the three modes:
    python mission5.py --mode camera --camera 0
    python mission5.py --mode tune --camera 0
    python mission5.py --mode mission --order R-B-Y --camera 0

All Mission 5 runtime tools are intentionally kept in this one file.

The ROV starts PAUSED. Press SPACE to start moving.

Keys (in the video window):
    SPACE  start / pause autonomy (pause = all thrusters zero)
    y      ball dropped -> move on to the next pole
    n      ball did NOT drop -> retry the same pole
    s      skip the current pole
    r      restart the sequence from the first pole
    m      show / hide the colour masks (for checking the HSV tuning)
    q/ESC  quit

For each pole the controller runs:
    SEARCH  -> yaw on the spot until the target colour is seen for a few frames
    REPOSITION -> nothing found after a full spin: drive forward a bit, search again
    ALIGN   -> yaw until the pole is centred
    APPROACH-> drive forward, keep steering onto the pole, slow down as it grows
    HIT     -> short blind forward push (pole fills the view, can't track it)
    BACK_OFF-> reverse away from the pole
    CONFIRM -> stop and wait for 'y' / 'n' (or auto-confirm with --auto-confirm)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour-pole detection
# ---------------------------------------------------------------------------

COLORS = ("red", "yellow", "blue")
DRAW_BGR = {"red": (0, 0, 255), "yellow": (0, 220, 255), "blue": (255, 80, 0)}

# OpenCV HSV uses H 0-179 and S/V 0-255. Red wraps around hue zero, so it
# needs two ranges.
DEFAULT_CONFIG = {
    "white_balance": True,
    "blur_ksize": 5,
    "min_area_frac": 0.001,
    "max_area_frac": 0.90,
    "min_aspect": 1.3,
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
    bbox: tuple
    center: tuple
    area: float
    err_x: float
    err_y: float
    height_frac: float
    width_frac: float
    area_frac: float
    score: float


def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        ranges = user.pop("ranges", {})
        cfg.update(user)
        cfg["ranges"].update(ranges)
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def white_balance_gains(bgr):
    sample = bgr[::4, ::4].reshape(-1, 3)
    med = np.median(sample, axis=0).astype(np.float32)
    gain = med.mean() / np.maximum(med, 1.0)
    return np.clip(gain, 0.5, 3.0)


def gray_world_white_balance(bgr, gain=None):
    if gain is None:
        gain = white_balance_gains(bgr)
    lut = np.clip(np.arange(256, dtype=np.float32)[:, None] * gain[None, :], 0, 255)
    return cv2.LUT(bgr, lut.astype(np.uint8).reshape(1, 256, 3))


class PoleDetector:
    def __init__(self, config=None):
        self.cfg = config if config is not None else load_config()
        self._wb_gain = None
        self._open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))

    def preprocess(self, bgr):
        if self.cfg.get("white_balance", True):
            gain = white_balance_gains(bgr)
            self._wb_gain = gain if self._wb_gain is None else 0.9 * self._wb_gain + 0.1 * gain
            bgr = gray_world_white_balance(bgr, self._wb_gain)
        ksize = int(self.cfg.get("blur_ksize", 5))
        if ksize >= 3:
            bgr = cv2.GaussianBlur(bgr, (ksize | 1, ksize | 1), 0)
        return bgr

    def mask(self, hsv, color):
        mask = None
        for lower, upper in self.cfg["ranges"][color]:
            part = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)

    def detect(self, bgr, return_masks=False):
        height, width = bgr.shape[:2]
        frame_area = float(height * width)
        hsv = cv2.cvtColor(self.preprocess(bgr), cv2.COLOR_BGR2HSV)
        results, masks = {}, {}
        for color in COLORS:
            mask = self.mask(hsv, color)
            masks[color] = mask
            results[color] = self._best_blob(mask, color, width, height, frame_area)
        return (results, masks) if return_masks else results

    def _best_blob(self, mask, color, width, height, frame_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            area_frac = area / frame_area
            if area_frac < self.cfg["min_area_frac"] or area_frac > self.cfg["max_area_frac"]:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = h / max(w, 1)
            touches_edge = y <= 2 or y + h >= height - 2
            if aspect < self.cfg["min_aspect"] and not touches_edge:
                continue
            solidity = area / max(w * h, 1)
            score = area * (0.5 + solidity) * min(aspect, 6.0)
            if best is None or score > best.score:
                center_x, center_y = x + w / 2.0, y + h / 2.0
                best = Detection(
                    color=color,
                    bbox=(x, y, w, h),
                    center=(center_x, center_y),
                    area=area,
                    err_x=(center_x - width / 2.0) / (width / 2.0),
                    err_y=(center_y - height / 2.0) / (height / 2.0),
                    height_frac=h / float(height),
                    width_frac=w / float(width),
                    area_frac=area_frac,
                    score=score,
                )
        return best


def draw_detections(frame, detections, target=None):
    for color, detection in detections.items():
        if detection is None:
            continue
        x, y, w, h = detection.bbox
        thickness = 4 if color == target else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), DRAW_BGR[color], thickness)
        label = f"{color} w={detection.width_frac:.2f}"
        cv2.putText(
            frame,
            label,
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            DRAW_BGR[color],
            2 if color == target else 1,
        )
    return frame

LETTER_TO_COLOR = {"R": "red", "Y": "yellow", "B": "blue"}


def parse_order(text):
    """'R-B-Y' -> ['red', 'blue', 'yellow']. Also accepts 'rby', 'R B Y', 'R,B,Y'."""
    letters = [ch for ch in text.upper() if ch.isalpha()]
    if sorted(letters) != ["B", "R", "Y"]:
        raise ValueError(f"order must use R, Y and B exactly once each, e.g. R-B-Y (got {text!r})")
    return [LETTER_TO_COLOR[ch] for ch in letters]


def order_str(order):
    return "-".join(c[0].upper() for c in order)


@dataclass
class Command:
    """Normalised thrust request, each in -1..1. surge +forward, yaw +right, heave +down."""
    surge: float = 0.0
    yaw: float = 0.0
    heave: float = 0.0


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


@dataclass
class ControlParams:
    search_yaw: float = 0.25      # yaw speed while looking for the pole
    search_turn_time: float = 30.0  # TUNE: ~time for one full 360 spin at search_yaw
    reposition_surge: float = 0.4   # nothing found after a full spin -> move and retry
    reposition_time: float = 3.0
    confirm_frames: int = 3       # consecutive sightings needed to leave SEARCH
    kp_yaw: float = 0.6           # yaw = kp_yaw * err_x
    kp_heave: float = 0.0         # set > 0 to also centre vertically
    align_tol: float = 0.15       # |err_x| below this -> start driving forward
    realign_tol: float = 0.35     # |err_x| above this while approaching -> stop and re-align
    approach_surge: float = 0.5   # forward speed when the pole is far away
    min_surge: float = 0.2        # forward speed just before the hit
    # Pole bbox width (fraction of frame width) at which we're close enough to hit.
    # Height is no use for this: a 1 m pole fills the frame height from ~1 m away.
    # TUNE: hold the ROV where the hitter just touches the pole, read w= on screen.
    hit_width_frac: float = 0.12
    hit_surge: float = 0.7
    hit_time: float = 1.0         # seconds of blind forward push
    backoff_surge: float = -0.4
    backoff_time: float = 2.0
    lost_timeout: float = 1.0     # seconds without the pole before going back to SEARCH


class MissionController:
    SEARCH, REPOSITION, ALIGN, APPROACH, HIT, BACK_OFF, CONFIRM, DONE = (
        "SEARCH", "REPOSITION", "ALIGN", "APPROACH", "HIT", "BACK_OFF", "CONFIRM", "DONE")

    def __init__(self, order, params=None, auto_confirm=False):
        self.order = list(order)
        self.p = params or ControlParams()
        self.auto_confirm = auto_confirm
        self.reset()

    def reset(self):
        self.index = 0
        self.results = {}           # color -> "hit" | "skipped"
        self._enter(self.SEARCH, None)  # timer starts on the first update()
        self._last_seen = None
        self._last_side = 1.0       # turn towards where the pole was last seen
        self._seen_count = 0

    @property
    def target(self):
        return self.order[self.index] if self.index < len(self.order) else None

    def _enter(self, state, now):
        self.state = state
        self.state_since = now

    def _next_target(self, now, result):
        self.results[self.target] = result
        self.index += 1
        self._seen_count = 0
        self._last_seen = None
        self._enter(self.DONE if self.target is None else self.SEARCH, now)

    # operator input -------------------------------------------------------
    def confirm(self, dropped, now):
        """Operator says whether the ball dropped. Only meaningful after a hit."""
        if self.state not in (self.CONFIRM, self.BACK_OFF, self.HIT):
            return
        if dropped:
            self._next_target(now, "hit")
        else:
            self._seen_count = 0
            self._enter(self.SEARCH, now)

    def resume(self):
        """Call when un-pausing so time spent paused doesn't count towards timers."""
        self.state_since = None
        self._last_seen = None

    def skip(self, now):
        if self.target is not None:
            self._next_target(now, "skipped")

    # main step --------------------------------------------------------------
    def update(self, detections, now):
        p = self.p
        det = detections.get(self.target) if self.target else None
        if det is not None:
            self._last_seen = now
            self._last_side = 1.0 if det.err_x >= 0 else -1.0
            self._seen_count += 1
        else:
            self._seen_count = 0
        if self.state_since is None:
            self.state_since = now
        elapsed = now - self.state_since

        if self.state == self.DONE:
            return Command()

        if self.state == self.HIT:
            if elapsed >= p.hit_time:
                self._enter(self.BACK_OFF, now)
            return Command(surge=p.hit_surge)

        if self.state == self.BACK_OFF:
            if elapsed >= p.backoff_time:
                if self.auto_confirm:
                    self._next_target(now, "hit")
                else:
                    self._enter(self.CONFIRM, now)
                return Command()
            return Command(surge=p.backoff_surge)

        if self.state == self.CONFIRM:
            return Command()

        if self.state in (self.SEARCH, self.REPOSITION):
            if det is not None and self._seen_count >= p.confirm_frames:
                self._enter(self.ALIGN, now)
            elif self.state == self.SEARCH:
                if elapsed >= p.search_turn_time:
                    self._enter(self.REPOSITION, now)
                    return Command(surge=p.reposition_surge)
                return Command(yaw=p.search_yaw * self._last_side)
            else:
                if elapsed >= p.reposition_time:
                    self._enter(self.SEARCH, now)
                    return Command()
                return Command(surge=p.reposition_surge)

        # ALIGN / APPROACH need the pole in view
        if det is None:
            if self._last_seen is None or now - self._last_seen > p.lost_timeout:
                self._enter(self.SEARCH, now)
                return Command(yaw=p.search_yaw * self._last_side)
            return Command()  # brief dropout: hold still and wait

        yaw = clamp(p.kp_yaw * det.err_x)
        heave = clamp(p.kp_heave * det.err_y)

        if self.state == self.ALIGN:
            if abs(det.err_x) < p.align_tol:
                self._enter(self.APPROACH, now)
            else:
                return Command(yaw=yaw, heave=heave)

        # APPROACH
        if det.width_frac >= p.hit_width_frac:
            self._enter(self.HIT, now)
            return Command(surge=p.hit_surge)
        if abs(det.err_x) > p.realign_tol:
            self._enter(self.ALIGN, now)
            return Command(yaw=yaw, heave=heave)
        # slow down linearly as the pole fills more of the frame
        closeness = clamp(det.width_frac / p.hit_width_frac, 0.0, 1.0)
        surge = p.approach_surge + (p.min_surge - p.approach_surge) * closeness
        return Command(surge=surge, yaw=yaw, heave=heave)


class ConsoleRov:
    """Stand-in for the real thruster link: prints commands when they change.

    To drive the real ROV, write a class with the same send()/stop() methods that
    maps Command.surge / yaw / heave onto your thrusters (e.g. pymavlink
    manual_control for ArduSub, or PWM values over serial to your controller)."""

    def __init__(self, min_period=0.3):
        self.min_period = min_period
        self._last = None
        self._last_t = 0.0

    def send(self, cmd):
        key = (round(cmd.surge, 2), round(cmd.yaw, 2), round(cmd.heave, 2))
        now = time.monotonic()
        if key != self._last and now - self._last_t >= self.min_period:
            print(f"CMD surge={cmd.surge:+.2f} yaw={cmd.yaw:+.2f} heave={cmd.heave:+.2f}")
            self._last, self._last_t = key, now

    def stop(self):
        self.send(Command())


def draw_hud(frame, ctrl, cmd, running, fps):
    H, W = frame.shape[:2]
    p = ctrl.p
    # centre line and align tolerance band
    cv2.line(frame, (W // 2, 0), (W // 2, H), (200, 200, 200), 1)
    for s in (-1, 1):
        x = int(W / 2 + s * p.align_tol * W / 2)
        cv2.line(frame, (x, 0), (x, H), (120, 120, 120), 1)
    # hit threshold: when the target box is this wide, the ROV goes for the hit
    half = int(p.hit_width_frac * W / 2)
    cv2.line(frame, (W // 2 - half, H - 30), (W // 2 + half, H - 30), (0, 255, 0), 3)

    parts = []
    for i, c in enumerate(ctrl.order):
        mark = {"hit": "[x]", "skipped": "[-]"}.get(ctrl.results.get(c), "[>]" if i == ctrl.index else "[ ]")
        parts.append(f"{mark}{c[0].upper()}")
    lines = [
        f"ORDER {order_str(ctrl.order)}   " + " ".join(parts),
        f"TARGET {ctrl.target.upper() if ctrl.target else '-'}   STATE {ctrl.state}",
        f"surge {cmd.surge:+.2f}  yaw {cmd.yaw:+.2f}  heave {cmd.heave:+.2f}   {fps:4.1f} fps",
    ]
    if ctrl.state == ctrl.CONFIRM:
        lines.append("Did the ball drop?  y = yes   n = retry")
    if not running:
        lines.append("PAUSED - press SPACE to start")
    y = 22
    for i, text in enumerate(lines):
        color = (0, 0, 255) if text.startswith("PAUSED") else (0, 255, 255) if "ball" in text else (255, 255, 255)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22
    if ctrl.target:
        cv2.rectangle(frame, (W - 40, 10), (W - 18, 32), DRAW_BGR[ctrl.target], -1)
    help_text = "SPACE start/pause  y/n confirm  s skip  r restart  m masks  q quit"
    cv2.putText(frame, help_text, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def mask_view(masks, size):
    W, H = size
    tiles = []
    for c in COLORS:
        t = cv2.cvtColor(cv2.resize(masks[c], (W // 3, H // 3)), cv2.COLOR_GRAY2BGR)
        cv2.putText(t, c, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_BGR[c], 2)
        tiles.append(t)
    strip = cv2.hconcat(tiles)
    return cv2.resize(strip, (W, strip.shape[0]))  # 3 * (W // 3) may be short of W


def open_source(src):
    source = str(src)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)  # DSHOW opens fast on Windows
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def run_camera_test(camera):
    """Open one camera at a time and show its raw live feed."""
    cap = open_source(camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Could not open camera {camera!r}")

    print(f"Camera {camera} is working. Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            cv2.imshow("Mission 5 - Camera Test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


TUNER_WINDOW = "Mission 5 - HSV Tuner"
TUNER_SLIDERS = ("H min", "H max", "S min", "S max", "V min", "V max")


def _set_tuner_sliders(color_range):
    (h0, s0, v0), (h1, s1, v1) = color_range
    for name, value in zip(TUNER_SLIDERS, (h0, h1, s0, s1, v0, v1)):
        cv2.setTrackbarPos(name, TUNER_WINDOW, int(value))


def _read_tuner_sliders():
    h0, h1, s0, s1, v0, v1 = (
        cv2.getTrackbarPos(name, TUNER_WINDOW) for name in TUNER_SLIDERS
    )
    return [[h0, s0, v0], [h1, s1, v1]]


def run_hsv_tuner(camera):
    """Tune red/yellow/blue HSV ranges using the live camera."""
    config = load_config()
    detector = PoleDetector(config)
    cap = open_source(camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Could not open camera/video {camera!r}")

    cv2.namedWindow(TUNER_WINDOW)
    for name in TUNER_SLIDERS:
        maximum = 179 if name.startswith("H") else 255
        cv2.createTrackbar(name, TUNER_WINDOW, 0, maximum, lambda _value: None)

    current = "red"
    _set_tuner_sliders(config["ranges"][current][0])
    state = {"click": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(TUNER_WINDOW, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            hsv = cv2.cvtColor(detector.preprocess(frame), cv2.COLOR_BGR2HSV)

            if state["click"] is not None:
                x, y = state["click"]
                state["click"] = None
                if y < hsv.shape[0] and x < hsv.shape[1]:
                    patch = hsv[max(0, y - 5):y + 6, max(0, x - 5):x + 6].reshape(-1, 3)
                    hue, saturation, value = np.median(patch, axis=0).astype(int)
                    print(f"Clicked HSV = ({hue}, {saturation}, {value})")
                    if current == "red" and hue > 90:
                        hue = 179 - hue
                    _set_tuner_sliders([
                        [max(0, hue - 8), max(0, saturation - 60), max(0, value - 70)],
                        [min(179, hue + 8), 255, 255],
                    ])

            color_range = _read_tuner_sliders()
            if current == "red":
                lower, upper = color_range
                config["ranges"]["red"] = [
                    color_range,
                    [[179 - upper[0], lower[1], lower[2]], [179, upper[1], upper[2]]],
                ]
            else:
                config["ranges"][current] = [color_range]

            mask = detector.mask(hsv, current)
            detection = detector.detect(frame)[current]
            view = frame.copy()
            if detection is not None:
                x, y, w, h = detection.bbox
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            status = (
                f"Editing {current.upper()}  1/2/3 switch  S save  "
                f"WB={'on' if config['white_balance'] else 'off'}"
            )
            cv2.putText(view, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)
            cv2.imshow(TUNER_WINDOW, cv2.hconcat([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                current = COLORS[key - ord("1")]
                _set_tuner_sliders(config["ranges"][current][0])
            elif key == ord("w"):
                config["white_balance"] = not config["white_balance"]
            elif key == ord("s"):
                save_config(config)
                print("Saved", CONFIG_PATH)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_mission(args):

    while True:
        try:
            order = parse_order(args.order or input("Referee order (e.g. R-B-Y): "))
            break
        except ValueError as e:
            print(e)
            args.order = None
    print("Order:", " -> ".join(order))

    cap = open_source(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video {args.camera!r}")

    detector = PoleDetector(load_config())
    ctrl = MissionController(order, auto_confirm=args.auto_confirm or args.no_gui)
    rov = ConsoleRov()
    running = args.no_gui
    show_masks = False
    writer = None
    fps, t_prev = 0.0, time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera/video ended.")
                break
            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(frame)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

            detections, masks = detector.detect(frame, return_masks=True)
            cmd = ctrl.update(detections, now) if running else Command()
            rov.send(cmd)

            if args.no_gui:
                if ctrl.state == ctrl.DONE:
                    print("Sequence complete:", ctrl.results)
                    break
                continue

            view = draw_detections(frame.copy(), detections, ctrl.target)
            draw_hud(view, ctrl, cmd, running, fps)
            if show_masks:
                view = cv2.vconcat([view, mask_view(masks, (view.shape[1], view.shape[0]))])
            cv2.imshow("Mission 5", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                running = not running
                if running:
                    ctrl.resume()
                print("RUNNING" if running else "PAUSED")
            elif key == ord("y"):
                ctrl.confirm(True, now)
            elif key == ord("n"):
                ctrl.confirm(False, now)
            elif key == ord("s"):
                ctrl.skip(now)
            elif key == ord("r"):
                ctrl.reset()
                print("Sequence restarted")
            elif key == ord("m"):
                show_masks = not show_masks
    except KeyboardInterrupt:
        pass
    finally:
        rov.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Results:", ctrl.results)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Mission 5 camera test, colour tuning, and pole-hitting controller"
    )
    parser.add_argument(
        "--mode",
        choices=("camera", "tune", "mission"),
        default="mission",
        help="camera = raw feed, tune = HSV tuner, mission = controller (default)",
    )
    parser.add_argument("--camera", default="0", help="camera index or video path (default 0)")
    parser.add_argument("--order", help="referee order, e.g. R-B-Y (asked for if omitted)")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="assume the ball dropped after each hit")
    parser.add_argument("--record", help="save raw mission camera video to this .mp4")
    parser.add_argument("--no-gui", action="store_true",
                        help="mission without a window; starts immediately")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "camera":
        run_camera_test(args.camera)
    elif args.mode == "tune":
        run_hsv_tuner(args.camera)
    else:
        run_mission(args)


if __name__ == "__main__":
    main()
