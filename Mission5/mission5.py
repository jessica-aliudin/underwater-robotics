"""Mission 5: find the coloured poles and hit them in the referee's order.

Run:
    python mission5.py --order R-B-Y --camera 1

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
import time
from dataclasses import dataclass

import cv2

from pole_detector import COLORS, DRAW_BGR, PoleDetector, draw_detections, load_config

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
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)  # DSHOW opens fast on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    return cap


def main():
    ap = argparse.ArgumentParser(description="Mission 5 colour pole hitting")
    ap.add_argument("--order", help="referee order, e.g. R-B-Y (asked for if omitted)")
    ap.add_argument("--camera", default="0", help="camera index or video file path (default 0)")
    ap.add_argument("--auto-confirm", action="store_true",
                    help="assume the ball dropped after each hit (no y/n needed)")
    ap.add_argument("--record", help="save the raw camera video to this .mp4 (for tuning later)")
    ap.add_argument("--no-gui", action="store_true",
                    help="no window; starts running immediately, Ctrl+C to stop (implies --auto-confirm)")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
