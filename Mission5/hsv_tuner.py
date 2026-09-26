"""Tune the HSV colour ranges on the live camera and save them to colors.json.

Run:
    python hsv_tuner.py --camera 1

Keys:
    1 / 2 / 3  pick colour to edit: red / yellow / blue
    w          toggle white balance (should match what mission5.py uses)
    c          click-sample mode: click on the pole to centre the sliders on it
    s          save colors.json
    q / ESC    quit (without saving)

Red wraps around hue 0, so it has two ranges: the sliders edit the low range
(0..H max) and the high range is mirrored automatically (180-H max .. 179).
"""

import argparse

import cv2
import numpy as np

from pole_detector import COLORS, CONFIG_PATH, PoleDetector, load_config, save_config
from mission5 import open_source

WIN = "HSV tuner"
SLIDERS = ["H min", "H max", "S min", "S max", "V min", "V max"]


def set_sliders(rng):
    (h0, s0, v0), (h1, s1, v1) = rng
    for name, val in zip(SLIDERS, (h0, h1, s0, s1, v0, v1)):
        cv2.setTrackbarPos(name, WIN, int(val))


def read_sliders():
    h0, h1, s0, s1, v0, v1 = (cv2.getTrackbarPos(n, WIN) for n in SLIDERS)
    return [[h0, s0, v0], [h1, s1, v1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0")
    args = ap.parse_args()

    cfg = load_config()
    det = PoleDetector(cfg)
    cap = open_source(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video {args.camera!r}")

    cv2.namedWindow(WIN)
    for n in SLIDERS:
        cv2.createTrackbar(n, WIN, 0, 179 if n.startswith("H") else 255, lambda v: None)
    current = "red"
    set_sliders(cfg["ranges"][current][0])
    state = {"hsv": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(det.preprocess(frame), cv2.COLOR_BGR2HSV)

        if state["click"] is not None:
            x, y = state["click"]
            state["click"] = None
            if y < hsv.shape[0] and x < hsv.shape[1]:
                patch = hsv[max(0, y - 5):y + 6, max(0, x - 5):x + 6].reshape(-1, 3)
                h, s, v = np.median(patch, axis=0).astype(int)
                print(f"clicked HSV = ({h}, {s}, {v})")
                if current == "red" and h > 90:
                    h = 179 - h  # edit red in its low-hue mirror
                set_sliders([[max(0, h - 8), max(0, s - 60), max(0, v - 70)],
                             [min(179, h + 8), 255, 255]])

        rng = read_sliders()
        if current == "red":
            lo, hi = rng
            cfg["ranges"]["red"] = [rng, [[179 - hi[0], lo[1], lo[2]], [179, hi[1], hi[2]]]]
        else:
            cfg["ranges"][current] = [rng]

        mask = det.mask(hsv, current)
        found = det.detect(frame)[current]
        view = frame.copy()
        if found is not None:
            x, y, w, h = found.bbox
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(view, f"editing {current.upper()}  (1/2/3 switch, s save)  WB={'on' if cfg['white_balance'] else 'off'}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow(WIN, cv2.hconcat([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("1"), ord("2"), ord("3")):
            current = COLORS[key - ord("1")]
            set_sliders(cfg["ranges"][current][0])
        elif key == ord("w"):
            cfg["white_balance"] = not cfg["white_balance"]
        elif key == ord("s"):
            save_config(cfg)
            print("saved", CONFIG_PATH)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
