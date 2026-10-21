"""New, separate integration runner.

Dry-run is the default. Existing teammate programs remain independently
runnable and are not imported by this file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mission4_hook_control import HookController
from mission5_adapter import Mission5Adapter
from serial_protocol import SerialTransport, ThrusterMixer, list_serial_ports


def open_camera(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def draw_status(frame, snapshot, pulses, dry_run, hook_open):
    lines = [
        f"Mission 5: {'RUNNING' if snapshot.running else 'PAUSED'}",
        f"Target: {snapshot.target or '-'}   State: {snapshot.state}",
        f"Command: surge {snapshot.surge:+.2f}  yaw {snapshot.yaw:+.2f}  heave {snapshot.heave:+.2f}",
        f"PWM: {pulses.right} {pulses.left} {pulses.vertical}",
        f"Hook: {'OPEN' if hook_open else 'CLOSED'}   Serial: {'DRY RUN' if dry_run else 'LIVE'}",
        "SPACE start/pause | X stop | Y/N confirm | S skip | R reset | O/C hook | Q quit",
    ]
    y = 26
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def run_mission5(args) -> int:
    dry_run = not args.live
    if args.live and not args.port:
        raise SystemExit("--port is required with --live; use --list-ports first")
    if args.send_hz <= 0:
        raise SystemExit("--send-hz must be greater than zero")

    transport = SerialTransport(args.port, args.baud, dry_run=dry_run)
    mixer = ThrusterMixer(
        span_us=args.span,
        reverse_right=args.reverse_right,
        reverse_left=args.reverse_left,
        reverse_vertical=args.reverse_vertical,
    )
    hook = HookController(transport)
    mission = Mission5Adapter(args.order, auto_confirm=args.auto_confirm)
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera or video {args.camera!r}")

    try:
        transport.open()
        transport.send_order(args.order)
        last_send = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera or video ended.")
                break
            now = time.monotonic()
            snapshot = mission.update_frame(frame, now)
            pulses = mixer.mix(snapshot.surge, snapshot.yaw, snapshot.heave, args.speed)

            # Repeating commands keeps the firmware watchdog alive. A paused
            # mission stays at neutral and never arms itself automatically.
            if mission.running and transport.armed and now - last_send >= 1.0 / args.send_hz:
                transport.send_thrusters(pulses)
                last_send = now

            draw_status(frame, snapshot, pulses, dry_run, hook.is_open)
            cv2.imshow("New integration - Mission 5", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if mission.running:
                    mission.pause()
                    transport.stop()
                    transport.disarm()
                else:
                    transport.arm()
                    mission.start()
            elif key == ord("x"):
                mission.pause()
                transport.stop()
                transport.disarm()
            elif key == ord("y"):
                mission.confirm(True, now)
            elif key == ord("n"):
                mission.confirm(False, now)
            elif key == ord("s"):
                mission.skip(now)
            elif key == ord("r"):
                mission.reset()
                transport.stop()
                transport.disarm()
            elif key == ord("o"):
                hook.open()
            elif key == ord("c"):
                hook.close()
    finally:
        mission.pause()
        try:
            if transport.ready:
                transport.stop()
        finally:
            transport.close()
            cap.release()
            cv2.destroyAllWindows()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Separate whole-bot integration additions")
    sub = parser.add_subparsers(dest="command", required=True)

    ports = sub.add_parser("ports", help="list serial ports and exit")
    ports.set_defaults(handler=lambda _args: print_ports())

    mission5 = sub.add_parser("mission5", help="run the new Mission 5 adapter")
    mission5.add_argument("--order", required=True, help="referee order such as R-B-Y")
    mission5.add_argument("--camera", default="0", help="camera index or video path")
    mission5.add_argument("--live", action="store_true", help="enable a real serial port")
    mission5.add_argument("--port", help="serial port, for example COM3")
    mission5.add_argument("--baud", type=int, default=115200)
    mission5.add_argument("--speed", type=float, default=0.35,
                          help="maximum autonomy output during initial testing (default 0.35)")
    mission5.add_argument("--span", type=int, default=250,
                          help="PWM range around 1500 during initial testing (default 250 us)")
    mission5.add_argument("--send-hz", type=float, default=20.0)
    mission5.add_argument("--auto-confirm", action="store_true")
    mission5.add_argument("--reverse-right", action="store_true")
    mission5.add_argument("--reverse-left", action="store_true")
    mission5.add_argument("--reverse-vertical", action="store_true")
    mission5.set_defaults(handler=run_mission5)
    return parser


def print_ports() -> int:
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found.")
    for device, description in ports:
        print(f"{device:15} {description}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

