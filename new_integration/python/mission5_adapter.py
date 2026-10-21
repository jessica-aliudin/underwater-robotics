"""Adapter around the existing standalone Mission 5 modules.

The original files in ``Mission5/`` are imported as-is and are not edited or
copied into this folder.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Mission5Snapshot:
    target: str | None
    state: str
    running: bool
    surge: float
    yaw: float
    heave: float
    detections: dict[str, Any]
    results: dict[str, str]


def _mission5_imports(repository_root: Path | None = None):
    root = repository_root or Path(__file__).resolve().parents[2]
    mission_dir = root / "Mission5"
    if not mission_dir.is_dir():
        raise FileNotFoundError(f"existing Mission5 folder not found at {mission_dir}")
    path = str(mission_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    from mission5 import Command, MissionController, parse_order
    from pole_detector import PoleDetector, load_config
    return Command, MissionController, parse_order, PoleDetector, load_config


class Mission5Adapter:
    def __init__(self, order: str, repository_root: Path | None = None,
                 auto_confirm: bool = False, detector: Any | None = None):
        (self.Command, controller_type, parse_order, detector_type,
         load_config) = _mission5_imports(repository_root)
        self.order = parse_order(order)
        self.controller = controller_type(self.order, auto_confirm=auto_confirm)
        self.detector = detector if detector is not None else detector_type(load_config())
        self.running = False

    def start(self) -> None:
        self.running = True
        self.controller.resume()

    def pause(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.running = False
        self.controller.reset()

    def confirm(self, dropped: bool, now: float | None = None) -> None:
        self.controller.confirm(dropped, time.monotonic() if now is None else now)

    def skip(self, now: float | None = None) -> None:
        self.controller.skip(time.monotonic() if now is None else now)

    def update_detections(self, detections: dict[str, Any],
                          now: float | None = None) -> Mission5Snapshot:
        now = time.monotonic() if now is None else now
        command = self.controller.update(detections, now) if self.running else self.Command()
        return Mission5Snapshot(
            target=self.controller.target,
            state=self.controller.state,
            running=self.running,
            surge=command.surge,
            yaw=command.yaw,
            heave=command.heave,
            detections=detections,
            results=dict(self.controller.results),
        )

    def update_frame(self, frame: Any, now: float | None = None) -> Mission5Snapshot:
        return self.update_detections(self.detector.detect(frame), now)

