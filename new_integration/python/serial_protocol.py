"""Safe laptop-side serial protocol for the new integration code.

This module does not import or modify either teammate control program. It only
defines the messages understood by ``new_rov_firmware.ino``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


NEUTRAL_US = 1500
MIN_US = 1100
MAX_US = 1900
DEFAULT_SPAN_US = 400


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_line(line: str) -> str:
    line = line.strip()
    if not line or "\n" in line or "\r" in line:
        raise ValueError("serial commands must be one non-empty line")
    return line


@dataclass(frozen=True)
class ThrusterPulses:
    right: int = NEUTRAL_US
    left: int = NEUTRAL_US
    vertical: int = NEUTRAL_US

    def limited(self) -> "ThrusterPulses":
        return ThrusterPulses(
            int(clamp(self.right, MIN_US, MAX_US)),
            int(clamp(self.left, MIN_US, MAX_US)),
            int(clamp(self.vertical, MIN_US, MAX_US)),
        )

    def encode(self) -> str:
        p = self.limited()
        return f"THRUST {p.right} {p.left} {p.vertical}"


@dataclass(frozen=True)
class ThrusterMixer:
    """Map normalised ROV motion onto two horizontal and one vertical motor.

    Positive surge means forward, positive yaw means turn right, and positive
    heave means move down. Reverse flags are kept here rather than hidden in the
    firmware, so lab calibration remains visible and testable.
    """

    span_us: int = DEFAULT_SPAN_US
    reverse_right: bool = False
    reverse_left: bool = False
    reverse_vertical: bool = False

    def mix(self, surge: float = 0.0, yaw: float = 0.0, heave: float = 0.0,
            speed: float = 1.0) -> ThrusterPulses:
        speed = clamp(speed, 0.0, 1.0)
        right = surge - yaw
        left = surge + yaw
        vertical = heave
        largest = max(1.0, abs(right), abs(left), abs(vertical))
        values = [right / largest, left / largest, vertical / largest]
        reversals = [self.reverse_right, self.reverse_left, self.reverse_vertical]
        pulses = []
        for value, reverse in zip(values, reversals):
            value = clamp(value * speed, -1.0, 1.0)
            if reverse:
                value = -value
            pulses.append(round(NEUTRAL_US + value * self.span_us))
        return ThrusterPulses(*pulses).limited()


class SerialTransport:
    """Small serial link with an explicit arm/disarm boundary.

    Dry-run is the default and is fully usable without pyserial. A custom
    ``serial_factory`` can be supplied by tests so no physical port is touched.
    """

    def __init__(self, port: Optional[str] = None, baud: int = 115200,
                 dry_run: bool = True,
                 serial_factory: Optional[Callable[..., object]] = None,
                 logger: Callable[[str], None] = print):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.serial_factory = serial_factory
        self.logger = logger
        self.connection = None
        self.armed = False
        self.sent: list[str] = []

    @property
    def ready(self) -> bool:
        return self.dry_run or self.connection is not None

    def open(self) -> None:
        if self.dry_run:
            self.logger("DRY RUN: serial port will not be opened")
            return
        if not self.port:
            raise ValueError("a serial port is required when --live is used")
        factory = self.serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is required for live operation") from exc
            factory = serial.Serial
        self.connection = factory(self.port, self.baud, timeout=0.05, write_timeout=0.2)

    def send_line(self, line: str) -> None:
        line = _clean_line(line)
        self.sent.append(line)
        if self.dry_run:
            self.logger("DRY RUN -> " + line)
            return
        if self.connection is None:
            raise RuntimeError("serial port is not open")
        self.connection.write((line + "\n").encode("ascii"))

    def arm(self) -> None:
        self.send_line("ARM")
        self.armed = True

    def stop(self) -> None:
        if self.ready:
            self.send_line("STOP")

    def disarm(self) -> None:
        if self.ready:
            self.send_line("DISARM")
        self.armed = False

    def send_thrusters(self, pulses: ThrusterPulses) -> None:
        if not self.armed:
            raise RuntimeError("refusing to send thrusters while disarmed")
        self.send_line(pulses.encode())

    def send_hook(self, opened: bool) -> None:
        self.send_line("HOOK OPEN" if opened else "HOOK CLOSE")

    def send_order(self, order: str) -> None:
        letters = [char for char in order.upper() if char.isalpha()]
        if sorted(letters) != ["B", "R", "Y"]:
            raise ValueError("order must contain R, Y and B exactly once")
        self.send_line("ORDER " + "-".join(letters))

    def close(self) -> None:
        try:
            if self.ready:
                self.disarm()
        finally:
            if self.connection is not None:
                self.connection.close()
                self.connection = None


def list_serial_ports() -> list[tuple[str, str]]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required to list serial ports") from exc
    return [(item.device, item.description) for item in list_ports.comports()]

