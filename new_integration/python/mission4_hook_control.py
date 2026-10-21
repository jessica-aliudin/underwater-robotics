"""Mission 4 hook commands for the new serial protocol."""

from dataclasses import dataclass

try:
    from .serial_protocol import SerialTransport
except ImportError:  # direct script/module use from this folder
    from serial_protocol import SerialTransport


@dataclass
class HookController:
    transport: SerialTransport
    is_open: bool = False

    def open(self) -> None:
        self.transport.send_hook(True)
        self.is_open = True

    def close(self) -> None:
        self.transport.send_hook(False)
        self.is_open = False

    def toggle(self) -> None:
        self.close() if self.is_open else self.open()

