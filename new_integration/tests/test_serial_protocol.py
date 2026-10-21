import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from mission4_hook_control import HookController
from serial_protocol import NEUTRAL_US, SerialTransport, ThrusterMixer, ThrusterPulses


class FakeSerial:
    def __init__(self, *_args, **_kwargs):
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def close(self):
        self.closed = True


class TestThrusterMixer(unittest.TestCase):
    def test_neutral(self):
        self.assertEqual(ThrusterMixer().mix(), ThrusterPulses())

    def test_forward_and_turn(self):
        mixer = ThrusterMixer(span_us=400)
        self.assertEqual(mixer.mix(surge=1.0), ThrusterPulses(1900, 1900, 1500))
        self.assertEqual(mixer.mix(yaw=1.0), ThrusterPulses(1100, 1900, 1500))

    def test_outputs_are_normalised_and_limited(self):
        pulses = ThrusterMixer(span_us=900).mix(surge=1.0, yaw=1.0, heave=-3.0)
        for pulse in (pulses.right, pulses.left, pulses.vertical):
            self.assertGreaterEqual(pulse, 1100)
            self.assertLessEqual(pulse, 1900)

    def test_reverse_flags_are_explicit(self):
        normal = ThrusterMixer(span_us=200).mix(surge=1.0)
        reversed_right = ThrusterMixer(span_us=200, reverse_right=True).mix(surge=1.0)
        self.assertEqual(normal.right - NEUTRAL_US, -(reversed_right.right - NEUTRAL_US))
        self.assertEqual(normal.left, reversed_right.left)


class TestTransport(unittest.TestCase):
    def test_disarmed_transport_refuses_thrusters(self):
        transport = SerialTransport(dry_run=True, logger=lambda _message: None)
        transport.open()
        with self.assertRaises(RuntimeError):
            transport.send_thrusters(ThrusterPulses(1550, 1550, 1500))

    def test_arm_send_stop_and_disarm(self):
        transport = SerialTransport(dry_run=True, logger=lambda _message: None)
        transport.open()
        transport.arm()
        transport.send_thrusters(ThrusterPulses(1600, 1400, 1500))
        transport.stop()
        transport.disarm()
        self.assertEqual(
            transport.sent,
            ["ARM", "THRUST 1600 1400 1500", "STOP", "DISARM"],
        )
        self.assertFalse(transport.armed)

    def test_live_transport_uses_injected_serial_factory(self):
        created = []

        def factory(*args, **kwargs):
            serial = FakeSerial(*args, **kwargs)
            created.append(serial)
            return serial

        transport = SerialTransport("COM_TEST", dry_run=False, serial_factory=factory)
        transport.open()
        transport.arm()
        transport.send_thrusters(ThrusterPulses())
        transport.close()
        self.assertEqual(created[0].writes[0], b"ARM\n")
        self.assertIn(b"THRUST 1500 1500 1500\n", created[0].writes)
        self.assertTrue(created[0].closed)

    def test_order_validation(self):
        transport = SerialTransport(dry_run=True, logger=lambda _message: None)
        transport.send_order("R-B-Y")
        self.assertEqual(transport.sent[-1], "ORDER R-B-Y")
        with self.assertRaises(ValueError):
            transport.send_order("R-R-Y")

    def test_hook_wrapper(self):
        transport = SerialTransport(dry_run=True, logger=lambda _message: None)
        hook = HookController(transport)
        hook.open()
        hook.close()
        self.assertEqual(transport.sent, ["HOOK OPEN", "HOOK CLOSE"])


if __name__ == "__main__":
    unittest.main()

