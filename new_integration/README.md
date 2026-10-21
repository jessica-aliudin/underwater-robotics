# New integration additions

This folder contains new work only. The existing programs written by Mansur,
Halim, and the earlier Mission 5 work remain untouched in their original
locations.

## What stays separate

- `Mission2_Control_Interface.py` is Mansur's original Mission 2 program.
- `python + arduino/py3.py` and `python + arduino/serial_receiver.ino` are
  Halim's original additions.
- `mission1.cpp` remains the standalone Mission 1 Wi-Fi program.
- `Mission5/` remains the existing standalone Mission 5 implementation and
  test suite.
- `new_integration/python/` contains only new Python code.
- `new_integration/esp32/` contains only new ESP32/Arduino firmware.

Python is never compiled into, pasted into, or combined with the ESP32 code.
The two programs run separately and communicate using short serial text
messages documented below.

## Safety-first serial protocol

Laptop to ESP32:

```text
ARM
DISARM
THRUST <right_us> <left_us> <vertical_us>
HOOK OPEN
HOOK CLOSE
ORDER <R-B-Y>
STOP
PING
```

The ESP32 starts disarmed. A valid `ARM` command is required before it accepts
thruster movement. If fresh `THRUST` commands stop arriving for 500 ms, the
firmware returns all thrusters to neutral and disarms itself.

## Install

From the repository root:

```powershell
py -m venv .venv-integration
.\.venv-integration\Scripts\python.exe -m pip install -r .\new_integration\requirements.txt
```

## Safe dry-run

Dry-run is the default. It opens the camera and Mission 5 adapter but prints
serial messages instead of opening a real serial port:

```powershell
.\.venv-integration\Scripts\python.exe .\new_integration\python\whole_bot_controller.py mission5 --camera 0 --order R-B-Y
```

Controls in the camera window:

- `Space`: start or pause Mission 5 autonomy
- `X`: emergency stop and disarm
- `Y` / `N`: confirm that the ball dropped or retry
- `S`: skip the current pole
- `R`: restart the sequence
- `O` / `C`: open or close the hook
- `Q` or `Esc`: stop and quit

Real serial output requires all three of these deliberate choices:

```powershell
.\.venv-integration\Scripts\python.exe .\new_integration\python\whole_bot_controller.py mission5 --camera 1 --order R-B-Y --live --port COM3
```

Check the camera, COM port, thruster direction, PWM limits, hook positions, and
emergency stop at low power before putting the ROV in the pool.

## Manual missions

Mission 2.2 and Mission 3 must remain under direct human control. The new
`mission2_docking_assist.py` and `mission3_line_assist.py` modules produce only
visual/operator guidance. They never generate thruster commands.

Mission 4 remains human controlled as well. `mission4_hook_control.py` adds a
small, explicit hook-command wrapper; the new ESP32 firmware implements the
matching `HOOK OPEN` and `HOOK CLOSE` messages.

## Tests

```powershell
.\.venv-integration\Scripts\python.exe -m unittest discover -s .\new_integration\tests -v
```

The tests use dry-run or synthetic images. They do not operate real motors.

