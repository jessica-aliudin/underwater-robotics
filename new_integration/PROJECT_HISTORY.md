# Existing work and ownership

This record exists so that new integration work is not confused with earlier
team contributions. No history has been rewritten or squashed.

| Commit | Branch/source | Existing contribution |
| --- | --- | --- |
| `f3b65ec` | Jessica `main` | Restored separate Mission 2 and Mission 5 programs. |
| `771e160` | Jessica `halim` | Halim added `py3.py` and the Arduino serial receiver. |
| `5b98a51` | Halim's fork | Implemented Wi-Fi connection and data fetching. |
| `ae2abbf` | Mansur's fork | Renamed and supplied the Mission 2 control interface. |
| `9b55984` | Jessica `main` | Merge commit for Mansur's Mission 2 pull request. |

## Protected existing files

The integration branch treats these files as read-only:

- `Mission2_Control_Interface.py`
- `python + arduino/py3.py`
- `python + arduino/serial_receiver.ino`
- `mission1.cpp`
- everything inside `team-repos/`

All new implementation is under `new_integration/` and should use clear commit
messages beginning with `docs:`, `test:`, `feat:`, or `fix:`. That makes the
new work easy to separate from the earlier commits in `git log`.

