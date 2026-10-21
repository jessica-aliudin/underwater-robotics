#pragma once

// These values belong only to the new firmware. Check every pin and pulse
// limit against the actual lab wiring before connecting propellers.
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long THRUSTER_TIMEOUT_MS = 500;

constexpr int THRUSTER_PINS[3] = {4, 5, 6};
constexpr int HOOK_PIN = 7;

constexpr int PWM_FREQUENCY_HZ = 50;
constexpr int PWM_RESOLUTION_BITS = 14;
constexpr int PWM_PERIOD_US = 20000;

constexpr int THRUSTER_NEUTRAL_US = 1500;
constexpr int THRUSTER_MIN_US = 1100;
constexpr int THRUSTER_MAX_US = 1900;

// Tune these two hook positions at low power with the linkage disconnected.
constexpr int HOOK_CLOSED_US = 1200;
constexpr int HOOK_OPEN_US = 1800;

