#include "serial_protocol.h"

// New firmware only. Halim's original serial_receiver.ino remains unchanged.

String inputLine;
bool armed = false;
bool failsafeReported = false;
unsigned long lastThrusterMessageMs = 0;

void writePulseUs(int pin, int pulseUs) {
  pulseUs = constrain(pulseUs, THRUSTER_MIN_US, THRUSTER_MAX_US);
  const uint32_t maximumDuty = (1UL << PWM_RESOLUTION_BITS) - 1;
  const uint32_t duty = static_cast<uint32_t>(pulseUs) * maximumDuty / PWM_PERIOD_US;
  ledcWrite(pin, duty);
}

void neutralThrusters() {
  for (int pin : THRUSTER_PINS) {
    writePulseUs(pin, THRUSTER_NEUTRAL_US);
  }
}

void setHook(bool open) {
  writePulseUs(HOOK_PIN, open ? HOOK_OPEN_US : HOOK_CLOSED_US);
  Serial.println(open ? "ACK HOOK OPEN" : "ACK HOOK CLOSE");
}

bool parseThreePulses(const String &arguments, int values[3]) {
  char extra = '\0';
  const int parsed = sscanf(arguments.c_str(), "%d %d %d %c",
                            &values[0], &values[1], &values[2], &extra);
  if (parsed != 3) {
    return false;
  }
  for (int value : {values[0], values[1], values[2]}) {
    if (value < THRUSTER_MIN_US || value > THRUSTER_MAX_US) {
      return false;
    }
  }
  return true;
}

void handleCommand(String message) {
  message.trim();
  if (message.length() == 0) {
    return;
  }

  const int firstSpace = message.indexOf(' ');
  String command = firstSpace < 0 ? message : message.substring(0, firstSpace);
  String arguments = firstSpace < 0 ? "" : message.substring(firstSpace + 1);
  command.toUpperCase();
  arguments.trim();

  if (command == "PING") {
    Serial.println("ACK PONG");
    return;
  }
  if (command == "ARM") {
    neutralThrusters();
    armed = true;
    failsafeReported = false;
    lastThrusterMessageMs = millis();
    Serial.println("ACK ARMED");
    return;
  }
  if (command == "STOP" || command == "DISARM") {
    neutralThrusters();
    armed = false;
    Serial.println("ACK DISARMED");
    return;
  }
  if (command == "HOOK") {
    arguments.toUpperCase();
    if (arguments == "OPEN") {
      setHook(true);
    } else if (arguments == "CLOSE") {
      setHook(false);
    } else {
      Serial.println("ERR HOOK expects OPEN or CLOSE");
    }
    return;
  }
  if (command == "ORDER") {
    // The laptop owns the Mission 5 state machine. The firmware records only
    // an acknowledgement so the operator can verify that the order arrived.
    arguments.toUpperCase();
    const bool validOrder = arguments.length() == 5 &&
                            arguments.charAt(1) == '-' && arguments.charAt(3) == '-' &&
                            arguments.indexOf('R') >= 0 && arguments.indexOf('Y') >= 0 &&
                            arguments.indexOf('B') >= 0;
    if (validOrder) {
      Serial.print("ACK ORDER ");
      Serial.println(arguments);
    } else {
      Serial.println("ERR ORDER expects R-B-Y format");
    }
    return;
  }
  if (command == "THRUST") {
    if (!armed) {
      Serial.println("ERR DISARMED");
      return;
    }
    int values[3];
    if (!parseThreePulses(arguments, values)) {
      neutralThrusters();
      armed = false;
      Serial.println("ERR invalid THRUST; DISARMED");
      return;
    }
    for (int i = 0; i < 3; ++i) {
      writePulseUs(THRUSTER_PINS[i], values[i]);
    }
    lastThrusterMessageMs = millis();
    failsafeReported = false;
    return;
  }

  Serial.print("ERR unknown command: ");
  Serial.println(command);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  inputLine.reserve(96);

  for (int pin : THRUSTER_PINS) {
    ledcAttach(pin, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS);
  }
  ledcAttach(HOOK_PIN, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS);
  neutralThrusters();
  setHook(false);

  // ESCs normally need a few seconds of neutral signal before they arm.
  delay(3000);
  Serial.println("READY DISARMED");
}

void loop() {
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\n') {
      handleCommand(inputLine);
      inputLine = "";
    } else if (character != '\r') {
      if (inputLine.length() < 95) {
        inputLine += character;
      } else {
        inputLine = "";
        neutralThrusters();
        armed = false;
        Serial.println("ERR command too long; DISARMED");
      }
    }
  }

  if (armed && millis() - lastThrusterMessageMs > THRUSTER_TIMEOUT_MS) {
    neutralThrusters();
    armed = false;
    if (!failsafeReported) {
      Serial.println("FAILSAFE timeout; DISARMED");
      failsafeReported = true;
    }
  }
}

