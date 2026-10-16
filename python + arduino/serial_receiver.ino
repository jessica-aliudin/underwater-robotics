const long BAUD_RATE = 115200;  
String inputLine = "";
int arr[3] = {0,0,0};
String nameidx[3] = {"THRUSTERA", "THURSTERB", "THRUSTERC"};
int thrusterpinp[3] = {4,5,6};
const int PINS[] = {4, 5, 6};
const int N = 3;
const int FREQ_HZ = 50;         // 20 ms period, standard servo/ESC signal
const int RES_BITS = 14;         // ESP32-S3 LEDC max is 14 bits
const int NEUTRAL_US = 1500;
const int TEST_DELTA_US = 100;  // low power: 1600 fwd / 1400 rev
const int RUN_MS = 2000;
const int MIN_US = 1100;
const int MAX_US = 1900;

void writeUs(int pin, int us) {
  uint32_t duty = (uint32_t)us * ((1UL << RES_BITS) - 1) / 20000UL;
  ledcWrite(pin, duty);
}

void setthruster(int idx, int A) {
  A = constrain(A, MIN_US, MAX_US);  // keep power in a safe range
  writeUs(PINS[idx], A); 
}

void handleMessage(const String &msg) {
  Serial.print("Received: ");
  Serial.println(msg);
  int spaceIndex = msg.indexOf(' ');
  String name = msg.substring(0,spaceIndex);
  String rest = msg.substring(spaceIndex+1);
  int count=0;
  int values[3];
  while (rest.length() > 0 && count < 3) {
    rest.trim();                                  // drop extra spaces
    int nextSpace = rest.indexOf(' ');
    if (nextSpace == -1) {                        // last number
      values[count++] = rest.toInt();
      break;
    }
    values[count++] = rest.substring(0, nextSpace).toInt();
    rest = rest.substring(nextSpace + 1);
  } 
  for(int i=0;i<3;i++)
  {
    setthruster(i,values[i]);
  }
}
void setup() {
  Serial.begin(BAUD_RATE);
  inputLine.reserve(64);

  for (int i = 0; i < N; i++) {
    ledcAttach(PINS[i], FREQ_HZ, RES_BITS);  // attach PWM to each thruster pin
    writeUs(PINS[i], NEUTRAL_US);            // send neutral so the ESC arms
  }
  delay(3000);                               // give the ESCs time to arm (beeps)

  Serial.println("Arduino ready");
}



void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      inputLine.trim();
      if (inputLine.length() > 0) {
        handleMessage(inputLine);
      }
      inputLine = "";
    } else {
      inputLine += c;
    }
  }
  
}


