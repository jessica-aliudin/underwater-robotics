#include <WiFi.h>
#include <HTTPClient.h>

// Mission 1 runs directly on the ESP32. It joins the referee's Wi-Fi, downloads
// the text file once, and prints the result to Serial Monitor.

// Replace these placeholders with the details provided in the lab. Keep real
// passwords out of GitHub; enter them locally just before uploading the code.
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// This must be the full address supplied by the referee, including the path.
const char* serverUrl = "http://ROV_TOWER_IP:8000/message.txt";

// Ask the tower for its message and print either the message or a useful error.
void fetchTowerData() {
  HTTPClient http;

  // A timeout keeps a missing tower from hanging the ESP32 forever.
  Serial.printf("[ROV] Sending GET request to: %s\n", serverUrl);
  http.begin(serverUrl);
  http.setTimeout(5000);

  int httpCode = http.GET();

  if (httpCode > 0) {
    if (httpCode == HTTP_CODE_OK) {
      String payload = http.getString();

      // Put clear separators around the answer so it is easy to spot in the log.
      Serial.println("\n==========================================");
      Serial.println(">>> RETRIEVED MISSION DATA <<<");
      Serial.println(payload);
      Serial.println("==========================================\n");
    } else {
      Serial.printf("[ROV] Server responded with HTTP status code: %d\n", httpCode);
    }
  } else {
    Serial.printf("[ROV] HTTP GET failed. Error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
}

void setup() {
  // setup() runs once whenever the ESP32 starts or is reset.
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n[ROV] Starting Wi-Fi setup...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  // Try for about 15 seconds (30 attempts x 500 ms), then fail cleanly.
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[ROV] Connected successfully!");
    Serial.print("[ROV] Assigned IP: ");
    Serial.println(WiFi.localIP());

    // Only contact the tower after Wi-Fi is definitely connected.
    fetchTowerData();
  } else {
    Serial.println("\n[ROV] Error: Failed to connect to Wi-Fi.");
  }
}
void loop() {
  // Mission 1 is a one-shot task, so there is nothing else to repeat here.
  delay(1000);
}
