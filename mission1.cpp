#include <WiFi.h>
#include <HTTPClient.h>

// Update these during the adjustment period
const char* ssid     = "Halim Putra iPhone";
const char* password = "halimputraw";

// Tower IP / Endpoint provided on-site
const char* serverUrl = "http://172.20.10.6:8000/message.txt";
void fetchTowerData() {
  HTTPClient http;
  
  Serial.printf("[ROV] Sending GET request to: %s\n", serverUrl);
  http.begin(serverUrl);
  http.setTimeout(5000); // 5-second timeout

  int httpCode = http.GET();

  if (httpCode > 0) {
    if (httpCode == HTTP_CODE_OK) {
      String payload = http.getString();
      
      // 3. Output data for the referee
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
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n[ROV] Starting Wi-Fi setup...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  // 1. Establish Wi-Fi Connection
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

    // 2. Fetch data via HTTP GET
    fetchTowerData();
  } else {
    Serial.println("\n[ROV] Error: Failed to connect to Wi-Fi.");
  }
}



void loop() {
  // Idle after mission task completion
  delay(1000);
}
