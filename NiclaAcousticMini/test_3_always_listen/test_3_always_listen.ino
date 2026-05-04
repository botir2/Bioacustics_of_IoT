/*
 * Nicla Voice - Weather Detector with Accuracy Tracking
 * Detects: Wind and Rain only
 * Microphone stays ON continuously for better detection
 */

#include "NDP.h"

// Detection counters
int windDetections = 0;
int rainDetections = 0;
int unknownDetections = 0;
int totalDetections = 0;

void handleDetection(char* label) {
  String labelStr = String(label);
  labelStr.toLowerCase(); // Convert to lowercase for easier matching
  totalDetections++;
  
  Serial.print("🔊 Detected: ");
  Serial.println(label);
  
  if (labelStr.indexOf("wind") >= 0) {
    windDetections++;
    nicla::leds.begin();
    nicla::leds.setColor(blue);
    Serial.println("🌬️ WIND detected!");
    delay(500);
    nicla::leds.setColor(off);
    nicla::leds.end();
  } 
  else if (labelStr.indexOf("rain") >= 0) {
    rainDetections++;
    nicla::leds.begin();
    nicla::leds.setColor(green);
    Serial.println("🌧️ RAIN detected!");
    delay(500);
    nicla::leds.setColor(off);
    nicla::leds.end();
  }
  else {
    unknownDetections++;
    Serial.print("⚠️ Unknown sound: ");
    Serial.println(label);
  }
  
  printQuickStats();
}

void ledRedBlink() {
  while (1) {
    nicla::leds.begin();
    nicla::leds.setColor(red);
    delay(200);
    nicla::leds.setColor(off);
    delay(200);
    nicla::leds.end();
  }
}

void printQuickStats() {
  Serial.print("   [Wind: ");
  Serial.print(windDetections);
  Serial.print(" | Rain: ");
  Serial.print(rainDetections);
  Serial.print(" | Unknown: ");
  Serial.print(unknownDetections);
  Serial.print(" | Total: ");
  Serial.print(totalDetections);
  Serial.println("]");
}

void printStats() {
  Serial.println("\n========== STATISTICS ==========");
  Serial.print("Total detections: ");
  Serial.println(totalDetections);
  
  Serial.print("  🌬️ Wind: ");
  Serial.print(windDetections);
  if (totalDetections > 0) {
    Serial.print(" (");
    Serial.print((windDetections * 100.0) / totalDetections, 1);
    Serial.println("%)");
  } else {
    Serial.println();
  }
  
  Serial.print("  🌧️ Rain: ");
  Serial.print(rainDetections);
  if (totalDetections > 0) {
    Serial.print(" (");
    Serial.print((rainDetections * 100.0) / totalDetections, 1);
    Serial.println("%)");
  } else {
    Serial.println();
  }
  
  Serial.print("  🔊 Unknown: ");
  Serial.print(unknownDetections);
  if (totalDetections > 0) {
    Serial.print(" (");
    Serial.print((unknownDetections * 100.0) / totalDetections, 1);
    Serial.println("%)");
  } else {
    Serial.println();
  }
  
  Serial.println("================================\n");
}

void setup() {
  Serial.begin(115200);
  
  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }
  
  nicla::begin();
  nicla::disableLDO();
  nicla::leds.begin();

  NDP.onError(ledRedBlink);
  NDP.onMatch(handleDetection);
  
  Serial.println("\n================================");
  Serial.println("Weather Detection Starting...");
  Serial.println("================================");
  Serial.println("Loading model packages...");
  
  NDP.begin("mcu_fw_120_v91.synpkg");
  NDP.load("dsp_firmware_v91.synpkg");
  NDP.load("ei_model.synpkg");
  
  Serial.println("✓ Packages loaded!");
  NDP.getInfo();
  
  Serial.println("\n🎤 Microphone is now CONTINUOUSLY LISTENING");
  Serial.println("Detecting: Wind and Rain");
  Serial.println("\nLED Colors:");
  Serial.println("  🔵 Blue = Wind");
  Serial.println("  🟢 Green = Rain");
  Serial.println("\nCommands:");
  Serial.println("  's' = Show detailed statistics");
  Serial.println("  'r' = Reset statistics");
  Serial.println("================================\n");
  
  // Turn on microphone and keep it on
  NDP.turnOnMicrophone();
  NDP.interrupts();
  
  Serial.println("✓ Ready! Make wind or rain sounds...\n");
  
  nicla::leds.end();
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') {
      printStats();
    }
    else if (cmd == 'r' || cmd == 'R') {
      windDetections = 0;
      rainDetections = 0;
      unknownDetections = 0;
      totalDetections = 0;
      Serial.println("\n✓ Statistics reset!\n");
    }
  }
  
  // Small delay to prevent tight loop
  delay(10);
}