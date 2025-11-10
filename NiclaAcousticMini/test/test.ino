#include <Arduino.h>
#include <Nicla_System.h>
#include <NDP.h>

uint8_t data[2048];

void setup() {
  Serial.begin(1000000);   // 1 Mbps: 16kHz*16bit mono ~256 kbps sig‘adi
  nicla::begin();
  NDP.begin("mcu_fw_120_v91.synpkg");
  NDP.load("dsp_firmware_v91.synpkg");
  NDP.load("ei_model.synpkg");
  NDP.turnOnMicrophone();
  if (NDP.getAudioChunkSize() > sizeof(data)) { while(1){} }
}

void loop() {
  unsigned int len = 0;
  NDP.extractData(data, &len);   // PCM LE bytes
  if (len) Serial.write(data, len);
}
