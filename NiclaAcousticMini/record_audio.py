"""
Nicla Voice Audio Capture Script
Receives audio data from Arduino and saves as WAV file
Run on your laptop after uploading Arduino code to Nicla Voice
"""

import serial
import wave
import struct
import os

# ===== CONFIGURATION - CHANGE THESE =====
SERIAL_PORT = 'COM3'  # Windows: COM3, COM4, etc.
                       # Linux/Mac: /dev/ttyACM0 or /dev/ttyUSB0
BAUD_RATE = 115200
SAMPLE_RATE = 16000
CHANNELS = 1

# WHERE TO SAVE THE WAV FILE
OUTPUT_FILE = 'C:/Arduino/background_sound.wav'  # Your specified path
# ========================================

def main():
    print("=" * 50)
    print("Nicla Voice Audio Recorder")
    print("=" * 50)
    print(f"Serial Port: {SERIAL_PORT}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Output Location: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 50)
    
    try:
        # Connect to Nicla Voice
        print("\nConnecting to Nicla Voice...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("✓ Connected successfully!")
        
        # Wait for start marker
        print("\nWaiting for recording to start...")
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(line)
                if '---WAV_DATA_START---' in line:
                    break
        
        print("\n🎤 CAPTURING AUDIO DATA...")
        print("Recording for 10 minutes 5 seconds...")
        audio_data = bytearray()
        
        # Read audio data
        chunk_count = 0
        while True:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                audio_data.extend(data)
                chunk_count += 1
                
                # Show progress every 100 chunks
                if chunk_count % 100 == 0:
                    print(f"  Received {len(audio_data):,} bytes ({len(audio_data)/1024:.1f} KB)")
                
                # Check for end marker
                try:
                    recent_str = data.decode('utf-8', errors='ignore')
                    if '---WAV_DATA_END---' in recent_str:
                        print("\n✓ Recording completed!")
                        break
                except:
                    pass
        
        # Remove text markers from audio data if present
        audio_data = bytes(audio_data)
        
        print(f"\nTotal data received: {len(audio_data):,} bytes")
        print(f"Duration: ~{len(audio_data)/(SAMPLE_RATE*2):.1f} seconds")
        
        # Save as WAV file
        print(f"\n💾 Saving to: {os.path.abspath(OUTPUT_FILE)}")
        with wave.open(OUTPUT_FILE, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        
        print("✓ Audio saved successfully!")
        print(f"\nYou can now play the file: {OUTPUT_FILE}")
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"\n❌ ERROR: Could not open serial port {SERIAL_PORT}")
        print(f"Error details: {e}")
        print("\nTips:")
        print("1. Check if Nicla Voice is connected via USB")
        print("2. Check the correct port in Arduino IDE (Tools → Port)")
        print("3. Close Arduino Serial Monitor if open")
        print("4. Try different port names (COM3, COM4, /dev/ttyACM0, etc.)")
    except KeyboardInterrupt:
        print("\n\n⚠ Recording interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()