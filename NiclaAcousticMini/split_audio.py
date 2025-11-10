"""
Audio Splitter - Split long audio into 5-second segments
Splits background_sound.wav into multiple noevent.X.wav files
"""

import wave
import os

# ===== CONFIGURATION =====
INPUT_FILE = 'C:/Arduino/background_sound.wav'  # Your recorded audio
OUTPUT_FOLDER = 'C:/Arduino/segments/'  # Where to save segments
OUTPUT_PREFIX = 'noevent'  # Filename prefix
SEGMENT_DURATION = 5  # Seconds per segment
# =========================

def split_audio():
    print("=" * 60)
    print("Audio Splitter - 5 Second Segments")
    print("=" * 60)
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Segment duration: {SEGMENT_DURATION} seconds")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: Input file not found!")
        print(f"Looking for: {INPUT_FILE}")
        print("Make sure you recorded the audio first!")
        return
    
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"\n✓ Created output folder: {OUTPUT_FOLDER}")
    
    # Open the input WAV file
    print("\n📂 Opening audio file...")
    with wave.open(INPUT_FILE, 'rb') as wav_file:
        # Get audio parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        total_frames = wav_file.getnframes()
        
        # Calculate duration
        total_duration = total_frames / framerate
        
        print(f"\n📊 Audio Info:")
        print(f"  Channels: {channels}")
        print(f"  Sample rate: {framerate} Hz")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        # Calculate frames per segment
        frames_per_segment = int(framerate * SEGMENT_DURATION)
        total_segments = int(total_frames / frames_per_segment)
        
        print(f"\n✂️ Will create {total_segments} segments of {SEGMENT_DURATION} seconds each")
        print(f"  Frames per segment: {frames_per_segment:,}")
        print("\n🎵 Processing...")
        
        # Split into segments
        segment_number = 1
        
        while True:
            # Read frames for one segment
            frames = wav_file.readframes(frames_per_segment)
            
            if len(frames) == 0:
                break  # End of file
            
            # Create output filename
            output_filename = f"{OUTPUT_PREFIX}.{segment_number}.wav"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Write segment to new file
            with wave.open(output_path, 'wb') as segment_file:
                segment_file.setnchannels(channels)
                segment_file.setsampwidth(sample_width)
                segment_file.setframerate(framerate)
                segment_file.writeframes(frames)
            
            # Show progress
            if segment_number % 10 == 0:
                print(f"  Created {segment_number} segments...")
            
            segment_number += 1
        
        print(f"\n✓ Successfully created {segment_number - 1} audio segments!")
        print(f"\n📁 Files saved in: {OUTPUT_FOLDER}")
        print("\nFile naming:")
        print(f"  {OUTPUT_PREFIX}.1.wav")
        print(f"  {OUTPUT_PREFIX}.2.wav")
        print(f"  {OUTPUT_PREFIX}.3.wav")
        print(f"  ...")
        print(f"  {OUTPUT_PREFIX}.{segment_number - 1}.wav")
        
        # Calculate some statistics
        print(f"\n📈 Statistics:")
        print(f"  Total segments: {segment_number - 1}")
        print(f"  Original file size: {os.path.getsize(INPUT_FILE) / (1024*1024):.2f} MB")
        
        # Calculate total size of segments
        total_size = 0
        for i in range(1, segment_number):
            filename = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}.{i}.wav")
            if os.path.exists(filename):
                total_size += os.path.getsize(filename)
        
        print(f"  Total segments size: {total_size / (1024*1024):.2f} MB")
        print(f"  Average segment size: {(total_size / (segment_number - 1)) / 1024:.2f} KB")

def main():
    try:
        split_audio()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()