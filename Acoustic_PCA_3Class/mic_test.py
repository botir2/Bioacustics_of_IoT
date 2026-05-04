import sounddevice as sd
import soundfile as sf
import numpy as np
import time

SAMPLE_RATE = 48000
DURATION = 5
OUTPUT_FILE = "mic_test_output.wav"


def show_input_devices():
    print("\nAvailable input devices:\n")
    devices = sd.query_devices()

    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"{i}: {dev['name']} | inputs={dev['max_input_channels']} | default_sr={dev['default_samplerate']}")
            input_devices.append(i)

    return input_devices


def record_test(device_index):
    print("\nRecording started...")
    print("Speak or play sound near the microphone.")
    print(f"Duration: {DURATION} seconds\n")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_index
    )

    sd.wait()

    audio = audio.flatten()

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    print("\nRecording finished.")
    print(f"RMS level:  {rms:.6f}")
    print(f"Peak level: {peak:.6f}")

    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
    print(f"\nSaved test file: {OUTPUT_FILE}")

    if peak < 0.001:
        print("\nResult: microphone signal is very weak or silent.")
    else:
        print("\nResult: microphone is recording sound.")


if __name__ == "__main__":
    input_devices = show_input_devices()

    if not input_devices:
        print("No input microphone devices found.")
        exit()

    print("\nChoose microphone device index from the list above.")
    selected = input("Enter device index: ")

    try:
        selected = int(selected)
    except ValueError:
        print("Invalid device index.")
        exit()

    if selected not in input_devices:
        print("Selected device is not an input microphone.")
        exit()

    try:
        record_test(selected)
    except Exception as e:
        print("\nMic error:")
        print(e)
        print("\nTry another device index, especially MME or DirectSound.")