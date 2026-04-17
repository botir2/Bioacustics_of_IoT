import subprocess
import numpy as np
import wave
import os
import signal
import sys

class Block:

    def __init__(self, device, duration, filename="temp.wav"):
        self.device = device
        self.duration = duration
        self.filename = filename

    def record_window(self):
        # remove old file if it exists
        if os.path.exists(self.filename):
            os.remove(self.filename)
        subprocess.run([
            "arecord",
            "-D", self.device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", "44100",
            "-d", str(self.duration),
            self.filename
        ])

    def load_audio(self):
        with wave.open(self.filename, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
        return audio

    def compute_energy(self, audio):
        return np.mean(np.abs(audio))

    ############################################################
    #
    #               Normalization
    #
    #
    def normalize_audio(self, audio):
        # convert to float
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio

        # scale between -1 and 1
        audio_norm = audio / max_val
        return audio_norm

    def compute_energy(self, audio):
        return np.mean(np.abs(audio))

    ############################################################
    #
    #               Block 3 quality screen
    #
    def quality_screen(self, audio):
        energy = np.mean(np.abs(audio))

        # 1️⃣ Silence check
        if energy < 600:
            return "DROP", "silence"

        # 2️⃣ Clipping check
        clip_ratio = np.mean(np.abs(audio) > 32000)

        if clip_ratio > 0.01:
            return "DROP", "clipping"

        # 3️⃣ Simple SNR proxy (signal variance)
        noise_level = np.std(audio)

        if noise_level < 100:
            return "DROP", "low_snr"
        return "KEEP", "ok"



    def run(self):

        # Try to record audio from microphone
        self.record_window()

        # If recording failed, create an empty WAV file
        if not os.path.exists(self.filename):
            print("Recording failed, creating empty WAV file")

            with wave.open(self.filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)

                # create 3 seconds of silence
                silent = np.zeros(44100 * self.duration, dtype=np.int16)
                wf.writeframes(silent.tobytes())

        # Load audio data
        audio = self.load_audio()

        # Compute signal energy RAW energy
        energy_raw  = self.compute_energy(audio)

        # NORMALIZED audio
        audio_norm = self.normalize_audio(audio)

        # Normalized energy
        energy_norm = self.compute_energy(audio_norm)

        # Block-3 quality screening
        decision, reason = self.quality_screen(audio)

        return audio, audio_norm, energy_raw, energy_norm, decision, reason


def cleanup(signum, frame):
    print("\nStopping program and cleaning audio processes...")

    # kill any running arecord processes
    subprocess.run(["pkill", "arecord"])

    print("Cleanup finished. Exiting.")
    sys.exit(0)

def main():

    WINDOW_SECONDS = 5
    DEVICE = "plughw:3,0"

    block = Block(device=DEVICE, duration=WINDOW_SECONDS)

    print("Block-1 running...")

    window_id = 0

    while True:
        audio, audio_norm, energy_raw, energy_norm, decision, reason = block.run()

        if audio is None:
            continue

        print(
            f"Window {window_id} | raw_energy={energy_raw:.2f} | "
            f"norm_energy={energy_norm:.4f} | decision={decision} ({reason})"
        )

        if decision == "DROP":
            window_id += 1
            continue
        window_id += 1


if __name__ == "__main__":
    # Capture Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, cleanup)
    main()



