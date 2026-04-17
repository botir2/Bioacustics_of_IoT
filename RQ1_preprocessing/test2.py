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
        self.sample_rate = 44100

    def record_window(self):
        # remove old file if it exists
        if os.path.exists(self.filename):
            os.remove(self.filename)

        subprocess.run([
            "arecord",
            "-D", self.device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(self.sample_rate),
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
    ############################################################
    def normalize_audio(self, audio):
        # convert to float
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio

        # scale between -1 and 1
        audio_norm = audio / max_val
        return audio_norm

    ############################################################
    #
    #               Block 3 quality screen
    #   Input to Block 4 should only be windows that pass here
    #
    ############################################################
    def quality_screen(self, audio):
        energy = np.mean(np.abs(audio))

        # 1) Silence check
        if energy < 600:
            return "DROP", "silence"

        # 2) Clipping check
        clip_ratio = np.mean(np.abs(audio) > 32000)
        if clip_ratio > 0.01:
            return "DROP", "clipping"

        # 3) Simple SNR proxy (signal variance)
        noise_level = np.std(audio)
        if noise_level < 100:
            return "DROP", "low_snr"

        return "KEEP", "ok"

    ############################################################
    #
    #               Block 4 cue extraction
    #   Low-cost waveform / simple spectral proxy cues
    #
    ############################################################
    def extract_block4_cues(self, audio):
        """
        Lightweight cues for environment-aware mode selection.
        Designed to stay Raspberry Pi friendly.
        """
        audio_f = audio.astype(np.float32)

        # Basic waveform-domain cues
        mean_abs = np.mean(np.abs(audio_f))
        std_val = np.std(audio_f)
        peak_abs = np.max(np.abs(audio_f)) if len(audio_f) > 0 else 0.0
        clip_ratio = np.mean(np.abs(audio_f) > 32000)

        # Zero-crossing rate (cheap cue for noisiness / high-frequency activity)
        if len(audio_f) > 1:
            zcr = np.mean((audio_f[:-1] * audio_f[1:]) < 0)
        else:
            zcr = 0.0

        # Crest factor proxy: peak / RMS
        rms = np.sqrt(np.mean(audio_f ** 2)) if len(audio_f) > 0 else 0.0
        crest_factor = peak_abs / (rms + 1e-8)

        # Very small FFT-based proxy to estimate low vs high-frequency dominance
        # This is still lightweight and practical on Raspberry Pi for short windows.
        if len(audio_f) > 0:
            spectrum = np.fft.rfft(audio_f)
            power = np.abs(spectrum) ** 2
            freqs = np.fft.rfftfreq(len(audio_f), d=1.0 / self.sample_rate)

            total_power = np.sum(power) + 1e-8
            low_band_ratio = np.sum(power[freqs < 500]) / total_power
            high_band_ratio = np.sum(power[freqs > 4000]) / total_power
            spectral_centroid = np.sum(freqs * power) / total_power
        else:
            low_band_ratio = 0.0
            high_band_ratio = 0.0
            spectral_centroid = 0.0

        return {
            "mean_abs": mean_abs,
            "std_val": std_val,
            "clip_ratio": clip_ratio,
            "zcr": zcr,
            "crest_factor": crest_factor,
            "low_band_ratio": low_band_ratio,
            "high_band_ratio": high_band_ratio,
            "spectral_centroid": spectral_centroid,
        }

    ############################################################
    #
    #               Block 4 condition gate
    #   Purpose:
    #   - identify a simple condition label for usable windows
    #   - choose processing mode: skip / light / full
    #
    ############################################################
    def condition_gate(self, audio):
        """
        Block 4:
        Uses simple threshold-based rules and proxy labels.

        Output 1: condition_label
        Output 2: mode_decision = light / full
        """

        cues = self.extract_block4_cues(audio)

        mean_abs = cues["mean_abs"]
        clip_ratio = cues["clip_ratio"]
        zcr = cues["zcr"]
        low_band_ratio = cues["low_band_ratio"]
        high_band_ratio = cues["high_band_ratio"]
        spectral_centroid = cues["spectral_centroid"]

        # 1) Harsh / degraded condition
        if clip_ratio > 0.003:
            condition_label = "clipped_or_harsh"
            mode_decision = "full"

        # 2) Possible low-frequency dominated rough condition (wind/rain proxy)
        elif low_band_ratio > 0.70 and zcr < 0.08:
            condition_label = "rough_noisy"
            mode_decision = "full"

        # 3) Possible insect/high-frequency dominated condition
        elif high_band_ratio > 0.35 and spectral_centroid > 3000:
            condition_label = "tonal_highfreq"
            mode_decision = "full"

        # 4) Low-activity but still usable window
        elif mean_abs < 1200:
            condition_label = "low_activity_background"
            mode_decision = "light"

        # 5) Default clean / baseline usable window
        else:
            condition_label = "baseline_clean"
            mode_decision = "light"

        return condition_label, mode_decision, cues

        # -------------------------------------------------------
        # Minimal proxy labels
        # -------------------------------------------------------
        #
        # These are not meant to be perfect semantic classes.
        # They are cheap first-pass proxies suitable for testing.
        #
        # baseline_clean        -> moderate, not harsh, not strongly low/high dominated
        # low_activity_background -> passed Block 3 but still weak / background-like
        # rough_noisy           -> noisy / unstable / difficult
        # tonal_highfreq        -> possible insect-like / high-frequency dominated
        # clipped_or_harsh      -> strong harshness / degraded
        #
        # -------------------------------------------------------

        # 1) Harsh / degraded condition
        if clip_ratio > 0.003:
            condition_label = "clipped_or_harsh"
            mode_decision = "full"

        # 2) Possible low-frequency dominated rough condition (wind/rain proxy)
        elif low_band_ratio > 0.70 and zcr < 0.08:
            condition_label = "rough_noisy"
            mode_decision = "full"

        # 3) Possible insect/high-frequency dominated condition
        elif high_band_ratio > 0.35 and spectral_centroid > 3000:
            condition_label = "tonal_highfreq"
            mode_decision = "full"

        # 4) Low-activity background-like usable window
        elif mean_abs < 1200:
            condition_label = "low_activity_background"
            mode_decision = "skip"

        # 5) Default clean / baseline usable window
        else:
            condition_label = "baseline_clean"
            mode_decision = "light"

        return condition_label, mode_decision, cues

    ############################################################
    #
    #               Optional extra preprocessing for FULL mode
    #   Placeholder only: keep lightweight for first version
    #
    ############################################################
    def robust_preprocess(self, audio):
        """
        Simple placeholder for additional preprocessing used in FULL mode.
        For now, just return normalized audio.
        You can later replace this with a low-cost high-pass filter,
        PCEN-lite, or another lightweight robustification step.
        """
        return self.normalize_audio(audio)

    def run(self):
        # Try to record audio from microphone
        self.record_window()

        # If recording failed, create an empty WAV file
        if not os.path.exists(self.filename):
            print("Recording failed, creating empty WAV file")

            with wave.open(self.filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)

                # create silence
                silent = np.zeros(self.sample_rate * self.duration, dtype=np.int16)
                wf.writeframes(silent.tobytes())

        # Load audio data
        audio = self.load_audio()

        # Compute signal energy RAW energy
        energy_raw = self.compute_energy(audio)

        # NORMALIZED audio
        audio_norm = self.normalize_audio(audio)

        # Normalized energy
        energy_norm = self.compute_energy(audio_norm)

        # -------------------------------------------------------
        # Block 3 quality screening
        # -------------------------------------------------------
        block3_decision, block3_reason = self.quality_screen(audio)

        # Default Block 4 outputs for windows rejected by Block 3
        condition_label = "not_applicable"
        mode_decision = "skip"
        block4_cues = None
        downstream_audio = None

        # -------------------------------------------------------
        # Block 4 only runs on usable windows that passed Block 3
        # -------------------------------------------------------
        if block3_decision == "KEEP":
            condition_label, mode_decision, block4_cues = self.condition_gate(audio)

            # skip = do not continue further processing
            if mode_decision == "skip":
                downstream_audio = None

            # light = forward raw/normalised window without extra robust preprocessing
            elif mode_decision == "light":
                downstream_audio = audio

            # full = apply extra preprocessing before downstream stage
            elif mode_decision == "full":
                downstream_audio = self.robust_preprocess(audio)

        return {
            "audio": audio,
            "audio_norm": audio_norm,
            "energy_raw": energy_raw,
            "energy_norm": energy_norm,
            "block3_decision": block3_decision,
            "block3_reason": block3_reason,
            "condition_label": condition_label,   # Block 4 output 1
            "mode_decision": mode_decision,       # Block 4 output 2
            "block4_cues": block4_cues,
            "downstream_audio": downstream_audio,
        }


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

    print("Block pipeline running...")

    window_id = 0

    while True:
        result = block.run()

        audio = result["audio"]
        if audio is None:
            continue

        print(
            f"Window {window_id} | "
            f"raw_energy={result['energy_raw']:.2f} | "
            f"norm_energy={result['energy_norm']:.4f} | "
            f"block3={result['block3_decision']} ({result['block3_reason']}) | "
            f"block4_label={result['condition_label']} | "
            f"mode={result['mode_decision']}"
        )

        # Optional: print Block 4 cues for debugging
        if result["block4_cues"] is not None:
            cues = result["block4_cues"]
            print(
                f"  cues: zcr={cues['zcr']:.4f}, "
                f"low_band={cues['low_band_ratio']:.3f}, "
                f"high_band={cues['high_band_ratio']:.3f}, "
                f"centroid={cues['spectral_centroid']:.1f}"
            )

        window_id += 1


if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    main()