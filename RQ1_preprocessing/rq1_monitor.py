#!/usr/bin/env python3
"""
RQ1 Pre-detection Pipeline — Continuous Monitor
================================================
Runs continuously on Raspberry Pi, recording and analysing one 5-second
audio window per cycle. Designed to operate unattended; survives SSH
disconnection, PC sleep, and individual run failures.

Usage:
    python3 rq1_monitor.py

Stop:
    Ctrl+C  (or:  kill <pid>  if running via nohup)

Paths (all under BASE_DIR):
    audio/      timestamped WAV files
    reports/    per-run human-readable text reports
    monitor.log continuous plain-text log
    summary.csv one CSV row per run
"""

import os
import csv
import time
import signal
import logging
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as spsignal
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = "/home/pi/pi-iot"
AUDIO_DIR   = os.path.join(BASE_DIR, "audio")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
LOG_PATH    = os.path.join(BASE_DIR, "monitor.log")
CSV_PATH    = os.path.join(BASE_DIR, "summary.csv")

SAMPLE_RATE  = 44100
CHANNELS     = 1
WINDOW_SEC   = 5
DEVICE       = "plughw:3,0"   # USB PnP Sound Device — update if card number changes

# ── Block 3 strict thresholds ─────────────────────────────────────────────────
TAU_E      = 0.015   # mean absolute amplitude floor
TAU_RMS    = 0.020   # RMS floor
TAU_PEAK   = 0.35    # raw peak floor
TAU_C      = 0.01    # maximum clipping rate
CLIP_LEVEL = 0.99    # sample magnitude considered clipped

# ── Block 4 thresholds ────────────────────────────────────────────────────────
TAU_AMP    = 0.05    # mean absolute amplitude
TAU_ZCR    = 0.15    # zero-crossing rate
TAU_CF     = 6.0     # crest factor
TAU_HF     = 0.55    # high-frequency ratio (energy >= 4 kHz)

# ── CSV columns ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp", "file", "duration_s", "sample_rate",
    "raw_peak", "e_raw", "rms", "clip_rate",
    "zcr", "crest_factor", "hf_ratio",
    "b3_decision", "b3_reason",
    "b4_mode", "b4_label", "b4_flags",
    "b5_status",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP — directories, logging, CSV header
# ═══════════════════════════════════════════════════════════════════════════════

def setup():
    for d in [BASE_DIR, AUDIO_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ]
    )

    # Write CSV header if file does not exist yet
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    logging.info("RQ1 monitor started — saving to %s", BASE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — Audio windowing
# Record one 5-second mono WAV window using arecord.
# Save with a timestamped filename so every run is preserved.
# ═══════════════════════════════════════════════════════════════════════════════

def block1_record(ts):
    filename = f"window_{ts}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)

    cmd = [
        "arecord",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-d", str(WINDOW_SEC),
        filepath,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=WINDOW_SEC + 5)
    if result.returncode != 0:
        raise RuntimeError(f"arecord failed: {result.stderr.decode().strip()}")

    # Load and convert to float64 [-1, 1]
    fs, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    x_raw = data.astype(np.float64) / np.iinfo(data.dtype).max

    return filepath, fs, x_raw


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — Signal preparation
# Retain both raw waveform and a normalised copy.
# The raw waveform is never overwritten.
# ═══════════════════════════════════════════════════════════════════════════════

def block2_normalise(x_raw):
    eps    = 1e-9
    peak   = float(np.max(np.abs(x_raw)))
    x_norm = x_raw / (peak + eps)
    return x_raw, x_norm, peak


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — Strict quality screening
# All measures derived from the raw waveform only.
# Two-stage logic:
#   Stage 1 — immediate DROP if clipping rate > TAU_C
#   Stage 2 — majority-vote: DROP if >= 2 of 3 weak-signal flags triggered
# ═══════════════════════════════════════════════════════════════════════════════

def block3_screen(x_raw):
    e_raw     = float(np.mean(np.abs(x_raw)))
    rms       = float(np.sqrt(np.mean(x_raw ** 2)))
    peak_raw  = float(np.max(np.abs(x_raw)))
    clip_rate = float(np.mean(np.abs(x_raw) >= CLIP_LEVEL))

    # Stage 1 — clipping
    if clip_rate > TAU_C:
        return "DROP", f"excessive clipping ({clip_rate:.4f} > {TAU_C})", \
               e_raw, rms, peak_raw, clip_rate

    # Stage 2 — weak-signal majority vote
    flag_e    = e_raw    < TAU_E
    flag_rms  = rms      < TAU_RMS
    flag_peak = peak_raw < TAU_PEAK
    n_flags   = sum([flag_e, flag_rms, flag_peak])

    flags_desc = []
    if flag_e:    flags_desc.append(f"E_raw {e_raw:.4f}<{TAU_E}")
    if flag_rms:  flags_desc.append(f"RMS {rms:.4f}<{TAU_RMS}")
    if flag_peak: flags_desc.append(f"peak {peak_raw:.4f}<{TAU_PEAK}")

    if n_flags >= 2:
        reason = "weak signal — " + " · ".join(flags_desc)
        return "DROP", reason, e_raw, rms, peak_raw, clip_rate

    return "KEEP", "ok", e_raw, rms, peak_raw, clip_rate


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — Condition-aware routing
# Input: retained normalised window from Block 3.
# Uses four low-cost cues to assign a routing label and mode (LIGHT or FULL).
# This is a mode-selection stage, not a weather or condition classifier.
# ═══════════════════════════════════════════════════════════════════════════════

def block4_route(x_norm, fs):
    N = len(x_norm)

    amp      = float(np.mean(np.abs(x_norm)))
    zcr      = float(np.mean(np.abs(np.diff(np.sign(x_norm)))) / 2)
    peak_n   = float(np.max(np.abs(x_norm)))
    rms_n    = float(np.sqrt(np.mean(x_norm ** 2)))
    cf       = peak_n / (rms_n + 1e-9)

    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_mag  = np.abs(np.fft.rfft(x_norm)) ** 2
    hf_ratio = float(np.sum(fft_mag[freqs >= 4000]) / (np.sum(fft_mag) + 1e-9))

    flag_amp = amp      > TAU_AMP
    flag_zcr = zcr      > TAU_ZCR
    flag_cf  = cf       > TAU_CF
    flag_hf  = hf_ratio > TAU_HF
    n_flags  = sum([flag_amp, flag_zcr, flag_cf, flag_hf])

    # Routing label
    if flag_hf and flag_zcr:
        label = "tonal/high-frequency dominated"
    elif flag_amp and not flag_cf:
        label = "rough/noisy"
    elif not flag_amp and not flag_zcr:
        label = "low-activity background"
    else:
        label = "baseline-clean"

    # Mode decision
    if label in ("rough/noisy", "tonal/high-frequency dominated") or n_flags >= 2:
        mode = "FULL"
    else:
        mode = "LIGHT"

    return mode, label, n_flags, amp, zcr, cf, hf_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — Optional enhancement
# Activated only when Block 4 selects FULL.
# Applies lightweight waveform-domain operations.
# Does not produce a KEEP/DROP decision.
# ═══════════════════════════════════════════════════════════════════════════════

def block5_enhance(x_norm, fs, label):
    x_enh = x_norm - np.mean(x_norm)                         # DC removal

    nyq   = fs / 2
    b, a  = spsignal.butter(4, [100 / nyq, min(12000 / nyq, 0.999)], btype="band")
    x_enh = spsignal.filtfilt(b, a, x_enh)                   # band limiting

    if label in ("rough/noisy", "tonal/high-frequency dominated"):
        x_enh = spsignal.savgol_filter(x_enh, window_length=5, polyorder=2)

    x_enh = x_enh / (np.max(np.abs(x_enh)) + 1e-9)          # re-normalisation
    return x_enh


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE-UP generator
# Produces a short academic-style paragraph for the text report.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_writeup(b3_dec, b3_reason, b4_mode, b4_label,
                     b5_status, e_raw, rms, peak_raw, zcr, cf, hf_ratio):
    if b3_dec == "DROP":
        return (
            f"The recorded window was assigned a DROP decision at Block 3 under the "
            f"current strict screening logic. The signal did not meet the minimum "
            f"energy requirements: {b3_reason}. No further processing was applied "
            f"to this window, and it was not forwarded to Block 4."
        )

    base = (
        f"The recorded window passed Block 3 screening with a raw peak of {peak_raw:.4f}, "
        f"a mean absolute amplitude of {e_raw:.5f}, and an RMS level of {rms:.5f}. "
        f"Under the current strict thresholds, the window contained sufficient acoustic "
        f"activity to be retained and forwarded to Block 4."
    )

    if b4_mode == "LIGHT":
        routing = (
            f" In Block 4, the retained normalised window was evaluated against four "
            f"low-cost cues (mean absolute amplitude, zero-crossing rate, crest factor, "
            f"and high-frequency ratio). The window was assigned the label "
            f"'{b4_label}' and routed to the LIGHT path, indicating that no strong "
            f"difficulty indicators were identified under the current provisional thresholds. "
            f"Block 5 was therefore not activated, and the retained window was forwarded "
            f"unchanged to downstream detection."
        )
    else:
        routing = (
            f" In Block 4, the retained normalised window was evaluated against four "
            f"low-cost cues. With a zero-crossing rate of {zcr:.4f}, crest factor of "
            f"{cf:.3f}, and high-frequency ratio of {hf_ratio:.4f}, the window was "
            f"assigned the label '{b4_label}' and routed to the FULL path. This indicates "
            f"that the window exhibited stronger difficulty indicators under the selected "
            f"cues. Block 4 should be interpreted as a lightweight condition-aware "
            f"mode-selection stage rather than a direct environmental classifier. "
            f"Because FULL was selected, Block 5 was activated and applied DC removal, "
            f"band limiting, and re-normalisation to produce a more robust processed "
            f"waveform for downstream detection. Block 5 did not introduce an additional "
            f"KEEP or DROP decision."
        )

    return base + routing


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT WRITER
# Writes a human-readable text report and appends one CSV row.
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(ts, filepath, fs, peak_raw, norm_peak,
                 e_raw, rms, clip_rate, zcr, cf, hf_ratio,
                 b3_dec, b3_reason, b4_mode, b4_label, b4_flags, b5_status,
                 writeup):

    report_path = os.path.join(REPORT_DIR, f"report_{ts}.txt")

    lines = [
        "RQ1 Pre-detection Pipeline — Run Report",
        f"Timestamp : {ts}",
        f"File      : {filepath}",
        f"Duration  : {WINDOW_SEC}.0 s",
        f"Sample rate: {fs} Hz",
        f"Format    : 16-bit Mono",
        "",
        "Measured signal values",
        f"  Raw peak amplitude         : {peak_raw:.6f}",
        f"  Normalised peak amplitude  : {norm_peak:.6f}",
        f"  Mean absolute amp (E_raw)  : {e_raw:.6f}",
        f"  RMS level                  : {rms:.6f}",
        f"  Clipping-related measure   : {clip_rate:.6f}",
        f"  Zero-crossing rate         : {zcr if zcr is not None else 'not run'}",
        f"  Crest factor               : {cf if cf is not None else 'not run'}",
        f"  High-frequency ratio       : {hf_ratio if hf_ratio is not None else 'not run'}",
        "",
        "Pipeline decisions",
        f"  Block 3 — Quality screening : {b3_dec}  ({b3_reason})",
        f"  Block 4 — Condition gate    : {b4_mode}",
        f"  Block 5 — Enhancement       : {b5_status}",
        "",
        "Write-up",
        writeup,
        "",
        "=" * 70,
        "",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
        f.flush()

    # Append to CSV
    row = {
        "timestamp"  : ts,
        "file"       : filepath,
        "duration_s" : WINDOW_SEC,
        "sample_rate": fs,
        "raw_peak"   : round(peak_raw,  6),
        "e_raw"      : round(e_raw,     6),
        "rms"        : round(rms,       6),
        "clip_rate"  : round(clip_rate, 6),
        "zcr"        : round(zcr,       6) if zcr      is not None else "",
        "crest_factor": round(cf,       4) if cf       is not None else "",
        "hf_ratio"   : round(hf_ratio,  6) if hf_ratio is not None else "",
        "b3_decision": b3_dec,
        "b3_reason"  : b3_reason,
        "b4_mode"    : b4_mode,
        "b4_label"   : b4_label,
        "b4_flags"   : b4_flags,
        "b5_status"  : b5_status,
    }

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)
        f.flush()

    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# ONE CYCLE — record + run all blocks + write report
# ═══════════════════════════════════════════════════════════════════════════════

def run_cycle():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info("── Cycle %s ──", ts)

    # Default placeholders
    zcr = cf = hf_ratio = None
    b4_mode = b4_label = "not run"
    b4_flags = ""
    b5_status = "not run"
    norm_peak = None

    # Block 1 — record
    filepath, fs, x_raw = block1_record(ts)
    logging.info("B1  recorded  %s", os.path.basename(filepath))

    # Block 2 — normalise
    x_raw, x_norm, peak_raw = block2_normalise(x_raw)
    norm_peak = float(np.max(np.abs(x_norm)))
    logging.info("B2  normalised  raw_peak=%.4f", peak_raw)

    # Block 3 — screen
    b3_dec, b3_reason, e_raw, rms, peak_raw2, clip_rate = block3_screen(x_raw)
    logging.info("B3  %s  %s", b3_dec, b3_reason)

    if b3_dec == "KEEP":

        # Block 4 — route
        b4_mode, b4_label, n_flags, amp, zcr, cf, hf_ratio = block4_route(x_norm, fs)
        b4_flags = n_flags
        logging.info("B4  %s  label=%s  flags=%d/4", b4_mode, b4_label, n_flags)

        if b4_mode == "FULL":
            # Block 5 — enhance
            block5_enhance(x_norm, fs, b4_label)
            b5_status = "activated"
            logging.info("B5  enhancement applied")
        else:
            b5_status = "not activated (LIGHT path)"
            logging.info("B5  skipped (LIGHT)")

    # Generate write-up
    writeup = generate_writeup(
        b3_dec, b3_reason, b4_mode, b4_label, b5_status,
        e_raw, rms, peak_raw, zcr, cf, hf_ratio
    )

    # Write report + CSV row
    report_path = write_report(
        ts, filepath, fs, peak_raw, norm_peak,
        e_raw, rms, clip_rate, zcr, cf, hf_ratio,
        b3_dec, b3_reason, b4_mode, b4_label, b4_flags, b5_status,
        writeup
    )

    logging.info("Report saved  → %s", os.path.basename(report_path))
    return b3_dec, b4_mode, b5_status


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — runs continuously until Ctrl+C or SIGTERM
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    setup()

    # Graceful shutdown on SIGTERM (e.g. from kill command)
    running = {"flag": True}
    def _stop(sig, frame):
        logging.info("Shutdown signal received — stopping after current cycle.")
        running["flag"] = False
    signal.signal(signal.SIGTERM, _stop)

    cycle = 0
    keep_count = drop_count = error_count = 0

    try:
        while running["flag"]:
            cycle += 1
            logging.info("═══ Cycle %d ═══", cycle)
            try:
                b3_dec, b4_mode, b5_status = run_cycle()
                if b3_dec == "KEEP":
                    keep_count += 1
                else:
                    drop_count += 1
            except Exception as e:
                error_count += 1
                logging.error("Cycle %d failed: %s — continuing.", cycle, e)

            logging.info(
                "Stats so far — cycles=%d  KEEP=%d  DROP=%d  errors=%d",
                cycle, keep_count, drop_count, error_count
            )

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt — shutting down cleanly.")

    logging.info(
        "Monitor stopped — total cycles=%d  KEEP=%d  DROP=%d  errors=%d",
        cycle, keep_count, drop_count, error_count
    )


if __name__ == "__main__":
    main()