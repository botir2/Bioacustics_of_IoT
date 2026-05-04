import os
import numpy as np
import pandas as pd
import librosa

FEATURE_MAP_PATH = "data/warblrb10k/features_map.csv"
SILENCE_DIR = "data/silence_wav"
OUTPUT_PATH = "data/warblrb10k/features_map_with_silence.csv"

SR = 16000


def extract_features(file_path, sr=SR):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    rms = librosa.feature.rms(y=y)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    return np.concatenate([
        [rms, zcr, centroid, bandwidth, rolloff],
        mfcc_mean
    ])


def main():
    base_df = pd.read_csv(FEATURE_MAP_PATH)

    rows = []

    wav_files = sorted([
        f for f in os.listdir(SILENCE_DIR)
        if f.lower().endswith(".wav")
    ])

    print(f"Found silence files: {len(wav_files)}")

    for i, filename in enumerate(wav_files):
        path = os.path.join(SILENCE_DIR, filename)

        try:
            features = extract_features(path)

            row = {
                "itemid": filename.replace(".wav", ""),
                "label": 2
            }

            for j, value in enumerate(features):
                row[f"f{j}"] = value

            rows.append(row)
            print(f"{i+1} processed: {filename}, label=2")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    silence_df = pd.DataFrame(rows)

    combined_df = pd.concat([base_df, silence_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDONE")
    print(f"Original rows: {len(base_df)}")
    print(f"Silence rows added: {len(silence_df)}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()