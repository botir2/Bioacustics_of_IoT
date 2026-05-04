import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import librosa


DATA_DIR = "data/warblrb10k"
WAV_DIR = os.path.join(DATA_DIR, "wav")
META_PATH = os.path.join(DATA_DIR, "warblrb10k_public_metadata.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "features_map.csv")

SR = 16000
N_WORKERS = max(1, mp.cpu_count() - 1)


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


def process_one(task):
    index, itemid, label = task
    wav_path = os.path.join(WAV_DIR, itemid + ".wav")

    if not os.path.exists(wav_path):
        return None, f"Missing: {wav_path}"

    try:
        features = extract_features(wav_path)

        output_row = {
            "itemid": itemid,
            "label": label
        }

        for j, value in enumerate(features):
            output_row[f"f{j}"] = value

        return output_row, f"{index} processed: {itemid}, label={label}"

    except Exception as e:
        return None, f"Error processing {itemid}: {e}"


def main():
    metadata = pd.read_csv(META_PATH)

    tasks = []
    for i, row in metadata.iterrows():
        itemid = str(row["itemid"])
        label = int(row["hasbird"])
        tasks.append((i + 1, itemid, label))

    rows = []

    print(f"Total tasks: {len(tasks)}")
    print(f"Using workers: {N_WORKERS}")

    with mp.Pool(processes=N_WORKERS) as pool:
        for result, message in pool.imap_unordered(process_one, tasks, chunksize=8):
            print(message)

            if result is not None:
                rows.append(result)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDONE")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total files processed: {len(out_df)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()