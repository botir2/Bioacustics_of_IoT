# -*- coding: utf-8 -*-
"""
train_classifier_3class.py

Train 3-class Random Forest classifier:
    0 = No-bird
    1 = Bird
    2 = Silence

Input:
    data/warblrb10k/features_map_with_silence.csv

Output:
    models/bird_no_bird_silence_random_forest.pkl
    models/bird_no_bird_silence_scaler.pkl

Run:
    python train_classifier_3class.py
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_MAP_PATH = "data/warblrb10k/features_map_with_silence.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "bird_no_bird_silence_random_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "bird_no_bird_silence_scaler.pkl")

CLASS_NAMES = ["No-bird", "Bird", "Silence"]


def main():
    if not os.path.exists(FEATURE_MAP_PATH):
        print("ERROR: features_map_with_silence.csv not found.")
        print(f"Expected path: {FEATURE_MAP_PATH}")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading 3-class feature map...")
    df = pd.read_csv(FEATURE_MAP_PATH)

    feature_cols = [c for c in df.columns if c.startswith("f")]

    if "label" not in df.columns:
        print("ERROR: 'label' column not found.")
        return

    if len(feature_cols) == 0:
        print("ERROR: feature columns f0, f1, ... not found.")
        return

    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    print(f"Total samples: {len(df)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"No-bird samples: {(y == 0).sum()}")
    print(f"Bird samples: {(y == 1).sum()}")
    print(f"Silence samples: {(y == 2).sum()}")

    if (y == 2).sum() == 0:
        print("\nWARNING: No silence samples found. Check label=2 rows.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining 3-class Random Forest...")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    print("\nEvaluation")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    print("\nConfusion matrix")
    print("Rows=true, columns=predicted")
    print("Labels: 0=No-bird, 1=Bird, 2=Silence")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    print("\nClassification report")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nDONE.")
    print(f"Model saved:  {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")


if __name__ == "__main__":
    main()
