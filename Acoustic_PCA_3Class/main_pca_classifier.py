# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import librosa
import joblib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QFrame
)
from PyQt6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 3-class feature map:
# 0 = No-bird
# 1 = Bird
# 2 = Silence
FEATURE_MAP_PATH = "data/warblrb10k/features_map_with_silence.csv"

MODEL_PATH = "models/bird_no_bird_silence_random_forest.pkl"
MODEL_SCALER_PATH = "models/bird_no_bird_silence_scaler.pkl"

CLASS_NAMES = {
    0: "NO-BIRD / BACKGROUND",
    1: "BIRD / KEEP",
    2: "SILENCE / DROP"
}

SILENCE_RMS_THRESHOLD = 0.005


class PCACanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6), facecolor="#0f172a")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.map_points = None
        self.map_labels = None
        self.trajectory_points = None
        self.index = 0
        self.prepare_empty_plot()

    def prepare_empty_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#111827")
        self.ax.set_title("PC1 x PC2 Acoustic PCA Map", color="white", fontsize=14)
        self.ax.set_xlabel("PC1", color="white")
        self.ax.set_ylabel("PC2", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, alpha=0.25)
        self.draw()

    def set_map(self, points, labels):
        self.map_points = points
        self.map_labels = labels
        self.trajectory_points = None
        self.index = 0
        self.draw_map()

    def set_trajectory(self, points):
        self.trajectory_points = points
        self.index = 0
        self.draw_map()

    def draw_map(self):
        self.ax.clear()
        self.ax.set_facecolor("#111827")
        self.ax.set_title("PC1 x PC2 Bird / No-bird / Silence Acoustic Map", color="white", fontsize=14)
        self.ax.set_xlabel("PC1", color="white")
        self.ax.set_ylabel("PC2", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, alpha=0.25)

        if self.map_points is not None:
            no_bird = self.map_labels == 0
            bird = self.map_labels == 1
            silence = self.map_labels == 2

            self.ax.scatter(
                self.map_points[no_bird, 0],
                self.map_points[no_bird, 1],
                s=18,
                alpha=0.40,
                edgecolors="none",
                label="No-bird"
            )

            self.ax.scatter(
                self.map_points[bird, 0],
                self.map_points[bird, 1],
                s=18,
                alpha=0.45,
                edgecolors="none",
                label="Bird"
            )

            self.ax.scatter(
                self.map_points[silence, 0],
                self.map_points[silence, 1],
                s=30,
                alpha=0.75,
                edgecolors="none",
                label="Silence"
            )

            legend = self.ax.legend(facecolor="#111827", edgecolor="#334155")
            for text in legend.get_texts():
                text.set_color("white")

        self.draw()

    def update_plot(self):
        if self.trajectory_points is None:
            return

        if self.index >= len(self.trajectory_points):
            return

        self.draw_map()
        current = self.trajectory_points[:self.index + 1]

        self.ax.plot(current[:, 0], current[:, 1], linewidth=2.0)
        self.ax.scatter(current[-1, 0], current[-1, 1], s=260, marker="o")
        self.ax.scatter(current[0, 0], current[0, 1], s=150, marker="s")

        self.index += 1
        self.draw()


def extract_features_from_array(y, sr=16000):
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


def extract_audio_features(file_path, sr=16000, window_sec=1.0, hop_sec=0.5):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    window = int(window_sec * sr)
    hop = int(hop_sec * sr)

    features = []
    rms_values = []

    for start in range(0, len(y) - window, hop):
        frame = y[start:start + window]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_values.append(rms)
        features.append(extract_features_from_array(frame, sr=sr))

    return np.array(features), np.array(rms_values)


def predict_whole_wav(features, rms_values, clf, clf_scaler):
    if features is None or len(features) == 0:
        return "ERROR", 0.0, "No valid audio windows"

    mean_rms = float(np.mean(rms_values))

    # Hard safety rule: very low energy is silence/drop
    if mean_rms < SILENCE_RMS_THRESHOLD:
        return "SILENCE / DROP", 100.0, f"Mean RMS={mean_rms:.5f} | RMS hard rule"

    X_scaled = clf_scaler.transform(features)
    probs = clf.predict_proba(X_scaled)
    preds = clf.predict(X_scaled)

    # Average class probability across windows
    mean_probs = np.mean(probs, axis=0)

    # Ensure class order comes from the trained model
    classes = list(clf.classes_)
    prob_by_class = {int(cls): float(mean_probs[i]) for i, cls in enumerate(classes)}

    no_bird_prob = prob_by_class.get(0, 0.0)
    bird_prob = prob_by_class.get(1, 0.0)
    silence_prob = prob_by_class.get(2, 0.0)

    final_class = max(prob_by_class, key=prob_by_class.get)
    label = CLASS_NAMES.get(final_class, f"CLASS {final_class}")
    confidence = prob_by_class[final_class] * 100.0

    no_bird_ratio = float(np.mean(preds == 0))
    bird_ratio = float(np.mean(preds == 1))
    silence_ratio = float(np.mean(preds == 2))

    detail = (
        f"Mean RMS={mean_rms:.5f} | "
        f"No-bird prob={no_bird_prob*100:.1f}% | "
        f"Bird prob={bird_prob*100:.1f}% | "
        f"Silence prob={silence_prob*100:.1f}% | "
        f"Windows: No-bird={no_bird_ratio*100:.1f}%, "
        f"Bird={bird_ratio*100:.1f}%, "
        f"Silence={silence_ratio*100:.1f}%"
    )

    return label, confidence, detail


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Acoustic PCA + 3-Class Classifier")
        self.resize(1220, 760)

        self.canvas = PCACanvas()

        self.pca_scaler = None
        self.pca = None

        self.clf = None
        self.clf_scaler = None

        self.status_label = QLabel("Status: Load saved map and classifier")
        self.variance_label = QLabel("PCA variance: -")
        self.file_label = QLabel("File: -")
        self.map_label = QLabel("Map: not loaded")
        self.model_label = QLabel("Model: not loaded")

        self.prediction_label = QLabel("Prediction: -")
        self.confidence_label = QLabel("Confidence: -")
        self.detail_label = QLabel("Detail: -")
        self.detail_label.setWordWrap(True)

        self.load_map_button = QPushButton("Load Saved Map")
        self.load_model_button = QPushButton("Load 3-Class Classifier")
        self.load_button = QPushButton("Load WAV + Predict")
        self.start_button = QPushButton("START Trajectory")
        self.stop_button = QPushButton("STOP")

        self.load_map_button.clicked.connect(self.load_saved_map)
        self.load_model_button.clicked.connect(self.load_classifier)
        self.load_button.clicked.connect(self.load_wav_and_predict)
        self.start_button.clicked.connect(self.start_animation)
        self.stop_button.clicked.connect(self.stop_animation)

        self.apply_style()

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(350)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(13)
        left_layout.setContentsMargins(18, 18, 18, 18)

        title = QLabel("Acoustic PCA\n+ 3-Class Classifier")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(title)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.load_map_button)
        left_layout.addWidget(self.load_model_button)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.stop_button)

        left_layout.addSpacing(16)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.map_label)
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.variance_label)
        left_layout.addWidget(self.file_label)

        left_layout.addSpacing(18)
        left_layout.addWidget(self.prediction_label)
        left_layout.addWidget(self.confidence_label)
        left_layout.addWidget(self.detail_label)

        left_layout.addStretch()

        note = QLabel(
            "Workflow:\n"
            "1. Load Saved Map\n"
            "2. Load 3-Class Classifier\n"
            "3. Load WAV + Predict\n"
            "4. START Trajectory\n\n"
            "Classes:\n"
            "0 = No-bird\n"
            "1 = Bird\n"
            "2 = Silence"
        )
        note.setObjectName("noteLabel")
        note.setWordWrap(True)
        left_layout.addWidget(note)

        left_panel.setLayout(left_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.canvas.update_plot)

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #020617;
            }

            #leftPanel {
                background-color: #0f172a;
                border-right: 1px solid #334155;
            }

            QLabel {
                color: #e5e7eb;
                font-size: 13px;
            }

            #titleLabel {
                color: #ffffff;
                font-size: 22px;
                font-weight: bold;
            }

            #noteLabel {
                color: #cbd5e1;
                background-color: #1e293b;
                border-radius: 10px;
                padding: 10px;
            }

            QPushButton {
                color: white;
                font-size: 15px;
                font-weight: bold;
                border-radius: 10px;
                padding: 12px;
            }
        """)

        self.load_map_button.setStyleSheet("background-color: #7c3aed;")
        self.load_model_button.setStyleSheet("background-color: #9333ea;")
        self.load_button.setStyleSheet("background-color: #2563eb;")
        self.start_button.setStyleSheet("background-color: #16a34a;")
        self.stop_button.setStyleSheet("background-color: #dc2626;")

        self.prediction_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #ffffff;"
        )
        self.confidence_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #e5e7eb;"
        )

    def load_saved_map(self):
        if not os.path.exists(FEATURE_MAP_PATH):
            self.status_label.setText("Status: features_map_with_silence.csv not found")
            return

        self.status_label.setText("Status: Loading 3-class map...")

        df = pd.read_csv(FEATURE_MAP_PATH)

        feature_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feature_cols].values
        labels = df["label"].values.astype(int)

        self.pca_scaler = StandardScaler()
        X_scaled = self.pca_scaler.fit_transform(X)

        self.pca = PCA(n_components=2)
        map_points = self.pca.fit_transform(X_scaled)

        variance = self.pca.explained_variance_ratio_.sum() * 100

        self.canvas.set_map(map_points, labels)

        n_no_bird = int((labels == 0).sum())
        n_bird = int((labels == 1).sum())
        n_silence = int((labels == 2).sum())

        self.variance_label.setText(f"PCA variance: {variance:.1f}%")
        self.map_label.setText(
            f"Map: {len(df)} loaded | No-bird={n_no_bird}, Bird={n_bird}, Silence={n_silence}"
        )
        self.status_label.setText("Status: 3-class map ready")

    def load_classifier(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_SCALER_PATH):
            self.status_label.setText("Status: 3-class model not found. Run train_classifier_3class.py first")
            return

        self.clf = joblib.load(MODEL_PATH)
        self.clf_scaler = joblib.load(MODEL_SCALER_PATH)

        self.model_label.setText("Model: 3-class Random Forest loaded")
        self.status_label.setText("Status: 3-class classifier ready")

    def load_wav_and_predict(self):
        if self.pca_scaler is None or self.pca is None:
            self.status_label.setText("Status: Load Saved Map first")
            return

        if self.clf is None or self.clf_scaler is None:
            self.status_label.setText("Status: Load 3-Class Classifier first")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV file",
            "",
            "Audio Files (*.wav *.mp3 *.flac)"
        )

        if not file_path:
            return

        self.status_label.setText("Status: Processing WAV...")
        self.file_label.setText("File: loading...")

        features, rms_values = extract_audio_features(file_path)

        if len(features) < 1:
            self.status_label.setText("Status: Audio too short")
            return

        # PCA visualisation
        X_pca_scaled = self.pca_scaler.transform(features)
        points = self.pca.transform(X_pca_scaled)
        self.canvas.set_trajectory(points)

        # 3-class Random Forest prediction
        label, confidence, detail = predict_whole_wav(
            features,
            rms_values,
            self.clf,
            self.clf_scaler
        )

        file_name = os.path.basename(file_path)
        self.file_label.setText(f"File: {file_name}")
        self.prediction_label.setText(f"Prediction: {label}")
        self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
        self.detail_label.setText(f"Detail: {detail}")
        self.status_label.setText("Status: WAV predicted")

    def start_animation(self):
        self.status_label.setText("Status: Trajectory running")
        self.timer.start(300)

    def stop_animation(self):
        self.status_label.setText("Status: Stopped")
        self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
