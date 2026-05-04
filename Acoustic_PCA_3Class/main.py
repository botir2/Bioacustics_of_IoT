# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QFrame
)
from PyQt6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_MAP_PATH = "data/warblrb10k/features_map_with_silence.csv"

# Raspberry Pi USB mic settings
MIC_DEVICE_INDEX = None      # None = default input device. If needed, set USB mic index, e.g. 2
MIC_SR = 16000               # good for bird/acoustic feature extraction
MIC_DURATION = 1.0           # record 1 second per update
MIC_UPDATE_MS = 1800         # update every 1.8 seconds


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
        self.ax.set_title("PC1 x PC2 Bird / No-bird Acoustic Map", color="white", fontsize=14)
        self.ax.set_xlabel("PC1", color="white")
        self.ax.set_ylabel("PC2", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, alpha=0.25)

        if self.map_points is not None:
            no_bird = self.map_labels == 0
            bird = self.map_labels == 1
            silence = self.map_labels == 2

            # More visible map points
            self.ax.scatter(
                self.map_points[no_bird, 0],
                self.map_points[no_bird, 1],
                s=14,
                alpha=0.42,
                edgecolors="none",
                label="No-bird"
            )

            self.ax.scatter(
                self.map_points[bird, 0],
                self.map_points[bird, 1],
                s=14,
                alpha=0.48,
                edgecolors="none",
                label="Bird"
            )
            self.ax.scatter(
                self.map_points[silence, 0],
                self.map_points[silence, 1],
                s=18,
                alpha=0.70,
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

        # More visible trajectory
        self.ax.plot(current[:, 0], current[:, 1], linewidth=2.0)
        self.ax.scatter(current[-1, 0], current[-1, 1], s=230, marker="o")
        self.ax.scatter(current[0, 0], current[0, 1], s=130, marker="s")

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

    for start in range(0, len(y) - window, hop):
        frame = y[start:start + window]
        features.append(extract_features_from_array(frame, sr=sr))

    return np.array(features)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Raspberry Pi Acoustic PCA Trajectory Visualiser")
        self.resize(1200, 730)

        self.canvas = PCACanvas()

        self.scaler = None
        self.pca = None
        self.live_points = []

        self.status_label = QLabel("Status: Load saved map")
        self.variance_label = QLabel("PCA variance: -")
        self.file_label = QLabel("File: -")
        self.map_label = QLabel("Map: not loaded")
        self.mic_label = QLabel("Mic: default USB mic")

        self.load_map_button = QPushButton("Load Saved Map")
        self.load_button = QPushButton("Load WAV")
        self.start_button = QPushButton("START WAV")
        self.stop_button = QPushButton("STOP WAV")
        self.start_mic_button = QPushButton("START MIC")
        self.stop_mic_button = QPushButton("STOP MIC")

        self.load_map_button.clicked.connect(self.load_saved_map)
        self.load_button.clicked.connect(self.load_wav)
        self.start_button.clicked.connect(self.start_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        self.start_mic_button.clicked.connect(self.start_mic)
        self.stop_mic_button.clicked.connect(self.stop_mic)

        self.apply_style()

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(300)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(14)
        left_layout.setContentsMargins(18, 18, 18, 18)

        title = QLabel("Acoustic PCA\nTrajectory")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(title)
        left_layout.addSpacing(15)
        left_layout.addWidget(self.load_map_button)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.start_mic_button)
        left_layout.addWidget(self.stop_mic_button)

        left_layout.addSpacing(20)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.map_label)
        left_layout.addWidget(self.variance_label)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.mic_label)
        left_layout.addStretch()

        note = QLabel(
            "Raspberry Pi workflow:\n"
            "1. Load Saved Map\n"
            "2. START MIC\n"
            "3. Play bird sound near USB mic\n\n"
            "Default mic is used.\n"
            "If needed, set MIC_DEVICE_INDEX."
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

        self.mic_timer = QTimer()
        self.mic_timer.timeout.connect(self.update_mic_point)

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
        self.load_button.setStyleSheet("background-color: #2563eb;")
        self.start_button.setStyleSheet("background-color: #16a34a;")
        self.stop_button.setStyleSheet("background-color: #dc2626;")
        self.start_mic_button.setStyleSheet("background-color: #0891b2;")
        self.stop_mic_button.setStyleSheet("background-color: #b45309;")

    def load_saved_map(self):
        if not os.path.exists(FEATURE_MAP_PATH):
            self.status_label.setText("Status: features_map.csv not found")
            return

        self.status_label.setText("Status: Loading saved map...")

        df = pd.read_csv(FEATURE_MAP_PATH)

        feature_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feature_cols].values
        labels = df["label"].values.astype(int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=2)
        map_points = self.pca.fit_transform(X_scaled)

        variance = self.pca.explained_variance_ratio_.sum() * 100

        self.canvas.set_map(map_points, labels)

        self.variance_label.setText(f"PCA variance: {variance:.1f}%")
        self.map_label.setText(f"Map: {len(df)} samples loaded")
        self.status_label.setText("Status: Map ready")

    def load_wav(self):
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

        features = extract_audio_features(file_path)

        if len(features) < 3:
            self.status_label.setText("Status: Audio too short")
            return

        if self.scaler is not None and self.pca is not None:
            X_scaled = self.scaler.transform(features)
            points = self.pca.transform(X_scaled)
        else:
            local_scaler = StandardScaler()
            X_scaled = local_scaler.fit_transform(features)

            local_pca = PCA(n_components=2)
            points = local_pca.fit_transform(X_scaled)

            variance = local_pca.explained_variance_ratio_.sum() * 100
            self.variance_label.setText(f"PCA variance: {variance:.1f}%")

        self.canvas.set_trajectory(points)

        file_name = os.path.basename(file_path)
        self.status_label.setText("Status: WAV ready")
        self.file_label.setText(f"File: {file_name}")

    def start_animation(self):
        self.status_label.setText("Status: WAV running")
        self.timer.start(300)

    def stop_animation(self):
        self.status_label.setText("Status: WAV stopped")
        self.timer.stop()

    def start_mic(self):
        if self.scaler is None or self.pca is None:
            self.status_label.setText("Status: Load Saved Map first")
            return

        self.live_points = []
        self.status_label.setText("Status: Mic running")
        self.mic_timer.start(MIC_UPDATE_MS)

    def stop_mic(self):
        self.status_label.setText("Status: Mic stopped")
        self.mic_timer.stop()

    def update_mic_point(self):
        try:
            audio = sd.rec(
                int(MIC_DURATION * MIC_SR),
                samplerate=MIC_SR,
                channels=1,
                dtype="float32",
                device=MIC_DEVICE_INDEX
            )
            sd.wait()

            y = audio.flatten()

            feature = extract_features_from_array(y, sr=MIC_SR).reshape(1, -1)

            X_scaled = self.scaler.transform(feature)
            point = self.pca.transform(X_scaled)[0]

            self.live_points.append(point)

            if len(self.live_points) > 80:
                self.live_points = self.live_points[-80:]

            self.canvas.set_trajectory(np.array(self.live_points))
            self.canvas.index = max(0, len(self.live_points) - 1)
            self.canvas.update_plot()

            rms = float(np.sqrt(np.mean(y ** 2)))
            self.status_label.setText(f"Status: Mic updated | RMS={rms:.4f}")

        except Exception as e:
            self.status_label.setText(f"Mic error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())