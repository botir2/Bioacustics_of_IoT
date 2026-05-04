import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# ---------- Dataset ----------

class AudioFolderDataset(Dataset):
    """
    Reads audio files from a directory layout like:

        root/
          class_a/*.wav
          class_b/*.wav
          ...

    and returns (features, label) pairs.
    """

    def __init__(self, root_dir, sample_rate=16000, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.transform = transform

        self.files = []
        self.labels = []
        self.class_to_idx = {}

        class_names = sorted(
            [d.name for d in os.scandir(root_dir) if d.is_dir()]
        )

        for idx, cls in enumerate(class_names):
            self.class_to_idx[cls] = idx
            pattern = os.path.join(root_dir, cls, "*.wav")
            for f in glob.glob(pattern):
                self.files.append(f)
                self.labels.append(idx)

        print(f"[{root_dir}] -> {len(self.files)} files, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(path)   # [channels, time]

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.transform is not None:
            features = self.transform(waveform)  # [1, n_mels, time]
        else:
            features = waveform

        return features, label


def make_transform(sample_rate=16000, n_mels=64):
    """Mel-spectrogram + log amplitude (Edge Impulse style)."""
    return nn.Sequential(
        MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels),
        AmplitudeToDB()
    )


# ---------- Simple 2D CNN model ----------

class AudioCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 64, 1, 1]
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, 1, F, T]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # [B, 64]
        x = self.classifier(x)
        return x


# ---------- Train / eval helpers ----------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # if [B, F, T] -> add channel dimension
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------- Main ----------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = make_transform(args.sample_rate, args.n_mels)

    train_dir = args.train_dir or os.path.join(args.data_root, "training")
    test_dir = args.test_dir or os.path.join(args.data_root, "testing")

    train_ds = AudioFolderDataset(train_dir, sample_rate=args.sample_rate, transform=transform)
    test_ds = AudioFolderDataset(test_dir, sample_rate=args.sample_rate, transform=transform)

    n_classes = len(train_ds.class_to_idx)
    model = AudioCNN(n_classes).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

            print(
                f"Epoch {epoch:02d}/{args.epochs} "
                f"| train loss {train_loss:.4f}, acc {train_acc:.3f} "
                f"| val loss {val_loss:.4f}, acc {val_acc:.3f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.model_path)
                print(f"  -> saved new best model to {args.model_path}")

    elif args.mode == "eval":
        # Just evaluate an existing model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Eval only | loss {val_loss:.4f}, acc {val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESC-50 / Edge Impulse style audio classifier")

    # Paths
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"C:\Users\bkarimov\OneDrive - University of Tasmania\Desktop\ESC-50-master\renamed",
        help="Root folder that contains 'training' and 'testing'"
    )
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Optional: explicit training folder (else data_root/training)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional: explicit testing folder (else data_root/testing)")

    # Feature extraction
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)

    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--model_path", type=str, default="esc50_cnn_best.pt")

    # Parse args (Jupyter safe)
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(args=[])

    main(args)
