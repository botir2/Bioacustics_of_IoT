# ===== Chainsaw training on Raspberry Pi (ARMv8 / aarch64-safe) =====
# Notes:
#  * XNNPACK is disabled to avoid picking ARMv8.2+ kernels that can SIGILL on Pi 3/4.
#  * We do NOT export TorchScript on the Pi to avoid JIT kernels that may use dotprod.
#  * Uses plain state_dict + saved threshold and produces training/PR plots.

import os, platform
# --- MUST set safe env vars BEFORE importing NumPy/Torch ---
os.environ.setdefault("PYTORCH_DISABLE_XNNPACK", "1")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
if "aarch64" in platform.machine().lower():
    os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchaudio, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- FIXED PATHS (edit here if you move files) ----------
META_CSV  = "/home/pi/chainsaw-pi/meta/esc50.csv"
AUDIO_DIR = "/home/pi/chainsaw-pi/audio"
SAVE_DIR  = "/home/pi/chainsaw-pi/"

# ---------- TRAINING HYPERPARAMS ----------
EPOCHS     = 100
BATCH_SIZE = 32
LR         = 1e-3
NEG_MULT   = 8
SR         = 16000
N_MELS     = 64


# ---------- MODEL ----------
class TinyCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        z = self.feature(x).squeeze(-1).squeeze(-1)
        return self.fc(z)

# ---------- DATA ----------
class ESC50BinaryDS(Dataset):
    def __init__(self, items, sr=SR, win_s=1.0, n_mels=N_MELS, train=True):
        self.items = items
        self.sr = sr
        self.win_s = win_s
        self.n_samples = int(sr*win_s)
        self.train = train
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024,
            win_length=int(0.025*sr), hop_length=int(0.010*sr),
            n_mels=n_mels, power=2.0)
        self.todb = torchaudio.transforms.AmplitudeToDB()
    def _logmel(self, wav):
        S = self.todb(self.mel(wav))
        return (S - S.mean(dim=(-2,-1), keepdim=True)) / (S.std(dim=(-2,-1), keepdim=True)+1e-6)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(0, keepdim=True)
        L = wav.shape[-1]
        if L >= self.n_samples:
            start = random.randint(0, L - self.n_samples) if self.train else max(0, (L - self.n_samples)//2)
            wav = wav[:, start:start+self.n_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.n_samples - L))
        X = self._logmel(wav).unsqueeze(0)  # [1,1,n_mels,T]
        return X.squeeze(0), torch.tensor(y, dtype=torch.long)

def build_items(meta, folds, neg_mult=NEG_MULT):
    pos = meta[(meta["category"]=="chainsaw") & (meta["fold"].isin(folds))]
    neg = meta[(meta["category"]!="chainsaw") & (meta["fold"].isin(folds))]
    pos_items = [(os.path.join(AUDIO_DIR, fn), 1) for fn in pos["filename"].tolist()]
    neg_files = neg["filename"].tolist()
    k = min(len(neg_files), neg_mult*len(pos_items)) if len(pos_items)>0 else len(neg_files)
    random.shuffle(neg_files)
    neg_items = [(os.path.join(AUDIO_DIR, fn), 0) for fn in neg_files[:k]]
    items = pos_items + neg_items
    random.shuffle(items)
    return items, len(pos_items), len(neg_items)

@torch.no_grad()
def eval_accuracy(dl, model, device="cpu"):
    model.eval(); correct=total=0
    for X,y in dl:
        X = X.to(device)
        y = y.to(device)
        p = model(X).argmax(1)
        correct += (p==y).sum().item(); total += y.numel()
    return correct/total if total else 0.0

@torch.no_grad()
def collect_probs(dl, model, device="cpu"):
    ys=[]; ps=[]
    model.eval()
    for X,y in dl:
        X = X.to(device)
        prob = torch.softmax(model(X), dim=1)[:,1].cpu().numpy()
        ps.append(prob); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

def pick_best_threshold(y_true, y_prob):
    ths = np.linspace(0.01, 0.99, 200)
    best_f1, best_th = 0.0, 0.5
    for th in ths:
        y_pred = (y_prob >= th).astype(np.int32)
        tp = np.sum((y_pred==1)&(y_true==1))
        fp = np.sum((y_pred==1)&(y_true==0))
        fn = np.sum((y_pred==0)&(y_true==1))
        prec = tp/max(1,tp+fp)
        rec  = tp/max(1,tp+fn)
        f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th, best_f1

def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def main():
    # Check paths
    assert os.path.exists(META_CSV),  f"esc50.csv not found at {META_CSV}"
    assert os.path.isdir(AUDIO_DIR),  f"audio/ not found at {AUDIO_DIR}"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print("ESC-50 found ✅", META_CSV)

    set_seed(42)
    meta = pd.read_csv(META_CSV)
    meta["fold"] = meta["fold"].astype(int)

    # Splits
    train_items, npos_tr, nneg_tr = build_items(meta, {1,2,3})
    val_items,   npos_v,  nneg_v  = build_items(meta, {4})
    test_items,  npos_t,  nneg_t  = build_items(meta, {5})
    print(f"Train: pos={npos_tr}, neg={nneg_tr} | Val: pos={npos_v}, neg={nneg_v} | Test: pos={npos_t}, neg={nneg_t}")

    train_dl = DataLoader(ESC50BinaryDS(train_items, train=True),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(ESC50BinaryDS(val_items,   train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(ESC50BinaryDS(test_items,  train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("Data ready ✅")

    device = "cpu"  # Pi CPU
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = torch.nn.CrossEntropyLoss()

    # Track history for plotting later
    hist = {"train_loss": [], "val_acc": []}

    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        model.train(); running=0.0; n=0
        for X,y in train_dl:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward(); opt.step()
            running += loss.item()*y.size(0); n += y.size(0)
        tr_loss = running/max(1,n)

        va = eval_accuracy(val_dl, model, device=device)
        hist["train_loss"].append(tr_loss)
        hist["val_acc"].append(va)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_acc={va:.3f}")
        if va > best_acc:
            best_acc = va
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval()
    print("Training complete ✅ | best val_acc =", best_acc)

    # Pick threshold on VAL
    y_true_v, y_prob_v = collect_probs(val_dl, model, device=device)
    best_th, best_f1 = pick_best_threshold(y_true_v, y_prob_v)
    print(f"Picked threshold (val F1): th={best_th:.3f}, F1={best_f1:.3f}")

    # Test metrics at that threshold
    y_true_t, y_prob_t = collect_probs(test_dl, model, device=device)
    y_pred_t = (y_prob_t >= best_th).astype(np.int32)
    tp = int(np.sum((y_pred_t==1)&(y_true_t==1)))
    fp = int(np.sum((y_pred_t==1)&(y_true_t==0)))
    tn = int(np.sum((y_pred_t==0)&(y_true_t==0)))
    fn = int(np.sum((y_pred_t==0)&(y_true_t==1)))
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    acc = (tp+tn)/max(1,tp+tn+fp+fn)
    print(f"TEST @ th={best_th:.2f} → acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}")

    # Save outputs (state_dict + threshold)
    SAVE = Path(SAVE_DIR)
    state_path = SAVE / "chainsaw_state.pth"
    torch.save(model.state_dict(), state_path)

    (SAVE / "best_threshold.txt").write_text(f"{best_th:.6f}")

    print("\nSaved:")
    print("  state_dict       :", state_path)
    print("  threshold        :", SAVE / 'best_threshold.txt')
    print("\nNext:")
    print(f"  Use state_dict for on-device inference. Threshold = {best_th:.2f}")

    import time

    print("\nWaiting 30 seconds before generating plots...")
    for s in range(30, 0, -1):
        print(f"Continuing in {s:02d}s", end="\r", flush=True)
        time.sleep(1)
    print(" " * 32, end="\r")  # clear the countdown line

    # ===== Save training curves & validation PR/threshold to files (headless-safe) =====
    import matplotlib
    matplotlib.use("Agg")  # no GUI on Pi
    import matplotlib.pyplot as plt

    SAVE.mkdir(parents=True, exist_ok=True)

    # 1) Save curves: validation accuracy + training loss
    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist["val_acc"]) + 1), hist["val_acc"], marker="o")
    ax.set_xlabel("Epoch");
    ax.set_ylabel("Validation accuracy");
    ax.set_title("Validation accuracy vs. epoch")
    ax.grid(True);
    fig.tight_layout()
    fig.savefig(SAVE / "val_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], marker="o")
    ax.set_xlabel("Epoch");
    ax.set_ylabel("Training loss");
    ax.set_title("Training loss vs. epoch")
    ax.grid(True);
    fig.tight_layout()
    fig.savefig(SAVE / "train_loss.png", dpi=150)
    plt.close(fig)

    # 2) Precision–Recall curve using VAL probs
    order = np.argsort(-y_prob_v)
    y_true_sort = y_true_v[order]
    y_prob_sort = y_prob_v[order]

    tp = np.cumsum(y_true_sort == 1)
    fp = np.cumsum(y_true_sort == 0)
    fn_total = (y_true_sort == 1).sum()
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / np.maximum(1, fn_total)

    # Average Precision (area under PR)
    #ap = float(np.trapz(prec, rec))
    ap = float(np.trapezoid(prec, rec))

    # Best F1 threshold (scan unique probs)
    best_f1_local, best_th_local = 0.0, 0.5
    for th in np.unique(y_prob_sort):
        y_pred = (y_prob_v >= th).astype(np.int32)
        TP = np.sum((y_pred == 1) & (y_true_v == 1))
        FP = np.sum((y_pred == 1) & (y_true_v == 0))
        FN = np.sum((y_pred == 0) & (y_true_v == 1))
        precision = TP / max(1, TP + FP)
        recall = TP / max(1, TP + FN)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1_local:
            best_f1_local, best_th_local = f1, float(th)

    # Persist PR plot + numbers
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_xlabel("Recall");
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve (AP={ap:.3f})")
    ax.grid(True);
    fig.tight_layout()
    fig.savefig(SAVE / "pr_curve.png", dpi=150)
    plt.close(fig)

    # Save arrays & summary so you can inspect later or re-threshold
    np.save(SAVE / "val_probs.npy", y_prob_v)
    np.save(SAVE / "val_labels.npy", y_true_v)
    with open(SAVE / "val_pr_summary.txt", "w") as f:
        f.write(f"AP={ap:.6f}\nBest_F1={best_f1_local:.6f}\nBest_th={best_th_local:.6f}\n")

    print(f"Saved plots and PR summary to: {SAVE}")

if __name__ == "__main__":
    main()
