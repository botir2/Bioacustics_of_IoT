#!/usr/bin/env python3
import os, numpy as np, torch, torch.nn as nn, torchaudio
from pathlib import Path

# ---- FIXED CONFIG ----
STATE_PTH  = "/home/pi/chainsaw-pi/chainsaw_state.pth"   # change if needed
AUDIO_PATH = "/home/pi/chainsaw-pi/chainsawTest.wav"                                          # e.g. "/home/pi/chainsaw-pi/audio/test.wav"
THRESH     = 0.30
SR         = 16000
N_MELS     = 64

# ---- model must match training ----
class TinyCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.10),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        z = self.feature(x).squeeze(-1).squeeze(-1)
        return self.fc(z)

def frontend(sr=SR, n_mels=N_MELS):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024,
        win_length=int(0.025*sr), hop_length=int(0.010*sr),
        n_mels=n_mels, power=2.0)
    to_db = torchaudio.transforms.AmplitudeToDB()
    def feat(x1c):
        S = to_db(mel(x1c))
        S = (S - S.mean((-2,-1),True)) / (S.std((-2,-1),True)+1e-6)
        return S.unsqueeze(0)   # [1,1,64,T]
    return feat

@torch.no_grad()
def slide_and_score(model, wav1c, sr=SR, win_s=1.0, hop_s=0.25, feat_fn=None):
    N = int(sr*win_s); H = int(sr*hop_s)
    if wav1c.shape[-1] < N:
        wav1c = torch.nn.functional.pad(wav1c, (0, N - wav1c.shape[-1]))
    times, probs = [], []
    for st in range(0, wav1c.shape[-1]-N+1, H):
        X = feat_fn(wav1c[:, st:st+N])
        p = torch.softmax(model(X), dim=1)[0,1].item()
        times.append(st/sr); probs.append(p)
    return np.array(times), np.array(probs)

def main():
    pth = Path(STATE_PTH).expanduser()
    assert pth.exists(), f"state_dict not found: {pth}"

    # load weights
    m = TinyCNN().eval()
    sd = torch.load(str(pth), map_location="cpu")
    m.load_state_dict(sd, strict=False)

    # dummy forward
    y = m(torch.randn(1,1,64,100))
    print("OK: loaded state_dict. Dummy forward shape:", tuple(y.shape))

    # optional scoring of an audio file
    if AUDIO_PATH:
        feat = frontend(SR, N_MELS)
        wav, sr0 = torchaudio.load(AUDIO_PATH)
        if sr0 != SR:
            wav = torchaudio.functional.resample(wav, sr0, SR)
        wav = wav.mean(0, keepdim=True)
        t, p = slide_and_score(m, wav, SR, 1.0, 0.25, feat)
        print(f"Frames: {len(p)} | max={p.max():.3f} | mean={p.mean():.3f}")

        th_on = float(THRESH)
        th_off = max(0.5*THRESH, 0.7*THRESH)
        on=False; on_st=off_st=0; events=[]
        for ti,pi in zip(t,p):
            if pi>=th_on: on_st+=1; off_st=0
            else:
                on_st=0
                if pi<=th_off: off_st+=1
                else: off_st=0
            if not on and on_st>=3:
                on=True; start=ti; peak=pi
            elif on:
                peak=max(peak,pi)
                if off_st>=2:
                    events.append((start,ti,peak)); on=False
        print("Events:")
        if not events: print("  (none)")
        else:
            for s,e,pk in events: print(f"  {s:6.2f}s → {e:6.2f}s (peak={pk:.2f})")
        print("Decision:", "CHAINSAW" if (len(events)>0 or (p>=th_on).any()) else "OTHER")

if __name__ == "__main__":
    main()
