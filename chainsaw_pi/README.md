# chainsaw_pi — Raspberry Pi Chainsaw Detector

Small CNN model that detects **chainsaw sound** on a **Raspberry Pi**.
Train, export to TorchScript, test one WAV, and evaluate datasets.

## Data (ESC-50)
We use the **ESC-50** dataset (2000 clips, 50 classes, 5-fold).  
Official repo: https://github.com/karoldvl/ESC-50 (Karol J. Piczak)  
License: **CC BY-NC 4.0**. We do **not** re-distribute ESC-50 audio here.  
Please download from the official repo and follow the license + Freesound terms.

## Folders
- `srcCode/` — ModelTrain.py, quatize.py, test.py, accuracyTest.py  
- `results/` — CSV/JSON metrics  
- `images/Image_results/` — plots (loss/accuracy/PR/ROC/spectrograms)  
- `models/` — saved models (.pth/.pt) — use Git LFS for big files  
- `samples/` — example audio (e.g., chainsawTest.wav)  
- `meta/` — best_threshold* and small metadata  
- `requirements.txt`

## Install
```bash
pip install -r chainsaw_pi/requirements.txt
# Install PyTorch for your platform: https://pytorch.org/get-started/locally/
