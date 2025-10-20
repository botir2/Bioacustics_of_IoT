import os
os.environ.setdefault("OPENBLAS_CORETYPE","ARMV8")
os.environ.setdefault("ATEN_DISABLE_XNNPACK","1")
os.environ.setdefault("OMP_NUM_THREADS","1")

import torch, torch.nn as nn
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False  # extra safety on ARM

class TinyCNN(nn.Module):
    def __init__(self, n_mels=64, n_classes=2):
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

m = TinyCNN().eval()
m.load_state_dict(torch.load("/home/pi/chainsaw-pi/chainsaw_state.pth", map_location="cpu"))

with torch.inference_mode():
    ex = torch.randn(1,1,64,100)
    ts = torch.jit.trace(m, ex)
    ts.save("/home/pi/chainsaw-pi/chainsaw_model_float_pi.pt")
print("Saved chainsaw_model_float_pi.pt")