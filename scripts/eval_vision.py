"""Re-check the fp16 vision weights on the SAME held-out split sat-vision-service used
(stratified 80/20, seed 42 on pooled EuroSAT tiles). Run from the sat-vision-service folder:
  cd ../sat-vision-service && python /path/to/retention-copilot/scripts/eval_vision.py
The raw 'test' parquet is NOT a valid held-out set here: it overlaps the pooled training data."""
import sys, json, torch, torch.nn as nn
from torchvision import models
sys.path.insert(0, ".")
from src.data import load_arrays_cached, stratified_split, CLASSES
from src.train import norm
X, y = load_arrays_cached(); _, te = stratified_split(y)
m = models.resnet18(weights=None); m.fc = nn.Linear(512, len(CLASSES))
sd = torch.load(sys.argv[1], map_location="cpu"); m.load_state_dict({k: v.float() for k, v in sd.items()}); m.eval()
ok = 0
for i in range(0, len(te), 256):
    idx = te[i:i+256]
    with torch.no_grad(): p = m(norm(X[idx])).argmax(1)
    ok += int((p == torch.from_numpy(y[idx])).sum())
r = {"heldout_tiles": len(te), "accuracy": round(ok / len(te), 4), "weights": "fp16 copy of sat-vision-service model"}
print(r); json.dump(r, open(sys.argv[2], "w"))
