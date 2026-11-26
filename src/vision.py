"""Site-tile land-use classifier: ResNet18 fine-tuned on EuroSAT (27,000 Sentinel-2 tiles, 10 classes).
Weights come from the sat-vision-service project (held-out accuracy 94.1%)."""
import io, os
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
CLASSES = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
PLAIN = {"AnnualCrop":"farmland","Forest":"forest","HerbaceousVegetation":"grassland","Highway":"highway","Industrial":"industrial area","Pasture":"pasture","PermanentCrop":"orchards","Residential":"residential area","River":"river","SeaLake":"sea or lake"}
# area types where a store/branch would have natural footfall
FOOTFALL = {"Residential": "high", "Industrial": "medium", "Highway": "medium"}
PATH = os.environ.get("VISION_MODEL", "results/vision_resnet18_fp16.pt")
_m = None
TF = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def model():
    global _m
    if _m is None:
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        sd = torch.load(PATH, map_location="cpu", weights_only=False)
        m.load_state_dict({k: v.float() for k, v in sd.items()}); m.eval(); _m = m
    return _m
def classify(png_bytes, k=3):
    x = TF(Image.open(io.BytesIO(png_bytes)).convert("RGB")).unsqueeze(0)
    with torch.no_grad(): p = torch.softmax(model()(x), 1)[0]
    top = p.topk(k)
    res = [{"class": CLASSES[i], "plain": PLAIN[CLASSES[i]], "prob": round(float(v), 4)} for v, i in zip(top.values, top.indices)]
    best = res[0]
    return {"top": res, "land_use": best["plain"], "confident": best["prob"] >= 0.7,
            "footfall": FOOTFALL.get(best["class"], "low") if best["prob"] >= 0.7 else "unknown"}
