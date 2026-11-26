"""Deep-learning churn model (PyTorch MLP) vs the gradient-boosting model, same stratified 80/20 split (seed 42).
Early stopping uses a validation slice carved from TRAIN only; the test set is touched once."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from src.churn_prep import load_raw, engineer, FEATURES, TARGET
torch.manual_seed(42); np.random.seed(42)
df = engineer(load_raw("data/sample_customers.csv"))
X = df[FEATURES].astype(float).values; y = df[TARGET].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)
sc = StandardScaler().fit(Xt)
T = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
w = torch.tensor([(yt == 0).sum() / (yt == 1).sum()], dtype=torch.float32)
lossf = nn.BCEWithLogitsLoss(pos_weight=w)
xt, yt_t, xv = T(Xt), torch.tensor(yt, dtype=torch.float32), T(Xv)
best, bs, pat = -1, None, 0
for ep in range(300):
    net.train(); perm = torch.randperm(len(xt))
    for i in range(0, len(xt), 64):
        b = perm[i:i+64]; opt.zero_grad(); lossf(net(xt[b]).squeeze(1), yt_t[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): a = roc_auc_score(yv, net(xv).squeeze(1).numpy())
    if a > best: best, bs, pat = a, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        pat += 1
        if pat >= 25: break
net.load_state_dict(bs); net.eval()
with torch.no_grad(): p = torch.sigmoid(net(T(Xte)).squeeze(1)).numpy()
# threshold chosen on validation, not test
with torch.no_grad(): pv = torch.sigmoid(net(xv).squeeze(1)).numpy()
ths = np.linspace(0.2, 0.9, 71); th = ths[int(np.argmax([accuracy_score(yv, pv > t) for t in ths]))]
m = {"mlp_test_accuracy": round(accuracy_score(yte, p > th), 4), "mlp_test_auc": round(roc_auc_score(yte, p), 4),
     "threshold_from_validation": round(float(th), 2), "epochs_run": ep + 1, "test_rows": int(len(yte)), "parameters": sum(q.numel() for q in net.parameters())}
pipe = joblib.load("results/churn_model.joblib")
pg = pipe.predict_proba(pd.DataFrame(Xte, columns=FEATURES))[:, 1]
m["gradient_boosting_same_split_accuracy_at_0.5"] = round(accuracy_score(yte, pg > 0.5), 4)
m["gradient_boosting_same_split_auc"] = round(roc_auc_score(yte, pg), 4)
m["note"] = "gradient boosting model was trained in churn-insight-platform; if its training split differs, its same-split number may be optimistic (shown for reference only)"
json.dump(m, open("results/dl_metrics.json", "w"), indent=1); print(m)
