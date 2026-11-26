"""Honest evaluation on public Arabic restaurant reviews (HF ar_res_reviews, 8,364 rows, binary).
Held-out = last 1,000 rows after a fixed shuffle (seed 42). Never used for tuning."""
import json, numpy as np, pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src import nlp
d = pd.read_parquet("data/ar_res_reviews.parquet").sample(frac=1, random_state=42).reset_index(drop=True)
te = d.iloc[-1000:]; tr = d.iloc[:-1000]
v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
lr = LogisticRegression(C=4, max_iter=2000).fit(v.fit_transform(tr.text), tr.polarity)
p_lr = lr.predict(v.transform(te.text))
t0 = time.time(); s = nlp.sentiment(te.text.tolist()); sec = time.time() - t0
lab = {x["label"] for x in s}; print("labels", lab)
p_bert = np.array([1 if x["label"].lower().startswith("pos") else 0 for x in s])
# neutral predictions count as wrong for a binary task: report both
neu = int(sum(x["label"].lower().startswith("neu") for x in s))
nn_mask = np.array([not x["label"].lower().startswith("neu") for x in s])
acc_nn = accuracy_score(te.polarity.values[nn_mask], p_bert[nn_mask])
m = {"camelbert_acc_when_not_neutral": round(float(acc_nn), 4), "heldout_rows": len(te),
     "tfidf_logreg_acc": round(accuracy_score(te.polarity, p_lr), 4),
     "tfidf_logreg_f1": round(f1_score(te.polarity, p_lr), 4),
     "camelbert_zero_shot_acc": round(accuracy_score(te.polarity, p_bert), 4),
     "camelbert_zero_shot_f1": round(f1_score(te.polarity, p_bert), 4),
     "camelbert_neutral_predictions": neu, "camelbert_seconds": round(sec, 1)}
json.dump(m, open("results/nlp_metrics.json", "w"), indent=1); print(m)
