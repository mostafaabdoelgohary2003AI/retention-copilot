"""Score customers, explain risk in plain language, recommend actions.
Explanations are rule-based readings of the model's known top drivers -
labeled as such, never presented as SHAP."""
import os
import joblib
import pandas as pd
from .churn_prep import engineer, FEATURES

MODEL_PATH = os.environ.get("CHURN_MODEL", "results/churn_model.joblib")
_pipe = None

def _load():
    global _pipe
    if _pipe is None:
        _pipe = joblib.load(MODEL_PATH)
    return _pipe

# plain-language driver rules grounded in the model's feature importance
DRIVERS = [
    ("complaints", lambda v: v >= 1,
     "filed a complaint", "Call them personally within 48 hours and resolve the open issue."),
    ("call_failures", lambda v: v >= 6,
     "experiencing frequent call failures", "Escalate to network quality team; offer a service credit."),
    ("seconds_of_use", lambda v: v < 500,
     "barely using the service", "Send a re-engagement offer matched to their usage pattern."),
    ("subscription_length", lambda v: v <= 6,
     "a new customer still in the fragile first months", "Enroll in the onboarding care program."),
    ("charge_amount", lambda v: v >= 7,
     "on a high tariff relative to usage", "Offer a plan-review to right-size their tariff."),
    ("customer_value", lambda v: v < 60,
     "low and falling account value", "Assign a retention agent before the next billing cycle."),
    ("frequency_of_use", lambda v: v < 8,
     "usage frequency is dropping", "Trigger the win-back campaign with a usage-based perk."),
]

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    eng = engineer(df.copy()).reindex(columns=FEATURES, fill_value=0)
    probs = _load().predict_proba(eng)[:, 1]
    out = df.copy()
    out["risk"] = probs
    out["risk_band"] = pd.cut(probs, [-0.01, 0.3, 0.7, 1.01], labels=["Low", "Medium", "High"])
    reasons, actions = [], []
    for _, row in df.iterrows():
        rs, acs = [], []
        for col, trig, reason, action in DRIVERS:
            if col in row and pd.notna(row[col]) and trig(row[col]):
                rs.append(reason); acs.append(action)
        reasons.append(rs[:3] or ["no single dominant driver - a combination of small signals"])
        actions.append(acs[:3] or ["include in the standard quarterly check-in"])
    out["reasons"] = reasons
    out["actions"] = actions
    return out

def summary(scored: pd.DataFrame) -> dict:
    band = scored["risk_band"].value_counts().to_dict()
    from collections import Counter
    c = Counter()
    for rs in scored["reasons"]:
        for r in rs:
            c[r] += 1
    return {"total": int(len(scored)),
            "high": int(band.get("High", 0)),
            "medium": int(band.get("Medium", 0)),
            "low": int(band.get("Low", 0)),
            "avg_risk": round(float(scored["risk"].mean()), 4),
            "top_reasons": c.most_common(5)}
