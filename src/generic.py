"""General churn path: map any churn CSV, train on it, score it with honest out-of-fold risk.

Nothing here is tuned per dataset. The model is a plain gradient boosting pipeline.
Risk scores for the uploaded rows come from cross-validation (each customer is scored by a
model that never saw that customer), so the dashboard does not reward memorising the file.
"""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MAX_ROWS, MIN_ROWS, MIN_PER_CLASS = 20000, 200, 30
MAX_CATEGORIES = 30
POS = {"1", "yes", "y", "true", "t", "churn", "churned", "left", "exited", "attrited", "attrited customer",
       "cancelled", "canceled", "lost", "closed"}
NEG = {"0", "no", "n", "false", "f", "stayed", "retained", "existing", "existing customer", "active", "current", "not churned"}
TARGET_HINTS = ["churn", "exited", "attrition", "attrited", "left", "cancel"]
LEAK_HINTS = ["churn score", "churn reason", "churn category", "churn label", "cltv"]


class DataError(ValueError):
    """A problem the user can fix. The message is shown on the page as written."""


def _clean_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise DataError("Two columns have the same name: " + ", ".join(sorted(set(df.columns[df.columns.duplicated()]))))
    return df


def suggest(df):
    """Guess which columns are the churn flag, customer id, tenure and value. Only guesses; the user confirms."""
    df = _clean_cols(df)
    low = {c: c.lower() for c in df.columns}
    target = next((c for h in TARGET_HINTS for c, l in low.items() if h in l and not any(k in l for k in LEAK_HINTS)
                   and df[c].nunique(dropna=True) == 2), None)
    cid = next((c for c, l in low.items() if re.search(r"(^|[^a-z])(customer ?id|cust ?id|id|msisdn|account)([^a-z]|$)", l)
                and df[c].nunique() > 0.9 * len(df)), None)
    tenure = next((c for c, l in low.items() if any(k in l for k in ["tenure", "subscription length", "months", "age of account"])
                   and pd.api.types.is_numeric_dtype(df[c])), None)
    value = next((c for c, l in low.items() if any(k in l for k in ["monthlycharges", "monthly charge", "customer value", "balance", "revenue", "arpu", "estimatedsalary"])
                  and pd.api.types.is_numeric_dtype(df[c])), None)
    return {"target": target, "id": cid, "tenure": tenure, "value": value}


def to_binary(s):
    """Turn a churn column into 0/1. Refuses anything it cannot read with certainty."""
    vals = s.dropna()
    if vals.empty:
        raise DataError("The churn column is empty.")
    if pd.api.types.is_bool_dtype(vals):
        return s.astype(float)
    uniq = {str(v).strip().lower() for v in vals.unique()}
    if len(uniq) != 2:
        raise DataError(f"The churn column must have exactly 2 different values (like Yes/No or 1/0). It has {len(uniq)}.")
    if uniq <= {"0", "1", "0.0", "1.0"}:
        return pd.to_numeric(s, errors="coerce")
    pos, neg = uniq & POS, uniq & NEG
    if len(pos) == 1 and len(neg) == 1:
        return s.map(lambda v: np.nan if pd.isna(v) else (1.0 if str(v).strip().lower() in pos else 0.0))
    raise DataError("I could not tell which value means the customer left. The two values are: "
                    + ", ".join(sorted(uniq)) + ". Use 1 for left and 0 for stayed, or Yes/No.")


def prepare(df, target, id_col=None):
    df = _clean_cols(df)
    for name, col in (("churn", target), ("customer id", id_col)):
        if col and col not in df.columns:
            raise DataError(f"The {name} column '{col}' is not in the file.")
    if not target:
        raise DataError("Choose which column says whether the customer left.")
    y = to_binary(df[target])
    keep = y.notna()
    df, y = df[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise DataError(f"Need at least {MIN_ROWS} customers with a known churn value to train a model. Found {len(df)}.")
    if len(df) > MAX_ROWS:
        raise DataError(f"Max {MAX_ROWS:,} customers per file. This file has {len(df):,}.")
    n1 = int(y.sum()); n0 = len(y) - n1
    if min(n0, n1) < MIN_PER_CLASS:
        raise DataError(f"Only {min(n0, n1)} customers are in the smaller group (left or stayed). Need at least {MIN_PER_CLASS}.")
    drop = {target} | ({id_col} if id_col else set())
    warnings = []
    leaks = [c for c in df.columns if c not in drop and any(k in c.lower() for k in LEAK_HINTS)]
    if leaks:
        drop |= set(leaks)
        warnings.append("Left out columns that look like they describe the outcome itself: " + ", ".join(leaks) + ".")
    X = df.drop(columns=list(drop))
    num, cat = [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(float); num.append(c); continue
        if not pd.api.types.is_numeric_dtype(s):
            conv = pd.to_numeric(s.astype(str).str.strip().replace("", np.nan), errors="coerce")
            if conv.notna().mean() >= 0.95 and s.notna().sum() > 0:
                X[c] = conv; num.append(c); continue
        if pd.api.types.is_numeric_dtype(X[c]):
            if X[c].nunique(dropna=True) > 1:
                num.append(c)
            continue
        if 1 < s.nunique(dropna=True) <= MAX_CATEGORIES:
            X[c] = s.astype("object").where(s.notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v).strip())
            cat.append(c)
    skipped = [c for c in X.columns if c not in num + cat]
    if skipped:
        warnings.append("Skipped columns that are not usable as inputs (free text, one value, or too many different values): " + ", ".join(skipped[:8]) + ("..." if len(skipped) > 8 else "") + ".")
    if not (num or cat):
        raise DataError("No usable input columns were left after removing the churn, id and unusable columns.")
    return X[num + cat], y, num, cat, df, warnings


def _pipe(num, cat):
    pre = ColumnTransformer([
        ("n", SimpleImputer(strategy="median"), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    return Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0))])


def _reason_tables(X, y, num, cat, feats):
    """Churn rate per value group for the top features, measured on the uploaded file."""
    base = float(y.mean()); tables = {}
    for f in feats:
        if f in num:
            try:
                bins = pd.qcut(X[f], 4, duplicates="drop")
            except ValueError:
                continue
            grp = pd.DataFrame({"g": bins, "y": y}).groupby("g", observed=True)["y"].agg(["mean", "size"])
            tables[f] = ("num", bins.cat.categories, grp, float(X[f].min()))
        else:
            grp = pd.DataFrame({"g": X[f], "y": y}).groupby("g")["y"].agg(["mean", "size"])
            tables[f] = ("cat", None, grp, None)
    return base, tables


def _row_reasons(row, base, tables):
    out = []
    for f, (kind, cats, grp, mn) in tables.items():
        v = row[f]
        if pd.isna(v):
            continue
        key = None
        if kind == "num":
            for iv in cats:
                if v in iv:
                    key = iv; break
        else:
            key = v
        if key is None or key not in grp.index:
            continue
        rate, n = float(grp.loc[key, "mean"]), int(grp.loc[key, "size"])
        if n >= 30 and rate >= 1.25 * base:
            label = f"{f} between {max(key.left, mn):g} and {key.right:g}" if kind == "num" else f"{f} = {v}"
            out.append((rate / base, f"{label} (customers like this left {rate:.0%} of the time, vs {base:.0%} overall)"))
    out.sort(key=lambda t: -t[0])
    return [t[1] for t in out[:3]]


def train_and_score(df, target, id_col=None, tenure=None, value=None, seed=0):
    X, y, num, cat, kept, warnings = prepare(df, target, id_col)
    # honest held-out metrics (25%), plus a majority-class baseline
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    p = _pipe(num, cat).fit(Xtr, ytr)
    pr = p.predict_proba(Xte)[:, 1]
    metrics = {"rows": int(len(y)), "churn_rate": round(float(y.mean()), 4), "heldout_rows": int(len(yte)),
               "auc": round(float(roc_auc_score(yte, pr)), 4),
               "accuracy": round(float(accuracy_score(yte, pr >= 0.5)), 4),
               "baseline_accuracy": round(float(max(yte.mean(), 1 - yte.mean())), 4),
               "inputs_used": len(num) + len(cat)}
    # top drivers by permutation importance on the held-out part
    sub = Xte.sample(min(len(Xte), 1500), random_state=seed)
    imp = permutation_importance(p, sub, yte.loc[sub.index], scoring="roc_auc", n_repeats=3, random_state=seed)
    order = np.argsort(-imp.importances_mean)
    feats = [X.columns[i] for i in order if imp.importances_mean[i] > 0][:6] or [X.columns[order[0]]]
    metrics["top_features"] = feats
    # out-of-fold risk for every uploaded row
    risk = cross_val_predict(_pipe(num, cat), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]
    base, tables = _reason_tables(X, y, num, cat, feats)
    reasons = [_row_reasons(r, base, tables) or ["no single strong driver among the top inputs"] for _, r in X.iterrows()]
    actions = [["Contact this customer first and ask what is wrong."] if r >= 0.7 else
               ["Add to the next retention campaign."] if r >= 0.3 else
               ["Keep in the normal check-in cycle."] for r in risk]
    out = pd.DataFrame({"risk": risk})
    out["risk_band"] = pd.cut(risk, [-0.01, 0.3, 0.7, 1.01], labels=["Low", "Medium", "High"])
    out["reasons"], out["actions"] = reasons, actions
    out["months"] = pd.to_numeric(kept[tenure], errors="coerce").fillna(0).astype(int) if tenure and tenure in kept else 0
    out["value"] = pd.to_numeric(kept[value], errors="coerce").fillna(0.0) if value and value in kept else 0.0
    out["complaints"] = 0
    out["cust"] = kept[id_col].astype(str) if id_col and id_col in kept else [str(i + 1) for i in range(len(kept))]
    metrics["warnings"] = warnings
    return out, metrics
