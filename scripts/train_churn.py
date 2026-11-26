"""Train the churn model (HistGradientBoosting) on data/sample_customers.csv (public UCI #563 data)
and write results/churn_model.joblib. The model file is not stored in git; run this first."""
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from src.churn_prep import load_raw, engineer, FEATURES, TARGET
df = engineer(load_raw("data/sample_customers.csv"))
m = HistGradientBoostingClassifier(random_state=42).fit(df[FEATURES], df[TARGET].astype(int))
joblib.dump(m, "results/churn_model.joblib"); print("trained on", len(df), "rows")
