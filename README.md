# Retention Copilot

A small web product that answers one business question: **which customers are about to leave, why, and what should we do?**

User journey: landing page -> create account / sign in -> upload a customer CSV (or use the 3,150-customer sample) -> dashboard (risk levels, risk by tenure, main reasons, contact list with suggested actions) -> Arabic assistant -> downloadable Word report.

## What is real
- ML model: gradient boosting trained on the public Iranian Churn dataset (UCI #563, 3,150 customers). Held-out ROC-AUC 0.992, accuracy 96.3% (see the churn-insight-platform repo for the benchmark).
- Accounts and run history in a real database (SQLite locally, Postgres with docker compose), with SQL migrations. Passwords are salted PBKDF2 hashes.
- DOCX report generated from your results.

## AI modules (all run in this repo)
| Module | Domain | Model / data | Honest result |
|---|---|---|---|
| Churn risk | ML | Gradient boosting, Iranian Churn (UCI 563, 3,150 customers) | 96.3% acc, AUC 0.9924 (held-out) |
| Churn risk (DL) | Deep learning | PyTorch MLP, same data, threshold chosen on a validation slice | 97.0% acc, AUC 0.9872 on 630 unseen customers (`scripts/train_dl.py`) |
| Comment reader | Arabic NLP | CAMeLBERT-MSA sentiment (public model, used as-is) + keyword topics | 73.7% on 1,000 unseen Arabic restaurant reviews (it also answers "neutral", which counts as wrong); 82.7% when it picks a side. A TF-IDF logistic regression trained on the same reviews gets 86.5% but only knows restaurants (`scripts/eval_nlp.py`) |
| Branch site check | Computer vision | ResNet18 fine-tuned on EuroSAT (from sat-vision-service) | 94.0% on 2,161 unseen tiles (`scripts/eval_vision.py`) |
| Win-back agents | Multi-agent + GenAI | KMeans Segmenter, rule Strategist, Qwen2.5-0.5B-Instruct Copywriter, rule Critic, Planner | The 0.5B model writes weak Arabic. The Critic rejected all 3 of its drafts in the sample run and the safe template was used. Use a larger model with `LLM_MODEL=...` for better drafts |
| Analytics | Data analytics | Segment tables from the scored customers | computed live |

Notes: the EuroSAT "test" parquet overlaps the pooled training data, so it is not used for evaluation. Memory: running every model in one process needs about 2 GB RAM. The campaign step takes about 1.5 minutes on a small CPU box.

## What is rule-based (not AI-generated)
- The plain-language reasons and actions are fixed rules over the model's known top drivers, not per-customer SHAP.
- The Arabic assistant matches question intents and fills templated Arabic answers with your numbers. No LLM is called.

## Run
    pip install -r requirements.txt
    python -m pytest -q
    uvicorn src.api:app --port 8000        # open http://localhost:8000
    docker compose up --build              # app + Postgres

## Deploy
The Dockerfile reads $PORT and defaults to 7860, so it deploys as-is to Hugging Face Spaces (Docker SDK) or Render.

Design: logo palette; Cairo (free) for headings. The Riyad Bank typeface is proprietary, so Cairo Black stands in for bold wide titles.

Built with AI pair-programming; numbers come from real runs.

## Files not in this repo
The Arabic reviews parquet is not committed. Run `python scripts/fetch_models.py` to download it and the models (public Hugging Face sources).

