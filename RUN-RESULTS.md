# Run results (real)
- Tests: 5/5 pass (full user journey: sign-up, sample analysis, filtered results, Arabic answer, DOCX report; 401 without login; bad login).
- Sample run: 3,150 real customers scored in about 2 seconds on CPU: 453 high risk (14%), 73 medium.
- Model quality (held-out, from churn-insight-platform): ROC-AUC 0.9924, accuracy 96.3%.
- DL churn MLP 97.0% acc / AUC 0.9872 (630 unseen customers); Arabic sentiment 73.7% (82.7% when not neutral) on 1,000 unseen reviews; vision 94.0% on 2,161 unseen tiles; campaign agents ran 91s on CPU with all LLM drafts rejected by the Critic.
- Not deployed to a public host yet (needs a hosting account). Runs locally and via docker compose.
