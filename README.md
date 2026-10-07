# Credit Card Fraud Detection — End-to-End ML Workflow

A fraud-scoring model built on a real, anonymized dataset of European card transactions. Framed as a realistic banking problem: extreme class imbalance (0.17% fraud) and asymmetric error costs.

## Highlights

- **Correct evaluation for imbalanced data** — PR-AUC and precision/recall, not misleading accuracy.
- **Three models compared** — interpretable logistic-regression baseline vs. XGBoost (cost-sensitive) vs. XGBoost + SMOTE.
- **No data leakage** — duplicates dropped before splitting; stratified 3-way train / validation / test split; scaler and SMOTE fit on training data only.
- **Honest threshold tuning** — decision threshold picked on the *validation* set (not the test set) and then evaluated on untouched test data.
- **Model persisted** — artifacts (`.joblib` + `model_metadata.json`) saved and ready for a serving layer.

## Results (held-out test set)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression (balanced weights, threshold 0.5) | 0.964 | 0.681 | 0.05 | 0.87 | 0.10 |
| XGBoost (cost-sensitive, threshold 0.5) | 0.964 | **0.823** | 0.93 | 0.80 | 0.86 |
| XGBoost + SMOTE (threshold 0.5) | 0.966 | 0.803 | 0.55 | 0.80 | 0.66 |
| **XGBoost (threshold tuned on validation set)** | 0.964 | **0.823** | **0.99** | 0.73 | **0.84** |

Read the logistic-regression row honestly: with `class_weight='balanced'` and a naive 0.5 cutoff it catches almost every fraud but drowns the review team in false alarms — precisely why threshold tuning matters.

At the val-tuned threshold the XGBoost model **catches 69 of 95 frauds with only 1 false alarm across ~56,700 test transactions.** In fraud-review terms, that's near-zero wasted analyst time at the cost of missing ~27% of fraud — a tradeoff a real fraud team would choose to renegotiate along the precision/recall curve.

## Dataset

*Credit Card Fraud Detection* — 284,807 transactions, 492 fraud. Features `V1`–`V28` are PCA components released by the data provider to protect confidentiality; only `Time` and `Amount` are untransformed. Originally from the ULB Machine Learning Group; widely distributed on Kaggle.

**Note:** `creditcard.csv` (~150 MB) is not committed to the repo (it exceeds GitHub's 100 MB file-size limit and is listed in `.gitignore`). Download it from Kaggle ("Credit Card Fraud Detection") and place it beside the notebook before running.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

The notebook executes end-to-end in a few minutes on a laptop CPU and produces the three saved artifacts: `fraud_xgb_model.joblib`, `amount_hour_scaler.joblib`, `model_metadata.json`.

## Roadmap — from notebook to production

This notebook is **phase 1**. Planned extensions that map to production ML Engineer requirements:

1. Refactor into `train.py` / `predict.py` modules
2. Serve behind a **FastAPI** microservice, containerized with **Docker**
3. Track experiments and versions with **MLflow**
4. Schedule retraining with **Airflow** + data-drift monitoring
5. **CI/CD** via GitHub Actions
6. Deploy to **AWS / GCP**

## Tech stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · matplotlib · seaborn
