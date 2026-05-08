# Loan ML + GDPR Compliance

Production-grade bank-loan-approval ML system with **explainable** decisions (SHAP), a **GDPR-compliant** audit trail, and **right-to-be-forgotten** data deletion.

> **Demo URLs**: _added after deployment_

## What it does

1. **Predicts** loan approval with XGBoost from five features: `age`, `income`, `credit_score`, `loan_amount`, `employment_years`.
2. **Explains** every decision via SHAP — both per-prediction (force values) and globally (feature importance).
3. **Logs** every action (predict / view / delete) into an encrypted audit table; sensitive payloads are encrypted at rest with Fernet.
4. **Honors GDPR**: customers can request a copy of their data (`/gdpr/data-request`) or have it deleted (`/gdpr/delete`). Deletions are themselves logged so the compliance team can prove the request was processed.
5. **Surfaces drift & fairness**: `/model-metrics` returns demographic parity (under-50 vs 50+ approval rates), the 4/5ths-rule check, and a credit-score drift estimate.

## Tech stack

| Layer | Tool |
| --- | --- |
| Model | XGBoost 2.0 |
| Explainability | SHAP 0.42 (TreeExplainer) |
| Backend | FastAPI 0.104 + uvicorn |
| Frontend | Streamlit 1.28 |
| ORM | SQLAlchemy 2.0 |
| DB | SQLite (local) · PostgreSQL (prod via `DATABASE_URL`) |
| Encryption | Fernet (`cryptography`) |

## Run locally (no Docker required)

```bash
# Backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-backend.txt
python train_model.py            # creates artifacts/
uvicorn api:app --reload         # http://localhost:8000

# Frontend (in a second terminal)
source venv/bin/activate
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run dashboard.py
```

The default backend URL the frontend looks for is `http://localhost:8000`. SQLite at `loan_ml.db` is used unless `DATABASE_URL` is set.

## API examples

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":35,"income":75000,"credit_score":720,"loan_amount":25000,"employment_years":5}'

# GDPR data request
curl -X POST http://localhost:8000/gdpr/data-request \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"cust_abc12345"}'

# GDPR deletion (right to be forgotten)
curl -X POST http://localhost:8000/gdpr/delete \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"cust_abc12345","reason":"Customer request"}'

# Audit trail
curl 'http://localhost:8000/audit-log?customer_id=cust_abc12345'
```

## GDPR features

| Capability | Implementation |
| --- | --- |
| Audit trail | `audit_log` table — every action logged with timestamp, user, encrypted details |
| Right of access | `POST /gdpr/data-request` returns predictions + audit + deletions + requests for a customer |
| Right to be forgotten | `POST /gdpr/delete` marks predictions as deleted, scrubs `input_features`, records the deletion |
| Encryption at rest | Fernet-encrypted JSON payloads in `predictions.input_features` and `audit_log.details` |
| Data minimization | Only the 5 declared features are stored; no PII beyond a customer_id |
| Auditability | Every GDPR action is itself written to `audit_log` |

## Files

```
loan-ml-gdpr/
├── api.py                    # FastAPI app
├── dashboard.py              # Streamlit UI
├── data_loader.py            # Synthetic dataset generator
├── model.py                  # XGBoost training + SHAP
├── audit_logger.py           # SQLAlchemy schema + Fernet encryption
├── gdpr_handler.py           # Data access + deletion logic
├── train_model.py            # Standalone training script
├── requirements.txt          # Frontend deps
├── requirements-backend.txt  # Backend deps
└── .gitignore
```

## License

MIT
