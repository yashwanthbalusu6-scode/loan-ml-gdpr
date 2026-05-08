"""FastAPI backend for the loan-approval ML system."""
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from audit_logger import (
    AuditLog,
    Deletion,
    Prediction,
    SessionLocal,
    decrypt_payload,
    encrypt_payload,
    init_db,
    log_audit,
)
from data_loader import FEATURE_NAMES, generate_synthetic_loan_data
from gdpr_handler import create_request, delete_customer_data, get_customer_data
from model import (
    explain_prediction,
    load_artifacts,
    make_explainer,
    plain_english,
    save_artifacts,
    train_model,
)


ARTIFACTS_DIR = os.getenv("MODEL_DIR", "artifacts")
APP_STATE: dict = {}


def get_session():
    sess = SessionLocal()
    try:
        yield sess
    finally:
        sess.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        model, X_bg = load_artifacts(ARTIFACTS_DIR)
    except Exception:
        df = generate_synthetic_loan_data(n=1000, seed=42, approval_rate=0.7)
        result = train_model(df)
        save_artifacts(result.model, result.X_train, ARTIFACTS_DIR)
        model, X_bg = result.model, result.X_train
    APP_STATE["model"] = model
    APP_STATE["X_bg"] = X_bg
    APP_STATE["explainer"] = make_explainer(model)
    yield
    APP_STATE.clear()


app = FastAPI(
    title="Loan ML + GDPR API",
    description="Bank loan approval with SHAP explainability and GDPR audit logging.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., ge=0)
    employment_years: int = Field(..., ge=0, le=70)
    customer_id: Optional[str] = None
    user_id: Optional[str] = None


class ShapDetail(BaseModel):
    feature: str
    value: float
    shap: float


class PredictResponse(BaseModel):
    customer_id: str
    prediction: int
    probability: float
    decision_text: str
    shap_values: List[ShapDetail]
    base_value: float
    explanation_text: str
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in APP_STATE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, sess=Depends(get_session)):
    if "model" not in APP_STATE:
        raise HTTPException(503, "Model not ready")
    customer_id = req.customer_id or f"cust_{uuid.uuid4().hex[:8]}"
    X_row = pd.DataFrame(
        [[req.age, req.income, req.credit_score, req.loan_amount, req.employment_years]],
        columns=FEATURE_NAMES,
    )
    proba = float(APP_STATE["model"].predict_proba(X_row)[0, 1])
    pred = int(proba >= 0.5)
    explanation = explain_prediction(APP_STATE["explainer"], X_row)
    shap_details = [
        ShapDetail(feature=f, value=float(X_row.iloc[0][f]), shap=float(s))
        for f, s in zip(explanation["feature_names"], explanation["shap_values"])
    ]
    decision_text = "APPROVED" if pred == 1 else "DENIED"
    explanation_text = plain_english(explanation, top_k=3)

    pred_row = Prediction(
        customer_id=customer_id,
        input_features=encrypt_payload(req.dict()),
        prediction=pred,
        probability=proba,
        shap_values={
            "feature_names": explanation["feature_names"],
            "shap_values": explanation["shap_values"],
            "base_value": explanation["base_value"],
        },
    )
    sess.add(pred_row)
    sess.flush()
    log_audit(
        sess,
        action="PREDICT",
        customer_id=customer_id,
        user_id=req.user_id,
        details={"prediction_id": pred_row.id, "prediction": pred, "probability": proba},
    )
    sess.commit()

    return PredictResponse(
        customer_id=customer_id,
        prediction=pred,
        probability=proba,
        decision_text=decision_text,
        shap_values=shap_details,
        base_value=explanation["base_value"],
        explanation_text=explanation_text,
        timestamp=pred_row.timestamp.isoformat(),
    )


class ExplainRequest(BaseModel):
    customer_id: str


@app.post("/explain")
def explain(req: ExplainRequest, sess=Depends(get_session)):
    p = sess.scalars(
        select(Prediction)
        .where(Prediction.customer_id == req.customer_id, Prediction.deleted == False)  # noqa: E712
        .order_by(Prediction.timestamp.desc())
    ).first()
    if p is None:
        raise HTTPException(404, "No prediction found for that customer")
    log_audit(sess, action="EXPLAIN_VIEWED", customer_id=req.customer_id,
              details={"prediction_id": p.id})
    return {
        "customer_id": req.customer_id,
        "prediction": p.prediction,
        "probability": p.probability,
        "shap": p.shap_values,
        "timestamp": p.timestamp.isoformat(),
    }


class GdprDeleteRequest(BaseModel):
    customer_id: str
    reason: Optional[str] = "GDPR right to be forgotten"
    requested_by: Optional[str] = "user"


@app.post("/gdpr/delete")
def gdpr_delete(req: GdprDeleteRequest, sess=Depends(get_session)):
    return delete_customer_data(sess, req.customer_id, who=req.requested_by, reason=req.reason)


class GdprDataRequest(BaseModel):
    customer_id: str


@app.post("/gdpr/data-request")
def gdpr_data_request(req: GdprDataRequest, sess=Depends(get_session)):
    return get_customer_data(sess, req.customer_id)


class GdprRequestCreate(BaseModel):
    customer_id: str
    request_type: Literal["access", "deletion"]


@app.post("/gdpr/request")
def gdpr_request_create(req: GdprRequestCreate, sess=Depends(get_session)):
    return create_request(sess, req.customer_id, req.request_type)


@app.get("/audit-log")
def audit_log_list(
    customer_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
    sess=Depends(get_session),
):
    q = select(AuditLog).order_by(AuditLog.timestamp.desc())
    if customer_id:
        q = q.where(AuditLog.customer_id == customer_id)
    if start:
        q = q.where(AuditLog.timestamp >= datetime.fromisoformat(start))
    if end:
        q = q.where(AuditLog.timestamp <= datetime.fromisoformat(end))
    rows = sess.scalars(q.limit(limit)).all()
    return [
        {
            "id": r.id,
            "customer_id": r.customer_id,
            "action": r.action,
            "user_id": r.user_id,
            "timestamp": r.timestamp.isoformat(),
            "details": decrypt_payload(r.details) if r.details else None,
        }
        for r in rows
    ]


@app.get("/model-metrics")
def model_metrics(sess=Depends(get_session)):
    if "model" not in APP_STATE:
        raise HTTPException(503, "Model not ready")
    df = generate_synthetic_loan_data(n=200, seed=99, approval_rate=0.7)
    X = df[FEATURE_NAMES]
    y_true = df["approved"].values
    y_pred = APP_STATE["model"].predict(X)
    accuracy = float((y_pred == y_true).mean())

    young = df[df["age"] < 50]
    older = df[df["age"] >= 50]
    rate_y = float((APP_STATE["model"].predict(young[FEATURE_NAMES]) == 1).mean()) if len(young) else 0.0
    rate_o = float((APP_STATE["model"].predict(older[FEATURE_NAMES]) == 1).mean()) if len(older) else 0.0
    parity_diff = abs(rate_y - rate_o)

    drift_score = float(
        abs(APP_STATE["X_bg"]["credit_score"].mean() - X["credit_score"].mean())
    ) / 100.0

    n_pred = sess.scalar(select(func.count(Prediction.id)))
    n_audit = sess.scalar(select(func.count(AuditLog.id)))
    n_deletions = sess.scalar(select(func.count(Deletion.id)))

    return {
        "accuracy": accuracy,
        "fairness": {
            "approval_rate_under_50": rate_y,
            "approval_rate_50_plus": rate_o,
            "demographic_parity_diff": parity_diff,
            "passes_4_5ths_rule": parity_diff < 0.2,
        },
        "drift": {
            "credit_score_drift": drift_score,
            "needs_retraining": drift_score > 0.5,
        },
        "audit_counts": {
            "predictions": int(n_pred or 0),
            "audit_log": int(n_audit or 0),
            "deletions": int(n_deletions or 0),
        },
    }
