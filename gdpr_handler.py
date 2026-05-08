"""GDPR primitives: data access, right-to-be-forgotten, request tracking.

Each operation writes to the audit log so the compliance team can prove
the request was processed.
"""
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from audit_logger import (
    AuditLog,
    Deletion,
    GdprRequest,
    Prediction,
    decrypt_payload,
    log_audit,
)


def get_customer_data(session: Session, customer_id: str) -> dict:
    preds = session.scalars(
        select(Prediction).where(
            Prediction.customer_id == customer_id, Prediction.deleted == False  # noqa: E712
        )
    ).all()
    audits = session.scalars(
        select(AuditLog).where(AuditLog.customer_id == customer_id)
    ).all()
    deletions = session.scalars(
        select(Deletion).where(Deletion.customer_id == customer_id)
    ).all()
    requests = session.scalars(
        select(GdprRequest).where(GdprRequest.customer_id == customer_id)
    ).all()

    log_audit(
        session,
        action="GDPR_ACCESS_REQUEST",
        customer_id=customer_id,
        details={"records_returned": len(preds), "audit_entries": len(audits)},
    )

    return {
        "customer_id": customer_id,
        "predictions": [
            {
                "id": p.id,
                "prediction": p.prediction,
                "probability": p.probability,
                "timestamp": p.timestamp.isoformat(),
                "input_features": decrypt_payload(p.input_features),
            }
            for p in preds
        ],
        "audit_log": [
            {
                "id": a.id,
                "action": a.action,
                "user_id": a.user_id,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in audits
        ],
        "deletions": [
            {
                "id": d.id,
                "deleted_at": d.deleted_at.isoformat(),
                "reason": d.reason,
                "who_deleted": d.who_deleted,
            }
            for d in deletions
        ],
        "requests": [
            {
                "id": r.id,
                "type": r.request_type,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in requests
        ],
    }


def delete_customer_data(
    session: Session,
    customer_id: str,
    who: str = "system",
    reason: str = "GDPR right to be forgotten",
) -> dict:
    result = session.execute(
        update(Prediction)
        .where(Prediction.customer_id == customer_id, Prediction.deleted == False)  # noqa: E712
        .values(deleted=True, input_features="<DELETED>", shap_values=None)
    )
    n_purged = result.rowcount or 0

    deletion = Deletion(customer_id=customer_id, reason=reason, who_deleted=who)
    session.add(deletion)
    session.flush()

    log_audit(
        session,
        action="GDPR_DELETION",
        customer_id=customer_id,
        user_id=who,
        details={"reason": reason, "records_purged": int(n_purged)},
    )
    session.commit()
    return {
        "customer_id": customer_id,
        "records_purged": int(n_purged),
        "deleted_at": deletion.deleted_at.isoformat(),
    }


def create_request(session: Session, customer_id: str, request_type: str) -> dict:
    req = GdprRequest(customer_id=customer_id, request_type=request_type)
    session.add(req)
    session.flush()
    log_audit(
        session,
        action="GDPR_REQUEST_CREATED",
        customer_id=customer_id,
        details={"type": request_type, "request_id": req.id},
    )
    session.commit()
    return {
        "id": req.id,
        "customer_id": customer_id,
        "type": request_type,
        "status": req.status,
        "created_at": req.created_at.isoformat(),
    }
