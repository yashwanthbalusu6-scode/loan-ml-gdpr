"""SQLAlchemy schema + audit logging primitives.

Sensitive payloads (raw input features, audit details) are encrypted at rest using Fernet.
DATABASE_URL env var selects backend (default: sqlite:///loan_ml.db).
"""
import json
import os
from datetime import datetime
from typing import Optional

from cryptography.fernet import Fernet
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///loan_ml.db")

engine_kwargs = {"future": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def _load_or_create_key() -> bytes:
    env_key = os.getenv("AUDIT_ENCRYPTION_KEY")
    if env_key:
        return env_key.encode() if isinstance(env_key, str) else env_key
    key_path = os.getenv("AUDIT_KEY_FILE", "/tmp/loan_ml_audit.key")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read().strip()
    new_key = Fernet.generate_key()
    try:
        with open(key_path, "wb") as f:
            f.write(new_key)
    except OSError:
        pass
    return new_key


_fernet = Fernet(_load_or_create_key())


def encrypt_payload(payload: dict) -> str:
    return _fernet.encrypt(json.dumps(payload, default=str).encode()).decode()


def decrypt_payload(token: str) -> dict:
    if not token or token == "<DELETED>":
        return {"_status": "deleted"}
    return json.loads(_fernet.decrypt(token.encode()).decode())


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    customer_id = Column(String, index=True, nullable=False)
    input_features = Column(Text, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    shap_values = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    deleted = Column(Boolean, default=False, index=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    customer_id = Column(String, index=True, nullable=True)
    action = Column(String, nullable=False)
    user_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class Deletion(Base):
    __tablename__ = "deletions"

    id = Column(Integer, primary_key=True)
    customer_id = Column(String, index=True, nullable=False)
    deleted_at = Column(DateTime, default=datetime.utcnow)
    reason = Column(String, nullable=True)
    who_deleted = Column(String, nullable=True)


class GdprRequest(Base):
    __tablename__ = "gdpr_requests"

    id = Column(Integer, primary_key=True)
    customer_id = Column(String, index=True, nullable=False)
    request_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def log_audit(
    session,
    action: str,
    customer_id: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[dict] = None,
) -> AuditLog:
    entry = AuditLog(
        customer_id=customer_id,
        action=action,
        user_id=user_id,
        details=encrypt_payload(details) if details else None,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry
