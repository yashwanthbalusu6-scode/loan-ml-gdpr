"""FastAPI backend skeleton — endpoints land in subsequent commits."""
from fastapi import FastAPI


app = FastAPI(
    title="Loan ML + GDPR API",
    description="Bank loan approval with SHAP explainability and GDPR audit logging.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}
