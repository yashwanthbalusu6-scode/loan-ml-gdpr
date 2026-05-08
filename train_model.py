"""One-shot script to train + persist the loan-approval model.

Used both during local dev (`python train_model.py`) and at container startup
when no pre-trained artifacts are bundled.
"""
import sys

from data_loader import generate_synthetic_loan_data
from model import save_artifacts, train_model


def main(out_dir: str = "artifacts") -> dict:
    df = generate_synthetic_loan_data(n=1000, seed=42, approval_rate=0.7)
    result = train_model(df)
    save_artifacts(result.model, result.X_train, out_dir)
    print(f"Model trained. Metrics: {result.metrics}")
    print(f"Saved to {out_dir}/")
    return result.metrics


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    main(out)
