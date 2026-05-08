"""Synthetic loan-application dataset generator.

We need realistic-looking data without touching real customers' financial records.
Score follows a logistic-style combination of features so the model has signal to learn,
plus Gaussian noise so it isn't trivially separable.
"""
from typing import Tuple

import numpy as np
import pandas as pd


FEATURE_NAMES = ["age", "income", "credit_score", "loan_amount", "employment_years"]


def generate_synthetic_loan_data(n: int = 1000, seed: int = 42, approval_rate: float = 0.7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 71, size=n)
    income = rng.integers(20_000, 200_001, size=n)
    credit_score = rng.integers(300, 851, size=n)
    loan_amount = rng.integers(5_000, 500_001, size=n)
    employment_years = np.minimum(rng.integers(0, 41, size=n), np.maximum(age - 18, 0))

    score = (
        (credit_score - 600) / 100.0
        + np.log1p(income / 50_000) * 1.5
        + np.log1p(employment_years) * 0.3
        - np.log1p(loan_amount / 100_000) * 1.2
        + rng.normal(0, 1.0, size=n)
    )
    threshold = float(np.quantile(score, 1 - approval_rate))
    approved = (score >= threshold).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income.astype(float),
        "credit_score": credit_score,
        "loan_amount": loan_amount.astype(float),
        "employment_years": employment_years,
        "approved": approved,
    })


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_NAMES].copy(), df["approved"].copy()
