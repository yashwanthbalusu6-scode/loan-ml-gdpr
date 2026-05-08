"""XGBoost model + SHAP explainer.

SHAP values are returned in log-odds space (the default for tree explainers).
Positive value pushes toward APPROVED; negative pushes toward DENIED.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data_loader import FEATURE_NAMES, split_features_target


@dataclass
class TrainResult:
    model: xgb.XGBClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metrics: Dict[str, float]


def train_model(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> TrainResult:
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(X_tr, y_tr)
    metrics = {
        "train_accuracy": float(accuracy_score(y_tr, model.predict(X_tr))),
        "test_accuracy": float(accuracy_score(y_te, model.predict(X_te))),
        "test_auc": float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])),
    }
    return TrainResult(model, X_tr, X_te, y_tr, y_te, metrics)


def make_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def explain_prediction(explainer: shap.TreeExplainer, X_row: pd.DataFrame) -> Dict:
    sv = explainer(X_row)
    values = np.asarray(sv.values)
    if values.ndim == 3:
        values = values[..., 1]
    base = np.atleast_1d(np.asarray(sv.base_values).flatten())[0]
    return {
        "feature_names": list(X_row.columns),
        "shap_values": [float(v) for v in values[0]],
        "base_value": float(base),
    }


def plain_english(shap_dict: Dict, top_k: int = 3) -> str:
    pairs: List[Tuple[str, float]] = list(zip(shap_dict["feature_names"], shap_dict["shap_values"]))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    fragments = []
    for name, val in pairs[:top_k]:
        direction = "raised" if val > 0 else "lowered"
        fragments.append(f"{name.replace('_', ' ')} {direction} approval likelihood by {abs(val):.2f}")
    return "; ".join(fragments) + "."


def save_artifacts(model: xgb.XGBClassifier, X_train: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    X_train.to_parquet(os.path.join(out_dir, "X_background.parquet"))


def load_artifacts(out_dir: str) -> Tuple[xgb.XGBClassifier, pd.DataFrame]:
    model = joblib.load(os.path.join(out_dir, "model.pkl"))
    X_bg = pd.read_parquet(os.path.join(out_dir, "X_background.parquet"))
    return model, X_bg
