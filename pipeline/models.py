"""
Model registry, preprocessing pipeline, and per-fold train/evaluate helpers.

Models:
    xgboost       – XGBClassifier
    random_forest – RandomForestClassifier
    lasso_lr      – LogisticRegression(penalty='l1', solver='saga')
    ridge_lr      – LogisticRegression(penalty='l2', solver='lbfgs')
    logistic      – LogisticRegression(penalty=None)   (unregularised)

Preprocessing (fit on training fold only):
    numeric : SimpleImputer(0) → StandardScaler
    categorical : OneHotEncoder(handle_unknown='ignore')

Metrics reported per fold:
    log_loss, brier_score, accuracy
"""

import time
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from xgboost import XGBClassifier
from .config import RANDOM_STATE

logger = logging.getLogger(__name__)

# ─── Model registry ───────────────────────────────────────────────────────────
# Each entry: model_name → (sklearn estimator, param grid for RandomizedSearchCV)

MODEL_REGISTRY: dict[str, tuple] = {
    "lasso_lr": (
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            tol=1e-3,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        {
            # 25 values: very fine log-spacing 1e-5 → 10; previous best was 0.01
            "clf__C": [
                0.00001, 0.00002, 0.00005,
                0.0001,  0.0002,  0.0005,
                0.001,   0.002,   0.005,
                0.007,   0.01,    0.015,  0.02,  0.03,  0.05,
                0.07,    0.1,     0.15,   0.2,   0.3,   0.5,
                0.7,     1.0,     3.0,    10.0,
            ],
        },
    ),
    "ridge_lr": (
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            tol=1e-3,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        {
            # 25 values: fine log-spacing 1e-5 → 10; previous best was 0.001 (was at boundary)
            "clf__C": [
                0.000005, 0.00001, 0.00002, 0.00005,
                0.0001,   0.0002,  0.0005,
                0.001,    0.002,   0.005,
                0.007,    0.01,    0.015,  0.02,  0.03,  0.05,
                0.07,     0.1,     0.15,   0.2,   0.3,   0.5,
                0.7,      1.0,     3.0,
            ],
        },
    ),
    "elasticnet": (
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            tol=5e-3,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        {
            # 15 C × 5 l1_ratio = 75 combos — all exhausted; covers full mixing range
            "clf__C": [
                0.00005, 0.0001, 0.0005,
                0.001,   0.002,  0.005,
                0.01,    0.02,   0.05,
                0.1,     0.2,    0.5,
                1.0,     3.0,    10.0,
            ],
            "clf__l1_ratio": [0.2, 0.4, 0.5, 0.6, 0.8],
        },
    ),
    "logistic": (
        LogisticRegression(
            # Unregularised: very large C
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            tol=1e-3,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        {},   # no tuning; unregularised LR is a baseline
    ),
    "xgboost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            # Conservative defaults to prevent gross overfitting before tuning
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            min_child_weight=10,
        ),
        {
            "clf__n_estimators":     [200, 500, 750, 1000, 1500],
            "clf__max_depth":        [3, 4, 5, 6, 7],
            "clf__learning_rate":    [0.005, 0.01, 0.02, 0.05, 0.1],
            "clf__subsample":        [0.6, 0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
            "clf__reg_lambda":       [0.1, 0.3, 0.5, 1, 3, 5, 10],
            "clf__min_child_weight": [1, 5, 10, 20],
            "clf__gamma":            [0, 0.1, 0.3, 0.5, 1.0],
        },
    ),
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


# ─── Preprocessing ────────────────────────────────────────────────────────────

def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer.
    Must be fitted on training data only inside each fold.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        # Only include cat columns that are actually present
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_full_pipeline(model_name: str, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessor + model.
    The 'clf' step is the estimator (used as prefix in param grid keys).
    """
    estimator, _ = MODEL_REGISTRY[model_name]
    preprocessor = make_preprocessor(num_cols, cat_cols)
    return Pipeline([
        ("pre", preprocessor),
        ("clf", estimator),
    ])


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Compute log loss, Brier score, and accuracy.
    y_prob should be the probability of class 1 (win), shape (n,).
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y_true, y_prob),
        "brier":    brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
    }


BASELINE_METRICS = {
    "log_loss": 0.6931,   # log(2): predict 0.5 always
    "brier":    0.2500,
    "accuracy": 0.5000,
}


# ─── Per-fold training ────────────────────────────────────────────────────────

def train_eval_fold(
    model_name:  str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict | None = None,
) -> dict[str, float]:
    """
    Fit a single model on (X_tr, y_tr) and evaluate on (X_val, y_val).
    params: dict of estimator hyperparameters (without 'clf__' prefix).
    Returns metrics dict.
    """
    estimator, _ = MODEL_REGISTRY[model_name]

    # Clone and set params
    from sklearn.base import clone
    est = clone(estimator)
    if params:
        est.set_params(**params)

    est.fit(X_tr, y_tr)
    y_prob = est.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_prob)
    return metrics


def run_cv(
    dataset_df: pd.DataFrame,
    folds: list[tuple[list[int], list[int]]],
    num_cols: list[str],
    cat_cols: list[str],
    best_params: dict[str, dict] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run all models across all folds.  Returns a long-form DataFrame with columns:
        model, fold, log_loss, brier, accuracy

    best_params: {model_name: {param: value}} – hyperparams without 'clf__' prefix.
                 If None, uses model defaults.
    """
    if model_names is None:
        model_names = MODEL_NAMES
    if best_params is None:
        best_params = {m: {} for m in model_names}

    records = []
    n_folds = len(folds)
    t_start = time.time()
    print(f"Rolling CV — {n_folds} folds × {len(model_names)} models", flush=True)

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        tr  = dataset_df.loc[tr_idx]
        val = dataset_df.loc[val_idx]

        # Fit preprocessor on training fold only
        pre = make_preprocessor(num_cols, cat_cols)
        X_tr  = pre.fit_transform(tr[num_cols + cat_cols])
        X_val = pre.transform(val[num_cols + cat_cols])
        y_tr  = tr["result"].values
        y_val = val["result"].values

        fold_metrics = {}
        for mname in model_names:
            params = best_params.get(mname, {})
            try:
                metrics = train_eval_fold(mname, X_tr, y_tr, X_val, y_val, params)
                records.append({"model": mname, "fold": fold_i, **metrics})
                fold_metrics[mname] = metrics["log_loss"]
            except Exception as exc:
                logger.warning("Fold %d, model %s failed: %s", fold_i, mname, exc)

        elapsed = time.time() - t_start
        eta = elapsed / (fold_i + 1) * (n_folds - fold_i - 1)
        scores = "  ".join(
            f"{m.split('_')[0]}={v:.4f}" for m, v in fold_metrics.items()
        )
        print(
            f"  fold {fold_i+1:>3}/{n_folds}  [{scores}]  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
            flush=True,
        )
        logger.info("CV progress: fold %d / %d", fold_i + 1, n_folds)

    print(f"CV complete in {time.time() - t_start:.0f}s", flush=True)
    return pd.DataFrame(records)


def summarise_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fold-level CV results to mean ± std per model.
    """
    return (
        cv_df.groupby("model")[["log_loss", "brier", "accuracy"]]
        .agg(["mean", "std"])
        .round(5)
    )


# ─── Ensemble ────────────────────────────────────────────────────────────────

def compute_ensemble_weights(
    cv_scores: dict[str, float],
) -> dict[str, float]:
    """
    Compute inverse-log-loss weights for ensemble averaging.

    Parameters
    ----------
    cv_scores : {model_label: mean_cv_log_loss}

    Returns
    -------
    {model_label: weight}  (weights sum to 1)
    """
    inv = {m: 1.0 / ll for m, ll in cv_scores.items()}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


def ensemble_predict(
    model_probs: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """
    Weighted average of model probabilities.

    Parameters
    ----------
    model_probs : {model_label: p_model}
    weights     : {model_label: weight} — if None, uses equal weights

    Returns
    -------
    (p_ensemble, model_std) : weighted mean and std of model predictions
    """
    labels = list(model_probs.keys())
    probs  = np.array([model_probs[l] for l in labels])

    if weights is None:
        w = np.ones(len(probs)) / len(probs)
    else:
        w = np.array([weights.get(l, 1.0 / len(probs)) for l in labels])
        w = w / w.sum()

    p_ens = float(np.dot(w, probs))
    p_std = float(np.std(probs))  # unweighted std — measure of disagreement
    return p_ens, p_std
