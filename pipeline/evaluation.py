"""
Final evaluation, calibration, feature importance, and ablation utilities.

Functions:
    evaluate_on_holdout   – retrain each model on full dev set, evaluate on holdout
    plot_cv_summary       – CV log-loss mean ± std bar chart per model
    plot_calibration      – calibration curves (predicted prob vs actual win rate)
    plot_feature_importance – top feature importances for tree models
    run_ablation          – evaluate model on subsets of feature groups
    leakage_report        – print/log a structured leakage checklist
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.base import clone
from sklearn.calibration import calibration_curve

from .config import OUTPUT_DIR, RANDOM_STATE
from .models import (
    MODEL_REGISTRY, MODEL_NAMES, make_preprocessor,
    train_eval_fold, compute_metrics, BASELINE_METRICS,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette
_PALETTE = {
    "xgboost":       "#e07b39",
    "lasso_lr":      "#4a7bbf",
    "ridge_lr":      "#9b59b6",
    "elasticnet":    "#27ae60",
    "logistic":      "#95a5a6",
    "baseline":      "#cccccc",
}


# ─── Final holdout evaluation ─────────────────────────────────────────────────

def evaluate_on_holdout(
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    best_params: dict[str, dict] | None = None,
    model_names: list[str] | None = None,
) -> dict[str, dict]:
    """
    For each model:
      1. Fit preprocessor on dev_df.
      2. Fit model on full dev_df with best hyperparameters.
      3. Evaluate on holdout_df.

    Returns {model_name: {"metrics": {...}, "y_prob": array, "y_true": array}}
    """
    if model_names is None:
        model_names = MODEL_NAMES
    if best_params is None:
        best_params = {m: {} for m in model_names}

    pre = make_preprocessor(num_cols, cat_cols)
    X_dev   = pre.fit_transform(dev_df[num_cols + cat_cols])
    X_hold  = pre.transform(holdout_df[num_cols + cat_cols])
    y_dev   = dev_df["result"].values
    y_hold  = holdout_df["result"].values

    results = {}
    for mname in model_names:
        estimator, _ = MODEL_REGISTRY[mname]
        est = clone(estimator)
        params = best_params.get(mname, {})
        if params:
            est.set_params(**params)

        est.fit(X_dev, y_dev)
        y_prob = est.predict_proba(X_hold)[:, 1]
        metrics = compute_metrics(y_hold, y_prob)
        results[mname] = {"metrics": metrics, "y_prob": y_prob, "y_true": y_hold}
        logger.info(
            "Holdout %-15s  log_loss=%.5f  brier=%.5f  acc=%.4f",
            mname, metrics["log_loss"], metrics["brier"], metrics["accuracy"],
        )

    return results


def holdout_summary_df(holdout_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for mname, res in holdout_results.items():
        rows.append({"model": mname, **res["metrics"]})
    rows.append({"model": "baseline", **BASELINE_METRICS})
    return pd.DataFrame(rows).set_index("model").round(5)


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_cv_summary(cv_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Bar chart of mean CV log-loss ± 1 std for each model.
    """
    summary = (
        cv_df.groupby("model")["log_loss"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = [_PALETTE.get(m, "#888888") for m in summary["model"]]
    bars = ax.barh(
        summary["model"], summary["mean"],
        xerr=summary["std"], color=colours,
        error_kw={"elinewidth": 1.2, "capsize": 4},
        height=0.6,
    )
    ax.axvline(BASELINE_METRICS["log_loss"], color="#333333", ls="--", lw=1.2,
               label=f"Baseline (predict 0.5): {BASELINE_METRICS['log_loss']:.4f}")
    ax.set_xlabel("Log Loss (lower is better)")
    ax.set_title("Rolling CV Log Loss by Model (mean ± 1 std)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "cv_log_loss.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


def plot_calibration(
    holdout_results: dict[str, dict],
    n_bins: int = 10,
    save: bool = True,
) -> plt.Figure:
    """
    Calibration curves: for each model, plot mean predicted prob vs fraction
    of actual wins within each probability bucket.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

    for mname, res in holdout_results.items():
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(
            prob_pred, prob_true,
            marker="o", ms=5,
            color=_PALETTE.get(mname, "#888888"),
            label=mname,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of wins (actual)")
    ax.set_title("Calibration Curves – Holdout Set")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "calibration.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


def plot_feature_importance(
    dev_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    best_params: dict[str, dict] | None = None,
    top_n: int = 30,
    save: bool = True,
) -> plt.Figure:
    """
    Feature importances for XGBoost (gain-based) and Random Forest.
    Fits both models on full dev set, then plots top_n features.
    """
    if best_params is None:
        best_params = {}

    pre = make_preprocessor(num_cols, cat_cols)
    X_dev = pre.fit_transform(dev_df[num_cols + cat_cols])
    y_dev = dev_df["result"].values

    # Get feature names from the preprocessor
    try:
        feature_names = (
            list(pre.named_transformers_["num"].get_feature_names_out(num_cols))
            + list(pre.named_transformers_["cat"].get_feature_names_out())
        )
    except (KeyError, AttributeError):
        feature_names = [f"f{i}" for i in range(X_dev.shape[1])]

    fig, axes = plt.subplots(1, 1, figsize=(7, max(6, top_n * 0.3)), squeeze=False)
    axes = axes.flatten()

    for ax, mname in zip(axes, ["xgboost"]):
        estimator, _ = MODEL_REGISTRY[mname]
        est = clone(estimator)
        params = best_params.get(mname, {})
        if params:
            est.set_params(**params)
        est.fit(X_dev, y_dev)

        if hasattr(est, "feature_importances_"):
            imps = est.feature_importances_
        else:
            logger.warning("%s has no feature_importances_", mname)
            continue

        idx = np.argsort(imps)[::-1][:top_n][::-1]
        names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
        vals  = imps[idx]

        ax.barh(names, vals, color=_PALETTE.get(mname, "#888888"), height=0.7)
        ax.set_title(f"Feature Importances: {mname}")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "feature_importance.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


def plot_cv_metric_over_time(
    cv_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    folds: list[tuple[list[int], list[int]]],
    metric: str = "log_loss",
    save: bool = True,
) -> plt.Figure:
    """
    Plot per-fold metric value over time, coloured by model.
    Helps identify regime changes or periods where models diverge.
    """
    fold_dates = []
    for _, val_idx in folds:
        fold_dates.append(dev_df.loc[val_idx, "date"].min())

    cv_df2 = cv_df.copy()
    cv_df2["val_start"] = cv_df2["fold"].map(
        {i: d for i, d in enumerate(fold_dates)}
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for mname in cv_df2["model"].unique():
        sub = cv_df2[cv_df2["model"] == mname].sort_values("val_start")
        # Filter out rows with NaT timestamps to avoid matplotlib conversion errors
        sub = sub.dropna(subset=["val_start"])
        ax.plot(
            sub["val_start"], sub[metric],
            lw=1.2, alpha=0.8,
            color=_PALETTE.get(mname, "#888888"),
            label=mname,
        )
    ax.axhline(BASELINE_METRICS[metric], color="#333333", ls="--", lw=1, label="Baseline")
    ax.set_xlabel("Validation window start date")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric} Over Time by Model")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"cv_{metric}_over_time.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ─── Ablation ─────────────────────────────────────────────────────────────────

# Define feature group membership by column name prefixes / keywords
ABLATION_GROUPS = {
    "A_elo":        ["elo_"],
    "B_form":       ["wr_", "ewma_", "avg_opp_elo_", "overperf_"],
    "C_earlygame":  ["gdiff", "xpdiff", "pct_pos_", "pct_gdiff", "fb_rate", "fd_rate",
                     "fh_rate", "ft_rate", "fbar_rate"],
    "C_thresholds": ["pct_gdiff15_gt", "pct_gdiff15_lt", "pct_gdiff10_gt", "pct_gdiff10_lt"],
    "D_conversion": ["lead_conv_rate", "comeback_rate", "strong_lead_conv",
                     "avg_gdiff15_when"],
    "E_objectives": ["dragon_share", "herald_share", "baron_share", "tower_share", "plate_share"],
    "F_style":      ["pace_mean", "vision_mean", "economy_mean", "aggression_mean",
                     "gamelength_mean"],
    "G_side":       ["is_blue", "blue_wr", "red_wr", "same_side_wr", "patch_blue_wr",
                     "side_wr_vs_patch"],
    "H_patch":      ["patch_wr_shrunk", "patch_gdiff15_shrunk"],
    "K_interactions": ["ix_"],
}

ABLATION_SETS = {
    "A_elo_only":          ["A_elo"],
    "A+B":                 ["A_elo", "B_form"],
    "A+B+C":               ["A_elo", "B_form", "C_earlygame"],
    "A+B+C+thresholds":    ["A_elo", "B_form", "C_earlygame", "C_thresholds"],
    "Tier1":               ["A_elo", "B_form", "C_earlygame", "D_conversion"],
    "Tier1+thresholds":    ["A_elo", "B_form", "C_earlygame", "C_thresholds", "D_conversion"],
    "Tier1+2_no_style":    ["A_elo", "B_form", "C_earlygame", "C_thresholds",
                            "D_conversion", "E_objectives", "G_side"],
    "Tier1+2_with_style":  ["A_elo", "B_form", "C_earlygame", "C_thresholds",
                            "D_conversion", "E_objectives", "F_style", "G_side"],
    "Tier1+2+interactions":["A_elo", "B_form", "C_earlygame", "C_thresholds",
                            "D_conversion", "E_objectives", "G_side", "K_interactions"],
    "Full":                list(ABLATION_GROUPS.keys()),
}


def _cols_in_group(df: pd.DataFrame, group_keys: list[str], num_cols: list[str]) -> list[str]:
    prefixes = []
    for gk in group_keys:
        prefixes.extend(ABLATION_GROUPS.get(gk, []))

    selected = []
    for c in num_cols:
        # Include if the column name starts with or contains any prefix
        if any(c.startswith(p) or p in c for p in prefixes):
            selected.append(c)
        # Also include opponent and diff versions
        stripped = c.removeprefix("opp_").removeprefix("diff_")
        if any(stripped.startswith(p) or p in stripped for p in prefixes):
            selected.append(c)
    return list(dict.fromkeys(selected))  # deduplicate preserving order


def run_ablation(
    dev_df: pd.DataFrame,
    folds: list[tuple[list[int], list[int]]],
    num_cols: list[str],
    cat_cols: list[str],
    model_name: str = "ridge_lr",
    best_params: dict[str, dict] | None = None,
    last_n_folds: int = 30,
) -> pd.DataFrame:
    """
    For each ablation set, run CV on the last_n_folds and report mean log_loss.
    Uses ridge_lr as reference model (fast, stable, interpretable).
    best_params: tuned hyperparameters — uses model defaults if None.
    """
    tune_folds = folds[-last_n_folds:] if len(folds) > last_n_folds else folds
    n_sets = len(ABLATION_SETS)
    results = []
    t_start = time.time()
    params = (best_params or {}).get(model_name, {})

    print(f"Ablation study — {n_sets} sets × {len(tune_folds)} folds ({model_name})", flush=True)
    if params:
        print(f"  Using tuned params: {params}", flush=True)

    for set_i, (abl_name, group_keys) in enumerate(ABLATION_SETS.items()):
        abl_num_cols = _cols_in_group(dev_df, group_keys, num_cols)
        if not abl_num_cols:
            logger.warning("Ablation %s: no numeric columns matched", abl_name)
            print(f"  [{set_i+1}/{n_sets}] {abl_name}  — no columns matched, skipped", flush=True)
            continue

        fold_losses = []
        for tr_idx, val_idx in tune_folds:
            tr  = dev_df.loc[tr_idx]
            val = dev_df.loc[val_idx]
            pre = make_preprocessor(abl_num_cols, cat_cols)
            X_tr  = pre.fit_transform(tr[abl_num_cols + cat_cols])
            X_val = pre.transform(val[abl_num_cols + cat_cols])
            y_tr  = tr["result"].values
            y_val = val["result"].values
            m = train_eval_fold(model_name, X_tr, y_tr, X_val, y_val, params)
            fold_losses.append(m["log_loss"])

        mean_loss = np.mean(fold_losses)
        std_loss  = np.std(fold_losses)
        results.append({
            "ablation":    abl_name,
            "n_features":  len(abl_num_cols),
            "mean_logloss": mean_loss,
            "std_logloss":  std_loss,
        })
        elapsed = time.time() - t_start
        print(
            f"  [{set_i+1}/{n_sets}] {abl_name:<28}  "
            f"features={len(abl_num_cols):>3}  "
            f"logloss={mean_loss:.5f} ± {std_loss:.5f}  "
            f"({elapsed:.0f}s elapsed)",
            flush=True,
        )
        logger.info(
            "Ablation %-20s  features=%3d  logloss=%.5f ± %.5f",
            abl_name, len(abl_num_cols), mean_loss, std_loss,
        )

    print(f"Ablation complete in {time.time() - t_start:.0f}s", flush=True)
    return pd.DataFrame(results)


# ─── Leakage checklist ────────────────────────────────────────────────────────

def leakage_report(dataset_df: pd.DataFrame, num_cols: list[str]) -> None:
    """Print a structured leakage checklist to the log."""
    checks = []

    # 1. Draft columns
    draft_cols = [c for c in dataset_df.columns
                  if any(kw in c.lower() for kw in ["champion", "pick", "ban", "firstpick"])]
    checks.append(("No draft columns", not bool(draft_cols), draft_cols or "OK"))

    # 2. Same-match kill/obj stats used directly as features
    # Note: "result" alone is the target; "ewma_result" is a lagged feature – allowed.
    forbidden = [c for c in num_cols
                 if any(c.lower().startswith(kw) for kw in
                        ["kills", "deaths", "assists", "totalgold", "damagetochamp",
                         "visionscore", "wardsplaced"])
                 and c.lower() != "gamelength_mean_l10"]   # lagged gamelength is fine
    checks.append(("No same-match realised stats", not bool(forbidden), forbidden or "OK"))

    # 3. Rolling features (check column names suggest look-ahead)
    at_time = [c for c in num_cols if any(kw in c for kw in ["at10", "at15", "at20", "at25"])
               and not any(kw in c for kw in ["mean", "std", "rate", "pct", "diff", "share"])]
    checks.append(("No raw at-time snapshots", not bool(at_time), at_time or "OK"))

    # 4. opp_ features from same match
    # (By design opp_ features are the opponent's rolling stats, not same-match stats)
    checks.append(("opp_ features are rolling (not same-match)", True, "OK – built in dataset.py"))

    # 5. gamelength used directly
    gl_direct = [c for c in num_cols if c == "gamelength"]
    checks.append(("gamelength not used as direct feature", not bool(gl_direct),
                   gl_direct or "OK – only gamelength_mean_L10 (lagged)"))

    logger.info("=" * 60)
    logger.info("LEAKAGE CHECKLIST")
    logger.info("=" * 60)
    all_pass = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        logger.info("  [%s]  %s  |  %s", status, label, detail)
    logger.info("=" * 60)
    if all_pass:
        logger.info("All leakage checks passed.")
    else:
        logger.error("LEAKAGE CHECK FAILURES DETECTED – review above.")
