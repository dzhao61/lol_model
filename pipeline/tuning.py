"""
Hyperparameter search on the development set using rolling-window CV folds.

Strategy:
- Only the most recent TUNE_LAST_N_FOLDS are used for tuning (faster,
  more relevant to current competitive meta).
- We use a manual RandomizedSearch over the param grid for each model,
  training and evaluating on each selected fold, then averaging log loss.
- Results are cached to disk so re-running doesn't redo all the work.
- The unregularised logistic model has no params to tune – it's skipped.
"""

import json
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from .config import RANDOM_STATE, N_TUNE_ITER, TUNE_LAST_N_FOLDS, CACHE_DIR
from .models import (
    MODEL_REGISTRY, MODEL_NAMES, make_preprocessor,
    train_eval_fold, BASELINE_METRICS,
)

logger = logging.getLogger(__name__)

_TUNE_CACHE = CACHE_DIR / "best_params.json"


def tune_all_models(
    dev_df: pd.DataFrame,
    folds: list[tuple[list[int], list[int]]],
    num_cols: list[str],
    cat_cols: list[str],
    n_iter: int = N_TUNE_ITER,
    last_n_folds: int = TUNE_LAST_N_FOLDS,
    force_retune: bool = False,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """
    Run hyperparameter search for all tunable models.

    Returns dict: {model_name: best_params} where best_params keys
    do NOT have the 'clf__' prefix (they are passed directly to the
    estimator's set_params).

    If a cache file exists and force_retune is False, loads from cache.
    cache_path: override default cache location (useful for draft model).
    """
    tune_cache = Path(cache_path) if cache_path else _TUNE_CACHE
    if tune_cache.exists() and not force_retune:
        logger.info("Loading cached hyperparameters from %s", tune_cache)
        with open(tune_cache) as f:
            return json.load(f)

    rng = random.Random(RANDOM_STATE)
    tune_folds = folds[-last_n_folds:] if len(folds) > last_n_folds else folds
    logger.info(
        "Tuning on %d folds (last %d of %d total)", len(tune_folds), last_n_folds, len(folds)
    )

    best_params_all: dict[str, dict] = {}

    for mname in MODEL_NAMES:
        _, param_grid = MODEL_REGISTRY[mname]

        if not param_grid:
            logger.info("%-15s  no params to tune – using defaults", mname)
            best_params_all[mname] = {}
            continue

        logger.info("Tuning %s (%d iterations × %d folds) ...", mname, n_iter, len(tune_folds))
        print(f"\n[{mname}] tuning {n_iter} combos × {len(tune_folds)} folds ...", flush=True)
        best_score, best_p = np.inf, {}

        # Sample param combinations
        param_combos = _sample_param_grid(param_grid, n_iter, rng)

        for combo_i, raw_params in enumerate(param_combos):
            # raw_params has 'clf__' prefix (from param grid keys)
            # strip prefix for set_params
            params = {k.replace("clf__", ""): v for k, v in raw_params.items()}
            fold_losses = []

            for tr_idx, val_idx in tune_folds:
                tr  = dev_df.loc[tr_idx]
                val = dev_df.loc[val_idx]
                pre = make_preprocessor(num_cols, cat_cols)
                X_tr  = pre.fit_transform(tr[num_cols + cat_cols])
                X_val = pre.transform(val[num_cols + cat_cols])
                y_tr  = tr["result"].values
                y_val = val["result"].values

                try:
                    m = train_eval_fold(mname, X_tr, y_tr, X_val, y_val, params)
                    fold_losses.append(m["log_loss"])
                except Exception as exc:
                    logger.debug("Combo %d fold error: %s", combo_i, exc)

            if fold_losses:
                mean_loss = np.mean(fold_losses)
                marker = " *" if mean_loss < best_score else ""
                print(
                    f"  combo {combo_i+1:>3}/{n_iter}  loss={mean_loss:.5f}{marker}",
                    flush=True,
                )
                if mean_loss < best_score:
                    best_score = mean_loss
                    best_p = params
            else:
                print(f"  combo {combo_i+1:>3}/{n_iter}  (all folds failed)", flush=True)

        logger.info(
            "%-15s  best log_loss=%.5f  params=%s", mname, best_score, best_p
        )
        print(f"[{mname}] done — best log_loss={best_score:.5f}  params={best_p}", flush=True)
        best_params_all[mname] = best_p

    # Cache results
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(tune_cache, "w") as f:
        json.dump(best_params_all, f, indent=2, default=_json_serialise)
    logger.info("Saved best params to %s", tune_cache)

    return best_params_all


def _sample_param_grid(
    grid: dict[str, list],
    n_iter: int,
    rng: random.Random,
) -> list[dict]:
    """
    Sample n_iter parameter combinations from the grid.
    If the full grid has fewer than n_iter combinations, return all of them.
    """
    keys = list(grid.keys())
    values = list(grid.values())
    all_combos = list(product(*values))

    if len(all_combos) <= n_iter:
        return [dict(zip(keys, c)) for c in all_combos]

    sampled = rng.sample(all_combos, n_iter)
    return [dict(zip(keys, c)) for c in sampled]


def _json_serialise(obj):
    """Handle numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def load_best_params(cache_path: Path | None = None) -> dict[str, dict] | None:
    """Load cached best params if available, else return None."""
    p = Path(cache_path) if cache_path else _TUNE_CACHE
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None
