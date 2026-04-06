"""
Time-based splitting and rolling cross-validation.

Design rules enforced here:
1. Splits are on unique game IDs sorted by date – both mirrored rows of a match
   always land in the same partition.
2. Rolling folds use strict date ranges: val rows have date ∈ [val_start, val_end)
   and train rows have date ∈ [val_start - train_days, val_start).
3. A final holdout (most recent HOLDOUT_FRAC of unique games) is separated first
   and must not be touched until all design choices are fixed.
"""

import numpy as np
import pandas as pd
import logging
from .config import HOLDOUT_FRAC, TRAIN_DAYS, VAL_DAYS, STRIDE_DAYS

logger = logging.getLogger(__name__)


def dev_holdout_split(
    df: pd.DataFrame,
    holdout_frac: float = HOLDOUT_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (dev, holdout) based on chronological gameid order.

    Both rows of each match always land in the same split because the
    split boundary is on unique gameids sorted by date.
    """
    unique_games = (
        df.drop_duplicates("gameid")
        .sort_values("date")["gameid"]
        .tolist()
    )
    n = len(unique_games)
    cutoff = int(n * (1 - holdout_frac))
    cutoff_date = (
        df[df["gameid"] == unique_games[cutoff]]["date"].iloc[0]
    )

    dev_games  = set(unique_games[:cutoff])
    hold_games = set(unique_games[cutoff:])

    dev_df  = df[df["gameid"].isin(dev_games)].copy().reset_index(drop=True)
    hold_df = df[df["gameid"].isin(hold_games)].copy().reset_index(drop=True)

    logger.info(
        "Split: dev=%d rows (%d matches) up to %s | holdout=%d rows (%d matches)",
        len(dev_df), len(dev_games), cutoff_date.date(),
        len(hold_df), len(hold_games),
    )

    # Sanity: no gameid should appear in both splits
    overlap = dev_games & hold_games
    assert not overlap, f"Leakage: {len(overlap)} gameids in both splits"

    return dev_df, hold_df


def rolling_folds(
    dev_df: pd.DataFrame,
    train_days: int = TRAIN_DAYS,
    val_days: int = VAL_DAYS,
    stride_days: int = STRIDE_DAYS,
    min_train_rows: int = 40,
    min_val_rows: int = 10,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate (train_indices, val_indices) for rolling-window CV.

    Windows:
        train: [val_start - train_days, val_start)
        val:   [val_start, val_start + val_days)

    stride advances val_start by stride_days each fold.

    Both rows of each gameid are always in the same fold (guaranteed by
    the date-based assignment and the fact that same-match rows share a date).
    """
    df = dev_df.sort_values(["date", "gameid"]).reset_index(drop=True)
    min_date = df["date"].min()
    max_date = df["date"].max()

    folds = []
    val_start = min_date + pd.Timedelta(days=train_days)

    while val_start + pd.Timedelta(days=val_days) <= max_date:
        train_start = val_start - pd.Timedelta(days=train_days)
        val_end     = val_start + pd.Timedelta(days=val_days)

        train_mask = (df["date"] >= train_start) & (df["date"] < val_start)
        val_mask   = (df["date"] >= val_start)   & (df["date"] < val_end)

        train_idx = df.index[train_mask].tolist()
        val_idx   = df.index[val_mask].tolist()

        if len(train_idx) >= min_train_rows and len(val_idx) >= min_val_rows:
            folds.append((train_idx, val_idx))

        val_start += pd.Timedelta(days=stride_days)

    logger.info(
        "Generated %d rolling folds (train=%dd, val=%dd, stride=%dd)",
        len(folds), train_days, val_days, stride_days,
    )
    if folds:
        sizes = [len(v) for _, v in folds]
        logger.info(
            "Val fold sizes: min=%d, median=%d, max=%d",
            min(sizes), int(np.median(sizes)), max(sizes),
        )
    return folds


def check_fold_leakage(
    dev_df: pd.DataFrame, folds: list[tuple[list[int], list[int]]]
) -> bool:
    """
    Verify that for every fold:
    - No gameid appears in both train and val.
    - Both rows of a gameid are in the same partition.

    Returns True if no leakage found.
    """
    ok = True
    for i, (tr_idx, val_idx) in enumerate(folds):
        tr_games  = set(dev_df.loc[tr_idx,  "gameid"])
        val_games = set(dev_df.loc[val_idx, "gameid"])
        overlap = tr_games & val_games
        if overlap:
            logger.error("Fold %d: %d gameids in both train and val!", i, len(overlap))
            ok = False

        # Check mirrored rows: for each gameid in val, both rows should be in val
        val_gid_counts = dev_df.loc[val_idx, "gameid"].value_counts()
        incomplete = val_gid_counts[val_gid_counts < 2]
        if not incomplete.empty:
            logger.warning(
                "Fold %d: %d gameids have only 1 row in val (cross-boundary match)",
                i, len(incomplete),
            )
    if ok:
        logger.info("Fold leakage check passed for all %d folds", len(folds))
    return ok


def describe_folds(
    dev_df: pd.DataFrame, folds: list[tuple[list[int], list[int]]]
) -> pd.DataFrame:
    """Return a summary DataFrame describing each fold's date range and size."""
    rows = []
    for i, (tr_idx, val_idx) in enumerate(folds):
        tr  = dev_df.loc[tr_idx]
        val = dev_df.loc[val_idx]
        rows.append({
            "fold":        i,
            "train_start": tr["date"].min().date(),
            "train_end":   tr["date"].max().date(),
            "val_start":   val["date"].min().date(),
            "val_end":     val["date"].max().date(),
            "train_rows":  len(tr),
            "val_rows":    len(val),
            "train_games": tr["gameid"].nunique(),
            "val_games":   val["gameid"].nunique(),
        })
    return pd.DataFrame(rows)
