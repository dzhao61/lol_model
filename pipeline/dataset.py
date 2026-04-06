"""
Build the final modeling dataset from the feature-engineered team DataFrame.

Steps:
1. Identify which columns are feature columns vs metadata.
2. For each row, merge in the opponent team's feature values (opp_ prefix).
   The opponent's features come from their own prior-match history,
   computed independently in features.py.  No leakage.
3. Compute diff_ = team_feat - opp_feat for all numeric feature columns.
4. Attach the categorical context columns (league, split, side, patch, playoffs).
5. Return a clean DataFrame ready for train/val splitting and preprocessing.

Leakage check:
- opp_ features are merged by (gameid, opp_teamid), meaning we pull the feature
  row that was built from the OPPONENT's own prior history.  Both rows of a match
  are built independently before being joined, so no cross-contamination.
"""

import numpy as np
import pandas as pd
import logging
from .config import META_COLS, CAT_COLS, TARGET, NODIFF_COLS, NOSYM_OPP_COLS
from .features import get_feature_cols

logger = logging.getLogger(__name__)


def build_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team and opponent features, add diff columns, return modeling df.

    Returns columns:
        [meta cols] + [team feature cols] + [opp_* cols] + [diff_* cols]
    """
    df = features_df.copy()

    feat_cols = get_feature_cols(df)
    # Only base team feature cols (no opp_/diff_ yet at this stage)
    feat_cols = [c for c in feat_cols if not c.startswith("opp_") and not c.startswith("diff_")]
    logger.info("Merging opponent features for %d feature columns", len(feat_cols))

    # Build opponent lookup: (gameid, teamid) → feature row
    opp_lookup = df[["gameid", "teamid"] + feat_cols].rename(
        columns={"teamid": "opp_teamid", **{c: f"opp_{c}" for c in feat_cols}}
    )

    result = df.merge(opp_lookup, on=["gameid", "opp_teamid"], how="left")

    n_missing_opp = result[[f"opp_{c}" for c in feat_cols[:3]]].isna().any(axis=1).sum()
    if n_missing_opp:
        logger.warning(
            "%d rows missing opponent features (incomplete game records)", n_missing_opp
        )

    # Diff features: team - opp, but skip columns where the diff is redundant:
    #   - Already-differential features (e.g. elo_diff = team_elo - opp_elo;
    #     diff_elo_diff = 2*elo_diff — perfectly correlated, wastes regularization)
    #   - Symmetric match-level cols identical for both teams (is_playoffs etc.)
    diff_added = 0
    for c in feat_cols:
        if c in NODIFF_COLS:
            continue
        opp_c = f"opp_{c}"
        if opp_c not in result.columns:
            continue
        if pd.api.types.is_numeric_dtype(result[c]) and pd.api.types.is_numeric_dtype(result[opp_c]):
            result[f"diff_{c}"] = result[c] - result[opp_c]
            diff_added += 1

    # Drop opp_ copies of symmetric match-level features (both teams share same value)
    sym_opp_to_drop = [f"opp_{c}" for c in NOSYM_OPP_COLS if f"opp_{c}" in result.columns]
    if sym_opp_to_drop:
        result.drop(columns=sym_opp_to_drop, inplace=True)
        logger.info("Dropped %d symmetric opp_ columns", len(sym_opp_to_drop))

    logger.info("Added %d diff_ columns", diff_added)

    # Drop any temporary helper columns
    result = result.loc[:, ~result.columns.str.startswith("_")]

    # Ensure result is sorted by date
    result = result.sort_values(["date", "gameid", "side"]).reset_index(drop=True)

    # Summary
    total_feats = len(_identify_feature_cols(result))
    logger.info(
        "Dataset built: %d rows, %d feature cols (team + opp + diff)",
        len(result), total_feats,
    )
    _log_leakage_warnings(result)
    return result


def get_model_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns (numeric_feature_cols, cat_feature_cols) for the modeling df.

    numeric_feature_cols: all engineered feature cols that are numeric
    cat_feature_cols: categorical columns to be one-hot encoded
    """
    feat_cols = _identify_feature_cols(df)

    num_cols = [
        c for c in feat_cols
        if c not in CAT_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    return num_cols, cat_cols


def _identify_feature_cols(df: pd.DataFrame) -> list[str]:
    """Delegate to the whitelist-based get_feature_cols."""
    return get_feature_cols(df)


def _log_leakage_warnings(df: pd.DataFrame) -> None:
    """
    Soft leakage checks: warn if any obviously forbidden column made it through.
    """
    forbidden_patterns = [
        "champion", "pick1", "pick2", "pick3", "pick4", "pick5",
        "ban1", "ban2", "ban3", "ban4", "ban5", "firstPick",
    ]
    found = [c for c in df.columns for p in forbidden_patterns if p in c.lower()]
    if found:
        logger.error(
            "LEAKAGE WARNING: draft-related columns found in dataset: %s", found
        )
