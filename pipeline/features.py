"""
Rolling feature engineering for the pre-draft prediction pipeline.

All features are computed from a team's PRIOR matches only.
Key mechanism: within-team sort → shift(1) → rolling.
Fresh groupby is created after each new column is added to df to avoid stale views.

Feature groups:
    B – Recent form        (win rates, EWMA, opp quality, overperformance)
    C – Early-game         (means, stds, rates, threshold features for gdiff15)
    D – Conversion/closing (basic + conditional: P(win | strong lead))
    E – Objective control  (dragon/herald/baron/tower shares)
    F – Style identity     (pace, vision, economy, aggression) [optional]
    G – Side history       (same-side WR vs patch baseline)
    H – Patch adaptation   (shrunk patch-specific WR and goldlead) [optional]
    J – Schedule/context   (days rest, match load, playoffs, game number)
    K – Interactions       (explicit pairwise products of key features) [optional]
"""

import time
import numpy as np
import pandas as pd
import logging
from .config import (
    WIN_SHORT, WIN_LONG, EWMA_ALPHA, CONV_K, PATCH_K,
    ELO_INIT, INCLUDE_PATCH_FEATURES, INCLUDE_STYLE_FEATURES,
    INCLUDE_INTERACTIONS, H2H_K,
)

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _by_team(df: pd.DataFrame):
    """Sort by (teamid, date, gameid), return (df_sorted, groupby)."""
    df_s = df.sort_values(["teamid", "date", "gameid"])
    g = df_s.groupby("teamid", sort=False)
    return df_s, g


def _roll(g, col: str, w: int, fill: float = 0.0) -> pd.Series:
    """Rolling mean over prior w rows (shift(1) excludes current)."""
    return g[col].transform(
        lambda x, w=w: pd.to_numeric(x, errors="coerce")
                         .shift(1).rolling(w, min_periods=1).mean()
    ).fillna(fill)


def _roll_std(g, col: str, w: int, fill: float = 0.0) -> pd.Series:
    return g[col].transform(
        lambda x, w=w: pd.to_numeric(x, errors="coerce")
                         .shift(1).rolling(w, min_periods=3).std()
    ).fillna(fill)


def _roll_thresh(g, col: str, w: int, threshold: float, above: bool = True) -> pd.Series:
    """Fraction of prior w rows where col > threshold (or < if above=False)."""
    if above:
        return g[col].transform(
            lambda x, w=w, t=threshold: (pd.to_numeric(x, errors="coerce").shift(1) > t)
                                          .rolling(w, min_periods=1).mean()
        ).fillna(0.5)
    else:
        return g[col].transform(
            lambda x, w=w, t=threshold: (pd.to_numeric(x, errors="coerce").shift(1) < t)
                                          .rolling(w, min_periods=1).mean()
        ).fillna(0.5)


def _roll_cond_mean(g, col: str, cond_col: str, w: int, fill: float = 0.0) -> pd.Series:
    """
    Rolling mean of `col` over prior w rows, conditioned on cond_col being True.
    Returns fill if no such rows exist.
    This is O(n*w) – used sparingly.
    """
    def _inner(x, col=col, cond_col=cond_col, w=w, fill=fill):
        # x is the index-aligned Series within a team group (sorted by date)
        col_s  = pd.to_numeric(df_ref[col ].reindex(x.index), errors="coerce").shift(1)
        cond_s = pd.to_numeric(df_ref[cond_col].reindex(x.index), errors="coerce").shift(1).astype(bool)

        result = pd.Series(fill, index=x.index)
        for i in range(len(x)):
            start = max(0, i - w)
            c_window = col_s.iloc[start:i]
            m_window = cond_s.iloc[start:i]
            vals = c_window[m_window]
            if len(vals) >= 1:
                result.iloc[i] = vals.mean()
        return result

    return g[col].transform(_inner).fillna(fill)


# ─── Main entry point ─────────────────────────────────────────────────────────

# Module-level reference used by _roll_cond_mean (set in compute_features)
df_ref: pd.DataFrame = None


def _step(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), printing elapsed time."""
    print(f"  {label} ...", end=" ", flush=True)
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"done ({time.time() - t0:.1f}s)", flush=True)
    return result


def compute_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires: elo_team, elo_opp, elo_diff, elo_expected, elo_overperf,
              elo_league_rel, elo_decayed_* (from elo.py).
    Returns df sorted by (date, gameid, side) with all rolling columns added.
    """
    global df_ref

    df = team_df.copy()
    df = df.sort_values(["date", "gameid", "side"]).reset_index(drop=True)
    df_ref = df  # for _roll_cond_mean closures

    n_rows = len(df)
    print(f"Feature engineering — {n_rows:,} rows", flush=True)

    t_total = time.time()

    _step("B  recent form", _group_b, df)
    _step("C  early-game (means + thresholds)", _group_c, df)
    _step("D  conversion/closing", _group_d, df)
    _step("E  objective control", _group_e, df)

    if INCLUDE_STYLE_FEATURES:
        _step("F  style", _group_f, df)

    _step("G  side history", _group_g, df)

    if INCLUDE_PATCH_FEATURES:
        _step("H  patch adaptation", _group_h, df)

    _step("J  schedule/context", _group_j, df)
    _step("P  head-to-head", _group_p, df)

    if INCLUDE_INTERACTIONS:
        _step("K  interactions", _group_k, df)

    # Drop helper columns
    helper_cols = [c for c in df.columns if c.startswith("_")]
    if helper_cols:
        df.drop(columns=helper_cols, inplace=True)

    # Drop raw OE input columns (used only for rolling feature input)
    from .config import RAW_INPUT_COLS
    raw_to_drop = [c for c in df.columns if c in set(RAW_INPUT_COLS)]
    if raw_to_drop:
        df.drop(columns=raw_to_drop, inplace=True)

    df_ref = None  # clear reference
    feat_count = len(get_feature_cols(df))
    print(f"Feature engineering complete — {feat_count} feature cols in {time.time() - t_total:.1f}s", flush=True)
    logger.info("Feature engineering complete. %d engineered feature cols.", feat_count)
    return df


# ─── Group B: Recent form ─────────────────────────────────────────────────────

def _group_b(df: pd.DataFrame) -> None:
    _, g = _by_team(df)

    # Win rates: L3, L5, L10
    for w, name in [(3, "L3"), (WIN_SHORT, "L5"), (WIN_LONG, "L10")]:
        df[f"wr_{name}"] = _roll(g, "result", w, fill=0.5)

    # Form trend: positive = team is heating up, negative = cooling down
    # Computed after wr_L5 and wr_L10 are in df
    df["form_trend"] = df["wr_L5"] - df["wr_L10"]

    # Exponentially weighted win rate
    df["ewma_result"] = g["result"].transform(
        lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA, min_periods=1).mean()
    ).fillna(0.5)

    # Schedule strength: avg opponent Elo
    for w, name in [(WIN_SHORT, "L5"), (WIN_LONG, "L10")]:
        df[f"avg_opp_elo_{name}"] = _roll(g, "elo_opp", w, fill=ELO_INIT)

    # Overperformance vs Elo expectation
    for w, name in [(WIN_SHORT, "L5"), (WIN_LONG, "L10")]:
        df[f"overperf_{name}"] = _roll(g, "elo_overperf", w, fill=0.0)

    # Total prior games played (proxy for feature reliability)
    df["n_games_played"] = g["result"].transform(
        lambda x: x.shift(1).expanding().count()
    ).fillna(0).astype(int)


# ─── Group C: Early-game ──────────────────────────────────────────────────────

# (src_col, short_name, compute_pos_rate, compute_thresholds)
# csdiff removed: noisy, high correlation with gdiff, low marginal signal
_EARLY_GAME_COLS = [
    ("golddiffat15", "gdiff15", True,  True),
    ("golddiffat10", "gdiff10", True,  True),
    ("xpdiffat15",   "xpdiff15", True, False),
    ("xpdiffat10",   "xpdiff10", True, False),
]

_FIRST_OBJ_COLS = [
    ("firstblood",  "fb"),
    ("firstdragon", "fd"),
    ("firstherald", "fh"),
    ("firsttower",  "ft"),
    ("firstbaron",  "fbar"),
]

# Thresholds for gdiff15 (gold) — separates dominant leads from marginal ones
_GDIFF15_THRESHOLDS = [1000, 2000]   # above these
_GDIFF15_DEFICITS   = [-1000]        # below these


def _group_c(df: pd.DataFrame) -> None:
    _, g = _by_team(df)

    for src, name, pos_rate, thresh in _EARLY_GAME_COLS:
        if src not in df.columns:
            continue

        # Rolling means (L5 and L10 only — no redundant L3 for diffs)
        for w, wname in [(WIN_SHORT, "L5"), (WIN_LONG, "L10")]:
            df[f"{name}_mean_{wname}"] = _roll(g, src, w, fill=0.0)

        # Rolling std (consistency / volatility)
        df[f"{name}_std_L10"] = _roll_std(g, src, WIN_LONG, fill=0.0)

        # Basic positive-rate
        if pos_rate:
            df[f"pct_pos_{name}_L10"] = _roll_thresh(g, src, WIN_LONG, 0.0, above=True)

        # Threshold features (gdiff15 and gdiff10 only)
        if thresh:
            for t in _GDIFF15_THRESHOLDS:
                col = f"pct_{name}_gt{t}_L10"
                df[col] = _roll_thresh(g, src, WIN_LONG, t, above=True)
            for t in _GDIFF15_DEFICITS:
                col = f"pct_{name}_lt{abs(t)}_L10"
                df[col] = _roll_thresh(g, src, WIN_LONG, t, above=False)

    # First-objective rates (L5 and L10)
    for src, name in _FIRST_OBJ_COLS:
        if src not in df.columns:
            continue
        for w, wname in [(WIN_SHORT, "L5"), (WIN_LONG, "L10")]:
            df[f"{name}_rate_{wname}"] = _roll(g, src, w, fill=0.5)


# ─── Group D: Conversion / closing ───────────────────────────────────────────

def _group_d(df: pd.DataFrame) -> None:
    """
    Basic:
        lead_conv_rate – P(win | golddiffat15 > 0), shrunk, cumulative
        comeback_rate  – P(win | golddiffat15 < 0), shrunk, cumulative

    Conditional (stronger signal):
        strong_lead_conv – P(win | golddiffat15 > +1000), shrunk
        avg_gdiff15_when_win  – mean golddiffat15 in wins (how big are winning leads?)
        avg_gdiff15_when_loss – mean golddiffat15 in losses (how far behind when losing?)
    """
    if "golddiffat15" not in df.columns:
        for col in ["lead_conv_rate", "comeback_rate",
                    "strong_lead_conv", "avg_gdiff15_when_win", "avg_gdiff15_when_loss"]:
            df[col] = 0.5
        return

    # Binary indicator columns for groupby transforms
    df["_ahead"]       = (df["golddiffat15"].astype(float) > 0).astype(float)
    df["_strong_lead"] = (df["golddiffat15"].astype(float) > 1000).astype(float)
    df["_behind"]      = (df["golddiffat15"].astype(float) < 0).astype(float)
    df["_ahead_win"]   = df["_ahead"]       * df["result"].astype(float)
    df["_strong_win"]  = df["_strong_lead"] * df["result"].astype(float)
    df["_behind_win"]  = df["_behind"]      * df["result"].astype(float)

    _, g = _by_team(df)

    c_ahead      = g["_ahead"      ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    c_strong     = g["_strong_lead"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    c_behind     = g["_behind"     ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    c_ahead_win  = g["_ahead_win"  ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    c_strong_win = g["_strong_win" ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    c_behind_win = g["_behind_win" ].transform(lambda x: x.shift(1).cumsum().fillna(0))

    k = CONV_K

    raw_lead   = c_ahead_win  / c_ahead.clip(lower=1e-9)
    raw_strong = c_strong_win / c_strong.clip(lower=1e-9)
    raw_back   = c_behind_win / c_behind.clip(lower=1e-9)

    df["lead_conv_rate"] = (
        (c_ahead  / (c_ahead  + k)) * raw_lead   + (k / (c_ahead  + k)) * 0.70
    ).fillna(0.70)

    df["strong_lead_conv"] = (
        (c_strong / (c_strong + k)) * raw_strong + (k / (c_strong + k)) * 0.75
    ).fillna(0.75)

    df["comeback_rate"] = (
        (c_behind / (c_behind + k)) * raw_back   + (k / (c_behind + k)) * 0.30
    ).fillna(0.30)

    # Average golddiffat15 in wins and losses (how decisive are their performances?)
    df["_gd15_in_win"]  = df["golddiffat15"].astype(float) * df["result"].astype(float)
    df["_win_flag"]     = df["result"].astype(float)
    df["_gd15_in_loss"] = df["golddiffat15"].astype(float) * (1 - df["result"].astype(float))
    df["_loss_flag"]    = (1 - df["result"].astype(float))

    _, g2 = _by_team(df)
    cum_gd_win   = g2["_gd15_in_win" ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    cum_wins     = g2["_win_flag"     ].transform(lambda x: x.shift(1).cumsum().fillna(0))
    cum_gd_loss  = g2["_gd15_in_loss"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    cum_losses   = g2["_loss_flag"   ].transform(lambda x: x.shift(1).cumsum().fillna(0))

    df["avg_gdiff15_when_win"]  = (cum_gd_win  / cum_wins.clip(lower=1e-9)).fillna(0.0)
    df["avg_gdiff15_when_loss"] = (cum_gd_loss / cum_losses.clip(lower=1e-9)).fillna(0.0)

    # Drop temp columns
    for c in ["_ahead", "_strong_lead", "_behind", "_ahead_win", "_strong_win",
              "_behind_win", "_gd15_in_win", "_win_flag", "_gd15_in_loss", "_loss_flag"]:
        df.drop(columns=[c], inplace=True, errors="ignore")


# ─── Group E: Objective control ───────────────────────────────────────────────

_OBJ_PAIRS = [
    ("dragons",      "opp_dragons",      "dragon"),
    ("heralds",      "opp_heralds",      "herald"),
    ("barons",       "opp_barons",       "baron"),
    ("towers",       "opp_towers",       "tower"),
    ("turretplates", "opp_turretplates", "plate"),
]


def _group_e(df: pd.DataFrame) -> None:
    for tcol, ocol, name in _OBJ_PAIRS:
        if tcol not in df.columns or ocol not in df.columns:
            continue
        t = df[tcol].astype(float)
        o = df[ocol].astype(float)
        share = (t / (t + o).replace(0, np.nan)).fillna(0.5)

        tmp = f"__{name}_share"
        df[tmp] = share
        _, g = _by_team(df)
        df[f"{name}_share_L{WIN_LONG}"] = _roll(g, tmp, WIN_LONG, fill=0.5)
        df[f"{name}_share_std_L{WIN_LONG}"] = _roll_std(g, tmp, WIN_LONG, fill=0.0)
        df.drop(columns=[tmp], inplace=True)


# ─── Group F: Style / identity ────────────────────────────────────────────────

_STYLE_COLS = [
    ("ckpm",       "pace"),
    ("vspm",       "vision"),
    ("earned_gpm", "economy"),
    ("team_kpm",   "aggression"),
]


def _group_f(df: pd.DataFrame) -> None:
    _, g = _by_team(df)
    for src, name in _STYLE_COLS:
        if src not in df.columns:
            continue
        # L10 only — L5 is too noisy and largely redundant
        df[f"{name}_mean_L10"] = _roll(g, src, WIN_LONG, fill=0.0)
    if "gamelength" in df.columns:
        df["gamelength_mean_L10"] = _roll(g, "gamelength", WIN_LONG, fill=0.0)


# ─── Group G: Side history ────────────────────────────────────────────────────

def _group_g(df: pd.DataFrame) -> None:
    df["is_blue"] = (df["side"] == "Blue").astype(int)

    for side_val, col in [("Blue", "blue_wr_hist"), ("Red", "red_wr_hist")]:
        side_mask = df["side"] == side_val
        df[col] = 0.5
        sub = df[side_mask].copy()
        if sub.empty:
            continue
        sub_s = sub.sort_values(["teamid", "date", "gameid"])
        g_s = sub_s.groupby("teamid", sort=False)
        wr = g_s["result"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        ).fillna(0.5)
        df.loc[side_mask, col] = wr.reindex(df.loc[side_mask].index)

    df["same_side_wr_hist"] = np.where(
        df["side"] == "Blue", df["blue_wr_hist"], df["red_wr_hist"]
    )
    df["patch_blue_wr"]   = _patch_blue_winrate(df)
    df["side_wr_vs_patch"] = df["same_side_wr_hist"] - df["patch_blue_wr"]


def _patch_blue_winrate(df: pd.DataFrame) -> pd.Series:
    blue = df[df["side"] == "Blue"][["gameid", "date", "patch", "result"]].copy()
    blue = blue.rename(columns={"result": "blue_won"})
    blue = blue.sort_values("date").reset_index(drop=True)

    g_p = blue.groupby("patch", sort=False)
    blue["_cum_bw"] = g_p["blue_won"].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    blue["_cum_bg"] = g_p["blue_won"].transform(
        lambda x: pd.Series(np.ones(len(x)), index=x.index).shift(1).cumsum().fillna(0)
    )
    blue["_patch_bwr"] = (
        blue["_cum_bw"] / blue["_cum_bg"].replace(0, np.nan)
    ).fillna(0.5)
    return df["gameid"].map(blue.set_index("gameid")["_patch_bwr"]).fillna(0.5)


# ─── Group H: Patch adaptation ────────────────────────────────────────────────

def _group_h(df: pd.DataFrame) -> None:
    k = PATCH_K
    df_tp = df.sort_values(["teamid", "patch", "date", "gameid"]).copy()
    df_tp["__ones"] = 1.0
    g_tp = df_tp.groupby(["teamid", "patch"], sort=False)

    df_tp["__patch_n"]    = g_tp["__ones"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    df_tp["__patch_wins"] = g_tp["result"].transform(lambda x: x.shift(1).cumsum().fillna(0))

    n = df_tp["__patch_n"]
    raw_wr = (df_tp["__patch_wins"] / n.replace(0, np.nan)).fillna(np.nan)
    overall_wr = df["wr_L10"].reindex(df_tp.index).fillna(0.5)

    df["patch_wr_shrunk"] = (
        (n / (n + k)) * raw_wr.fillna(overall_wr) + (k / (n + k)) * overall_wr
    ).reindex(df.index).fillna(0.5)

    # patch_gdiff15: dropped per recommendation (noisy, correlated with wr)
    # Keep only patch_wr_shrunk as the patch adaptation signal


# ─── Group J: Schedule / context ─────────────────────────────────────────────

def _count_prior_in_window(dates_sorted: np.ndarray, window_days: int) -> np.ndarray:
    n = len(dates_sorted)
    result = np.zeros(n, dtype=int)
    delta = np.timedelta64(window_days, "D")
    for i in range(1, n):
        cutoff = dates_sorted[i] - delta
        lo = np.searchsorted(dates_sorted[:i], cutoff)
        result[i] = i - lo
    return result


def _group_j(df: pd.DataFrame) -> None:
    df_bt, _ = _by_team(df)

    last_date = df_bt.groupby("teamid", sort=False)["date"].transform(lambda x: x.shift(1))
    days_since = (df_bt["date"] - last_date).dt.total_seconds() / 86400
    df["days_since_last"] = days_since.reindex(df.index).fillna(-1)

    for col_name, days in [("matches_last_7d", 7), ("matches_last_14d", 14)]:
        counts_series = np.zeros(len(df), dtype=int)
        for _, grp_idx in df_bt.groupby("teamid", sort=False).groups.items():
            grp_rows = df_bt.loc[grp_idx]
            local_dates = grp_rows["date"].values.astype("datetime64[ns]")
            local_counts = _count_prior_in_window(local_dates, days)
            for pos, ridx in enumerate(grp_rows.index):
                counts_series[ridx] = local_counts[pos]
        df[col_name] = counts_series

    if "playoffs" in df.columns:
        df["is_playoffs"] = pd.to_numeric(df["playoffs"], errors="coerce").fillna(0).astype(int)
    if "game" in df.columns:
        df["game_number"] = pd.to_numeric(df["game"], errors="coerce").fillna(1).astype(int)

    # League tier: ordinal competitive strength
    # Tier 1 = top global leagues; Tier 2 = strong regional; Tier 3 = developing
    if "league" in df.columns:
        df["league_tier"] = df["league"].map(_LEAGUE_TIER_MAP).fillna(3).astype(int)


# Tier map — defined at module level so it can be inspected / updated easily
_LEAGUE_TIER_MAP: dict[str, int] = {
    # Tier 1: major global competitions
    "LCK": 1, "LPL": 1, "LEC": 1, "LCS": 1,
    "Worlds": 1, "MSI": 1,
    # Tier 2: strong secondary leagues
    "LCK CL": 2, "LCKC": 2,
    "LDL": 2,
    "LCS A": 2, "LCSA": 2,
    "NACL": 2,
    "VCS": 2,
    "PCS": 2,
    "CBLOL": 2, "CBLOL A": 2,
    "LFL": 2,
    "NLC": 2,
    "PRM": 2,
    "LLA": 2, "LAS": 2,
    "TCL": 2,
    "LJL": 2,
    "LCO": 2,
    "EBL": 2,
    "EMEA Masters": 2,
    # Everything else defaults to Tier 3 via fillna(3)
}


# ─── Group P: Head-to-head history ───────────────────────────────────────────

def _group_p(df: pd.DataFrame) -> None:
    """
    Compute cumulative head-to-head win rate for each (teamid, opp_teamid) pair.

    Leakage safety: shift(1) within each (teamid, opp_teamid) group ensures the
    current match is never included in its own feature value.

    Shrinkage: shrink toward the team's own rolling win rate (wr_L10) rather than
    a fixed 0.5 prior — a team that's generally 70% good should have a 70% prior
    against a new opponent, not 50%.

    Features:
        h2h_wr       – shrunk H2H win rate vs this specific opponent
        h2h_n_games  – number of prior H2H meetings (reflects feature reliability)
    """
    if "opp_teamid" not in df.columns:
        df["h2h_wr"]      = 0.5
        df["h2h_n_games"] = 0
        return

    # Sort within each (teamid, opp_teamid) pair chronologically
    df_s = df.sort_values(["teamid", "opp_teamid", "date", "gameid"]).copy()
    g = df_s.groupby(["teamid", "opp_teamid"], sort=False)

    # Cumulative wins / games played against this opponent — shifted to exclude current
    df_s["_h2h_wins"]  = g["result"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    df_s["_h2h_games"] = g["result"].transform(
        lambda x: pd.Series(np.ones(len(x)), index=x.index).shift(1).cumsum().fillna(0)
    )

    raw_h2h = df_s["_h2h_wins"] / df_s["_h2h_games"].clip(lower=1e-9)

    # Prior = team's recent win rate (already computed in Group B)
    prior = df_s["wr_L10"].fillna(0.5) if "wr_L10" in df_s.columns else pd.Series(0.5, index=df_s.index)

    k = H2H_K
    df_s["_h2h_wr"] = (
        (df_s["_h2h_games"] / (df_s["_h2h_games"] + k)) * raw_h2h
        + (k / (df_s["_h2h_games"] + k)) * prior
    ).fillna(prior)

    # Map back to df's original index order
    df["h2h_wr"]      = df_s["_h2h_wr"].reindex(df.index).fillna(0.5)
    df["h2h_n_games"] = df_s["_h2h_games"].reindex(df.index).fillna(0)


# ─── Group K: Interactions ────────────────────────────────────────────────────

def _group_k(df: pd.DataFrame) -> None:
    """
    Explicit pairwise interaction terms between the most predictive features.

    All use pre-computed rolling features (already in df) so there is no
    additional leakage risk beyond what the base features already have.

    Selected interactions (ordered by expected predictive value):
    1. elo_diff × gdiff15_mean_L10      – strong team with consistent early leads
    2. elo_diff × wr_L5                 – strong team in good recent form
    3. gdiff15_mean_L10 × dragon_share  – early gold + macro control
    4. lead_conv_rate × gdiff15_mean_L10 – closing ability × lead magnitude
    5. elo_diff × overperf_L5          – quality × recent outperformance
    6. side_wr_vs_patch × elo_diff     – side-adjusted strength
    7. elo_diff × ewma_result          – quality × momentum
    8. elo_decayed_diff × gdiff15_mean_L10 – decayed strength × early game
    9. avg_opp_elo_L10 × wr_L10        – winning against strong opponents
    10. strong_lead_conv × gdiff15_mean_L10 – elite closers with big leads
    """
    # Helper: safe multiply with fillna(0)
    def _mult(a, b) -> pd.Series:
        return (df[a].fillna(0) * df[b].fillna(0))

    def _add_ix(name: str, a: str, b: str) -> None:
        if a in df.columns and b in df.columns:
            df[f"ix_{name}"] = _mult(a, b)

    _add_ix("elo_x_gdiff15",     "elo_diff",        "gdiff15_mean_L10")
    _add_ix("elo_x_wr5",         "elo_diff",        "wr_L5")
    _add_ix("gdiff15_x_dragon",  "gdiff15_mean_L10","dragon_share_L10")
    _add_ix("conv_x_gdiff15",    "lead_conv_rate",  "gdiff15_mean_L10")
    _add_ix("elo_x_overperf5",   "elo_diff",        "overperf_L5")
    _add_ix("side_x_elo",        "side_wr_vs_patch","elo_diff")
    _add_ix("elo_x_ewma",        "elo_diff",        "ewma_result")
    _add_ix("opp_elo_x_wr10",    "avg_opp_elo_L10", "wr_L10")
    _add_ix("strong_conv_x_gd15","strong_lead_conv","gdiff15_mean_L10")

    # Decayed Elo interaction (only if decayed Elo is enabled)
    if "elo_decayed_diff" in df.columns:
        _add_ix("dec_elo_x_gdiff15", "elo_decayed_diff", "gdiff15_mean_L10")

    # Patch win rate × Elo (patch advantage for strong teams)
    if "patch_wr_shrunk" in df.columns:
        _add_ix("patch_wr_x_elo", "patch_wr_shrunk", "elo_diff")


# ─── Utility ─────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return engineered feature columns using the FEATURE_PREFIXES whitelist.
    Includes opp_ and diff_ prefixed versions automatically.
    """
    from .config import FEATURE_PREFIXES
    result = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        base = c.removeprefix("opp_").removeprefix("diff_")
        if any(base.startswith(p) or c.startswith(p) for p in FEATURE_PREFIXES):
            result.append(c)
    return result
