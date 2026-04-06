"""
Draft-based feature engineering for game-level prediction.

Must be called AFTER compute_elo() and BEFORE compute_features().
compute_features() drops pick1-5 via RAW_INPUT_COLS; this module uses
them first to produce aggregated numeric features.

Feature groups:
    L – Champion meta     : patch-level champion win rates (how strong is the pick?)
    M – Champion comfort  : team's rolling win rate with each picked champion
    N – Series context    : game number, score within series, must-win flag
        has_first_pick    : did this team have first pick in draft?
"""

import time
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PICK_COLS    = ["pick1", "pick2", "pick3", "pick4", "pick5"]
CHAMP_META_K = 10   # shrinkage for patch champion WR — 50/50 at 10 games on patch
CHAMP_COMF_K = 15   # shrinkage for team comfort — higher = more conservative


# ─── Main entry point ─────────────────────────────────────────────────────────

def compute_draft_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add draft features to team_df.

    Returns a new DataFrame with additional columns:
        champ_meta_mean, champ_meta_min, champ_meta_std   (Group L)
        champ_comfort_mean, champ_comfort_min             (Group M)
        series_game_num, series_score_diff, series_must_win (Group N)
        has_first_pick                                    (draft order)
    """
    df = team_df.copy()
    t_total = time.time()
    print(f"Draft feature engineering — {len(df):,} rows", flush=True)

    has_picks = all(c in df.columns for c in PICK_COLS)

    if has_picks:
        print("  L  champion meta (patch win rates) ...", end=" ", flush=True)
        t = time.time()
        meta_df = _champ_meta_features(df)
        df = df.merge(meta_df, on=["gameid", "teamid"], how="left")
        df["champ_meta_mean"] = df["champ_meta_mean"].fillna(0.5)
        df["champ_meta_min"]  = df["champ_meta_min"].fillna(0.5)
        df["champ_meta_std"]  = df["champ_meta_std"].fillna(0.0)
        print(f"done ({time.time()-t:.1f}s)", flush=True)

        print("  M  champion comfort (team rolling WR per champion) ...", end=" ", flush=True)
        t = time.time()
        comfort_df = _champ_comfort_features(df)
        df = df.merge(comfort_df, on=["gameid", "teamid"], how="left")
        df["champ_comfort_mean"] = df["champ_comfort_mean"].fillna(0.5)
        df["champ_comfort_min"]  = df["champ_comfort_min"].fillna(0.5)
        print(f"done ({time.time()-t:.1f}s)", flush=True)
    else:
        logger.warning("Pick columns not found — skipping champion features (filling 0.5)")
        for col in ["champ_meta_mean", "champ_meta_min", "champ_meta_std",
                    "champ_comfort_mean", "champ_comfort_min"]:
            df[col] = 0.5

    print("  N  series context ...", end=" ", flush=True)
    t = time.time()
    _series_context_features(df)
    print(f"done ({time.time()-t:.1f}s)", flush=True)

    print("     first pick ...", end=" ", flush=True)
    _first_pick_feature(df)
    print("done", flush=True)

    n_draft_cols = len([c for c in df.columns if c.startswith(("champ_", "series_", "has_first"))])
    print(
        f"Draft features complete — {n_draft_cols} new cols in {time.time()-t_total:.1f}s",
        flush=True,
    )
    return df


# ─── Group L: Champion meta (patch win rate) ──────────────────────────────────

def _champ_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of a team's 5 picks, look up the champion's cumulative win rate
    on the current patch (using only games BEFORE the current one).
    Shrink toward 0.5 to handle sparse early-patch data.

    Returns: DataFrame (gameid, teamid, champ_meta_mean, champ_meta_min, champ_meta_std)
    """
    available = [c for c in PICK_COLS if c in df.columns]
    id_cols = ["gameid", "teamid", "date", "patch", "result"]

    picks = (
        df[id_cols + available]
        .melt(id_vars=id_cols, value_vars=available,
              var_name="slot", value_name="champion")
        .dropna(subset=["champion"])
    )
    picks["champion"] = picks["champion"].astype(str).str.strip()
    picks = picks[picks["champion"] != ""]

    # Sort per (champion, patch) chronologically — shift(1) excludes current game
    picks = picks.sort_values(["champion", "patch", "date", "gameid"]).reset_index(drop=True)

    g = picks.groupby(["champion", "patch"], sort=False)
    picks["_cum_wins"]  = g["result"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    picks["_cum_games"] = g["result"].transform(
        lambda x: pd.Series(np.ones(len(x)), index=x.index).shift(1).cumsum().fillna(0)
    )

    k = CHAMP_META_K
    raw = picks["_cum_wins"] / picks["_cum_games"].clip(lower=1e-9)
    picks["_meta_wr"] = (
        (picks["_cum_games"] / (picks["_cum_games"] + k)) * raw
        + (k / (picks["_cum_games"] + k)) * 0.5
    ).fillna(0.5)

    agg = (
        picks.groupby(["gameid", "teamid"])["_meta_wr"]
        .agg(champ_meta_mean="mean", champ_meta_min="min", champ_meta_std="std")
        .reset_index()
    )
    agg["champ_meta_std"] = agg["champ_meta_std"].fillna(0.0)
    return agg


# ─── Group M: Champion comfort (team rolling WR per champion) ─────────────────

def _champ_comfort_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of a team's 5 picks, look up the team's historical win rate when
    playing that champion (all prior games, shrunk toward 0.5).
    High comfort = team frequently picks and wins with this champion.

    Returns: DataFrame (gameid, teamid, champ_comfort_mean, champ_comfort_min)
    """
    available = [c for c in PICK_COLS if c in df.columns]
    id_cols = ["gameid", "teamid", "date", "result"]

    picks = (
        df[id_cols + available]
        .melt(id_vars=id_cols, value_vars=available,
              var_name="slot", value_name="champion")
        .dropna(subset=["champion"])
    )
    picks["champion"] = picks["champion"].astype(str).str.strip()
    picks = picks[picks["champion"] != ""]

    # Sort per (teamid, champion) chronologically
    picks = picks.sort_values(["teamid", "champion", "date", "gameid"]).reset_index(drop=True)

    g = picks.groupby(["teamid", "champion"], sort=False)
    picks["_cum_wins"]  = g["result"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    picks["_cum_games"] = g["result"].transform(
        lambda x: pd.Series(np.ones(len(x)), index=x.index).shift(1).cumsum().fillna(0)
    )

    k = CHAMP_COMF_K
    raw = picks["_cum_wins"] / picks["_cum_games"].clip(lower=1e-9)
    picks["_comfort"] = (
        (picks["_cum_games"] / (picks["_cum_games"] + k)) * raw
        + (k / (picks["_cum_games"] + k)) * 0.5
    ).fillna(0.5)

    agg = (
        picks.groupby(["gameid", "teamid"])["_comfort"]
        .agg(champ_comfort_mean="mean", champ_comfort_min="min")
        .reset_index()
    )
    return agg


# ─── Group N: Series context ──────────────────────────────────────────────────

def _series_context_features(df: pd.DataFrame) -> None:
    """
    Compute series context features in-place:

        series_game_num  : game number within the series (1, 2, 3 ...)
        series_score_diff: team wins - losses in series BEFORE this game
                           (0 for game 1, +1/-1 for game 2, etc.)
        series_must_win  : 1 if team has lost more games than won in series

    Series identification (vectorised):
        Sort rows by (team-pair-key, date, gameid).
        A new series starts whenever game_n == 1 for a given pair.
        team-pair-key = (league, unordered teamid pair)

    This correctly handles:
        - Bo1 leagues: every game has game_n=1 → each game is its own series
        - Bo3 / Bo5: game_n increments 1→2→3 within a series
    """
    if "game" not in df.columns or "opp_teamid" not in df.columns:
        df["series_game_num"]   = 1
        df["series_score_diff"] = 0
        df["series_must_win"]   = 0
        return

    cols = ["gameid", "teamid", "opp_teamid", "league", "date", "game", "result"]
    dfw = df[[c for c in cols if c in df.columns]].copy()
    dfw["_game_n"] = pd.to_numeric(dfw["game"], errors="coerce").fillna(1).astype(int)

    # Canonical unordered pair key
    t1 = dfw[["teamid", "opp_teamid"]].min(axis=1).astype(str)
    t2 = dfw[["teamid", "opp_teamid"]].max(axis=1).astype(str)
    league_str = dfw["league"].astype(str) if "league" in dfw.columns else pd.Series("", index=dfw.index)
    dfw["_pair"] = t1 + "__" + t2 + "__" + league_str

    # Assign series IDs on UNIQUE GAMES first.
    # Each game has 2 team rows both with the same game_n.  If we ran the cumsum
    # on all rows, both rows of game 1 would each trigger a "series start", splitting
    # a single series into multiple fragments.  Deduplicating to one row per gameid
    # and then merging back avoids this.
    unique_games = (
        dfw.drop_duplicates("gameid")[["gameid", "_pair", "date", "_game_n"]]
        .sort_values(["_pair", "date", "gameid"])
        .reset_index(drop=True)
    )
    pair_change = unique_games["_pair"] != unique_games["_pair"].shift(1)
    game_reset  = unique_games["_game_n"] == 1
    unique_games["_series_id"] = (pair_change | game_reset).cumsum()

    # Propagate series_id back to every team row
    dfw = dfw.merge(unique_games[["gameid", "_series_id"]], on="gameid", how="left")

    # Within each (series_id, teamid): cumulative wins/losses BEFORE each game
    dfw = dfw.sort_values(["_series_id", "date", "gameid"])
    g = dfw.groupby(["_series_id", "teamid"], sort=False)
    dfw["_ser_wins"]   = g["result"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    dfw["_ser_losses"] = g["result"].transform(lambda x: (1 - x).shift(1).cumsum().fillna(0))
    dfw["_score_diff"] = dfw["_ser_wins"] - dfw["_ser_losses"]
    dfw["_must_win"]   = (dfw["_ser_losses"] > dfw["_ser_wins"]).astype(int)

    # Map back to df via (gameid, teamid) — unique key
    lookup = dfw.drop_duplicates(["gameid", "teamid"]).set_index(["gameid", "teamid"])
    mi = pd.MultiIndex.from_arrays([df["gameid"], df["teamid"]], names=["gameid", "teamid"])

    df["series_game_num"]   = lookup["_game_n"].reindex(mi).fillna(1).astype(int).values
    df["series_score_diff"] = lookup["_score_diff"].reindex(mi).fillna(0).values
    df["series_must_win"]   = lookup["_must_win"].reindex(mi).fillna(0).astype(int).values


# ─── First pick ───────────────────────────────────────────────────────────────

def _first_pick_feature(df: pd.DataFrame) -> None:
    """
    Add has_first_pick (1 = team had first pick, 0 = opponent had first pick).
    Fills 0.5 when unknown (e.g. older data without firstPick column).
    """
    if "firstPick" in df.columns:
        df["has_first_pick"] = pd.to_numeric(df["firstPick"], errors="coerce").fillna(0.5)
    else:
        df["has_first_pick"] = 0.5
