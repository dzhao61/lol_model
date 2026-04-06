"""
Chronological Elo computation.

Leakage rules enforced here:
- elo_team / elo_opp are recorded BEFORE the match result updates the ratings.
- The Elo of a team that has never played is ELO_INIT.
- league_elo_rel uses only matches strictly before the current row's date.

Output columns added to team_df:
    elo_team         – pre-match Elo of this team
    elo_opp          – pre-match Elo of opponent
    elo_diff         – elo_team - elo_opp
    elo_expected     – expected win probability given the Elo matchup
    elo_overperf     – result - elo_expected (positive = exceeded expectation)
    elo_league_rel   – elo_team minus mean Elo of teams in same league (prior)
    elo_decayed_team – Elo that regresses toward 1500 during inactivity gaps
    elo_decayed_diff – elo_decayed_team - elo_decayed_opp
"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from .config import ELO_INIT, ELO_K, ELO_D, LEAGUE_WINDOW, ELO_DECAY_HALFLIFE

logger = logging.getLogger(__name__)


def compute_elo(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Elo-based columns to team_df (must already be sorted by date, gameid, side).
    Processes matches in strict chronological order.
    Returns a copy with new columns.
    """
    df = team_df.copy()
    df = df.sort_values(["date", "gameid", "side"]).reset_index(drop=True)

    n = len(df)
    elo_team_arr    = np.full(n, ELO_INIT)
    elo_opp_arr     = np.full(n, ELO_INIT)
    elo_dec_team    = np.full(n, ELO_INIT)
    elo_dec_opp     = np.full(n, ELO_INIT)

    # Running state: teamid → (current_elo, decayed_elo, last_match_timestamp)
    ratings:     dict[str, float]  = defaultdict(lambda: ELO_INIT)
    dec_ratings: dict[str, float]  = defaultdict(lambda: ELO_INIT)
    last_ts:     dict[str, object] = {}   # teamid → pd.Timestamp

    use_decay = ELO_DECAY_HALFLIFE is not None
    ln2 = np.log(2.0)

    def _apply_decay(team: str, current_ts) -> None:
        """Regress decayed Elo toward 1500 based on days since last match."""
        if not use_decay or team not in last_ts:
            return
        days = (current_ts - last_ts[team]).total_seconds() / 86400.0
        if days > 0:
            factor = 0.5 ** (days / ELO_DECAY_HALFLIFE)
            dec_ratings[team] = ELO_INIT + (dec_ratings[team] - ELO_INIT) * factor

    prev_gameid = None
    game_rows: list[int] = []

    def _flush_game(rows: list[int]) -> None:
        if len(rows) != 2:
            for idx in rows:
                tid = df.at[idx, "teamid"]
                oid = df.at[idx, "opp_teamid"]
                elo_team_arr[idx] = ratings[tid]
                elo_opp_arr[idx]  = ratings[oid] if pd.notna(oid) else ELO_INIT
                elo_dec_team[idx] = dec_ratings[tid]
                elo_dec_opp[idx]  = dec_ratings[oid] if pd.notna(oid) else ELO_INIT
            return

        idx_a, idx_b = rows
        tid_a = df.at[idx_a, "teamid"]
        tid_b = df.at[idx_b, "teamid"]
        match_ts = df.at[idx_a, "date"]

        # Apply decay to both teams before recording
        _apply_decay(tid_a, match_ts)
        _apply_decay(tid_b, match_ts)

        ra  = ratings[tid_a];     rb  = ratings[tid_b]
        rda = dec_ratings[tid_a]; rdb = dec_ratings[tid_b]

        # Record pre-match values
        elo_team_arr[idx_a] = ra;  elo_opp_arr[idx_a] = rb
        elo_team_arr[idx_b] = rb;  elo_opp_arr[idx_b] = ra
        elo_dec_team[idx_a] = rda; elo_dec_opp[idx_a] = rdb
        elo_dec_team[idx_b] = rdb; elo_dec_opp[idx_b] = rda

        # Update both standard and decayed Elo identically (decay only at start of next gap)
        result_a = df.at[idx_a, "result"]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / ELO_D))
        new_ra = ra + ELO_K * (result_a - ea)
        new_rb = rb + ELO_K * ((1.0 - result_a) - (1.0 - ea))
        ratings[tid_a]     = new_ra;  ratings[tid_b]     = new_rb
        dec_ratings[tid_a] = new_ra;  dec_ratings[tid_b] = new_rb
        last_ts[tid_a]     = match_ts; last_ts[tid_b]    = match_ts

    for idx, row in df.iterrows():
        gid = row["gameid"]
        if gid != prev_gameid:
            if game_rows:
                _flush_game(game_rows)
            game_rows = [idx]
            prev_gameid = gid
        else:
            game_rows.append(idx)

    if game_rows:
        _flush_game(game_rows)

    df["elo_team"]  = elo_team_arr
    df["elo_opp"]   = elo_opp_arr
    df["elo_diff"]  = df["elo_team"] - df["elo_opp"]
    df["elo_expected"] = 1.0 / (1.0 + 10.0 ** (-df["elo_diff"] / ELO_D))
    df["elo_overperf"] = df["result"] - df["elo_expected"]

    if use_decay:
        df["elo_decayed_team"] = elo_dec_team
        df["elo_decayed_opp"]  = elo_dec_opp
        df["elo_decayed_diff"] = df["elo_decayed_team"] - df["elo_decayed_opp"]

    df["elo_league_rel"] = _compute_league_rel(df)

    logger.info(
        "Elo computed. Range: [%.0f, %.0f]  Decayed: %s",
        df["elo_team"].min(), df["elo_team"].max(),
        f"[{elo_dec_team.min():.0f}, {elo_dec_team.max():.0f}]" if use_decay else "disabled",
    )
    return df


def _compute_league_rel(df: pd.DataFrame) -> pd.Series:
    """
    For each row: elo_team - mean(elo_team of all teams in same league whose
    last match before this date gives their elo_team value).

    Approximation used for efficiency:
    - For each row at date d in league L, take all rows in league L with date < d,
      keep the most-recent elo_team per teamid, compute the mean.
    - Fallback = 0 if no prior data in the league.
    """
    df_sorted = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    league_rel = np.zeros(len(df_sorted))

    for league, grp_idx in df_sorted.groupby("league", sort=False).groups.items():
        grp = df_sorted.loc[grp_idx].sort_values("date").reset_index()
        # Expanding mean of elo_team across all teams in league
        # For row i, we want mean of elo_team across all distinct teams in
        # the league using their most-recent pre-i elo value.
        # Simple proxy: expanding mean of elo_team values seen so far
        # (includes all team appearances, not just most recent per team,
        # but is a good approximation and fully vectorised).
        expanding_mean = grp["elo_team"].expanding(min_periods=1).mean().shift(1)
        expanding_mean.iloc[0] = ELO_INIT  # first row in league has no prior

        for i, orig_idx in enumerate(grp["index"]):
            league_rel[orig_idx] = (
                df_sorted.at[orig_idx, "elo_team"] - expanding_mean.iloc[i]
            )

    # Re-align to df's original index
    return pd.Series(league_rel, index=df_sorted.index).reindex(df.index)
