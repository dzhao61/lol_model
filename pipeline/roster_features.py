"""
Roster stability features for the pre-draft prediction pipeline.

Must be called with (team_df, player_df) BEFORE compute_features().
player_df is the raw Oracle's Elixir player rows (position != 'team').

Features added to team_df (Group R):
    roster_overlap          – fraction of players shared with previous game (0–1)
                              1.0 = identical roster, <1.0 = substitution(s)
    games_since_roster_change – games played since last roster change
                              0 = this is the first game after a change,
                              high = stable roster

Betting context:
    A freshly substituted team (roster_overlap < 1.0) has unreliable rolling
    stats (win rates, early game metrics) because the underlying players changed.
    The model should have lower confidence in its estimate for such teams.
    `games_since_roster_change` lets the model learn this dampening automatically.
    Typical pattern: teams perform worse in the 1–3 games after a sub-in.
"""

import time
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum playerid coverage required to compute roster comparison.
# If < MIN_COVERAGE fraction of players have a valid playerid, fall back to
# playername-based matching (less reliable across years due to name changes).
MIN_ID_COVERAGE = 0.80


def compute_roster_features(
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add roster stability features to team_df.

    Parameters
    ----------
    team_df   : one row per team per match (position == 'team')
    player_df : one row per player per match (position != 'team')

    Returns team_df copy with additional columns:
        roster_overlap, games_since_roster_change
    """
    t0 = time.time()
    print("Roster features ...", end=" ", flush=True)

    df = team_df.copy()

    # ── Build roster sets ──────────────────────────────────────────────────────
    # For each (gameid, teamid), collect the frozenset of player identifiers.
    id_col = _choose_id_column(player_df)
    roster_map = _build_roster_map(player_df, id_col)

    # ── Sort team_df chronologically per team ──────────────────────────────────
    df_s = df.sort_values(["teamid", "date", "gameid"]).copy()

    # ── Compute per-row overlap with previous game ─────────────────────────────
    overlaps          = np.ones(len(df_s), dtype=float)   # default = stable
    games_since       = np.zeros(len(df_s), dtype=int)

    prev_roster : dict[str, frozenset] = {}   # teamid → last known roster
    prev_change : dict[str, int]       = {}   # teamid → games since last change counter

    for pos, (idx, row) in enumerate(df_s.iterrows()):
        tid = row["teamid"]
        gid = row["gameid"]
        key = (gid, tid)

        current = roster_map.get(key, frozenset())

        if tid in prev_roster and len(current) > 0 and len(prev_roster[tid]) > 0:
            # Jaccard-style overlap: |intersection| / 5
            overlap = len(current & prev_roster[tid]) / 5.0
            overlaps[pos] = overlap
            if overlap < 1.0:
                prev_change[tid] = 0    # reset counter
            else:
                prev_change[tid] = prev_change.get(tid, 0) + 1
        else:
            # First game for this team — treat as stable (no prior to compare)
            overlaps[pos] = 1.0
            prev_change[tid] = prev_change.get(tid, 0) + 1

        if len(current) > 0:
            prev_roster[tid] = current

        games_since[pos] = prev_change.get(tid, 0)

    df_s["roster_overlap"]             = overlaps
    df_s["games_since_roster_change"]  = games_since

    # Map back to df's original row order
    df["roster_overlap"]            = df_s["roster_overlap"].reindex(df.index).fillna(1.0)
    df["games_since_roster_change"] = df_s["games_since_roster_change"].reindex(df.index).fillna(0).astype(int)

    n_changes = (df["roster_overlap"] < 1.0).sum()
    print(
        f"done ({time.time()-t0:.1f}s) — "
        f"{n_changes:,} roster changes detected ({n_changes/len(df)*100:.1f}% of games)",
        flush=True,
    )
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _choose_id_column(player_df: pd.DataFrame) -> str:
    """Use playerid if well-populated, else fall back to playername."""
    if "playerid" in player_df.columns:
        coverage = player_df["playerid"].notna().mean()
        if coverage >= MIN_ID_COVERAGE:
            return "playerid"
    logger.info("playerid coverage below %.0f%% — using playername", MIN_ID_COVERAGE * 100)
    return "playername"


def _build_roster_map(
    player_df: pd.DataFrame,
    id_col: str,
) -> dict[tuple, frozenset]:
    """
    Build a dict: (gameid, teamid) → frozenset of player identifiers.
    Only includes rows where the identifier is non-null.
    """
    sub = player_df[["gameid", "teamid", id_col]].dropna(subset=[id_col]).copy()
    sub[id_col] = sub[id_col].astype(str).str.strip()

    roster_map: dict[tuple, frozenset] = {}
    for (gid, tid), grp in sub.groupby(["gameid", "teamid"], sort=False):
        roster_map[(gid, tid)] = frozenset(grp[id_col].tolist())

    return roster_map
