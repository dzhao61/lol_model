"""
Load and clean raw Oracle's Elixir CSVs.

Responsibilities:
- Concatenate all yearly files
- Filter to team-level rows (position == 'team')
- Rename columns with spaces
- Parse dates
- Build opp_teamid mapping (no leakage: just cross-reference within each gameid)
- Return (team_df, player_df)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from .config import DATA_DIR, YEARS, COL_RENAME, META_COLS

logger = logging.getLogger(__name__)

# Columns that will never enter the feature matrix (draft / same-match stats)
_FORBIDDEN_FEATURE_COLS = [
    # draft
    "firstPick", "champion",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5",
    # same-match kill/death/assist (only usable as lagged summaries)
    "kills", "deaths", "assists", "teamkills", "teamdeaths",
    "doublekills", "triplekills", "quadrakills", "pentakills",
    "firstbloodkill", "firstbloodassist", "firstbloodvictim",
    # same-match realised econ/vision
    "totalgold", "goldspent", "earnedgold", "earnedgoldshare",
    "minionkills", "monsterkills", "monsterkillsownjungle", "monsterkillsenemyjungle",
    "cspm", "total_cs",
    "wardsplaced", "wpm", "wardskilled", "wcpm",
    "damagetochampions", "damagemitigatedperminute", "damagetotowers",
    # same-match at-time snapshots (can only be used as lagged)
    "goldat10", "xpat10", "csat10", "opp_goldat10", "opp_xpat10", "opp_csat10",
    "killsat10", "assistsat10", "deathsat10", "opp_killsat10", "opp_assistsat10", "opp_deathsat10",
    "goldat15", "xpat15", "csat15", "opp_goldat15", "opp_xpat15", "opp_csat15",
    "killsat15", "assistsat15", "deathsat15", "opp_killsat15", "opp_assistsat15", "opp_deathsat15",
    "goldat20", "xpat20", "csat20", "opp_goldat20", "opp_xpat20", "opp_csat20",
    "goldat25", "xpat25", "csat25", "opp_goldat25", "opp_xpat25", "opp_csat25",
    "killsat20", "assistsat20", "deathsat20", "opp_killsat20", "opp_assistsat20", "opp_deathsat20",
    "killsat25", "assistsat25", "deathsat25", "opp_killsat25", "opp_assistsat25", "opp_deathsat25",
    # gamelength and final tallies (same-match)
    "gamelength",
    "inhibitors", "opp_inhibitors",
    "elementaldrakes", "opp_elementaldrakes",
    "infernals", "mountains", "clouds", "oceans", "chemtechs", "hextechs",
    "dragons (type unknown)", "elders", "opp_elders",
]


def _csv_path(year: int) -> Path:
    return DATA_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def load_all(use_years: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all yearly OE CSVs, returning (team_df, player_df).

    team_df  – one row per team per match (position == 'team')
    player_df – one row per player per match (position != 'team')
                needed only for Tier-3 roster features

    Leakage safety:
    - opp_teamid is derived purely from the other team in the same gameid.
    - No cross-match information is mixed.
    """
    years = use_years or YEARS
    frames = []
    for yr in years:
        p = _csv_path(yr)
        if not p.exists():
            logger.warning("Missing file: %s", p)
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df["year"] = yr
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to load %s: %s", p, exc)

    if not frames:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")

    raw = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw rows from %d files", len(raw), len(frames))

    raw = _rename_cols(raw)
    raw = _parse_date(raw)
    raw = _cast_numerics(raw)

    player_df = raw[raw["position"] != "team"].copy()
    team_df = raw[raw["position"] == "team"].copy()

    team_df = _build_opponent_mapping(team_df)
    team_df = _sort_and_validate(team_df)

    logger.info(
        "Team rows: %d  |  Player rows: %d  |  Unique matches: %d",
        len(team_df),
        len(player_df),
        team_df["gameid"].nunique(),
    )
    return team_df, player_df


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COL_RENAME)


def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad:
        logger.warning("Dropping %d rows with unparseable dates", n_bad)
        df = df.dropna(subset=["date"])
    return df


def _cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns that should be numeric but may be read as object (mixed types
    across years).
    """
    numeric_candidates = [
        "result", "playoffs", "game",
        "golddiffat10", "xpdiffat10", "csdiffat10",
        "golddiffat15", "xpdiffat15", "csdiffat15",
        "firstblood", "firstdragon", "firstherald", "firstbaron", "firsttower",
        "firsttothreetowers", "firstmidtower",
        "dragons", "opp_dragons", "heralds", "opp_heralds",
        "barons", "opp_barons", "towers", "opp_towers",
        "turretplates", "opp_turretplates",
        "void_grubs", "opp_void_grubs",
        "team_kpm", "ckpm", "earned_gpm", "gspd", "gpr",
        "vspm", "visionscore", "controlwardsbought",
        "dpm", "damageshare", "damagetakenperminute",
        "gamelength",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_opponent_mapping(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opp_teamid column.
    For each gameid, there are exactly 2 team rows (Blue + Red).
    opp_teamid for each row = the other row's teamid.
    """
    # Within each gameid, pair the two teamids
    pairs = (
        team_df.groupby("gameid")["teamid"]
        .apply(list)
        .reset_index()
    )
    # Keep only games with exactly 2 teams
    pairs = pairs[pairs["teamid"].map(len) == 2]
    valid_gameids = set(pairs["gameid"])

    n_dropped = team_df["gameid"].nunique() - len(valid_gameids)
    if n_dropped:
        logger.warning("Dropping %d gameids with ≠2 team rows", n_dropped)
    team_df = team_df[team_df["gameid"].isin(valid_gameids)].copy()

    # Build lookup: (gameid, teamid) → opp_teamid
    lookup = {}
    for _, row in pairs.iterrows():
        gid = row["gameid"]
        a, b = row["teamid"]
        lookup[(gid, a)] = b
        lookup[(gid, b)] = a

    team_df["opp_teamid"] = team_df.apply(
        lambda r: lookup.get((r["gameid"], r["teamid"]), np.nan), axis=1
    )
    n_missing = team_df["opp_teamid"].isna().sum()
    if n_missing:
        logger.warning("opp_teamid missing for %d rows", n_missing)

    return team_df


def _sort_and_validate(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.sort_values(["date", "gameid", "side"]).reset_index(drop=True)

    # Drop rows where result is unknown
    n_before = len(team_df)
    team_df = team_df.dropna(subset=["result"])
    team_df["result"] = team_df["result"].astype(int)
    n_after = len(team_df)
    if n_before != n_after:
        logger.warning("Dropped %d rows with missing result", n_before - n_after)

    return team_df


def forbidden_cols() -> list[str]:
    """Return the list of columns that must never be used as model features."""
    return list(_FORBIDDEN_FEATURE_COLS)
