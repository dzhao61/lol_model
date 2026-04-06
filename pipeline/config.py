"""
Global configuration for the pre-draft LoL prediction pipeline.
All paths, constants, and knobs live here.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "OE Public Match Data"
CACHE_DIR  = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs"

# ─── Data ─────────────────────────────────────────────────────────────────────
YEARS = list(range(2014, 2027))

# Column renaming: OE has column names with spaces
COL_RENAME = {
    "team kpm":   "team_kpm",
    "earned gpm": "earned_gpm",
    "total cs":   "total_cs",
}

# ─── Elo ──────────────────────────────────────────────────────────────────────
ELO_INIT = 1500.0
ELO_K    = 20.0
ELO_D    = 400.0   # logistic scale factor

# ─── Rolling windows ──────────────────────────────────────────────────────────
WIN_SHORT     = 5
WIN_LONG      = 10
EWMA_ALPHA    = 0.3
LEAGUE_WINDOW = 90   # days for league-relative Elo

# ─── Shrinkage ────────────────────────────────────────────────────────────────
CONV_K  = 10   # conversion / closing features
PATCH_K = 5    # patch adaptation features
H2H_K   = 5   # H2H shrinkage — most pairs meet only 4–6 times, so stay conservative

# ─── Validation ───────────────────────────────────────────────────────────────
HOLDOUT_FRAC  = 0.20
TRAIN_DAYS    = 180
VAL_DAYS      = 14
STRIDE_DAYS   = 14

# How many of the most-recent folds to use for hyperparameter tuning.
# Using all ~280 folds is slow; recent folds are more relevant anyway.
TUNE_LAST_N_FOLDS = 30

# ─── Models ───────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_TUNE_ITER   = 100      # RandomizedSearchCV iterations per model

# ─── Feature groups to include ────────────────────────────────────────────────
INCLUDE_PATCH_FEATURES   = True
INCLUDE_STYLE_FEATURES   = True   # Group F – ablate to test if they add signal
INCLUDE_INTERACTIONS     = True   # Group K – explicit pairwise interaction terms
INCLUDE_ROSTER_FEATURES  = False  # Tier 3 – needs player rows; off by default

# ─── Elo time-decay ───────────────────────────────────────────────────────────
# Decayed Elo: before each match, Elo regresses toward 1500 with this half-life.
# Teams that haven't played recently move toward average.
# Set to None to disable (use standard Elo only).
ELO_DECAY_HALFLIFE = 365.0   # days; None = no decay

# ─── Metadata columns (never used as model features) ─────────────────────────
META_COLS = [
    "gameid", "date", "year", "teamid", "teamname", "opp_teamid",
    "league", "split", "playoffs", "patch", "game", "side",
    "result", "datacompleteness",
]

# ─── Categorical features for one-hot encoding ────────────────────────────────
CAT_COLS = ["league", "split", "side", "patch"]
# playoffs is binary 0/1 – keep numeric

# ─── Target ───────────────────────────────────────────────────────────────────
TARGET = "result"

# ─── Raw OE columns used ONLY as input to rolling feature computation ─────────
# These must never enter the model feature matrix directly.
RAW_INPUT_COLS = [
    # identifiers / admin
    "url", "participantid", "position", "playername", "playerid",
    # teamname kept — needed for display/lookup; not a model feature (not in FEATURE_PREFIXES)
    # draft (forbidden as features)
    "firstPick", "champion",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5",
    # Elo-derived intermediate columns (contain current-match result → leakage)
    "elo_overperf",   # = result - elo_expected; only rolled version (overperf_L5/L10) is safe
    # elo_expected is fine as a feature (pure function of pre-match Elos, no leakage)

    # same-match stats used only to build rolling features
    "golddiffat10", "xpdiffat10", "csdiffat10",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "golddiffat20", "xpdiffat20", "csdiffat20",
    "golddiffat25", "xpdiffat25", "csdiffat25",
    "firstblood", "firstdragon", "firstherald", "firstbaron", "firsttower",
    "firstmidtower", "firsttothreetowers",
    "dragons", "opp_dragons", "heralds", "opp_heralds",
    "barons", "opp_barons", "towers", "opp_towers",
    "turretplates", "opp_turretplates",
    "void_grubs", "opp_void_grubs", "atakhans", "opp_atakhans",
    "ckpm", "team_kpm", "vspm", "earned_gpm",
    "gspd", "gpr", "dpm", "damageshare", "damagetakenperminute",
    "visionscore", "controlwardsbought",
    "gamelength",
    # columns that only appear in player rows (will be NaN on team rows)
    "kills", "deaths", "assists", "teamkills", "teamdeaths",
    "doublekills", "triplekills", "quadrakills", "pentakills",
    "firstbloodkill", "firstbloodassist", "firstbloodvictim",
    "totalgold", "earnedgold", "earned_gpm", "earnedgoldshare",
    "goldspent", "total_cs", "minionkills", "monsterkills",
    "monsterkillsownjungle", "monsterkillsenemyjungle", "cspm",
    "wardsplaced", "wpm", "wardskilled", "wcpm",
    "damagetochampions", "damagemitigatedperminute", "damagetotowers",
    "goldat10", "xpat10", "csat10", "opp_goldat10", "opp_xpat10", "opp_csat10",
    "goldat15", "xpat15", "csat15", "opp_goldat15", "opp_xpat15", "opp_csat15",
    "goldat20", "xpat20", "csat20", "opp_goldat20", "opp_xpat20", "opp_csat20",
    "goldat25", "xpat25", "csat25", "opp_goldat25", "opp_xpat25", "opp_csat25",
    "killsat10", "assistsat10", "deathsat10", "opp_killsat10", "opp_assistsat10", "opp_deathsat10",
    "killsat15", "assistsat15", "deathsat15", "opp_killsat15", "opp_assistsat15", "opp_deathsat15",
    "killsat20", "assistsat20", "deathsat20", "opp_killsat20", "opp_assistsat20", "opp_deathsat20",
    "killsat25", "assistsat25", "deathsat25", "opp_killsat25", "opp_assistsat25", "opp_deathsat25",
    "inhibitors", "opp_inhibitors",
    "elementaldrakes", "opp_elementaldrakes",
    "infernals", "mountains", "clouds", "oceans", "chemtechs", "hextechs",
    "dragons (type unknown)", "elders", "opp_elders",
    "totalgold", "earnedgold", "earnedgoldshare", "goldspent",
]

# Prefixes that identify ENGINEERED feature columns (built by features.py)
FEATURE_PREFIXES = (
    # Group A (Elo – standard + decayed)
    "elo_",
    # Group B (form)
    "wr_", "ewma_", "avg_opp_elo_", "overperf_",
    "form_trend",        # new: wr_L5 - wr_L10 (hot/cooling indicator)
    "n_games_played",    # new: total prior matches (reliability weight)
    # Group C (early-game – means, stds, rates, thresholds)
    "gdiff", "xpdiff", "csdiff",
    "pct_pos_", "pct_gdiff15_", "pct_gdiff10_",
    "fb_rate", "fd_rate", "fh_rate", "ft_rate", "fbar_rate",
    # Group D (conversion – standard + conditional)
    "lead_conv_rate", "comeback_rate",
    "strong_lead_conv", "avg_gdiff15_when_",
    # Group E (objectives)
    "dragon_share", "herald_share", "baron_share", "tower_share", "plate_share",
    "vgrubs_diff", "atakhan_diff",
    # Group F (style)
    "pace_mean", "vision_mean", "economy_mean", "aggression_mean", "gamelength_mean",
    # Group G (side)
    "is_blue", "blue_wr_hist", "red_wr_hist", "same_side_wr_hist",
    "patch_blue_wr", "side_wr_vs_patch",
    # Group H (patch)
    "patch_wr_shrunk", "patch_gdiff15_shrunk", "patch_dragon_share_shrunk",
    # Group J (schedule + context)
    "days_since_last", "matches_last_", "is_playoffs", "game_number",
    "league_tier",       # new: ordinal competitive tier (1=T1, 2=T2, 3=regional)
    # Group K (interactions)
    "ix_",
    # Group P (head-to-head history)
    "h2h_wr", "h2h_n_games",
    # Group R (roster stability)
    "roster_overlap", "games_since_roster_change",
    # Group L (champion meta — draft model only)
    "champ_meta_",
    # Group M (champion comfort — draft model only)
    "champ_comfort_",
    # Group N (series context — draft model only)
    "series_game_num", "series_score_diff", "series_must_win",
    # First pick (draft model only)
    "has_first_pick",
)

# ─── Feature redundancy controls ─────────────────────────────────────────────
# Columns where diff_ = team - opp is mathematically redundant or zero-sum:
#   - Already-differential features (e.g. elo_diff = team_elo - opp_elo, so
#     diff_elo_diff = elo_diff - (-elo_diff) = 2*elo_diff: perfectly correlated)
#   - Symmetric match-level features identical for both teams
NODIFF_COLS = {
    # Elo differentials (already encode team vs opp gap)
    "elo_diff", "elo_expected", "elo_decayed_diff",
    # Elo individual ratings: diff_elo_team = elo_team - opp_elo_team = elo_diff (redundant)
    # diff_elo_opp = elo_opp - opp_elo_opp = -elo_diff (redundant)
    "elo_team", "elo_opp", "elo_decayed_team", "elo_decayed_opp",
    # Symmetric: both teams share the same value
    "is_playoffs", "game_number", "patch_blue_wr",
    # is_blue: diff_is_blue ≈ -2*(is_blue - 0.5), nearly redundant with is_blue itself
    "is_blue",
    # Series: score_diff is zero-sum (opp = -team), diff_ = 2× team value
    "series_score_diff",
    # H2H: h2h_n_games is the same from both sides of the same match
    "h2h_n_games",
    # League tier: both teams in same league → diff always 0
    "league_tier",
}

# opp_ versions to drop for symmetric match-level columns (both teams have same value)
# Also includes columns where opp_ is perfectly (anti-)correlated with an existing feature.
NOSYM_OPP_COLS = {
    "is_playoffs", "game_number", "patch_blue_wr",
    # Both teams are in the same game number of the series
    "series_game_num",
    # Both teams in the same league tier
    "league_tier",
    # opp_elo_diff = -elo_diff (perfectly anti-correlated → redundant)
    # opp_elo_decayed_diff similarly redundant
    "elo_diff", "elo_decayed_diff",
}
