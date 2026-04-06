"""
Predict win probability for an upcoming LoL match.

Usage:
    python -m scripts.predict_match \
        --team1 "T1" --team2 "Gen.G" \
        --league LCK --split Spring --playoffs 0 \
        --patch 25.S1.3 --game 1

Output:
    p_model: probability that team1 wins (0–1)
    Confidence metadata: n_games for each team, H2H record, etc.
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import CACHE_DIR, NODIFF_COLS, NOSYM_OPP_COLS, H2H_K
from pipeline.features import get_feature_cols


# ─── Load artefacts ───────────────────────────────────────────────────────────

def load_bundle():
    p = CACHE_DIR / "model_bundle.pkl"
    if not p.exists():
        raise FileNotFoundError(
            "cache/model_bundle.pkl not found.\n"
            "Run the notebook through Section 10 first to generate it."
        )
    with open(p, "rb") as f:
        return pickle.load(f)


def load_feat_df():
    p = CACHE_DIR / "feat_df_for_inference.pkl"
    if not p.exists():
        raise FileNotFoundError(
            "cache/feat_df_for_inference.pkl not found.\n"
            "Run the notebook through Section 10 first to generate it."
        )
    return pd.read_pickle(p)


# ─── Team lookup ──────────────────────────────────────────────────────────────

def find_team(feat_df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Find all rows for a team by name (case-insensitive, partial match allowed).
    Returns sorted rows (oldest → newest).
    """
    mask = feat_df["teamname"].str.lower() == name.lower()
    rows = feat_df[mask]

    if len(rows) == 0:
        # Try partial match
        mask = feat_df["teamname"].str.lower().str.contains(name.lower(), regex=False)
        rows = feat_df[mask]

    if len(rows) == 0:
        available = sorted(feat_df["teamname"].unique())
        raise ValueError(
            f"Team '{name}' not found in training data.\n"
            f"Close matches to check:\n"
            + "\n".join(f"  {t}" for t in available if name.lower()[:3] in t.lower())
            + f"\n\nRun list_teams() to see all {len(available)} teams."
        )

    return rows.sort_values(["date", "gameid"])


def list_teams(feat_df: pd.DataFrame, league_filter: str | None = None) -> pd.Series:
    """Print all team names in the training data."""
    teams = feat_df.groupby("teamname").agg(
        n_games=("result", "count"),
        last_game=("date", "max"),
        league=("league", "last"),
    ).sort_values("last_game", ascending=False)

    if league_filter:
        teams = teams[teams["league"].str.lower() == league_filter.lower()]

    return teams


# ─── H2H computation ──────────────────────────────────────────────────────────

def compute_h2h(
    feat_df: pd.DataFrame,
    team_id: str,
    opp_id: str,
    prior_wr: float,
) -> tuple[float, int]:
    """
    Compute shrunk H2H win rate for team_id vs opp_id.
    Shrinks toward prior_wr (team's own recent win rate) with K=H2H_K.
    Returns (h2h_wr, n_prior_meetings).
    """
    h2h = feat_df[
        (feat_df["teamid"] == team_id) &
        (feat_df["opp_teamid"] == opp_id)
    ]
    n    = len(h2h)
    wins = int(h2h["result"].sum()) if n > 0 else 0
    raw  = wins / n if n > 0 else prior_wr
    shrunk = (n / (n + H2H_K)) * raw + (H2H_K / (n + H2H_K)) * prior_wr
    return float(shrunk), n


# ─── Build inference row ──────────────────────────────────────────────────────

def build_inference_row(
    team_row:  pd.Series,
    opp_row:   pd.Series,
    feat_cols: list[str],
    context:   dict,
    h2h_wr:    float,
    opp_h2h_wr: float,
    h2h_n:     int,
) -> pd.Series:
    """
    Build a single dataset-format row for model inference.

    Mirrors the structure of build_dataset() output:
        team feature cols + opp_{col} cols + diff_{col} cols + context
    """
    row = {}

    # ── Team A feature values ──
    for col in feat_cols:
        if col in team_row.index:
            row[col] = team_row[col]
        else:
            row[col] = 0.0

    # ── Override H2H with correct matchup values ──
    row["h2h_wr"]      = h2h_wr
    row["h2h_n_games"] = h2h_n

    # ── opp_ columns from Team B ──
    for col in feat_cols:
        if col in NOSYM_OPP_COLS:
            continue
        opp_col = f"opp_{col}"
        val = opp_row[col] if col in opp_row.index else 0.0
        row[opp_col] = val

    row["opp_h2h_wr"]      = opp_h2h_wr
    row["opp_h2h_n_games"] = h2h_n

    # ── diff_ columns ──
    for col in feat_cols:
        if col in NODIFF_COLS:
            continue
        opp_col  = f"opp_{col}"
        diff_col = f"diff_{col}"
        if opp_col in row:
            row[diff_col] = row[col] - row[opp_col]

    # ── Context features ──
    row["league"]             = context["league"]
    row["split"]              = context["split"]
    row["side"]               = context.get("side", "Blue")
    row["patch"]              = context["patch"]
    row["is_playoffs"]        = int(context.get("playoffs", 0))
    row["game_number"]        = int(context.get("game", 1))
    row["series_game_num"]    = int(context.get("game", 1))
    row["series_score_diff"]  = int(context.get("series_score_diff", 0))
    row["series_must_win"]    = int(context.get("series_must_win", 0))
    row["is_blue"]            = 1 if context.get("side", "Blue") == "Blue" else 0

    # days_since_last: days from team's last game to today
    if "date" in team_row.index and pd.notna(team_row["date"]):
        days = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(team_row["date"]).tz_localize("UTC") if pd.Timestamp(team_row["date"]).tzinfo is None else pd.Timestamp.now(tz="UTC") - pd.Timestamp(team_row["date"])).days
        row["days_since_last"] = float(days)
    else:
        row["days_since_last"] = 7.0   # default 1 week

    return pd.Series(row)


# ─── Main prediction function ─────────────────────────────────────────────────

def predict_match(
    team1:    str,
    team2:    str,
    context:  dict,
    bundle:   dict | None = None,
    feat_df:  pd.DataFrame | None = None,
    verbose:  bool = True,
) -> dict:
    """
    Predict P(team1 wins) for an upcoming match.

    Parameters
    ----------
    team1, team2 : team names (must match Oracle's Elixir teamname column)
    context      : dict with keys: league, split, playoffs, patch, game,
                   optionally: series_score_diff, series_must_win
    bundle       : loaded model bundle (loaded from cache if None)
    feat_df      : feature dataframe (loaded from cache if None)

    Returns
    -------
    dict with p_model, metadata, and per-side predictions
    """
    if bundle is None:
        bundle = load_bundle()
    if feat_df is None:
        feat_df = load_feat_df()

    pre       = bundle["preprocessor"]
    clf       = bundle["classifier"]
    num_cols  = bundle["num_cols"]
    cat_cols  = bundle["cat_cols"]
    all_cols  = num_cols + cat_cols

    # ── Find teams ──
    rows1 = find_team(feat_df, team1)
    rows2 = find_team(feat_df, team2)
    last1 = rows1.iloc[-1]
    last2 = rows2.iloc[-1]

    # ── Feature columns used at training time ──
    # Only use base feature cols (no opp_/diff_ — we build those ourselves)
    feat_cols = [
        c for c in get_feature_cols(feat_df)
        if not c.startswith("opp_") and not c.startswith("diff_")
    ]

    # ── H2H ──
    h2h_1v2, h2h_n = compute_h2h(feat_df, last1["teamid"], last2["teamid"], last1.get("wr_L10", 0.5))
    h2h_2v1, _     = compute_h2h(feat_df, last2["teamid"], last1["teamid"], last2.get("wr_L10", 0.5))

    # ── Average predictions over Blue and Red side assignments ──
    # Side is unknown pre-draft — averaging removes this uncertainty.
    probs_by_side = {}
    for side in ["Blue", "Red"]:
        ctx = {**context, "side": side}
        inf_row = build_inference_row(
            last1, last2, feat_cols, ctx, h2h_1v2, h2h_2v1, h2h_n,
        )

        # Align to model's expected columns (fill missing with 0)
        X = pd.DataFrame([inf_row])
        for col in all_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[all_cols]

        X_t   = pre.transform(X)
        p     = clf.predict_proba(X_t)[0, 1]
        probs_by_side[side] = float(p)

    p_model = (probs_by_side["Blue"] + probs_by_side["Red"]) / 2.0

    result = {
        "team1":          last1.get("teamname", team1),
        "team2":          last2.get("teamname", team2),
        "p_model":        round(p_model, 4),
        "p_blue_side":    round(probs_by_side["Blue"], 4),
        "p_red_side":     round(probs_by_side["Red"],  4),
        "elo_team1":      round(float(last1.get("elo_team", 1500)), 1),
        "elo_team2":      round(float(last2.get("elo_team", 1500)), 1),
        "wr_L10_team1":   round(float(last1.get("wr_L10", 0.5)), 3),
        "wr_L10_team2":   round(float(last2.get("wr_L10", 0.5)), 3),
        "n_games_team1":  int(last1.get("n_games_played", 0)),
        "n_games_team2":  int(last2.get("n_games_played", 0)),
        "h2h_record":     f"{int(h2h_n)} prior meetings",
        "h2h_wr_team1":   round(h2h_1v2, 3),
        "last_game_team1": str(last1.get("date", ""))[:10],
        "last_game_team2": str(last2.get("date", ""))[:10],
        "model":          bundle["model_name"],
        "context":        context,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Pre-draft prediction")
        print(f"{'='*55}")
        print(f"  {result['team1']:<20}  vs  {result['team2']}")
        print(f"  P({result['team1']} wins): {p_model:.1%}  "
              f"(Blue side: {probs_by_side['Blue']:.1%} | Red side: {probs_by_side['Red']:.1%})")
        print(f"")
        print(f"  Elo:          {result['elo_team1']:.0f}  vs  {result['elo_team2']:.0f}")
        print(f"  Form (L10):   {result['wr_L10_team1']:.1%}  vs  {result['wr_L10_team2']:.1%}")
        print(f"  H2H:          {result['h2h_wr_team1']:.1%} win rate ({result['h2h_record']})")
        print(f"  Games in data:{result['n_games_team1']}  vs  {result['n_games_team2']}")
        print(f"  Last game:    {result['last_game_team1']}  vs  {result['last_game_team2']}")
        print(f"  League:       {context.get('league')}  |  Patch: {context.get('patch')}")
        print(f"  Model:        {bundle['model_name']}")

        if result["n_games_team1"] < 20 or result["n_games_team2"] < 20:
            print(f"\n  ⚠ WARNING: One team has fewer than 20 games in training data.")
            print(f"    Prediction relies heavily on Elo — treat with lower confidence.")
        def _days_ago(ts) -> int:
            t = pd.Timestamp(ts)
            now = pd.Timestamp.now(tz="UTC")
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            return (now - t).days

        if _days_ago(last1["date"]) > 14:
            print(f"\n  ⚠ WARNING: {result['team1']} last game was >14 days ago.")
            print(f"    Update your Oracle's Elixir data before trading.")
        if _days_ago(last2["date"]) > 14:
            print(f"\n  ⚠ WARNING: {result['team2']} last game was >14 days ago.")
            print(f"    Update your Oracle's Elixir data before trading.")

        print(f"{'='*55}\n")

    return result


def series_win_prob(p_game: float, fmt: str = "Bo3") -> float:
    """
    Convert per-game win probability to series win probability.

    Bo1:  p_game (trivial)
    Bo3:  p²(3 − 2p)         [first to 2 wins]
    Bo5:  10p³ − 15p⁴ + 6p⁵  [first to 3 wins]
    """
    p = float(p_game)
    if fmt == "Bo1":
        return p
    elif fmt == "Bo3":
        return p**2 * (3 - 2*p)
    elif fmt == "Bo5":
        return 10*p**3 - 15*p**4 + 6*p**5
    else:
        raise ValueError(f"Unknown format: {fmt}. Use Bo1, Bo3, or Bo5.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict LoL match outcome")
    parser.add_argument("--team1",    required=True)
    parser.add_argument("--team2",    required=True)
    parser.add_argument("--league",   default="LCK")
    parser.add_argument("--split",    default="Spring")
    parser.add_argument("--playoffs", type=int, default=0)
    parser.add_argument("--patch",    default="25.S1.3")
    parser.add_argument("--game",     type=int, default=1)
    parser.add_argument("--format",   default="Bo3", help="Bo1/Bo3/Bo5 for series prob")
    parser.add_argument("--list-teams", action="store_true")
    args = parser.parse_args()

    feat_df = load_feat_df()

    if args.list_teams:
        print(list_teams(feat_df).to_string())
        return

    context = {
        "league":   args.league,
        "split":    args.split,
        "playoffs": args.playoffs,
        "patch":    args.patch,
        "game":     args.game,
    }

    bundle = load_bundle()
    result = predict_match(args.team1, args.team2, context, bundle, feat_df)

    if args.format != "Bo1":
        p_series = series_win_prob(result["p_model"], args.format)
        print(f"  Series win prob ({args.format}): {p_series:.1%}")
        result["p_series"] = round(p_series, 4)
        result["series_format"] = args.format


if __name__ == "__main__":
    main()
