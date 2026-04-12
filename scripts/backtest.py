"""
Comprehensive backtest of the pre-draft LoL market-making strategy
using historical Polymarket game outcomes and Elo-implied market prices.

Prerequisites
-------------
1. cache/poly_historical_markets.csv  — run scripts/fetch_poly_history.py
2. cache/model_bundle*.pkl            — run lol_predraft_research.ipynb
3. cache/feat_df_for_inference.pkl    — same notebook

Market price proxy
------------------
Polymarket's public API does not archive price history for resolved markets.
We use `elo_expected` from feat_df as the market price proxy — the Elo-model's
pre-match win probability. This is a meaningful comparator because:
  - Elo is the dominant prior used by sharp bettors and prediction markets.
  - "Edge" in this backtest = how much our rich feature model beats Elo alone.
  - Any positive edge over Elo translates to real bankroll growth if Elo ≈ market.

Usage
-----
    python -m scripts.backtest [options]

    --min-volume FLOAT    Minimum Polymarket market volume (default 50)
    --kelly FLOAT         Fractional Kelly multiplier (default 0.25)
    --half-spread FLOAT   Half-spread for quoting (default 0.015)
    --bankroll FLOAT      Starting bankroll in USDC (default 10000)
    --output PATH         Output CSV path (default outputs/backtest_results.csv)
    --no-charts           Skip matplotlib charts

Methodology notes
-----------------
- Rolling features in feat_df are pre-game (no feature lookahead).
- Model coefficients trained on all data — standard frozen-model backtest bias.
- H2H rates use full history (not truncated to pre-match) — minor caveat.
"""

import sys
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import CACHE_DIR, NODIFF_COLS, NOSYM_OPP_COLS, H2H_K
from pipeline.features import get_feature_cols
from pipeline.betting import QuoteParams, market_maker_quotes, taker_fee_per_share

# Import helpers from predict_match without triggering its CLI
from scripts.predict_match import build_inference_row, compute_h2h

HIST_CSV    = CACHE_DIR / "poly_historical_markets.csv"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ─── Team name normalisation ──────────────────────────────────────────────────
# Keys are lowercased Polymarket names; values are OE teamname strings.
_ALIASES: dict[str, str] = {
    # LCK
    "t1":                     "T1",
    "gen.g":                  "Gen.G",
    "geng":                   "Gen.G",
    "kt rolster":             "KT Rolster",
    "kt":                     "KT Rolster",
    "dk":                     "Dplus KIA",
    "dplus kia":              "Dplus KIA",
    "damwon kia":             "Dplus KIA",
    "nongshim":               "Nongshim RedForce",
    "nongshim redforce":      "Nongshim RedForce",
    "ns redforce":            "Nongshim RedForce",
    "hanwha":                 "Hanwha Life Esports",
    "hanwha life":            "Hanwha Life Esports",
    "hanwha life esports":    "Hanwha Life Esports",
    "hle":                    "Hanwha Life Esports",
    "fredit brion":           "Brion",
    "ok brion":               "Brion",
    "brion":                  "Brion",
    "drx":                    "DRX",
    "kwangdong freecs":       "Kwangdong Freecs",
    "kwangdong":              "Kwangdong Freecs",
    "liiv sandbox":           "Liiv SANDBOX",
    "liiv":                   "Liiv SANDBOX",
    "focus":                  "Focus",
    "bnk fearx":              "BNK FearX",
    "fearx":                  "BNK FearX",
    # LPL
    "jdg":                    "JD Gaming",
    "jd gaming":              "JD Gaming",
    "blg":                    "Bilibili Gaming",
    "bilibili gaming":        "Bilibili Gaming",
    "weibo gaming":           "Weibo Gaming",
    "weibo":                  "Weibo Gaming",
    "wbg":                    "Weibo Gaming",
    "lng":                    "LNG Esports",
    "lng esports":            "LNG Esports",
    "edg":                    "Edward Gaming",
    "edward gaming":          "Edward Gaming",
    "top esports":            "Top Esports",
    "tes":                    "Top Esports",
    "te":                     "Top Esports",
    "ig":                     "Invictus Gaming",
    "invictus gaming":        "Invictus Gaming",
    "al":                     "Anyone's Legend",
    "anyone's legend":        "Anyone's Legend",
    "nip":                    "NIP",
    "ninjas in pyjamas":      "NIP",
    "fpx":                    "FunPlus Phoenix",
    "funplus phoenix":        "FunPlus Phoenix",
    "rng":                    "Royal Never Give Up",
    "royal never give up":    "Royal Never Give Up",
    "ra":                     "Royal Academy",
    "omg":                    "Oh My God",
    "oh my god":              "Oh My God",
    "v5":                     "Ninjas in Pyjamas (LPL)",
    "ultra prime":            "Ultra Prime",
    "up":                     "Ultra Prime",
    "woa":                    "Wolves",
    # LEC
    "g2 esports":             "G2 Esports",
    "g2":                     "G2 Esports",
    "fnatic":                 "Fnatic",
    "fnc":                    "Fnatic",
    "team vitality":          "Team Vitality",
    "vitality":               "Team Vitality",
    "vit":                    "Team Vitality",
    "mad lions":              "MAD Lions",
    "mad":                    "MAD Lions",
    "astralis":               "Astralis",
    "koi":                    "KOI",
    "rogue":                  "Rogue",
    "excel esports":          "Excel Esports",
    "excel":                  "Excel Esports",
    "xls":                    "Excel Esports",
    "sk gaming":              "SK Gaming",
    "sk":                     "SK Gaming",
    "karmine corp":           "Karmine Corp",
    "kc":                     "Karmine Corp",
    "team bds":               "Team BDS",
    "bds":                    "Team BDS",
    "natus vincere":          "Natus Vincere",
    "navi":                   "Natus Vincere",
    "gentle mates":           "Gentle Mates",
    "gm":                     "Gentle Mates",
    "heretics":               "Team Heretics",
    "team heretics":          "Team Heretics",
    # LCS
    "cloud9":                 "Cloud9",
    "c9":                     "Cloud9",
    "team liquid":            "Team Liquid",
    "tl":                     "Team Liquid",
    "100 thieves":            "100 Thieves",
    "100t":                   "100 Thieves",
    "flyquest":               "FlyQuest",
    "fly":                    "FlyQuest",
    "evil geniuses":          "Evil Geniuses",
    "eg":                     "Evil Geniuses",
    "nrg":                    "NRG",
    "dignitas":               "Dignitas",
    "dig":                    "Dignitas",
    "immortals":              "Immortals",
    "imt":                    "Immortals",
    "optic gaming":           "Optic Gaming",
    "optic":                  "Optic Gaming",
    "shopify rebellion":      "Shopify Rebellion",
    "shopify":                "Shopify Rebellion",
    "sr":                     "Shopify Rebellion",
    "golden guardians":       "Golden Guardians",
    "gg":                     "Golden Guardians",
    "tsm":                    "TSM",
    "clg":                    "Counter Logic Gaming",
    "counter logic gaming":   "Counter Logic Gaming",
    "misfits":                "Misfits Gaming",
    "misfits gaming":         "Misfits Gaming",
    # International
    "psg talon":              "PSG Talon",
    "cfg":                    "Chiefs",
    "chiefs":                 "Chiefs",
    "loud":                   "LOUD",
    "pain gaming":            "paiN Gaming",
    "pain":                   "paiN Gaming",
    "fluxo":                  "Fluxo",
    "isurus":                 "Isurus",
}


def _normalize_team(name: str, known_teams: set[str]) -> str | None:
    """
    Resolve a Polymarket team name to an Oracle's Elixir teamname.

    Steps:
    1. Alias table lookup (handles common abbreviations and alternate spellings)
    2. Exact case-insensitive match against known OE teams
    3. Substring match (one name is contained within the other)
    """
    raw = name.strip()
    raw_l = raw.lower()

    # 1. Alias table
    if alias := _ALIASES.get(raw_l):
        if alias in known_teams:
            return alias
        # alias may differ in spacing/capitalisation — try case-insensitive
        for t in known_teams:
            if t.lower() == alias.lower():
                return t

    # 2. Exact case-insensitive
    for t in known_teams:
        if t.lower() == raw_l:
            return t

    # 3. Substring (e.g. "Kwangdong" in "Kwangdong Freecs")
    for t in known_teams:
        t_l = t.lower()
        if raw_l in t_l or t_l in raw_l:
            return t

    return None


# ─── Historical prediction ────────────────────────────────────────────────────

def _team_row_at(
    feat_df: pd.DataFrame,
    team_name: str,
    cutoff: pd.Timestamp,
) -> pd.Series | None:
    """
    Return the team's most-recent feat_df row with date <= cutoff.
    Falls back to the earliest available row if no pre-cutoff data exists.
    """
    mask = feat_df["teamname"].str.lower() == team_name.lower()
    rows = feat_df[mask]
    if rows.empty:
        return None
    before = rows[rows["date"] <= cutoff]
    if before.empty:
        return rows.sort_values("date").iloc[0]  # best available
    return before.sort_values("date").iloc[-1]


def _predict_game(
    team1: str,
    team2: str,
    feat_df: pd.DataFrame,
    bundle: dict,
    cutoff: pd.Timestamp,
) -> float | None:
    """
    Predict P(team1 wins) using rolling features as of `cutoff`.

    Averages Blue- and Red-side predictions to remove side assignment
    uncertainty (same approach as predict_match.py).
    """
    r1 = _team_row_at(feat_df, team1, cutoff)
    r2 = _team_row_at(feat_df, team2, cutoff)
    if r1 is None or r2 is None:
        return None

    pre      = bundle["preprocessor"]
    clf      = bundle["classifier"]
    all_cols = bundle["num_cols"] + bundle["cat_cols"]

    feat_cols = [
        c for c in get_feature_cols(feat_df)
        if not c.startswith("opp_") and not c.startswith("diff_")
    ]

    # H2H (full history — minor caveat documented in module docstring)
    h2h_1v2, h2h_n = compute_h2h(
        feat_df,
        str(r1.get("teamid", "")),
        str(r2.get("teamid", "")),
        float(r1.get("wr_L10", 0.5)),
    )
    h2h_2v1, _ = compute_h2h(
        feat_df,
        str(r2.get("teamid", "")),
        str(r1.get("teamid", "")),
        float(r2.get("wr_L10", 0.5)),
    )

    # Context: infer from the team row
    days_since = max(0.0, (cutoff - r1["date"]).total_seconds() / 86400)
    context = {
        "league":   str(r1.get("league",   "LCK")),
        "split":    str(r1.get("split",    "Spring")),
        "playoffs": int(r1.get("playoffs", 0)),
        "patch":    str(r1.get("patch",    "14")),
        "game":     1,
    }

    probs = []
    for side in ["Blue", "Red"]:
        ctx = {**context, "side": side}
        inf_row = build_inference_row(r1, r2, feat_cols, ctx, h2h_1v2, h2h_2v1, h2h_n)
        # Override days_since_last: build_inference_row uses Timestamp.now(),
        # but for historical games we want the actual gap to the match date.
        inf_row["days_since_last"] = days_since

        X = pd.DataFrame([inf_row])
        for col in all_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[all_cols]
        probs.append(float(clf.predict_proba(pre.transform(X))[0, 1]))

    return round((probs[0] + probs[1]) / 2.0, 4)


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def _print_calibration(sim_df: pd.DataFrame) -> None:
    """Print a text calibration table (model prob bucket → actual win rate)."""
    df = sim_df.copy()
    df["prob_bucket"] = pd.cut(
        df["p_model"],
        bins=[0, .15, .25, .35, .45, .55, .65, .75, .85, 1.0],
        labels=["<15%","15-25%","25-35%","35-45%","45-55%","55-65%","65-75%","75-85%",">85%"],
    )
    cal = df.groupby("prob_bucket", observed=True).agg(
        n=("result", "count"),
        actual_wr=("result", "mean"),
        avg_p_model=("p_model", "mean"),
        avg_p_market=("p_market", "mean"),
    ).round(3)

    print(f"\n  Calibration (model prob → actual win rate vs Elo baseline):")
    print(f"  {'Bucket':<9}  {'N':>4}  {'Model':>7}  {'Elo':>7}  {'Actual':>7}  {'Model Err':>9}  {'Elo Err':>8}")
    print(f"  {'-'*59}")
    for bucket, row in cal.iterrows():
        if row["n"] == 0:
            continue
        model_err  = row["avg_p_model"]  - row["actual_wr"]
        market_err = row["avg_p_market"] - row["actual_wr"]
        print(
            f"  {str(bucket):<9}  {row['n']:>4}  "
            f"{row['avg_p_model']:>7.3f}  {row['avg_p_market']:>7.3f}  "
            f"{row['actual_wr']:>7.3f}  {model_err:>+9.3f}  {market_err:>+8.3f}"
        )


def _print_edge_analysis(sim_df: pd.DataFrame) -> None:
    """Print P&L and accuracy broken down by |edge| bucket."""
    sim_df = sim_df.copy()
    sim_df["edge_abs"]   = sim_df["edge"].abs()
    bins   = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 1.0]
    labels = ["0-2%", "2-5%", "5-10%", "10-15%", "15-20%", ">20%"]
    sim_df["edge_bucket"] = pd.cut(sim_df["edge_abs"], bins=bins, labels=labels)

    tab = sim_df.groupby("edge_bucket", observed=True).agg(
        n=("edge_abs", "count"),
        dir_bets=(    "bet_direction", lambda x: (x != "PASS").sum()),
        model_acc=(   "model_correct", "mean"),
        dir_pnl_sum=( "directional_pnl", "sum"),
    ).round(4)

    print(f"\n  P&L by |edge| bucket:")
    print(f"  {'Bucket':<8}  {'N':>5}  {'Dir bets':>8}  {'Model acc':>9}  {'P&L':>9}")
    print(f"  {'-'*50}")
    for bucket, r in tab.iterrows():
        print(
            f"  {str(bucket):<8}  {r['n']:>5}  {r['dir_bets']:>8}  "
            f"{r['model_acc']:>9.1%}  ${r['dir_pnl_sum']:>+8.2f}"
        )


def _try_save_charts(sim_df: pd.DataFrame) -> None:
    """Generate and save equity curve + calibration scatter (optional)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("LoL Market-Making Backtest", fontweight="bold")

        # ── 1. Equity curve ──
        ax = axes[0]
        ax.plot(sim_df.index, sim_df["cumulative_pnl"], lw=1.5, color="steelblue")
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_title("Cumulative P&L (USDC)")
        ax.set_xlabel("Game #")
        ax.set_ylabel("USDC")
        ax.grid(True, alpha=0.3)

        # ── 2. Calibration scatter ──
        ax = axes[1]
        bins = np.arange(0, 1.05, 0.1)
        labels = np.arange(0.05, 1.0, 0.1)
        sim_df2 = sim_df.copy()
        sim_df2["pb"] = pd.cut(sim_df2["p_model"], bins=bins, labels=labels)
        cal = sim_df2.groupby("pb", observed=True).agg(
            n=("result", "count"), wr=("result", "mean"), avg_p=("p_model", "mean")
        ).dropna()
        ax.scatter(cal["avg_p"], cal["wr"], s=cal["n"] * 3, color="steelblue", alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        ax.set_title("Model Calibration")
        ax.set_xlabel("Predicted P(team1)")
        ax.set_ylabel("Actual win rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── 3. Edge distribution ──
        ax = axes[2]
        ax.hist(sim_df["edge"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", lw=1, linestyle="--")
        ax.set_title("Edge Distribution (model − market)")
        ax.set_xlabel("Edge")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = OUTPUTS_DIR / "backtest_charts.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Charts saved → {chart_path}")

    except ImportError:
        print("  (matplotlib not available — skipping charts)")
    except Exception as e:
        print(f"  (chart save failed: {e})")


# ─── Main backtest ────────────────────────────────────────────────────────────

def run_backtest(
    min_volume:   float = 50.0,
    min_pts:      int   = 0,       # unused — kept for CLI compat
    price_col:    str   = "elo",   # unused explicitly; elo_expected is always used
    kelly_frac:   float = 0.25,
    half_spread:  float = 0.015,
    bankroll:     float = 10000.0,
    output_path:  Path | None = None,
    save_charts:  bool  = True,
) -> pd.DataFrame:
    """
    Full backtest pipeline: load data → predict → simulate → analyse.
    """
    # ── 1. Load historical market data ────────────────────────────────────────
    if not HIST_CSV.exists():
        print(f"ERROR: {HIST_CSV} not found.")
        print(f"Run first:  python -m scripts.fetch_poly_history")
        sys.exit(1)

    print("=" * 60)
    print("  LoL Market-Making Backtest")
    print("=" * 60)

    raw = pd.read_csv(HIST_CSV, dtype={"yes_token_id": str})
    print(f"\nRaw markets loaded:        {len(raw)}")
    print(f"  result known (resolved): {raw['result'].notna().sum()}")

    df = raw.copy()
    df = df[df["result"].notna()].copy()
    df = df[df["volume"] >= min_volume].copy()
    df["result"]   = df["result"].astype(int)
    df["end_dt"]   = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    df             = df[df["end_dt"].notna()].copy()
    df             = df.sort_values("end_dt").reset_index(drop=True)

    print(f"\nAfter filters (vol≥{min_volume}, result known):")
    print(f"  Markets:  {len(df)}")
    if len(df) == 0:
        print("Nothing to backtest. Run fetch_poly_history.py --force to rebuild the dataset.")
        return pd.DataFrame()

    # ── 2. Load model bundle ──────────────────────────────────────────────────
    print("\nLoading model …")
    bundle_candidates = sorted(CACHE_DIR.glob("model_bundle*.pkl"))
    if not bundle_candidates:
        print("ERROR: No model bundle in cache/. Run the training notebook first.")
        sys.exit(1)

    # Prefer model_bundle_xgboost > model_bundle > anything else
    def _priority(p: Path) -> int:
        n = p.stem
        if "xgboost" in n:    return 0
        if n == "model_bundle": return 1
        return 2

    bundle_candidates.sort(key=_priority)
    with open(bundle_candidates[0], "rb") as f:
        bundle = pickle.load(f)
    print(f"  Bundle: {bundle_candidates[0].name}  (model: {bundle.get('model_name','?')})")

    print("Loading feat_df …")
    feat_df = pd.read_pickle(CACHE_DIR / "feat_df_for_inference.pkl")
    print(f"  feat_df shape: {feat_df.shape}")
    known_teams = {t for t in feat_df["teamname"].unique() if isinstance(t, str)}

    # ── 3. Normalise team names ───────────────────────────────────────────────
    print("\nNormalising team names …")
    df["oe_team1"] = df["team1"].apply(lambda n: _normalize_team(n, known_teams))
    df["oe_team2"] = df["team2"].apply(lambda n: _normalize_team(n, known_teams))

    unmatched_mask = df["oe_team1"].isna() | df["oe_team2"].isna()
    if unmatched_mask.any():
        unk = set(
            df.loc[df["oe_team1"].isna(), "team1"].tolist() +
            df.loc[df["oe_team2"].isna(), "team2"].tolist()
        )
        print(f"  Unmatched teams (dropping): {sorted(unk)}")

    df = df[~unmatched_mask].copy()
    print(f"  Matched: {len(df)} / {len(raw[raw['result'].notna()])}")

    if len(df) == 0:
        print("No matched teams. Add entries to _ALIASES in backtest.py.")
        return pd.DataFrame()

    # Year/date coverage
    df["year"] = df["end_dt"].dt.year
    print(f"  Year range: {df['year'].min()} – {df['year'].max()}")
    print("  Per year:   " + "  ".join(
        f"{y}: {n}" for y, n in df.groupby("year").size().items()
    ))

    # ── 4. Generate model predictions + Elo-proxy market prices ─────────────
    print("\nGenerating model predictions and Elo market proxies …")
    predictions:  list[float | None] = []
    elo_proxies:  list[float | None] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        cutoff = row["end_dt"]
        r1     = _team_row_at(feat_df, row["oe_team1"], cutoff)
        r2     = _team_row_at(feat_df, row["oe_team2"], cutoff)

        # Elo-implied win probability = market proxy
        p_elo = float(r1.get("elo_expected", 0.5)) if r1 is not None else None
        elo_proxies.append(p_elo)

        # Full model prediction
        p_model = _predict_game(row["oe_team1"], row["oe_team2"], feat_df, bundle, cutoff)
        predictions.append(p_model)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(df)}")

    df["p_model"] = predictions
    df["p_elo"]   = elo_proxies
    n_nopred = df["p_model"].isna().sum()
    if n_nopred:
        print(f"  Skipped (team data missing): {n_nopred}")
    df = df[df["p_model"].notna() & df["p_elo"].notna()].copy()
    print(f"  Final dataset: {len(df)} rows")

    if len(df) == 0:
        return pd.DataFrame()

    # ── 5. Merge and simulate ─────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    p_model  = df["p_model"].astype(float)
    # Market proxy = Elo-implied probability (see module docstring for rationale)
    p_market = df["p_elo"].astype(float).clip(0.02, 0.98)
    results  = df["result"].astype(int)

    print(f"\n  Market proxy (Elo-implied) stats:")
    print(f"    mean={p_market.mean():.3f}  std={p_market.std():.3f}  "
          f"min={p_market.min():.3f}  max={p_market.max():.3f}")

    params = QuoteParams(
        half_spread=half_spread,
        min_edge=0.02,
        kelly_frac=kelly_frac,
    )

    # Inline directional-only simulation using fixed Kelly (no compounding).
    # Stake = bankroll × kelly_size on every bet, regardless of prior outcomes.
    records  = []
    cum_pnl  = 0.0

    for idx in df.index:
        p_m   = float(p_model.loc[idx])
        p_mk  = float(p_market.loc[idx])
        res   = int(results.loc[idx])

        q     = market_maker_quotes(p_m, p_mk, params)
        fee   = taker_fee_per_share(p_mk)

        dir_pnl = 0.0
        stake   = 0.0

        if q.bet_direction != "PASS" and q.kelly_size > 0:
            stake = bankroll * q.kelly_size   # fixed: always use initial bankroll
            if q.bet_direction == "YES":
                dir_pnl = stake * (res - p_mk) - stake * fee
            else:
                dir_pnl = stake * ((1 - res) - (1 - p_mk)) - stake * fee

        cum_pnl += dir_pnl

        records.append({
            "p_model":         round(p_m,  4),
            "p_market":        round(p_mk, 4),
            "edge":            round(q.edge, 4),
            "taker_fee":       round(fee, 5),
            "bet_direction":   q.bet_direction,
            "kelly_size":      q.kelly_size,
            "stake":           round(stake, 2),
            "result":          res,
            "directional_pnl": round(dir_pnl, 4),
            "total_pnl":       round(dir_pnl, 4),
            "cumulative_pnl":  round(cum_pnl, 4),
        })

    sim_df = pd.DataFrame(records)
    if sim_df.empty:
        print("No bets placed — edge threshold too high or no predictions provided.")
        return sim_df

    # ── 6. Enrich simulation results ─────────────────────────────────────────
    sim_df["question"]       = df["question"].values
    sim_df["team1"]          = df["team1"].values
    sim_df["team2"]          = df["team2"].values
    sim_df["end_date"]       = df["end_date"].values
    sim_df["volume"]         = df["volume"].values
    sim_df["year"]           = df["year"].values
    sim_df["p_market"]       = p_market.values   # = p_elo

    sim_df["model_correct"]  = (
        (sim_df["p_model"] > 0.5).astype(int) == sim_df["result"]
    ).astype(int)
    sim_df["elo_correct"] = (
        (sim_df["p_market"] > 0.5).astype(int) == sim_df["result"]
    ).astype(int)

    # ── 7. Print full analysis ────────────────────────────────────────────────
    model_acc  = sim_df["model_correct"].mean()
    elo_acc    = sim_df["elo_correct"].mean()
    n_total    = len(sim_df)
    n_bets     = (sim_df["bet_direction"] != "PASS").sum()
    edge_mean  = sim_df["edge"].abs().mean()
    edge_std   = sim_df["edge"].std()
    bets_df    = sim_df[sim_df["bet_direction"] != "PASS"]
    win_rate   = (bets_df["directional_pnl"] > 0).mean() if len(bets_df) > 0 else 0.0
    total_dir  = sim_df["directional_pnl"].sum()
    avg_stake  = bets_df["stake"].mean() if len(bets_df) > 0 else 0.0
    roi        = total_dir / bankroll if bankroll > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Directional Betting Summary  (fixed Kelly, bankroll=${bankroll:.0f})")
    print(f"{'='*60}")
    print(f"  Games processed:     {n_total:>6}")
    print(f"  Bets placed:         {n_bets:>6}  ({n_bets/n_total:.0%} of games)")
    print(f"  Avg stake:          ${avg_stake:>7.2f}")
    print(f"  Win rate (bets):     {win_rate:>7.1%}")
    print(f"  Total P&L:          ${total_dir:>+8.2f}  ({roi:>+.1%} of bankroll)")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"  Accuracy & Edge  (market proxy = elo_expected)")
    print(f"{'='*60}")
    print(f"  Total games:              {n_total:>6}")
    print(f"  Directional bets:         {n_bets:>6}  ({n_bets/n_total:.0%})")
    print(f"  Model accuracy:           {model_acc:>7.1%}")
    print(f"  Elo accuracy (baseline):  {elo_acc:>7.1%}")
    print(f"  Model lift over Elo:      {model_acc - elo_acc:>+7.1%}")
    print(f"  Mean |edge| (model−elo):  {edge_mean:>7.1%}")
    print(f"  Edge std:                 {edge_std:>7.1%}")

    _print_calibration(sim_df)
    _print_edge_analysis(sim_df)

    # By year
    yr_tab = sim_df.groupby("year").agg(
        n=("directional_pnl", "count"),
        dir_pnl=("directional_pnl", "sum"),
        model_acc=("model_correct", "mean"),
    ).round(3)
    print(f"\n  P&L by year:")
    print(f"  {'Year':<6}  {'N':>4}  {'P&L':>9}  {'Model acc':>10}")
    print(f"  {'-'*34}")
    for yr, r in yr_tab.iterrows():
        print(f"  {int(yr):<6}  {r['n']:>4}  ${r['dir_pnl']:>+8.2f}  {r['model_acc']:>10.1%}")

    # Top winning and losing games
    bets = sim_df[sim_df["bet_direction"] != "PASS"].copy()
    if len(bets) >= 5:
        print(f"\n  Top 5 directional wins:")
        top = bets.nlargest(5, "directional_pnl")[
            ["team1","team2","p_model","p_market","edge","stake","directional_pnl","result"]
        ]
        for _, r in top.iterrows():
            print(
                f"    {r['team1']:<18} vs {r['team2']:<18}  "
                f"model={r['p_model']:.3f}  mkt={r['p_market']:.3f}  "
                f"edge={r['edge']:+.3f}  pnl=${r['directional_pnl']:+.2f}"
            )
        print(f"\n  Top 5 directional losses:")
        bot = bets.nsmallest(5, "directional_pnl")[
            ["team1","team2","p_model","p_market","edge","stake","directional_pnl","result"]
        ]
        for _, r in bot.iterrows():
            print(
                f"    {r['team1']:<18} vs {r['team2']:<18}  "
                f"model={r['p_model']:.3f}  mkt={r['p_market']:.3f}  "
                f"edge={r['edge']:+.3f}  pnl=${r['directional_pnl']:+.2f}"
            )

    # ── 8. Save results ───────────────────────────────────────────────────────
    if output_path is None:
        output_path = OUTPUTS_DIR / "backtest_results.csv"

    save_cols = [
        "question", "team1", "team2", "end_date", "year",
        "p_model", "p_market", "edge", "result",
        "bet_direction", "kelly_size", "stake",
        "directional_pnl", "total_pnl", "cumulative_pnl",
        "model_correct", "elo_correct", "volume",
    ]
    sim_df[[c for c in save_cols if c in sim_df.columns]].to_csv(
        output_path, index=False
    )
    print(f"\n  Results CSV → {output_path}")

    if save_charts:
        _try_save_charts(sim_df)

    return sim_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest LoL Polymarket market-making strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--min-volume",    type=float, default=50.0,
                        help="Min market volume to include")
    parser.add_argument("--kelly",         type=float, default=0.25,
                        help="Fractional Kelly multiplier")
    parser.add_argument("--half-spread",   type=float, default=0.015,
                        help="Market-making half-spread")
    parser.add_argument("--bankroll",      type=float, default=10000.0,
                        help="Starting bankroll in USDC")
    parser.add_argument("--output",        type=str,   default=None,
                        help="Output CSV path")
    parser.add_argument("--no-charts",     action="store_true",
                        help="Skip matplotlib chart generation")
    args = parser.parse_args()

    run_backtest(
        min_volume   = args.min_volume,
        kelly_frac   = args.kelly,
        half_spread  = args.half_spread,
        bankroll     = args.bankroll,
        output_path  = Path(args.output) if args.output else None,
        save_charts  = not args.no_charts,
    )
