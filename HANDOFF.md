# Agent Handoff: LoL Polymarket Market-Making System

> Written after a session that completed a comprehensive quantitative research audit + implementation sprint.
> Read this before touching anything else.

---

## What This Project Is

A League of Legends match prediction + Polymarket market-making system. Pre-draft model predicts P(team1 wins). Market-making layer posts two-sided limit orders (BUY YES token + BUY NO token) on Polymarket's CLOB, collecting maker rebates and liquidity rewards.

**Stack:** Python 3.11, scikit-learn, XGBoost, Streamlit, py-clob-client, Oracle's Elixir data.

---

## Repository Map

```
lol_model/
├── pipeline/
│   ├── config.py            # All constants, feature flags, paths — start here
│   ├── betting.py           # Fee math, fair value blend, quote engine, position tracking
│   ├── models.py            # Model registry, CV runner, ensemble functions
│   ├── features.py          # Rolling feature engineering (Groups B-K, P)
│   ├── elo.py               # Elo rating system with decay
│   ├── validation.py        # Rolling CV splits + holdout split
│   ├── evaluation.py        # Holdout eval, calibration curves, ablation study
│   ├── tuning.py            # Hyperparameter search (RandomizedSearch over rolling folds)
│   ├── roster_features.py   # Group R: roster stability features (recently enabled)
│   └── dataset.py           # Assembles team_df + player_df into model-ready dataset
├── scripts/
│   ├── predict_match.py     # Inference: load bundle → build feature row → predict
│   ├── polymarket_client.py # Polymarket CLOB API wrapper (auth, orderbook, posting)
│   └── run_maker.py         # CLI end-to-end workflow (predict → quote → post → monitor)
├── app.py                   # Streamlit terminal UI (main interactive tool)
├── lol_predraft_research.ipynb  # Training notebook — run this to rebuild model bundles
├── cache/
│   ├── model_bundle_{model}.pkl    # Per-model trained bundles (xgboost, ridge_lr, etc.)
│   ├── feat_df_for_inference.pkl   # Feature dataframe for team lookups at inference time
│   └── best_params.json            # Cached tuned hyperparameters
└── data/OE Public Match Data/      # Oracle's Elixir CSV files (2014-2026)
```

---

## Current Model Performance

| Model | Holdout Log-Loss | Notes |
|-------|-----------------|-------|
| XGBoost | **0.6290** | Best; uses nonlinear interactions |
| Ridge LR | 0.6332 | Most stable linear model |
| Lasso LR | 0.6331 | Similar to Ridge |
| ElasticNet | 0.6330 | Redundant with Ridge |
| Logistic (unreg.) | 0.6334 | Weakest; consider dropping |
| Baseline (predict 0.5) | 0.6931 | Random guess |

**Feature ablation highlights:**
- Elo alone: 0.6409
- + form (A+B): 0.6366
- + early-game (A+B+C): 0.6343
- Full feature set: ~0.6290

Realistic ceiling for pre-draft prediction: ~0.615-0.620. About 0.01 log-loss left to find.

---

## What Was Implemented in the Last Session

### pipeline/betting.py
- **`PositionState` dataclass** — tracks YES/NO shares, avg cost, capital deployed; has `update_fill()`, `inventory_skew()`, `net_exposure()` methods
- **`compute_half_spread(model_std, hours_to_match, edge_abs)`** — dynamic spread (base 1.5¢, widens near match and with model disagreement)
- **`compute_alpha(model_std, n_games_min, hours_to_match)`** — model confidence weight in [0.30, 0.80], default 0.65
- **`compute_fair_value(p_model, p_market, ...)`** — Bayesian blend: `p_fair = α × p_model + (1-α) × p_market`, returns `(p_fair, alpha_used)`
- **`market_maker_quotes()`** — now uses blended fair value (not raw model) as quote center; new params: `model_std`, `n_games_min`, `hours_to_match`, `blend_alpha`
- **`Quote` dataclass** — added `p_model` (raw) and `alpha` (blend weight) fields
- **Reward score bug fixed** — was passing `size=0` which always returned 0; now passes `size=1`
- **`QuoteParams.half_spread` default** — changed 0.03 → 0.015 (at 0.03 = max qualifying spread → reward score was literally zero)

### pipeline/models.py
- **`compute_ensemble_weights(cv_scores)`** — inverse-log-loss weights summing to 1
- **`ensemble_predict(model_probs, weights)`** — returns `(p_ensemble, model_std)`
- **XGBoost grid expanded**: `n_estimators` up to 1500, `learning_rate` down to 0.005, `reg_lambda` down to 0.1, `colsample_bytree` down to 0.5, new `gamma` [0-1.0], `max_depth` up to 7

### pipeline/config.py
- `INCLUDE_ROSTER_FEATURES = True` (was False — free signal, infrastructure already existed)

### scripts/run_maker.py
- **`monitor_fills()` function** — polls every 30s, detects fills, cancels remaining side if market moves >2¢ against filled position (adverse selection defense)
- **`--monitor` / `--monitor-interval` CLI flags**
- **Defaults changed**: `--half-spread 0.015` (was 0.03), `--size 30` (was 100)
- Quote summary now shows `p_fair (blended)` and `α`

### app.py
- Quote builder: `p_model_raw` → `compute_fair_value()` → `p_fair` (blended)
- Model std from ensemble stored in `st.session_state["_model_std"]`, passed to `compute_fair_value()`
- Consensus row → Ensemble row showing σ (model disagreement)
- Defaults: half-spread 2¢, quote size 30

---

## What Still Needs to Be Done

### 1. RETRAIN — Most Important (run the notebook)

The `cache/` bundles were saved **before** roster features were enabled and before the XGBoost grid was expanded. The app loads stale bundles.

**Steps:**
1. Open `lol_predraft_research.ipynb`
2. Verify `dataset.py` passes `player_df` to `compute_roster_features()` — check if this call is already wired or needs adding
3. Run all cells through the bundle-saving section
4. Set `force_retune=True` on the XGBoost tuning cell so it re-runs with the expanded grid
5. Verify new holdout log-loss (should improve ~0.001-0.003 from roster features + better XGB params)

**Key function to check in dataset.py:**
```python
# Should look something like this — verify it's present:
from pipeline.roster_features import compute_roster_features
team_df = compute_roster_features(team_df, player_df)
```

### 2. Compute Proper Ensemble Weights

`ensemble_predict()` currently uses **equal weights** because proper weights require CV scores. After retraining:

```python
from pipeline.models import compute_ensemble_weights, summarise_cv
import json

cv_summary = summarise_cv(cv_df)
cv_scores = cv_summary["log_loss"]["mean"].to_dict()
# cv_scores = {"xgboost": 0.6290, "ridge_lr": 0.6332, ...}

weights = compute_ensemble_weights(cv_scores)
# Save to cache
with open("cache/ensemble_weights.json", "w") as f:
    json.dump(weights, f)
```

Then load in `app.py` and `run_maker.py`:
```python
import json
from pathlib import Path
weights_path = CACHE_DIR / "ensemble_weights.json"
ensemble_weights = json.load(open(weights_path)) if weights_path.exists() else None
p_ens, model_std = ensemble_predict(model_probs, weights=ensemble_weights)
```

### 3. Wire `hours_to_match` Through

`compute_fair_value()` and `compute_half_spread()` accept `hours_to_match` but neither caller passes it — defaulting to 168h (7 days). The market end date is available on the `LoLMarket` object.

**In `run_maker.py`** (after fetching market):
```python
from datetime import datetime, timezone
end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
hours_to_match = max(0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
# Then pass to market_maker_quotes:
quote = market_maker_quotes(p_model, p_market, params, hours_to_match=hours_to_match)
```

**In `app.py`** — same pattern using `sel_market.end_date`.

### 4. Wire `model_std` Into `run_maker.py`

The CLI only runs one model bundle (`load_bundle()` returns one). It should load all bundles for ensemble:

```python
# In run_maker.py, replace single-bundle load with:
import pickle, glob
all_bundles = {}
for path in (ROOT / "cache").glob("model_bundle_*.pkl"):
    key = path.stem.replace("model_bundle_", "")
    all_bundles[key] = pickle.load(open(path, "rb"))

# After predicting with each bundle:
all_preds = {k: predict_match(args.team1, args.team2, context, b, feat_df, verbose=False)["p_model"]
             for k, b in all_bundles.items()}
from pipeline.models import ensemble_predict
p_ensemble, model_std = ensemble_predict(all_preds)
# Use p_ensemble instead of p_model, pass model_std to market_maker_quotes
```

### 5. Wire `PositionState` Into Post Flow

`PositionState` is defined in `betting.py` and imported in `run_maker.py`, but never instantiated. After posting, create one and update it in `monitor_fills()`:

```python
position = PositionState()
# When bid fills:
position.update_fill("YES", args.size, t1_price)
# When ask fills:
position.update_fill("NO", args.size, t2_price)
# Compute skew for re-quoting:
skew = position.inventory_skew(p_fair, max_position=bankroll*0.15, half_spread=args.half_spread)
```

### 6. Minor: Drop Redundant Models (Optional)

ElasticNet and unregularized Logistic are within 0.004 log-loss of Ridge and add compute time. Consider removing from `MODEL_REGISTRY` in `models.py`. A 2-model ensemble (XGBoost + Ridge) is likely as good.

---

## Critical System Facts (Read Before Touching Orders)

### Polymarket Order Posting
- Posts **two BUY orders**: BUY YES token + BUY NO token (NOT buy+sell)
- YES token pays $1 if team1 wins; NO token pays $1 if team2 wins
- Both tokens together always pay $1 → guaranteed profit if both fill
- Minimum order size: **5 shares** (Polymarket hard limit)
- Orders must be **below current asks** to rest as makers (clamping logic in `run_maker.py` and `app.py`)
- `POLY_SIGNATURE_TYPE=1` in `.env` — Magic/email login; **do not change to 2** (causes 400 errors)
- `POLY_FUNDER` must match the wallet address shown in Polymarket profile settings

### .env Required Keys
```
POLY_API_KEY=...
POLY_API_SECRET=...
POLY_PASSPHRASE=...
POLY_FUNDER=0x...        # wallet address — must match Polymarket profile
POLY_SIGNATURE_TYPE=1    # Magic/email; NOT 2
```

### Fee Structure
- **Maker**: zero fees + rebate of `0.25 × 0.03 × p × (1-p)` per share
- **Taker**: fee of `0.03 × p × (1-p)` per share (~75 bps at p=0.50)
- **Liquidity rewards**: $1,550/game pre-period for LCK/LPL/LEC/LCS tier AB
  - Quadratic score: `((max_spread - spread) / max_spread)² × size`
  - Max qualifying spread: **3 cents from market mid**
  - At half_spread=1.5¢: score = 25 per share — **this is the main income source**

### Bankroll Context
- Current bankroll: ~$200
- Quote size: 30 shares per side (recently reduced from 100)
- At p=0.50: 30 shares × $0.50 = $15 per side = 7.5% of bankroll per fill
- Max tolerable loss per market: ~$30 (15% of bankroll)

---

## Running the System

**Streamlit app (main UI):**
```bash
streamlit run app.py
```

**CLI dry run:**
```bash
python -m scripts.run_maker \
    --team1 "T1" --team2 "Gen.G" \
    --league LCK --split Spring --patch 25.S1.3 \
    --condition-id 0xABC123...
```

**CLI live run with fill monitoring:**
```bash
python -m scripts.run_maker \
    --team1 "T1" --team2 "Gen.G" \
    --league LCK --split Spring --patch 25.S1.3 \
    --condition-id 0xABC123... \
    --live --monitor
```

**Cancel all orders before match starts:**
```bash
python -m scripts.run_maker --cancel --condition-id 0xABC123...
```

**List active LoL markets:**
```bash
python -m scripts.run_maker --list-markets --query "T1"
```

---

## Prioritized Next Steps

| Priority | Task | Effort | File |
|----------|------|--------|------|
| 🔴 Critical | Retrain with roster features + expanded XGB grid | Run notebook | `lol_predraft_research.ipynb` |
| 🔴 Critical | Compute + save ensemble weights from CV scores | 30 min | `models.py` + `cache/` |
| 🟡 High | Wire `hours_to_match` from market end date | 30 min | `run_maker.py`, `app.py` |
| 🟡 High | Wire `model_std` from multi-bundle ensemble into CLI | 1 hr | `run_maker.py` |
| 🟡 High | Wire `PositionState` into posting + monitoring flow | 1 hr | `run_maker.py` |
| 🟢 Medium | Wire `compute_half_spread()` into quote params | 30 min | `run_maker.py`, `app.py` |
| 🟢 Medium | Add realistic backtest with fill probability model | 4 hr | `betting.py` new function |
| 🟢 Medium | Add ECE calibration monitoring to evaluation | 2 hr | `evaluation.py` |
| ⚪ Low | Drop ElasticNet + Logistic from registry | 15 min | `models.py` |
| ⚪ Low | Add streak features (win/loss streaks) | 1 hr | `features.py` Group B |
