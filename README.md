# LoL Market Maker

A pre-draft prediction model and market-making terminal for LoL esports matches on Polymarket.

## What it does

- Loads 13 years of Oracle's Elixir professional match data
- Engineers 100+ rolling features (Elo, form, H2H, objectives, style, draft) with strict leakage controls
- Trains an ensemble of 5 models (XGBoost + Ridge/Lasso/ElasticNet/Logistic)
- Surfaces live Polymarket bid/ask prices and model-vs-market edges in a Streamlit terminal
- Computes symmetric maker quotes and Kelly-sized directional bets, and posts orders to the CLOB

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create `.env`**
```
POLY_PRIVATE_KEY=0x...
POLY_FUNDER=0x...
```

**3. Ensure the model cache exists**

Run `lol_predraft_research.ipynb` through to the end (cell 31) to build all `.pkl` artefacts in `cache/`. This is a one-time step; the cache persists across sessions.

---

## Running the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Workflow
1. Select **League** and **Match** from the top bar
2. Prices auto-load from the Polymarket orderbook (live, 60s cache)
3. Press **Run all models** to generate predictions
4. Review the **Model Signals** table — per-model series probability and edge vs market
5. Use the **Quote Builder** to configure model, series format, and Kelly fraction
6. Press **Post** to submit maker quotes to Polymarket

---

## CLI (headless)

Predict and quote without the UI:

```bash
# Dry run — shows quotes, posts nothing
python -m scripts.run_maker \
  --team1 "T1" --team2 "Gen.G" \
  --league LCK --split Spring --patch 25.S1.3 \
  --condition-id 0xABC123... \
  --bankroll 200 --size 100

# Live — posts real orders
python -m scripts.run_maker ... --live

# Cancel all open orders for a market
python -m scripts.run_maker --cancel --condition-id 0xABC123...

# List active LCK markets
python -m scripts.run_maker --list-markets --query "LCK"
```

---

## Project structure

```
app.py                      Streamlit terminal UI
pipeline/
  config.py                 Global constants and feature flags
  load_data.py              Load and clean Oracle's Elixir CSVs
  elo.py                    Chronological Elo ratings (leakage-safe)
  features.py               Rolling feature engineering (groups B–K)
  draft_features.py         Draft-phase features (groups L–N)
  roster_features.py        Roster change detection
  dataset.py                Final dataset assembly + diff columns
  models.py                 Model registry and preprocessing pipelines
  validation.py             Time-based rolling cross-validation
  tuning.py                 Hyperparameter search
  evaluation.py             Holdout evaluation and plots
  betting.py                Polymarket fee math and Kelly sizing
scripts/
  predict_match.py          Match win probability inference
  polymarket_client.py      Polymarket CLOB API wrapper
  run_maker.py              CLI market-making workflow
  test_connection.py        Polymarket connectivity check
lol_predraft_research.ipynb Research notebook and model training
data/                       Oracle's Elixir CSVs (2014–2026)
cache/                      Computed artefacts (.pkl) — gitignored
outputs/                    Evaluation charts and results CSVs
```
