# LoL Polymarket Trading Guide

Complete end-to-end workflow for operating the market-making system — from finding a match to post-game closeout.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pre-Trade Checklist](#pre-trade-checklist)
3. [Step 1 — Find the Market](#step-1--find-the-market)
4. [Step 2 — Read the Signals](#step-2--read-the-signals)
5. [Step 3 — Configure Quotes](#step-3--configure-quotes)
6. [Step 4 — Risk Check Before Posting](#step-4--risk-check-before-posting)
7. [Step 5 — Post Orders](#step-5--post-orders)
8. [Step 6 — Monitor Active Orders](#step-6--monitor-active-orders)
9. [Step 7 — Manage Fills and Exposure](#step-7--manage-fills-and-exposure)
10. [Step 8 — Close Before Match Start](#step-8--close-before-match-start)
11. [Step 9 — Post-Match Settlement](#step-9--post-match-settlement)
12. [Risk Management Reference](#risk-management-reference)
13. [Key Numbers to Track](#key-numbers-to-track)
14. [Common Scenarios and Responses](#common-scenarios-and-responses)
15. [Quick Reference — App vs CLI](#quick-reference--app-vs-cli)

---

## System Overview

You are a **maker-only liquidity provider** on Polymarket's prediction market CLOB. You post two resting BUY orders — one for each team's outcome token — and earn money three ways:

| Income source | When | Approximate size |
|---|---|---|
| **Spread capture** | Both sides fill | `(1 − t1_price − t2_price) × shares` |
| **Maker rebate** | Either side filled by a taker | `0.25 × 0.03 × p × (1−p)` per share |
| **Liquidity rewards** | Orders rest within 3¢ of mid | Up to $1,550/game (at 10% market share) |

**You never need token inventory.** Both orders are funded by USDC collateral already in your Polymarket wallet. If both fill, one token always pays $1 (the winner), so your maximum payout equals your combined cost plus the spread.

---

## Pre-Trade Checklist

Run through this before every trade:

- [ ] Match is **pre-draft** (models are pre-draft only — no live-game data)
- [ ] Match is **at least 2 hours away** (dynamic spread widens dangerously inside 2h)
- [ ] Both teams have **≥ 20 games** in training data (warning shown in app)
- [ ] Neither team has been **inactive > 14 days** (stale roster/form signal)
- [ ] USDC balance ≥ `(t1_price + t2_price) × shares` (enough to fund both orders)
- [ ] No open orders on this market from a previous session (cancel before re-quoting)
- [ ] Polymarket market is **active, not closed or resolved**

---

## Step 1 — Find the Market

### App (preferred)
The Streamlit app auto-loads active LoL markets from the Gamma API. Select from the dropdown at the top of the left panel. Team names are parsed from the question text and matched to Oracle's Elixir names.

### CLI
```bash
python -m scripts.run_maker --list-markets
python -m scripts.run_maker --list-markets --query "T1"
```

### What to note
- `condition_id` — the unique market identifier; needed for all CLI commands
- `end_date` — the market expiry (usually match start time); `hours_to_match` is computed from this
- YES token = team listed first in the question; NO token = team listed second

> **Team name mismatch?** Use the "Override team / context" expander in the app to manually map the Polymarket team name to the Oracle's Elixir name used in training data.

---

## Step 2 — Read the Signals

### Model signals panel (app right column)

| Field | What it means |
|---|---|
| **p1 / p2** | Per-model win probability for team 1 / team 2 |
| **Edge YES / Edge NO** | `p_model − p_market` for each side. Positive = model thinks that side is underpriced |
| **Ensemble row** | Inverse-log-loss weighted average across all 5 models |
| **σ (sigma)** | Standard deviation of predictions across models. High σ = disagreement = less confidence |

### Interpreting the signals

| Situation | Meaning | Action |
|---|---|---|
| All models agree, σ < 0.02 | High confidence | Can quote tighter; Rewards mode viable |
| Models split, σ > 0.05 | Genuine uncertainty | Widen spread; use dynamic spread |
| Edge > 5¢ | Model and market disagree significantly | Consider Directional mode |
| Edge < 1¢ | No real edge | Pure Rewards mode; don't skew |
| n_games < 20 warning | Small sample size | Be cautious, widen manually |
| Last played > 14 days | Stale form data | Reduce size; model may be outdated |

### Fair value blend
The quote center is not your raw model output — it's a Bayesian blend:

```
p_fair = α × p_model + (1 − α) × p_market
```

`α` is computed dynamically (`pipeline/betting.py:compute_alpha`):
- Default ≈ 0.65 (trust model 65%, market 35%)
- Shrinks toward 0.30 when σ is high (less model confidence)
- Shrinks toward 0.30 as match approaches (market becomes more informed)
- Never goes above 0.80

Blended fair value is shown as "Blended fair value (α=XX%)" in the stats strip beneath the order cards.

---

## Step 3 — Configure Quotes

### Quote mode (most important setting)

**Rewards mode** — center quotes on market mid
- Both orders sit within `half_spread` of market mid
- Both Q-scores are nonzero → full Q_min two-sided rewards
- Best when: edge < 3¢, you want passive income from liquidity rewards
- Risk: if your model is right and market is wrong, you're leaving directional edge uncaptured

**Directional mode** — center quotes on blended fair value
- Orders shift toward your model's view of fair value
- Q-scores may drop to zero if fair value is > 3¢ from market mid
- Best when: edge > 4–5¢, you want to express a view
- Risk: potential rewards disqualification; one side may be far from the book

### Settings to adjust

| Setting | Typical value | When to change |
|---|---|---|
| **Half-spread** | 1.5–2¢ | Widen to 2.5–3¢ near match (< 12h); tighten to 1¢ for tight, liquid markets |
| **Shares / side** | 25–30 | Reduce to 10–15 for uncertain markets; never exceed your USDC / 2 |
| **Kelly frac** | 0.25 | Increase to 0.35 only if very high confidence (σ < 0.01, edge > 5¢) |
| **Quote mode** | Rewards | Switch to Directional only when edge > 4¢ |
| **Prob type** | Per-game | Use Series only if the Polymarket question is "Who wins the series?" |

### Dynamic spread (automatic)
The CLI and `compute_half_spread` (`pipeline/betting.py:167`) automatically widens your base spread:

| Condition | Extra spread added |
|---|---|
| σ > 0.02 (model disagreement) | Up to +1¢ |
| Match < 24h away | +0.5¢ |
| Match < 12h away | +1¢ |
| Match < 2h away | +2¢ |

This is computed automatically. In the app, the UI shows the resulting dynamic spread in the "Half-spread" output field.

### Scenario analysis panel
Before posting, always read the Scenario Analysis table beneath the order cards:

| Row | Best case | Model EV | Worst case |
|---|---|---|---|
| **Both fill** | locked-in spread | locked-in spread | locked-in spread |
| **Only T1 fills** | +profit if T1 wins | weighted EV | −cost if T1 loses |
| **Only T2 fills** | +profit if T2 wins | weighted EV | −cost if T2 loses |
| **Neither fills** | $0 | $0 | $0 |

**Q-scores**: Check that both Q YES and Q NO are green (> 0). A red Q-score means that order is outside the 3¢ qualifying window and earns zero liquidity rewards. In Rewards mode both should always be green. In Directional mode one or both may be red.

---

## Step 4 — Risk Check Before Posting

Calculate your exposure before hitting Post:

```
Max single-side loss = token_price × shares
  e.g. 48¢ × 30 = $14.40

Combined capital reserved = (t1_price + t2_price) × shares
  e.g. (0.48 + 0.48) × 30 = $28.80

As % of bankroll ($200) = 14.4%
```

### Hard limits (respect these)

| Rule | Why |
|---|---|
| Single-market max loss ≤ 15% of bankroll ($30) | Avoid ruin from a single bad fill |
| Total open orders ≤ 40% of bankroll ($80) | Keep capital buffer for other markets |
| Never post if USDC balance < combined cost | Order will reject; wastes latency |
| Never post inside 30 min of match start | Cannot cancel in time if something goes wrong |

### Edge sanity check
Before posting in Directional mode, verify:
- **Net taker edge > 0**: `|p_model − p_market| − taker_fee`. If this is negative, there's no edge even if you take aggressively.
- **Edge ≥ min_edge (2¢)**: Below this, the Kelly size is zero (PASS). No reason to use Directional mode.

---

## Step 5 — Post Orders

### App
Click the **🚀 Post** button. The app calls `post_two_sided_quote` which submits both BUY orders atomically in a single `post_orders` batch API call (`scripts/polymarket_client.py:post_two_sided_quote`).

You'll see:
- Bid order ID (YES token BUY)
- Ask order ID (NO token BUY)
- Active order summary bar with prices and age timer

### CLI
```bash
python -m scripts.run_maker \
  --team1 "T1" --team2 "Gen.G" \
  --league LCK --split Spring --patch 25.S1.3 \
  --condition-id 0xABC123... \
  --live
```
Add `--monitor` to start the fill-monitoring loop after posting.

### Price clamping (automatic)
Both the app and CLI automatically clamp your posted prices to sit strictly below the current best ask for each token:
- YES bid is clamped to `min(your_bid, current_YES_ask − 1¢)`
- NO bid is clamped to `min(your_no_bid, (1 − current_YES_bid) − 1¢)`

This ensures your orders REST in the book as makers, not immediately cross as takers. If clamping fires, the app shows a yellow warning box.

---

## Step 6 — Monitor Active Orders

### What to watch post-posting

| Signal | Check every | How |
|---|---|---|
| Orders still resting | 5–10 min | App "Active" order bar; or CLI open orders |
| Market mid movement | 5–10 min | App refresh; watch `p_market` drift |
| Fill notifications | Immediate | Polymarket web/app notifications; CLI `--monitor` |
| Hours remaining | Each check | Shown in app stats strip; auto-refreshes |

### CLI fill monitor
```bash
python -m scripts.run_maker ... --live --monitor --monitor-interval 30
```
Polls every 30 seconds. If one side fills and the market moves > 2¢ against you, it automatically cancels the remaining order (adverse selection defense). Runs for 30 minutes by default (60 × 30s).

### Normal behavior
- Orders rest for hours, possibly the entire pre-match period.
- Low fill rate is expected and fine — liquidity rewards accrue on resting orders.
- You do not need both sides to fill to profit; rewards pay on each hour of qualifying liquidity.

---

## Step 7 — Manage Fills and Exposure

### If only one side fills

This is the main risk scenario. You now hold a directional position.

**Calculate your exposure:**
```
If YES (T1 token) fills at t1_price:
  Profit if T1 wins  = (1 − t1_price) × shares
  Loss if T1 loses   = t1_price × shares
  Net EV             = p_model × profit − (1−p_model) × loss
```

**Response options:**

| Situation | Action |
|---|---|
| EV still positive (model favors your filled side) | Hold. Let it ride. |
| EV negative (market moved against you) | Cancel remaining order; consider closing the position if you can |
| Market mid moved > 3¢ against your fill | Strong adverse selection signal; cancel remaining order immediately |
| Both sides still open after 12h | Normal; do nothing, keep monitoring |

**The fill monitor handles this automatically** with a 2¢ threshold. You can adjust `--monitor-interval` and the 2¢ threshold is hardcoded in `scripts/run_maker.py:monitor_fills`.

### If both sides fill

You hold YES + NO tokens for the same market. The payout is exactly $1 regardless of outcome — one always wins. Your profit is locked in:
```
Profit = 1.00 − t1_price − t2_price
e.g.  = 1.00 − 0.48 − 0.48 = $0.04  per share
Total = $0.04 × 30 shares = $1.20
```
No further action needed. Tokens resolve automatically after the match.

### Re-quoting after a fill

If one side fills and you want to restore two-sided exposure, you can post a new order on the filled side at the current price. Be careful:
- New market price may be different from your original quote
- Recalculate the new combined spread to confirm it's still < $1.00
- If the market moved significantly, your model may no longer justify the trade

---

## Step 8 — Close Before Match Start

**Always cancel all open orders before the match begins.**

This is critical. Once a match starts, the outcome is being determined, but Polymarket may still accept orders if the market hasn't been paused. You don't want to fill at pre-draft prices after the game has started.

### Set a reminder
Check `hours_to_match` in the app (shown in the stats strip). Cancel when it hits ~30 minutes.

### Cancel all orders

**App**: Click **❌ Cancel all** button.

**CLI**:
```bash
python -m scripts.run_maker --cancel --condition-id 0xABC123...
```

**Verify**: After cancelling, check that no open orders remain (check Polymarket app or use `get_open_orders`).

### GTD alternative (auto-expire)
The CLI supports `OrderType.GTD` with an expiration timestamp if you want orders to auto-expire at a set time. This is not yet wired into the app UI but is available in the Python client. Useful for setting-and-forgetting.

---

## Step 9 — Post-Match Settlement

You do **not** need to do anything after the match ends. Polymarket automatically resolves markets.

- If you hold a token from a partial fill, it will either pay $1 (if your team won) or expire worthless (if they lost).
- If both sides filled (locked spread), both tokens resolve and your $1 payout is automatically credited.
- Resolved USDC returns to your Polymarket USDC balance and can be withdrawn anytime.

---

## Risk Management Reference

### Position sizing formula
```
Max stake per market = min(bankroll × 0.15, $30)

Shares to post = max_stake / token_price
  e.g. $30 / $0.48 = 62.5  → round down to 60

Default (conservative): 30 shares @ ~50¢ = $15 per side = 7.5% bankroll
```

### Kelly sizing (for directional bets)
Used automatically in CLI/app when `|edge| ≥ min_edge (2¢)`:
```
Kelly fraction = edge / (1 − p_market)   (simplified)
Fractional Kelly = Kelly × 0.25          (default kelly_frac)
Stake = bankroll × fractional_kelly
```
```
Example: p_model=0.60, p_market=0.50, kelly_frac=0.25
Kelly = 0.10 / 0.50 = 0.20
Fractional = 0.20 × 0.25 = 0.05  → 5% of bankroll = $10
```

### Bankroll management rules

| Rule | Value | Reasoning |
|---|---|---|
| Single trade max loss | 15% ($30) | 7 consecutive losses to ruin |
| Total reserved capital | ≤ 40% ($80) | Keep 60% liquid for opportunities |
| Stop trading if bankroll < | $100 | Below this, reward potential < risk |
| Reduce size when bankroll falls | < $150 | Drop to 15 shares / side |

### Adverse selection risk
You are posting maker orders — sophisticated bettors ("informed takers") may pick you off when they have better information (roster changes, injury news, bootcamp results). Defenses:
1. **Dynamic spread** automatically widens near match time
2. **Fill monitor** cancels the remaining side if market moves 2¢ against a one-sided fill
3. **Cancel before match** eliminates exposure to any last-minute news
4. **Rewards mode** centers on market mid, not your model — you express no view and are less likely to be adversely selected

---

## Key Numbers to Track

Maintain a simple spreadsheet across trades:

| Column | What to record |
|---|---|
| Date | Match date |
| Market | Team1 vs Team2 |
| p_model | Ensemble prediction |
| p_market | Market mid at time of posting |
| Edge | p_model − p_market |
| Quote mode | Rewards / Directional |
| t1_price | YES buy price posted |
| t2_price | NO buy price posted |
| Shares | Size per side |
| Capital reserved | (t1+t2) × shares |
| Fill outcome | Both / T1 only / T2 only / Neither |
| PnL | Realized profit/loss |
| Rewards earned | Estimated liquidity reward |
| Total | PnL + rewards |

### Session-level metrics to review weekly

| Metric | Target |
|---|---|
| Fill rate — both sides | > 20% of trades |
| Fill rate — one side | < 40% (high one-sided fills = adverse selection) |
| Average spread captured | ≥ 3¢ per both-fill |
| Avg rewards per game | $1–5 (at 10% market share, 2¢ half-spread) |
| Win rate on one-side fills | ≥ 55% (means your edge metric is real) |
| Net PnL per game | Positive after 20+ games |

---

## Common Scenarios and Responses

### "Market mid is very far from my model (> 10¢)"
- Check: is the model picking up something the market hasn't priced in? Or is there a soft data issue (wrong team names, stale data)?
- Verify team names match Oracle's Elixir data via the override expander
- If confident: use Directional mode, size by Kelly, accept rewards disqualification
- If not sure: use Rewards mode at 50% normal size

### "Q-score is zero in the scenario panel"
- You're in Directional mode and your fair value center is > 3¢ from market mid
- Either switch to Rewards mode, or widen spread to close the gap
- In Directional mode with zero Q: you earn spread only, no rewards — make sure your edge justifies this

### "My YES order filled but not NO"
- You now hold a YES token position. Calculate your EV (shown in scenario analysis)
- If market has moved < 2¢: keep the NO order open, wait for fill
- If market has moved > 2¢ against you: cancel the NO order and accept partial exposure
- If next day and still unfilled: re-evaluate and either re-post the NO order at current price, or keep the directional position if still justified

### "Both sides filled immediately after posting"
- Your prices were likely too high (close to the current asks). The orders crossed the book as takers.
- Check for the yellow "price clamped" warning — if present, prices were adjusted down
- If both fill immediately, you still made the spread, but paid taker fees instead of earning maker rebates
- Next time, reduce prices by 1–2¢ from the current bid

### "Match starts in < 2h and I have open orders"
- Cancel immediately via the app ❌ Cancel button or CLI `--cancel`
- Do not re-post — dynamic spread widens +2¢ inside 2h, making rewards negligible
- If you already have a one-sided fill from an earlier session, that position will resolve naturally

### "I missed cancelling and match has started"
- Cancel immediately with `--cancel`
- Any fills after match start are at your model's pre-draft probability — may not reflect actual game state
- Accept whatever position you have and let it resolve

---

## Quick Reference — App vs CLI

| Task | App | CLI |
|---|---|---|
| Find markets | Dropdown (auto-loaded) | `--list-markets` |
| Run models | Auto on market select | `--team1 X --team2 Y` |
| Post orders | 🚀 Post button | `--live` |
| Dry run preview | Always shown before Post | Default (no `--live`) |
| Cancel orders | ❌ Cancel all button | `--cancel --condition-id X` |
| Fill monitoring | Not wired (use CLI) | `--live --monitor` |
| Override teams | Override expander | `--team1` / `--team2` flags |
| Adjust half-spread | Settings panel | `--half-spread 0.015` |
| Adjust size | Settings panel | `--size 30` |
| Series vs game prob | Series/Per-game radio | `--format Bo3` |

---

*Last updated: 2026-04-06. Reflects model v2 (5-model ensemble, roster features, inverse log-loss weights).*
