"""
Polymarket market-making utilities for pre-draft LoL match prediction.

Context
-------
You are acting as a retail market maker on low-liquidity Polymarket prediction
markets for LoL esports matches that resolve in 1+ week. The markets are binary
(YES/NO tokens priced 0–1). Your model provides a calibrated probability; the
market provides a public price.

How Polymarket works for market makers
---------------------------------------
- **CLOB (Central Limit Order Book)**: post limit orders at any price.
- **Tick sizes**: variable per market (0.1 / 0.01 / 0.001 / 0.0001); 0.01 typical.
- **Fees**: Makers pay ZERO fees. Takers pay a curve-based fee:
      taker_fee_per_share = fee_rate × p × (1 − p)
  For Sports / Esports (LoL), fee_rate = 0.03.
  At p=0.50: fee = 0.0075/share (75 bps). At p=0.30/0.70: fee = 0.0063/share.
- **Maker rebates**: Makers earn 25% of the taker fee when their resting order
  is taken: rebate_per_share = 0.25 × fee_rate × p × (1 − p).
  At p=0.50: rebate = 0.001875/share (~19 bps).
- **Liquidity Rewards**: Separate daily program rewarding tight two-sided
  quoting. For LoL pre-game (A/B tier): $1,550/game distributed pro-rata to
  makers based on a quadratic spread-weighted score. This is often the dominant
  income source for a market maker in low-liquidity LoL markets.
- **Settlement**: YES token pays $1 at resolution, NO token pays $0.
- **Inventory risk**: if filled on one side, you hold a directional position
  for up to 1+ week. Price can move against you (odds update).

Two income streams
------------------
1. **Spread / rebate income**: Post bid/ask → get taken → earn maker rebate.
   EV = rebate_per_share × (fill probability each side).
2. **Liquidity Rewards**: Daily USDC based on Q_min score (tighter quotes +
   larger size = higher score). For A/B tier LoL: $1,550/game pre-period.

Directional betting (taking liquidity)
---------------------------------------
When |p_model − p_market| is large, cross the spread as a taker to capture
directional value. You pay the full taker fee, so you need genuine edge:
    net_edge = |p_model − p_market| − taker_fee_per_share(p_market)
Use fractional Kelly to size.

Market-making strategy
----------------------
1. Compute fair value (model probability p_model).
2. Quote bid = p_model − half_spread, ask = p_model + half_spread.
3. Earn rebate income + liquidity rewards when filled on both sides.
4. When model disagrees with market significantly, skew quotes toward the
   value-positive side (inventory management).
5. Size directional bets via fractional Kelly when taking (not making).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ─── Fee constants (LoL/Esports = Sports category) ────────────────────────────

TICK_SIZE            = 0.01     # typical for LoL markets (verify per market)
MIN_PRICE            = 0.01     # Polymarket price floor
MAX_PRICE            = 0.99     # Polymarket price ceiling

SPORTS_TAKER_RATE    = 0.03     # Sports/Esports taker fee rate
MAKER_REBATE_FRAC    = 0.25     # Makers receive 25% of collected taker fees
MAKER_FEE_RATE       = 0.0      # Makers are NEVER charged fees

# ─── Liquidity Rewards (April 2026 program) ───────────────────────────────────

LOL_PRE_REWARD_TIER_AB = 1550.0  # $ per game, pre-period — LCK, LPL, LEC Playoffs, LCS
LOL_PRE_REWARD_TIER_C  = 150.0   # $ per game, pre-period — ERLs, national leagues
LOL_LIVE_REWARD_TIER_AB = 3950.0 # $ per game, in-play (not our focus)

# Max qualifying spread for liquidity rewards (configured per market; 3 cents typical)
DEFAULT_MAX_REWARD_SPREAD = 0.03


# ─── Core fee functions ───────────────────────────────────────────────────────

def taker_fee_per_share(p: float, rate: float = SPORTS_TAKER_RATE) -> float:
    """
    Taker fee in USDC per share traded at price p.

    Formula: fee = rate × p × (1 − p)
    Symmetric around 0.5: fee at 0.30 equals fee at 0.70.

    Parameters
    ----------
    p    : share price (0–1)
    rate : fee rate (0.03 for Sports/Esports)

    Returns
    -------
    float : USDC fee per share (e.g. 0.0075 at p=0.50 with rate=0.03)
    """
    p = float(np.clip(p, 0.0, 1.0))
    return float(rate * p * (1.0 - p))


def maker_rebate_per_share(p: float, rate: float = SPORTS_TAKER_RATE) -> float:
    """
    Maker rebate in USDC per share when a resting order at price p is taken.
    Equals 25% of the taker fee paid by the other side.

    Parameters
    ----------
    p    : share price
    rate : underlying taker fee rate

    Returns
    -------
    float : USDC rebate per share (e.g. 0.001875 at p=0.50)
    """
    return float(MAKER_REBATE_FRAC * taker_fee_per_share(p, rate))


def compute_edge(p_model: float, p_market: float) -> float:
    """
    Signed model vs market edge (no fee adjustment).
    Positive → model thinks YES is underpriced.
    Negative → model thinks NO is underpriced.
    """
    return float(p_model) - float(p_market)


# ─── Kelly criterion ─────────────────────────────────────────────────────────

def kelly_fraction(
    p_model:    float,
    p_market:   float,
    fractional: float = 0.25,
    as_maker:   bool  = False,
) -> float:
    """
    Fractional Kelly criterion for a Polymarket binary bet.

    Polymarket fees depend on which side you are:
      - **Taker** (crossing the spread): pay fee = rate × p × (1−p) per share.
      - **Maker** (resting limit order filled): pay zero fees, earn rebate.

    Kelly for a YES bet at effective price p_eff:
        b = (1 − p_eff) / p_eff          # net payout per $1 staked
        f* = (b × p_model − (1−p_model)) / b

    Parameters
    ----------
    p_model    : model probability of YES
    p_market   : current market mid price
    fractional : Kelly multiplier (0.25 = quarter-Kelly)
    as_maker   : if True, apply rebate (not fee) to effective price

    Returns
    -------
    float : recommended bankroll fraction (0 if no positive net edge)
    """
    p  = float(p_model)
    q  = 1.0 - p
    pm = float(np.clip(p_market, 1e-6, 1.0 - 1e-6))

    if as_maker:
        # Maker receives a rebate → effective cost is reduced
        adj      = -maker_rebate_per_share(pm)   # negative = reduces cost
    else:
        # Taker pays a fee → effective cost is increased
        adj      = taker_fee_per_share(pm)        # positive = raises cost

    if p > pm:   # bet YES
        # Effective price = what we actually pay per share after fee/rebate
        p_eff = pm + adj
        p_eff = float(np.clip(p_eff, 1e-6, 1.0 - 1e-6))
        b     = (1.0 - p_eff) / p_eff
        edge  = p - p_eff                 # model prob minus effective cost
        f_star = (b * p - q) / b
    else:        # bet NO — buy NO token at (1−pm)
        p_eff_no = (1.0 - pm) + adj      # cost per NO share including fee/rebate
        p_eff_no = float(np.clip(p_eff_no, 1e-6, 1.0 - 1e-6))
        b        = (1.0 - p_eff_no) / p_eff_no
        p_no     = q                      # model prob of NO
        edge     = p_no - p_eff_no
        f_star   = (b * p_no - p) / b

    if edge <= 0.0:
        return 0.0

    f_star = max(0.0, f_star)
    return round(float(fractional * f_star), 4)


# ─── Liquidity Rewards ────────────────────────────────────────────────────────

def order_reward_score(
    spread:     float,
    size:       float,
    max_spread: float = DEFAULT_MAX_REWARD_SPREAD,
    in_game_multiplier: float = 1.0,
) -> float:
    """
    Quadratic score for a single resting order per Polymarket's formula:

        S(v, s) = ((v − s) / v)² × b × size

    where v = max qualifying spread (cents), s = spread from mid, b = multiplier.
    Returns 0 if the order is outside the qualifying spread window.

    Parameters
    ----------
    spread     : distance of this order from the market midpoint (in price units)
    size       : order size in shares
    max_spread : qualifying spread boundary (e.g. 0.03 = 3 cents)
    in_game_multiplier : b factor (default 1.0 for pre-game)

    Returns
    -------
    float : order score (higher = better)
    """
    if spread >= max_spread or size <= 0:
        return 0.0
    return ((max_spread - spread) / max_spread) ** 2 * in_game_multiplier * size


def two_sided_reward_score(
    bid_spread: float,
    bid_size:   float,
    ask_spread: float,
    ask_size:   float,
    max_spread: float = DEFAULT_MAX_REWARD_SPREAD,
    scaling_c:  float = 3.0,
) -> float:
    """
    Q_min score for a two-sided quote per Polymarket's methodology.

    Two-sided quoting is boosted by taking min(Q_one, Q_two).
    Single-sided quoting scores at Q_one_or_two / c (currently c=3).
    Valid when midpoint is in [0.10, 0.90].

    Parameters
    ----------
    bid_spread, bid_size : quote on the buy side
    ask_spread, ask_size : quote on the sell side
    max_spread           : qualifying spread boundary
    scaling_c            : single-side penalty (default 3.0)

    Returns
    -------
    float : Q_min score for this quote set
    """
    q_bid = order_reward_score(bid_spread, bid_size, max_spread)
    q_ask = order_reward_score(ask_spread, ask_size, max_spread)

    # Two-sided: boosted by min; single-sided: penalised by c
    return max(min(q_bid, q_ask), max(q_bid, q_ask) / scaling_c)


def estimate_liquidity_reward(
    bid_spread:     float,
    bid_size:       float,
    ask_spread:     float,
    ask_size:       float,
    market_share:   float = 0.10,
    tier:           str   = "AB",
    max_spread:     float = DEFAULT_MAX_REWARD_SPREAD,
) -> float:
    """
    Estimate expected liquidity reward for one game's pre-period.

    We don't know the total Q_min across all makers (requires live market data),
    so we parameterise by `market_share` — your assumed fraction of total
    qualifying liquidity. A 10% share is conservative for low-liquidity LoL.

    Parameters
    ----------
    bid_spread, bid_size : your resting bid parameters
    ask_spread, ask_size : your resting ask parameters
    market_share         : assumed fraction of total reward pool you capture
    tier                 : "AB" (LCK/LPL/LEC/LCS) or "C" (ERLs)
    max_spread           : qualifying spread boundary

    Returns
    -------
    float : expected USDC reward for this game's pre-period
    """
    pool = LOL_PRE_REWARD_TIER_AB if tier == "AB" else LOL_PRE_REWARD_TIER_C
    q    = two_sided_reward_score(bid_spread, bid_size, ask_spread, ask_size, max_spread)
    # If our score is zero we earn nothing; otherwise scale by assumed share
    if q <= 0:
        return 0.0
    return float(market_share * pool)


# ─── Market-making quote engine ───────────────────────────────────────────────

@dataclass
class QuoteParams:
    """Configuration for market-making quotes."""
    half_spread:      float = 0.03   # distance from mid to each side (3 cents)
    min_edge:         float = 0.02   # minimum |model − market| to post directional intent
    inventory_skew:   float = 0.0    # positive = skew ask up (long), negative = skew bid down
    max_inventory:    float = 0.10   # max bankroll fraction in a single position
    kelly_frac:       float = 0.25   # fractional Kelly for sizing directional bets
    max_reward_spread: float = DEFAULT_MAX_REWARD_SPREAD  # qualifying spread for rewards


@dataclass
class Quote:
    """Output of the market-making quote engine."""
    bid:              float          # price to buy YES
    ask:              float          # price to sell YES
    mid:              float          # model fair value
    market_mid:       float          # observed market price
    edge:             float          # signed model vs market edge
    net_taker_edge:   float          # edge minus taker fee (break-even check)
    bet_direction:    str            # "YES", "NO", or "PASS"
    kelly_size:       float          # recommended position size (taker bet fraction)
    bid_reward_score: float          # liquidity reward score for the bid side
    ask_reward_score: float          # liquidity reward score for the ask side
    rationale:        str            # human-readable explanation


def market_maker_quotes(
    p_model:  float,
    p_market: float,
    params:   QuoteParams | None = None,
) -> Quote:
    """
    Compute bid/ask quotes for a Polymarket YES/NO market.

    The model acts as the source of fair value. Quotes are snapped to tick size.
    When the model's edge vs market is large, quotes are skewed to accumulate
    inventory on the value side.

    Taker fee: fee_rate × p × (1−p) — curve-based, highest at p=0.50.
    Maker rebate: 25% of taker fee — paid when our resting order is taken.

    Parameters
    ----------
    p_model  : model probability of YES (your fair value)
    p_market : current Polymarket market mid price
    params   : QuoteParams controlling spread, edge threshold, skew

    Returns
    -------
    Quote dataclass with bid, ask, sizing, reward scores, and rationale
    """
    if params is None:
        params = QuoteParams()

    p_model  = float(np.clip(p_model,  0.02, 0.98))
    p_market = float(np.clip(p_market, 0.02, 0.98))

    edge     = compute_edge(p_model, p_market)
    fee      = taker_fee_per_share(p_market)
    net_edge = abs(edge) - fee           # positive = worth taking as a taker

    # Inventory skew: lean toward value side
    skew = -np.sign(edge) * abs(edge) * 0.3 + params.inventory_skew
    mid  = _snap(p_model)
    bid  = _snap(mid - params.half_spread + skew)
    ask  = _snap(mid + params.half_spread + skew)

    # Hard bounds
    bid = float(np.clip(bid, MIN_PRICE, mid - TICK_SIZE))
    ask = float(np.clip(ask, mid + TICK_SIZE, MAX_PRICE))

    # Reward scores for each side (measure tightness vs qualifying spread)
    bid_spread = abs(p_market - bid)
    ask_spread = abs(ask - p_market)
    bid_score  = order_reward_score(bid_spread, 0, params.max_reward_spread)  # size=0: score per share
    ask_score  = order_reward_score(ask_spread, 0, params.max_reward_spread)

    # Directional sizing: Kelly only if net_edge > 0 and |edge| >= min_edge
    if abs(edge) >= params.min_edge:
        bet_dir = "YES" if edge > 0 else "NO"
        k_size  = kelly_fraction(p_model, p_market, params.kelly_frac, as_maker=False)
        k_size  = min(k_size, params.max_inventory)
    else:
        bet_dir = "PASS"
        k_size  = 0.0

    rebate = maker_rebate_per_share(p_market)
    rationale = (
        f"Model={p_model:.3f}  Market={p_market:.3f}  Edge={edge:+.3f}  "
        f"TakerFee={fee:.4f}  NetEdge={net_edge:+.4f}  Rebate={rebate:.5f}  "
        f"→ {bet_dir}  Kelly={k_size:.3f}"
    )

    return Quote(
        bid=bid, ask=ask, mid=mid,
        market_mid=p_market,
        edge=edge,
        net_taker_edge=net_edge,
        bet_direction=bet_dir,
        kelly_size=k_size,
        bid_reward_score=bid_score,
        ask_reward_score=ask_score,
        rationale=rationale,
    )


# ─── Backtest / P&L simulation ────────────────────────────────────────────────

def simulate_market_making(
    predictions:   pd.Series,
    market_prices: pd.Series,
    results:       pd.Series,
    bankroll:      float        = 1000.0,
    params:        QuoteParams | None = None,
    tier:          str          = "AB",
    market_share:  float        = 0.10,
    verbose:       bool         = False,
) -> pd.DataFrame:
    """
    Simulate P&L from the market-making strategy on historical data.

    Models two income streams per game:
      1. Directional taker bet (if |edge| >= min_edge): Kelly-sized, pays taker fee.
      2. Liquidity rewards: estimated based on assumed market share of the reward pool.

    Taker fee formula: SPORTS_TAKER_RATE × p × (1 − p) per share.
    Maker rebate:      25% of taker fee (earned on maker fills — not modelled
                       separately here since fill probability is unknown).

    Parameters
    ----------
    predictions   : model probabilities indexed by game rows
    market_prices : market mid prices at time of quoting
    results       : actual outcomes (1 = YES resolves, 0 = NO)
    bankroll      : starting bankroll in USDC
    params        : QuoteParams
    tier          : "AB" (LCK/LPL/LEC/LCS) or "C" (ERLs)
    market_share  : assumed fraction of liquidity rewards captured (default 10%)
    verbose       : print per-bet details

    Returns
    -------
    DataFrame with columns: bet_dir, edge, kelly_size, stake, directional_pnl,
                             liquidity_reward, total_pnl, cumulative_pnl, bankroll
    """
    if params is None:
        params = QuoteParams()

    records  = []
    cum_pnl  = 0.0
    cash     = bankroll

    for idx in predictions.index:
        p_m   = float(predictions.loc[idx])
        p_mk  = float(market_prices.loc[idx])
        res   = int(results.loc[idx])

        q     = market_maker_quotes(p_m, p_mk, params)
        fee   = taker_fee_per_share(p_mk)

        # ── Income stream 2: liquidity rewards (independent of direction) ──
        liq_reward = estimate_liquidity_reward(
            bid_spread=abs(p_mk - q.bid),
            bid_size=cash * 0.05 / p_mk,        # rough size at 5% of bankroll
            ask_spread=abs(q.ask - p_mk),
            ask_size=cash * 0.05 / (1.0 - p_mk),
            market_share=market_share,
            tier=tier,
            max_spread=params.max_reward_spread,
        )

        # ── Income stream 1: directional taker bet ─────────────────────────
        dir_pnl = 0.0
        stake   = 0.0

        if q.bet_direction != "PASS" and q.kelly_size > 0:
            stake = cash * q.kelly_size

            if q.bet_direction == "YES":
                # Buy YES at p_mk, pay taker fee
                # PnL = stake × (outcome − p_mk) − stake × fee
                dir_pnl = stake * (res - p_mk) - stake * fee
            else:
                # Buy NO at (1 − p_mk), pay taker fee
                dir_pnl = stake * ((1 - res) - (1 - p_mk)) - stake * fee

        total_pnl = dir_pnl + liq_reward
        cum_pnl  += total_pnl
        cash     += total_pnl

        records.append({
            "idx":             idx,
            "p_model":         round(p_m,  4),
            "p_market":        round(p_mk, 4),
            "edge":            round(q.edge, 4),
            "net_taker_edge":  round(q.net_taker_edge, 4),
            "taker_fee":       round(fee, 5),
            "bet_direction":   q.bet_direction,
            "kelly_size":      q.kelly_size,
            "stake":           round(stake, 2),
            "result":          res,
            "directional_pnl": round(dir_pnl, 4),
            "liquidity_reward":round(liq_reward, 4),
            "total_pnl":       round(total_pnl, 4),
            "cumulative_pnl":  round(cum_pnl, 4),
            "bankroll":        round(cash, 2),
        })

        if verbose:
            print(
                f"  {q.bet_direction:<3}  p_m={p_m:.3f}  p_mk={p_mk:.3f}  "
                f"edge={q.edge:+.3f}  fee={fee:.4f}  stake={stake:.1f}  "
                f"dir={dir_pnl:+.2f}  liq={liq_reward:.2f}  "
                f"total={total_pnl:+.2f}  cum={cum_pnl:+.2f}",
            )

    sim_df = pd.DataFrame(records)

    if len(sim_df) == 0:
        print("No bets placed — edge threshold too high or no predictions provided.")
        return sim_df

    total_bets      = len(sim_df)
    bets_placed     = (sim_df["bet_direction"] != "PASS").sum()
    win_rate        = (sim_df["directional_pnl"] > 0).mean()
    total_dir_pnl   = sim_df["directional_pnl"].sum()
    total_liq       = sim_df["liquidity_reward"].sum()
    total_pnl       = sim_df["total_pnl"].sum()
    roi             = total_pnl / bankroll
    avg_fee         = sim_df["taker_fee"].mean()

    print(f"\n{'='*60}")
    print(f"  Betting simulation summary")
    print(f"{'='*60}")
    print(f"  Games processed:    {total_bets:>6}")
    print(f"  Directional bets:   {bets_placed:>6}  ({bets_placed/total_bets:.0%} of games)")
    print(f"  Win rate (dir bets):{win_rate:>7.1%}")
    print(f"  Directional P&L:   {total_dir_pnl:>+9.2f} USDC")
    print(f"  Liquidity rewards: {total_liq:>+9.2f} USDC  ({market_share:.0%} est. share, {tier} tier)")
    print(f"  Total P&L:         {total_pnl:>+9.2f} USDC")
    print(f"  ROI:               {roi:>+9.2%}")
    print(f"  Avg taker fee:     {avg_fee:>9.5f}  (vs old flat 0.00050)")
    print(f"  Final bankroll:    {sim_df['bankroll'].iloc[-1]:>9.2f} USDC")
    print(f"{'='*60}")

    return sim_df


# ─── Batch quote generation ───────────────────────────────────────────────────

def generate_quotes(
    predictions:   pd.Series,
    market_prices: pd.Series,
    params:        QuoteParams | None = None,
    tier:          str                = "AB",
    market_share:  float              = 0.10,
) -> pd.DataFrame:
    """
    Generate bid/ask quotes and reward estimates for a batch of upcoming matches.

    Parameters
    ----------
    predictions   : model probabilities indexed by match identifier
    market_prices : Polymarket current market prices for same matches
    params        : QuoteParams
    tier          : reward tier for liquidity rewards estimate
    market_share  : assumed reward pool share

    Returns
    -------
    DataFrame sorted by |edge| descending with quote details and reward estimates
    """
    if params is None:
        params = QuoteParams()

    rows = []
    for idx in predictions.index:
        p_m  = float(predictions.loc[idx])
        p_mk = float(market_prices.loc[idx])
        q    = market_maker_quotes(p_m, p_mk, params)
        fee  = taker_fee_per_share(p_mk)

        rew = estimate_liquidity_reward(
            bid_spread=abs(p_mk - q.bid),
            bid_size=100,     # illustrative 100-share quote
            ask_spread=abs(q.ask - p_mk),
            ask_size=100,
            market_share=market_share,
            tier=tier,
            max_spread=params.max_reward_spread,
        )

        rows.append({
            "match":           idx,
            "model_prob":      round(p_m,   4),
            "market_price":    round(p_mk,  4),
            "edge":            round(q.edge, 4),
            "net_taker_edge":  round(q.net_taker_edge, 4),
            "taker_fee":       round(fee, 5),
            "bid":             q.bid,
            "ask":             q.ask,
            "bet_direction":   q.bet_direction,
            "kelly_size":      q.kelly_size,
            "est_liq_reward":  round(rew, 2),
            "rationale":       q.rationale,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("edge", key=abs, ascending=False).reset_index(drop=True)
    return df


# ─── Fee comparison helper ────────────────────────────────────────────────────

def fee_table(prices: list[float] | None = None) -> pd.DataFrame:
    """
    Print taker fee and maker rebate at common share prices.
    Useful for calibrating min_edge thresholds.

    Returns a DataFrame showing fee/rebate at each price.
    """
    if prices is None:
        prices = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    for p in prices:
        fee    = taker_fee_per_share(p)
        rebate = maker_rebate_per_share(p)
        rows.append({
            "price":           p,
            "taker_fee":       round(fee, 5),
            "taker_fee_bps":   round(fee * 10000, 1),
            "maker_rebate":    round(rebate, 6),
            "maker_rebate_bps":round(rebate * 10000, 1),
            "min_edge_to_bet": round(fee, 5),   # need |edge| > fee to take profitably
        })
    return pd.DataFrame(rows)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _snap(price: float) -> float:
    """Snap price to Polymarket tick size (0.01) and clip to valid range."""
    snapped = round(round(price / TICK_SIZE) * TICK_SIZE, 4)
    return float(np.clip(snapped, MIN_PRICE, MAX_PRICE))
