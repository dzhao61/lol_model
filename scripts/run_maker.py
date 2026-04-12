"""
End-to-end market-making workflow for a single LoL match.

Steps:
1. Predict win probability from your model
2. Look up the Polymarket market and current price
3. Compute optimal bid/ask quotes
4. Optionally post orders

Usage (dry run — no orders posted):
    python -m scripts.run_maker \
        --team1 "T1" --team2 "Gen.G" \
        --league LCK --split Spring --patch 25.S1.3 \
        --condition-id 0xABC123... \
        --bankroll 200 --size 100

Usage (live — posts real orders):
    python -m scripts.run_maker ... --live

Cancel all open orders for a market:
    python -m scripts.run_maker --cancel --condition-id 0xABC123...

List recent LoL markets:
    python -m scripts.run_maker --list-markets [--query "T1"]
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predict_match  import predict_match, load_bundle, load_feat_df, series_win_prob
from scripts.polymarket_client import (
    get_client, find_lol_markets, get_market_by_condition_id,
    get_mid_price, print_market_summary, post_two_sided_quote,
    post_single_side_quote, cancel_all_orders, get_open_orders,
    check_order_scoring, check_reward_percentages, check_reward_earnings,
    check_market_rewards, find_reward_markets, print_reward_markets,
    merge_positions,
)
from pipeline.betting import (
    market_maker_quotes, QuoteParams, fee_table, taker_fee_per_share,
    PositionState, compute_half_spread,
)
from pipeline.models import ensemble_predict


# ─── Fill monitoring ─────────────────────────────────────────────────────────

def monitor_fills(
    condition_id:  str,
    posted_bid_id: str | None,
    posted_ask_id: str | None,
    yes_token_id:  str,
    no_token_id:   str | None = None,
    bid_price:     float = 0.0,
    ask_price:     float = 0.0,
    p_fair:        float = 0.5,
    size:          float = 30.0,
    interval:      int   = 30,
    max_checks:    int   = 60,
    adverse_threshold: float = 0.02,
) -> None:
    """
    Three-tier fill monitor with Avellaneda-Stoikov inventory management.

    Tier 1 (continuous): Quote skew is handled at post time via PositionState.
    Tier 2 (on single fill): Cancel unfilled side, re-post it tighter (0.5¢ closer
        to current mid) to attract the complementary fill and hedge the position.
    Tier 3 (adverse selection): If after Tier 2 repost the market moves >2¢ against
        the filled position, cancel all and exit — we were picked off by informed flow.

    When both sides fill: calls merge_positions to recycle capital to USDC.
    Drift warning: if mid moves >1.5¢ with no fill, warns to re-quote.
    """
    import time
    from scripts.polymarket_client import get_orderbook_snapshot

    TICK = 0.01
    DRIFT_WARN  = 0.015   # warn if mid drifts this far with no fill
    TIGHTER_ADJ = 0.005   # Tier 2: re-post this much closer to mid

    position = PositionState()
    bid_alive = posted_bid_id is not None
    ask_alive = posted_ask_id is not None

    print(f"\nMonitoring fills (every {interval}s, up to {max_checks} checks)...")
    print(f"  Bid order: {posted_bid_id[:12] if posted_bid_id else 'none'}")
    print(f"  Ask order: {posted_ask_id[:12] if posted_ask_id else 'none'}")
    print(f"  Press Ctrl+C to stop.\n")

    snap = get_orderbook_snapshot(yes_token_id)
    original_mid = snap.get("mid") or p_fair

    # State for three-tier logic
    fill_mid: float | None = None   # mid captured at fill detection time (Tier 3 anchor)
    requote_id: str | None = None   # order ID of Tier 2 re-posted order
    requote_side: str | None = None # "YES" or "NO"

    for check in range(max_checks):
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            return

        open_orders = get_open_orders(condition_id)
        open_ids = {o.get("id") or o.get("orderID") for o in open_orders}

        bid_filled = bid_alive and posted_bid_id not in open_ids
        ask_filled = ask_alive and posted_ask_id not in open_ids

        # Detect fills and record position
        if bid_filled:
            position.update_fill("YES", size, p_fair)
            bid_alive = False
            print(f"  [{check+1}] BID FILLED — bought {size:.0f} YES @ {bid_price:.3f}  "
                  f"net_exposure={position.net_exposure(p_fair):+.2f}")

        if ask_filled:
            position.update_fill("NO", size, 1.0 - p_fair)
            ask_alive = False
            print(f"  [{check+1}] ASK FILLED — bought {size:.0f} NO @ {ask_price:.3f}  "
                  f"net_exposure={position.net_exposure(p_fair):+.2f}")

        # ── Both filled → fully hedged, merge and exit ────────────────────────
        if not bid_alive and not ask_alive:
            if position.is_flat or (position.yes_shares > 0 and position.no_shares > 0):
                print(f"\n  Both sides filled — spread captured!")
                merge_positions(condition_id, min(position.yes_shares, position.no_shares))
            return

        # Fetch current mid for drift/adverse checks
        current_snap = get_orderbook_snapshot(yes_token_id)
        current_mid  = current_snap.get("mid") or original_mid

        # ── Tier 2: one side just filled → aggressive requote ─────────────────
        if (bid_filled or ask_filled) and fill_mid is None:
            fill_mid = current_mid   # anchor Tier 3 to THIS moment
            if no_token_id:
                if bid_filled and ask_alive:
                    # YES filled, NO still resting — re-post NO tighter
                    no_mid = 1.0 - current_mid
                    new_ask_price = round(min(no_mid + TIGHTER_ADJ, 0.99), 2)
                    print(f"  Tier 2: YES filled → cancelling NO, re-posting tighter @ {new_ask_price:.2f}...")
                    cancel_all_orders(condition_id)
                    ask_alive = False
                    requote_id = post_single_side_quote(no_token_id, condition_id, new_ask_price, size)
                    requote_side = "NO"
                    print(f"  Re-quote posted: {str(requote_id)[:12] if requote_id else 'failed'}")

                elif ask_filled and bid_alive:
                    # NO filled, YES still resting — re-post YES tighter
                    new_bid_price = round(max(current_mid - TIGHTER_ADJ, 0.01), 2)
                    print(f"  Tier 2: NO filled → cancelling YES, re-posting tighter @ {new_bid_price:.2f}...")
                    cancel_all_orders(condition_id)
                    bid_alive = False
                    requote_id = post_single_side_quote(yes_token_id, condition_id, new_bid_price, size)
                    requote_side = "YES"
                    print(f"  Re-quote posted: {str(requote_id)[:12] if requote_id else 'failed'}")
            else:
                # No no_token_id available — fall back to cancel only
                print(f"  One side filled (no_token_id not set, skipping Tier 2 re-quote)")

        # ── Tier 3: adverse selection check after Tier 2 requote ─────────────
        if fill_mid is not None and requote_id is not None:
            move = current_mid - fill_mid
            adverse = False
            if requote_side == "NO" and move < -adverse_threshold:
                # Bought YES (bid filled), market dropped → adverse
                print(f"  ⚠ ADVERSE: bought YES, market dropped {move:.3f} since fill")
                adverse = True
            elif requote_side == "YES" and move > adverse_threshold:
                # Bought NO (ask filled), market rose → adverse
                print(f"  ⚠ ADVERSE: bought NO, market rose {move:+.3f} since fill")
                adverse = True
            if adverse:
                cancel_all_orders(condition_id)
                print(f"  Tier 3 triggered — all orders cancelled.")
                return

            # Check if Tier 2 requote filled (hedge complete)
            if requote_id not in open_ids:
                print(f"\n  Tier 2 re-quote filled — fully hedged!")
                merge_positions(condition_id, size)
                return

        # ── Drift warning: no fill, mid has moved away ────────────────────────
        if fill_mid is None and bid_alive and ask_alive:
            drift = abs(current_mid - original_mid)
            if drift > DRIFT_WARN:
                print(f"  [{check+1}] ⚠ Mid drifted {drift*100:.1f}¢ from entry — consider re-quoting")
            else:
                print(f"  [{check+1}] Both orders resting  mid={current_mid:.3f}  drift={drift*100:.1f}¢")

    print(f"\nMax checks reached ({max_checks}). Stopping monitor.")


# ─── Main workflow ────────────────────────────────────────────────────────────

def run(args):
    # ── 1. List markets mode ──────────────────────────────────────────────────
    if args.list_markets:
        print("Searching for active LoL markets on Polymarket...")
        markets = find_lol_markets(query=args.query or "")
        if not markets:
            print("No active LoL markets found.")
            return
        print(f"\nFound {len(markets)} market(s):\n")
        for i, m in enumerate(markets):
            print(f"  [{i+1}] {m.question}")
            print(f"       condition_id: {m.condition_id}")
            print(f"       YES token:    {m.yes_token_id[:20]}...")
            print(f"       Ends:         {m.end_date[:10]}")
            print()
        return

    # ── 2. Check scoring mode ─────────────────────────────────────────────────
    if args.check_scoring:
        if not args.order_ids:
            print("ERROR: --order-ids required for --check-scoring  (e.g. --order-ids 0xABC 0xDEF)")
            sys.exit(1)
        check_order_scoring(args.order_ids)
        return

    # ── 2b. Find reward markets mode ─────────────────────────────────────────
    if args.find_reward_markets:
        print(f"Scanning all reward markets (max capital/side=${args.max_capital:.0f}, "
              f"min daily rate=${args.min_daily_rate:.0f})...")
        markets_rw = find_reward_markets(
            max_capital_per_side=args.max_capital,
            min_daily_rate=args.min_daily_rate,
        )
        print_reward_markets(markets_rw)
        return

    # ── 2c. Auto-scout: pick best reward market automatically ─────────────────
    if args.auto_scout:
        print(f"Auto-scout: finding best reward market (max_capital=${args.max_capital:.0f}, "
              f"min_daily_rate=${args.min_daily_rate:.0f})...")
        markets_rw = find_reward_markets(
            max_capital_per_side=args.max_capital,
            min_daily_rate=args.min_daily_rate,
        )
        if not markets_rw:
            print("No qualifying reward markets found. Adjust --max-capital or --min-daily-rate.")
            return
        # Score: high daily rate, low competition
        best = max(markets_rw, key=lambda m: m.daily_rate / max(m.competitiveness, 0.1))
        print(f"  → Selected: {best.question[:65]}")
        print(f"     Daily rate: ${best.daily_rate:.0f}  Comp: {best.competitiveness:.1f}  Capital/side: ${best.capital_per_side:.0f}")
        args.condition_id = best.condition_id
        # Fall through to any-market quote mode below

    # ── 2b. Check earnings mode ───────────────────────────────────────────────
    if args.check_earnings:
        check_reward_percentages(args.condition_id)
        check_reward_earnings(args.date or None)
        return

    # ── 2c. Market rewards mode ───────────────────────────────────────────────
    if args.market_rewards:
        if not args.condition_id:
            print("ERROR: --condition-id required for --market-rewards")
            sys.exit(1)
        check_market_rewards(args.condition_id)
        return

    # ── 2b. Open orders mode ──────────────────────────────────────────────────
    if args.open_orders:
        if not args.condition_id:
            print("ERROR: --condition-id required for --open-orders")
            sys.exit(1)
        orders = get_open_orders(args.condition_id)
        if not orders:
            print("No open orders found for this market.")
            return
        print(f"\nOpen orders ({len(orders)}):\n")
        for o in orders:
            oid  = o.get("id") or o.get("orderID") or "?"
            side = o.get("side", "?")
            price = o.get("price", "?")
            size  = o.get("size") or o.get("originalSize", "?")
            asset = o.get("asset_id", o.get("tokenId", ""))[:12]
            print(f"  {oid}  side={side}  price={price}  size={size}  token={asset}...")
        print()
        return

    # ── 3. Cancel mode ────────────────────────────────────────────────────────
    if args.cancel:
        if not args.condition_id:
            print("ERROR: --condition-id required for --cancel")
            sys.exit(1)
        print(f"Cancelling all orders for {args.condition_id[:16]}...")
        cancel_all_orders(args.condition_id)
        return

    # ── 4. Quote mode ─────────────────────────────────────────────────────────
    lol_mode = bool(args.team1 and args.team2)
    any_market_mode = bool(args.condition_id and not lol_mode)

    if not lol_mode and not any_market_mode:
        print("ERROR: provide --team1 and --team2 (LoL mode) or --condition-id alone (any-market mode)")
        sys.exit(1)

    p_model = None
    model_std = 0.0

    if lol_mode:
        context = {
            "league":   args.league,
            "split":    args.split,
            "playoffs": args.playoffs,
            "patch":    args.patch,
            "game":     args.game,
        }

        # Step 1: model prediction (ensemble over all bundles)
        print("Step 1 — Loading models and predicting (ensemble)...")
        feat_df = load_feat_df()

        weights_path = ROOT / "cache" / "ensemble_weights.json"
        ensemble_weights = json.load(open(weights_path)) if weights_path.exists() else None

        all_preds: dict[str, float] = {}
        first_pred = None
        for bpath in sorted((ROOT / "cache").glob("model_bundle_*.pkl")):
            key = bpath.stem.replace("model_bundle_", "")
            if not key:
                continue
            bundle = pickle.load(open(bpath, "rb"))
            try:
                r = predict_match(args.team1, args.team2, context, bundle, feat_df,
                                  verbose=(first_pred is None))
                all_preds[key] = r["p_model"]
                if first_pred is None:
                    first_pred = r
            except Exception as e:
                print(f"  [{key}] failed: {e}")

        if not all_preds:
            print("ERROR: all model predictions failed.")
            sys.exit(1)

        p_model, model_std = ensemble_predict(all_preds, weights=ensemble_weights)
        weight_mode = "weighted" if ensemble_weights else "equal weights"
        print(f"\n  Ensemble: p={p_model:.3f}  σ={model_std:.4f}  ({len(all_preds)} models, {weight_mode})")
        for k, p in sorted(all_preds.items()):
            print(f"    {k:<15} {p:.3f}")

        # Optional: show series probability
        if args.format != "Bo1":
            p_series = series_win_prob(p_model, args.format)
            print(f"  Series win prob ({args.format}): {p_series:.1%}\n")
            print(f"  NOTE: If the Polymarket market is for the SERIES winner,")
            print(f"  use p_series={p_series:.3f} instead of p_model={p_model:.3f}\n")
    else:
        print("Step 1 — Any-market mode: fair value = current mid (no directional model)")

    # Step 2: get market price
    p_market = None
    market   = None

    yes_ask_live = None  # track current ask for clamping

    if args.condition_id:
        print("Step 2 — Fetching Polymarket price...")
        market = get_market_by_condition_id(args.condition_id)
        if market:
            from scripts.polymarket_client import get_orderbook_snapshot
            snap = get_orderbook_snapshot(market.yes_token_id)
            p_market     = snap.get("mid")
            yes_ask_live = snap["asks"][0][0] if snap.get("asks") else None
            yes_bid_live = snap["bids"][0][0] if snap.get("bids") else None
            print_market_summary(market, p_model=p_model)
        else:
            print(f"  Market {args.condition_id[:16]}... not found.")
    else:
        print("Step 2 — No --condition-id provided. Skipping price fetch.")
        print("  Find your market on polymarket.com → Esports → LoL")
        print("  Copy the condition_id from the URL and re-run with --condition-id\n")

    if p_market is None:
        p_market = float(input("Enter current Polymarket mid price (e.g. 0.55): ").strip())

    # In any-market mode, fair value = current mid (no directional signal)
    if p_model is None:
        p_model = p_market

    # Compute hours until match from market end date
    hours_to_match = 168.0
    if market and market.end_date:
        try:
            end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
            hours_to_match = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
        except Exception:
            pass

    # Step 3: compute quotes
    print("Step 3 — Computing quotes...")

    # Dynamic spread: widens near match and when models disagree
    dyn_spread = compute_half_spread(
        model_std=model_std,
        hours_to_match=hours_to_match,
        edge_abs=abs(p_model - p_market),
        base=args.half_spread,
    )

    params = QuoteParams(
        half_spread   = dyn_spread,
        min_edge      = args.min_edge,
        max_inventory = 0.05,
        kelly_frac    = 0.25,
    )

    quote = market_maker_quotes(p_model, p_market, params,
                                model_std=model_std,
                                hours_to_match=hours_to_match)

    fee    = taker_fee_per_share(p_market)
    bankroll = args.bankroll

    print(f"\n{'─'*55}")
    print(f"  Quote summary")
    print(f"{'─'*55}")
    print(f"  p_model:          {p_model:.3f}  (σ={model_std:.4f})")
    print(f"  p_fair (blended): {quote.mid:.3f}  (α={quote.alpha:.2f})")
    print(f"  p_market (mid):   {p_market:.3f}")
    print(f"  Edge:             {quote.edge:+.3f}")
    print(f"  Hours to match:   {hours_to_match:.1f}h")
    print(f"  Half-spread:      {dyn_spread:.4f}  (dynamic, base={args.half_spread:.4f})")
    print(f"  Taker fee:        {fee:.4f}  (at this price)")
    print(f"  Net taker edge:   {quote.net_taker_edge:+.4f}")
    print(f"")
    print(f"  BID (buy YES):    {quote.bid:.3f}")
    print(f"  ASK (sell YES):   {quote.ask:.3f}")
    print(f"  Spread:           {quote.ask - quote.bid:.3f}  ({(quote.ask-quote.bid)*100:.1f} cents)")
    print(f"")
    if quote.bet_direction != "PASS":
        stake = bankroll * quote.kelly_size
        print(f"  Directional bet:  {quote.bet_direction}  (Kelly size: {quote.kelly_size:.3f})")
        print(f"  Stake:            ${stake:.2f}  (of ${bankroll:.0f} bankroll)")
    else:
        print(f"  Directional bet:  PASS  (edge below threshold)")
    print(f"{'─'*55}\n")

    # Step 4: show fee table for reference
    if args.show_fees:
        print("Polymarket Sports/Esports fee structure:")
        print(fee_table().to_string(index=False))
        print()

    # Step 5: post orders
    if not market:
        print("Skipping order posting — no market loaded.")
        return

    if args.live:
        print(f"Step 4 — Posting LIVE orders (size={args.size} shares each side)...")
        confirm = input(
            f"  Confirm: post BID @ {quote.bid:.3f} and ASK @ {quote.ask:.3f} "
            f"for {args.size} shares each? [y/N] "
        ).strip().lower()

        if confirm != "y":
            print("  Cancelled — no orders posted.")
            return

        TICK = 0.01
        # Clamp to stay below current ask (ensures resting maker orders, not taker crosses)
        t1 = quote.bid
        t2 = round(1.0 - quote.ask, 2)
        if yes_ask_live is not None:
            t1 = min(t1, round(yes_ask_live - TICK, 2))
        if yes_bid_live is not None:
            no_ask_mkt = round(1.0 - yes_bid_live, 2)
            t2 = min(t2, round(no_ask_mkt - TICK, 2))

        posted = post_two_sided_quote(
            token_id     = market.yes_token_id,
            no_token_id  = market.no_token_id,
            condition_id = market.condition_id,
            t1_price     = t1,
            t2_price     = t2,
            size         = args.size,
            dry_run      = False,
        )
        print(f"\nOrders posted successfully.")
        print(f"  Bid order ID: {posted.bid_order_id}")
        print(f"  Ask order ID: {posted.ask_order_id}")
        print(f"\n⚠ Remember to cancel orders before match starts:")
        print(f"  python -m scripts.run_maker --cancel --condition-id {market.condition_id}")

        # Optional: monitor fills for adverse selection
        if args.monitor:
            monitor_fills(
                condition_id=market.condition_id,
                posted_bid_id=posted.bid_order_id,
                posted_ask_id=posted.ask_order_id,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                bid_price=posted.bid_price,
                ask_price=posted.ask_price,
                p_fair=quote.mid,
                size=args.size,
                interval=args.monitor_interval,
            )
    else:
        print("Step 4 — Dry run (add --live to post real orders):")
        TICK = 0.01
        t1 = quote.bid
        t2 = round(1.0 - quote.ask, 2)
        if yes_ask_live is not None:
            t1 = min(t1, round(yes_ask_live - TICK, 2))
        if yes_bid_live is not None:
            no_ask_mkt = round(1.0 - yes_bid_live, 2)
            t2 = min(t2, round(no_ask_mkt - TICK, 2))
        post_two_sided_quote(
            token_id     = market.yes_token_id,
            no_token_id  = market.no_token_id,
            condition_id = market.condition_id,
            t1_price     = t1,
            t2_price     = t2,
            size         = args.size,
            dry_run      = True,
        )
        print(f"\nAdd --live to submit these orders to Polymarket.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoL pre-draft market maker")

    # Match identity
    parser.add_argument("--team1",        default=None)
    parser.add_argument("--team2",        default=None)
    parser.add_argument("--league",       default="LCK")
    parser.add_argument("--split",        default="Spring")
    parser.add_argument("--playoffs",     type=int, default=0)
    parser.add_argument("--patch",        default="25.S1.3")
    parser.add_argument("--game",         type=int, default=1)
    parser.add_argument("--format",       default="Bo3",
                        help="Bo1/Bo3/Bo5 — prints series win prob alongside game prob")

    # Polymarket
    parser.add_argument("--condition-id", default=None,
                        help="Polymarket condition_id for the market")

    # Quoting
    parser.add_argument("--half-spread",  type=float, default=0.015)
    parser.add_argument("--min-edge",     type=float, default=0.02)
    parser.add_argument("--bankroll",     type=float, default=200.0)
    parser.add_argument("--size",         type=float, default=30.0,
                        help="Shares per side for quotes (default: 30)")

    # Modes
    parser.add_argument("--live",         action="store_true",
                        help="Post real orders (default is dry run)")
    parser.add_argument("--cancel",       action="store_true",
                        help="Cancel all orders for --condition-id")
    parser.add_argument("--list-markets", action="store_true",
                        help="List active LoL markets")
    parser.add_argument("--query",        default=None,
                        help="Filter for --list-markets (e.g. 'T1')")
    parser.add_argument("--open-orders",   action="store_true",
                        help="List open orders for --condition-id with full order IDs")
    parser.add_argument("--check-scoring",  action="store_true",
                        help="Check whether orders are earning liquidity rewards")
    parser.add_argument("--order-ids",      nargs="+", default=[],
                        help="Order IDs to check with --check-scoring (e.g. 0xABC 0xDEF)")
    parser.add_argument("--check-earnings", action="store_true",
                        help="Show real-time reward % share and daily earnings")
    parser.add_argument("--date",           default=None,
                        help="Date for --check-earnings (YYYY-MM-DD, defaults to today)")
    parser.add_argument("--market-rewards", action="store_true",
                        help="Show reward config (pool size, max spread) for --condition-id")
    parser.add_argument("--show-fees",    action="store_true",
                        help="Print the fee/rebate table")
    parser.add_argument("--monitor",      action="store_true",
                        help="Monitor fills after posting (adverse selection defense)")
    parser.add_argument("--monitor-interval", type=int, default=30,
                        help="Seconds between fill checks (default: 30)")
    parser.add_argument("--find-reward-markets", action="store_true",
                        help="Scan all reward markets ranked by daily rate")
    parser.add_argument("--auto-scout",          action="store_true",
                        help="Auto-pick best reward market by rate/competitiveness and post quotes")
    parser.add_argument("--max-capital",   type=float, default=1000.0,
                        help="Max capital per side in USD for --find-reward-markets / --auto-scout (default: 1000)")
    parser.add_argument("--min-daily-rate", type=float, default=10.0,
                        help="Minimum daily reward pool in USD (default: 10)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
