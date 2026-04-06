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
    cancel_all_orders, get_open_orders,
)
from pipeline.betting import (
    market_maker_quotes, QuoteParams, fee_table, taker_fee_per_share,
    PositionState, compute_half_spread,
)
from pipeline.models import ensemble_predict


# ─── Fill monitoring ─────────────────────────────────────────────────────────

def monitor_fills(
    condition_id: str,
    posted_bid_id: str | None,
    posted_ask_id: str | None,
    yes_token_id: str,
    p_fair: float = 0.5,
    size: float = 30.0,
    interval: int = 30,
    max_checks: int = 60,
) -> None:
    """
    Poll for fills and implement adverse selection defense.

    After posting orders, checks every `interval` seconds.
    If one side fills and the market moves against us by > 2 cents,
    cancels the remaining side to prevent being picked off.

    Parameters
    ----------
    condition_id   : market condition ID
    posted_bid_id  : order ID for the bid (YES buy)
    posted_ask_id  : order ID for the ask (NO buy)
    yes_token_id   : YES token ID for price checks
    interval       : seconds between checks
    max_checks     : max number of checks before stopping
    """
    import time
    from scripts.polymarket_client import get_orderbook_snapshot

    position = PositionState()
    bid_alive = posted_bid_id is not None
    ask_alive = posted_ask_id is not None

    print(f"\nMonitoring fills (every {interval}s, up to {max_checks} checks)...")
    print(f"  Bid order: {posted_bid_id[:12] if posted_bid_id else 'none'}")
    print(f"  Ask order: {posted_ask_id[:12] if posted_ask_id else 'none'}")
    print(f"  Press Ctrl+C to stop.\n")

    initial_snap = get_orderbook_snapshot(yes_token_id)
    initial_mid = initial_snap.get("mid")

    for check in range(max_checks):
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            return

        # Check which orders are still open
        open_orders = get_open_orders(condition_id)
        open_ids = {o.get("id") for o in open_orders}

        bid_filled = bid_alive and posted_bid_id not in open_ids
        ask_filled = ask_alive and posted_ask_id not in open_ids

        if bid_filled:
            position.update_fill("YES", size, p_fair)
            print(f"  [{check+1}] BID FILLED (YES buy)  exposure={position.net_exposure(p_fair):+.2f}")
            bid_alive = False
        if ask_filled:
            position.update_fill("NO", size, 1.0 - p_fair)
            print(f"  [{check+1}] ASK FILLED (NO buy)  exposure={position.net_exposure(p_fair):+.2f}")
            ask_alive = False

        if not bid_alive and not ask_alive:
            print(f"\n  Both sides filled — spread captured!")
            return

        # One side filled: check for adverse selection
        if (bid_filled or ask_filled) and (bid_alive or ask_alive):
            snap = get_orderbook_snapshot(yes_token_id)
            current_mid = snap.get("mid")

            if initial_mid and current_mid:
                move = current_mid - initial_mid
                print(f"  Market moved: {move:+.3f} since order post")

                # Adverse selection: market moved against our filled position
                adverse = False
                if bid_filled and move < -0.02:
                    # We bought YES, market dropped → adverse
                    adverse = True
                    print(f"  ⚠ ADVERSE: bought YES but market dropped {move:.3f}")
                elif ask_filled and move > 0.02:
                    # We bought NO, market rose → adverse
                    adverse = True
                    print(f"  ⚠ ADVERSE: bought NO but market rose {move:+.3f}")

                if adverse:
                    remaining = "bid" if bid_alive else "ask"
                    print(f"  Cancelling remaining {remaining} to limit exposure...")
                    cancel_all_orders(condition_id)
                    print(f"  Done — adverse selection defense triggered.")
                    return

        if not bid_filled and not ask_filled:
            print(f"  [{check+1}] Both orders still resting...")

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

    # ── 2. Cancel mode ────────────────────────────────────────────────────────
    if args.cancel:
        if not args.condition_id:
            print("ERROR: --condition-id required for --cancel")
            sys.exit(1)
        print(f"Cancelling all orders for {args.condition_id[:16]}...")
        cancel_all_orders(args.condition_id)
        return

    # ── 3. Predict + quote mode ───────────────────────────────────────────────
    if not args.team1 or not args.team2:
        print("ERROR: --team1 and --team2 are required")
        sys.exit(1)

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
    parser.add_argument("--show-fees",    action="store_true",
                        help="Print the fee/rebate table")
    parser.add_argument("--monitor",      action="store_true",
                        help="Monitor fills after posting (adverse selection defense)")
    parser.add_argument("--monitor-interval", type=int, default=30,
                        help="Seconds between fill checks (default: 30)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
