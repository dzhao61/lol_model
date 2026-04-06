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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predict_match  import predict_match, load_bundle, load_feat_df, series_win_prob
from scripts.polymarket_client import (
    get_client, find_lol_markets, get_market_by_condition_id,
    get_mid_price, print_market_summary, post_two_sided_quote,
    cancel_all_orders, get_open_orders,
)
from pipeline.betting import market_maker_quotes, QuoteParams, fee_table, taker_fee_per_share


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

    # Step 1: model prediction
    print("Step 1 — Loading model and predicting...")
    bundle  = load_bundle()
    feat_df = load_feat_df()
    pred    = predict_match(args.team1, args.team2, context, bundle, feat_df, verbose=True)
    p_model = pred["p_model"]

    # Optional: show series probability
    if args.format != "Bo1":
        p_series = series_win_prob(p_model, args.format)
        print(f"  Series win prob ({args.format}): {p_series:.1%}\n")
        print(f"  NOTE: If the Polymarket market is for the SERIES winner,")
        print(f"  use p_series={p_series:.3f} instead of p_model={p_model:.3f}\n")

    # Step 2: get market price
    p_market = None
    market   = None

    if args.condition_id:
        print("Step 2 — Fetching Polymarket price...")
        market = get_market_by_condition_id(args.condition_id)
        if market:
            p_market = get_mid_price(market.yes_token_id)
            print_market_summary(market, p_model=p_model)
        else:
            print(f"  Market {args.condition_id[:16]}... not found.")
    else:
        print("Step 2 — No --condition-id provided. Skipping price fetch.")
        print("  Find your market on polymarket.com → Esports → LoL")
        print("  Copy the condition_id from the URL and re-run with --condition-id\n")

    if p_market is None:
        p_market = float(input("Enter current Polymarket mid price (e.g. 0.55): ").strip())

    # Step 3: compute quotes
    print("Step 3 — Computing quotes...")

    params = QuoteParams(
        half_spread   = args.half_spread,
        min_edge      = args.min_edge,
        max_inventory = 0.05,
        kelly_frac    = 0.25,
    )

    quote = market_maker_quotes(p_model, p_market, params)

    fee    = taker_fee_per_share(p_market)
    bankroll = args.bankroll

    print(f"\n{'─'*55}")
    print(f"  Quote summary")
    print(f"{'─'*55}")
    print(f"  p_model:          {p_model:.3f}")
    print(f"  p_market (mid):   {p_market:.3f}")
    print(f"  Edge:             {quote.edge:+.3f}")
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

        posted = post_two_sided_quote(
            token_id  = market.yes_token_id,
            bid_price = quote.bid,
            ask_price = quote.ask,
            size      = args.size,
            dry_run   = False,
        )
        print(f"\nOrders posted successfully.")
        print(f"  Bid order ID: {posted.bid_order_id}")
        print(f"  Ask order ID: {posted.ask_order_id}")
        print(f"\n⚠ Remember to cancel orders before match starts:")
        print(f"  python -m scripts.run_maker --cancel --condition-id {market.condition_id}")
    else:
        print("Step 4 — Dry run (add --live to post real orders):")
        post_two_sided_quote(
            token_id  = market.yes_token_id,
            bid_price = quote.bid,
            ask_price = quote.ask,
            size      = args.size,
            dry_run   = True,
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
    parser.add_argument("--half-spread",  type=float, default=0.03)
    parser.add_argument("--min-edge",     type=float, default=0.02)
    parser.add_argument("--bankroll",     type=float, default=200.0)
    parser.add_argument("--size",         type=float, default=100.0,
                        help="Shares per side for quotes")

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

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
