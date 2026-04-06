"""
Polymarket CLOB client wrapper for LoL market-making.

Handles:
- Authentication (derived from private key)
- Finding LoL markets by team names
- Fetching current mid price
- Posting bid/ask quotes (GTC limit orders)
- Cancelling orders before match start
"""

import json
import os
import re
import sys
import time
import requests
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL


# ─── Client setup ─────────────────────────────────────────────────────────────

_client: ClobClient | None = None


def get_client() -> ClobClient:
    """Return a singleton authenticated ClobClient."""
    global _client
    if _client is not None:
        return _client

    private_key = os.getenv("POLY_PRIVATE_KEY")
    funder      = os.getenv("POLY_FUNDER")

    if not private_key or not funder:
        raise EnvironmentError("POLY_PRIVATE_KEY and POLY_FUNDER must be set in .env")

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        signature_type=0,   # EOA — standard for Google/Magic login keys
        funder=funder,
    )

    # Derive CLOB API credentials from the private key
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    _client = client
    return client


# ─── Market lookup ────────────────────────────────────────────────────────────

@dataclass
class LoLMarket:
    condition_id:  str
    question:      str
    yes_token_id:  str
    no_token_id:   str
    end_date:      str
    active:        bool


# LoL-specific word-boundary patterns — prevent substring matches like
# "lol" inside "Lightfoot" or "lec" inside "election"
_LOL_PATTERNS = [
    re.compile(r"\bleague of legends\b", re.I),
    re.compile(r"\blck\b",  re.I),
    re.compile(r"\blpl\b",  re.I),
    re.compile(r"\blec\b",  re.I),
    re.compile(r"\blcs\b",  re.I),
    re.compile(r"\bvcs\b",  re.I),
    re.compile(r"\bpcs\b",  re.I),
    re.compile(r"\bcblol\b",re.I),
    re.compile(r"\bmsi\b",  re.I),  # Mid-Season Invitational
    re.compile(r"\bworlds\b",re.I), # LoL Worlds
    re.compile(r"\bgen\.?g\b",     re.I),
    re.compile(r"\bcloud9\b",      re.I),
    re.compile(r"\bteam liquid\b", re.I),
    re.compile(r"\bfnatic\b",      re.I),
    re.compile(r"\bg2 esports\b",  re.I),
    re.compile(r"\bt1\b",          re.I),
    re.compile(r"\bfaker\b",       re.I),
]


def _is_lol_market(question: str, description: str = "") -> bool:
    """Return True only if the market is clearly a LoL/esports market."""
    text = question + " " + description
    return any(p.search(text) for p in _LOL_PATTERNS)


_MATCH_WINNER_RE = re.compile(
    r"LoL:\s+.+\s+vs\s+.+\s+\(BO\d\)", re.I
)
_PROP_SKIP_RE = re.compile(
    r"\bGame [1-9]\b"
    r"|\bO/U\b"
    r"|\bhandicap\b"
    r"|\bodd[/ ]?even\b"
    r"|\bboth teams\b"
    r"|\bany player\b"
    r"|\btotal kills\b"
    r"|\bGames Total\b",
    re.I,
)


def find_lol_markets(query: str = "", league: str = "", limit: int = 50) -> list[LoLMarket]:
    """
    Search for active LoL head-to-head match markets using the Gamma events API.

    Only series/match winner markets are returned (e.g. "LoL: T1 vs Gen.G (BO3)").
    Season winner, props (Game 1 winner, O/U, etc.) are excluded.

    Parameters
    ----------
    query  : optional text filter applied to question/slug (team name, league…)
    league : optional league code filter, e.g. "LCK", "LPL", "LEC", "LCS"
    limit  : max match markets to return

    Returns
    -------
    List of LoLMarket objects representing series/match winner markets.
    """
    results = []

    # ── Primary: Gamma events API with esports tag ────────────────────────────
    try:
        offset = 0
        page_size = 100

        while len(results) < limit:
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "tag_slug": "esports",
                    "active":   "true",
                    "closed":   "false",
                    "limit":    page_size,
                    "offset":   offset,
                },
                timeout=10,
            )
            resp.raise_for_status()
            events = resp.json() if isinstance(resp.json(), list) else resp.json().get("events", [])

            if not events:
                break

            for ev in events:
                ev_title = ev.get("title", "")
                ev_slug  = ev.get("slug", "")

                # Must be a LoL event
                if not _is_lol_market(ev_title, ev_slug):
                    continue

                # Optional league filter (applied to event title / slug)
                if league:
                    lc = league.lower()
                    ev_title_l = ev_title.lower()
                    ev_slug_l  = ev_slug.lower()
                    if lc not in ev_title_l and lc not in ev_slug_l:
                        continue
                    # Exclude sub-leagues that contain the league name as a prefix.
                    # e.g. "LCK Challenger" must not appear when filtering for "LCK".
                    _SUBLEVEL_SUFFIXES = ("challenger", "academy", "amateur", "rising", "next")
                    if any(
                        f"{lc} {suffix}" in ev_title_l or f"{lc}-{suffix}" in ev_slug_l
                        for suffix in _SUBLEVEL_SUFFIXES
                    ):
                        continue

                # Optional free-text query
                if query:
                    ql = query.lower()
                    if ql not in ev_title.lower() and ql not in ev_slug.lower():
                        continue

                # Only keep the head-to-head match winner market
                for m in ev.get("markets", []):
                    q = m.get("question", "")

                    # POSITIVE filter: must match "LoL: A vs B (BON)" format
                    if not _MATCH_WINNER_RE.match(q):
                        continue
                    # NEGATIVE filter: skip any prop that slipped through
                    if _PROP_SKIP_RE.search(q):
                        continue

                    raw_ids = m.get("clobTokenIds", []) or []
                    if isinstance(raw_ids, str):
                        try:
                            raw_ids = json.loads(raw_ids)
                        except (json.JSONDecodeError, ValueError):
                            raw_ids = []

                    # Parse outcomes (also JSON-encoded string) to validate token order.
                    # Gamma API returns outcomes in same positional order as clobTokenIds.
                    # Question format: "LoL: TeamA vs TeamB (BON) - ..."
                    # outcomes[0] should correspond to TeamA (team1 in the question).
                    raw_outcomes = m.get("outcomes", []) or []
                    if isinstance(raw_outcomes, str):
                        try:
                            raw_outcomes = json.loads(raw_outcomes)
                        except (json.JSONDecodeError, ValueError):
                            raw_outcomes = []

                    # Extract team1 from question to cross-check outcomes[0]
                    _q_match = re.match(r"LoL:\s+(.+?)\s+vs\s+(.+?)\s+\(BO\d\)", q, re.I)
                    if (
                        _q_match
                        and len(raw_outcomes) >= 2
                        and len(raw_ids) >= 2
                    ):
                        team1_q = _q_match.group(1).strip().lower()
                        out0    = raw_outcomes[0].strip().lower()
                        out1    = raw_outcomes[1].strip().lower()
                        # If outcomes[0] is closer to team2 than team1, swap
                        if out1 == team1_q or (
                            out0 != team1_q and team1_q in out1 and team1_q not in out0
                        ):
                            raw_ids = [raw_ids[1], raw_ids[0]]

                    yes_id = raw_ids[0] if len(raw_ids) > 0 else ""
                    no_id  = raw_ids[1] if len(raw_ids) > 1 else ""

                    results.append(LoLMarket(
                        condition_id = m.get("conditionId", ""),
                        question     = q,
                        yes_token_id = yes_id,
                        no_token_id  = no_id,
                        end_date     = m.get("endDateIso") or ev.get("endDate", ""),
                        active       = True,
                    ))

                    if len(results) >= limit:
                        break

                if len(results) >= limit:
                    break

            offset += page_size

    except Exception as e:
        print(f"Gamma events API unavailable ({e}), falling back to CLOB search...")

    # ── Fallback: CLOB API with strict word-boundary filtering ─────────────────
    if not results:
        client = get_client()
        raw    = client.get_markets()
        for m in raw.get("data", []):
            q    = m.get("question", "")
            desc = m.get("description", "") or ""

            if not _is_lol_market(q, desc):
                continue
            if query and query.lower() not in q.lower() and query.lower() not in desc.lower():
                continue
            if not m.get("active", False):
                continue

            tokens    = m.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            no_token  = next((t for t in tokens if t.get("outcome", "").upper() == "NO"),  None)
            # LoL markets use team names — fall back to positional
            if not yes_token and len(tokens) >= 1:
                yes_token = tokens[0]
            if not no_token and len(tokens) >= 2:
                no_token = tokens[1]
            if not yes_token or not no_token:
                continue

            results.append(LoLMarket(
                condition_id = m.get("condition_id", ""),
                question     = m.get("question", ""),
                yes_token_id = yes_token.get("token_id", ""),
                no_token_id  = no_token.get("token_id", ""),
                end_date     = m.get("end_date_iso", ""),
                active       = True,
            ))
            if len(results) >= limit:
                break

    return results


def get_market_by_condition_id(condition_id: str) -> LoLMarket | None:
    """Look up a specific market by its condition_id."""
    client = get_client()
    m = client.get_market(condition_id)
    if not m:
        return None

    tokens = m.get("tokens", [])

    # LoL markets use team names as outcomes (not YES/NO).
    # Try YES/NO first; fall back to positional (first token = team1/YES).
    yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
    no_token  = next((t for t in tokens if t.get("outcome", "").upper() == "NO"),  None)
    if not yes_token and len(tokens) >= 1:
        yes_token = tokens[0]
    if not no_token and len(tokens) >= 2:
        no_token = tokens[1]

    return LoLMarket(
        condition_id = m.get("condition_id", ""),
        question     = m.get("question", ""),
        yes_token_id = yes_token.get("token_id", "") if yes_token else "",
        no_token_id  = no_token.get("token_id",  "") if no_token  else "",
        end_date     = m.get("end_date_iso", ""),
        active       = m.get("active", False),
    )


# ─── Orderbook / price ────────────────────────────────────────────────────────

def _book_side(entries) -> list[tuple[float, float]]:
    """Convert a list of OrderSummary or dicts to (price, size) float tuples."""
    result = []
    for e in (entries or []):
        if hasattr(e, "price"):
            result.append((float(e.price), float(e.size)))
        else:
            result.append((float(e["price"]), float(e["size"])))
    return result


def _best_bids(entries, depth: int = 5) -> list[tuple[float, float]]:
    """
    Return the top-of-book bids (highest prices first).

    py_clob_client returns bids in ASCENDING order (worst/lowest first,
    best/highest last). Reverse to get best-first, take depth entries.
    """
    raw = _book_side(entries)
    # Sort descending by price to be safe, then take top depth
    raw.sort(key=lambda x: x[0], reverse=True)
    return raw[:depth]


def _best_asks(entries, depth: int = 5) -> list[tuple[float, float]]:
    """
    Return the top-of-book asks (lowest prices first).

    py_clob_client returns asks in DESCENDING order (worst/highest first,
    best/lowest last). Sort ascending to get best-first, take depth entries.
    """
    raw = _book_side(entries)
    raw.sort(key=lambda x: x[0])
    return raw[:depth]


def get_mid_price(token_id: str) -> float | None:
    """
    Get current mid price for a token (average of best bid and best ask).

    Returns None if the book is empty.
    """
    client = get_client()
    book   = client.get_order_book(token_id)

    bids = _best_bids(book.bids if hasattr(book, "bids") else book.get("bids", []), depth=1)
    asks = _best_asks(book.asks if hasattr(book, "asks") else book.get("asks", []), depth=1)

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    elif best_bid is not None:
        return best_bid
    elif best_ask is not None:
        return best_ask
    return None


def get_orderbook_snapshot(token_id: str, depth: int = 5) -> dict:
    """Return top-of-book snapshot: best N bids (descending) and asks (ascending)."""
    client = get_client()
    book   = client.get_order_book(token_id)

    bids = _best_bids(book.bids if hasattr(book, "bids") else book.get("bids", []), depth)
    asks = _best_asks(book.asks if hasattr(book, "asks") else book.get("asks", []), depth)
    mid  = get_mid_price(token_id)

    return {"bids": bids, "asks": asks, "mid": mid}


# ─── Order posting ────────────────────────────────────────────────────────────

@dataclass
class PostedQuote:
    bid_order_id:  str | None
    ask_order_id:  str | None
    bid_price:     float
    ask_price:     float
    size:          float
    token_id:      str


def post_two_sided_quote(
    token_id:  str,
    bid_price: float,
    ask_price: float,
    size:      float,
    dry_run:   bool = False,
) -> PostedQuote:
    """
    Post a bid and ask for the YES token.

    Parameters
    ----------
    token_id  : YES token id from LoLMarket
    bid_price : price to BUY YES (e.g. 0.47)
    ask_price : price to SELL YES (e.g. 0.53)
    size      : number of shares for each side
    dry_run   : if True, print orders but don't submit

    Returns
    -------
    PostedQuote with order IDs
    """
    if bid_price >= ask_price:
        raise ValueError(
            f"Crossed market: bid={bid_price} >= ask={ask_price}. "
            "This would cause an immediate loss on every fill."
        )

    client = get_client()

    bid_id = None
    ask_id = None

    if dry_run:
        print(f"[DRY RUN] Would post BID  {size:.0f} shares @ {bid_price:.3f}")
        print(f"[DRY RUN] Would post ASK  {size:.0f} shares @ {ask_price:.3f}")
    else:
        # Post bid (buy YES)
        bid_order = client.create_and_post_order(
            OrderArgs(token_id=token_id, side=BUY,  price=bid_price, size=size),
            order_type=OrderType.GTC,
        )
        bid_id = bid_order.get("orderID") if bid_order else None

        time.sleep(0.3)   # brief pause between orders

        # Post ask (sell YES)
        ask_order = client.create_and_post_order(
            OrderArgs(token_id=token_id, side=SELL, price=ask_price, size=size),
            order_type=OrderType.GTC,
        )
        ask_id = ask_order.get("orderID") if ask_order else None

        print(f"Posted BID  {size:.0f} @ {bid_price:.3f}  [id: {str(bid_id)[:8]}...]")
        print(f"Posted ASK  {size:.0f} @ {ask_price:.3f}  [id: {str(ask_id)[:8]}...]")

    return PostedQuote(
        bid_order_id=bid_id,
        ask_order_id=ask_id,
        bid_price=bid_price,
        ask_price=ask_price,
        size=size,
        token_id=token_id,
    )


# ─── Order management ─────────────────────────────────────────────────────────

def cancel_all_orders(condition_id: str) -> None:
    """Cancel all open orders for a market. Call this before match start."""
    client = get_client()
    result = client.cancel_market_orders(market=condition_id)
    print(f"Cancelled all orders for market {condition_id[:12]}... — {result}")


def get_open_orders(condition_id: str) -> list[dict]:
    """Return list of open orders for this market."""
    client = get_client()
    from py_clob_client.clob_types import OpenOrderParams
    orders = client.get_orders(OpenOrderParams(market=condition_id))
    return orders if isinstance(orders, list) else []


def print_market_summary(market: LoLMarket, p_model: float | None = None) -> None:
    """Print a human-readable summary of a market and current orderbook."""
    snap = get_orderbook_snapshot(market.yes_token_id)

    print(f"\n{'─'*60}")
    print(f"Market:  {market.question}")
    print(f"ID:      {market.condition_id[:16]}...")
    print(f"Ends:    {market.end_date[:10]}")
    print(f"")
    print(f"Orderbook (YES token):")
    if snap["asks"]:
        for p, s in reversed(snap["asks"][:3]):
            print(f"  ASK  {p:.3f}  ({s:.0f} shares)")
    print(f"  --- mid: {snap['mid']:.3f} ---" if snap["mid"] else "  --- no mid ---")
    if snap["bids"]:
        for p, s in snap["bids"][:3]:
            print(f"  BID  {p:.3f}  ({s:.0f} shares)")
    if p_model is not None:
        print(f"")
        print(f"Your model:  p_model = {p_model:.3f}")
        edge = p_model - (snap["mid"] or 0.5)
        print(f"Edge vs mid: {edge:+.3f}")
    print(f"{'─'*60}\n")
