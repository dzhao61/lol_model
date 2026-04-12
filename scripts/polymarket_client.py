"""
Polymarket CLOB client wrapper for market-making across all categories.

Handles:
- Authentication (derived from private key)
- Finding markets by team names (LoL) or condition ID (any market)
- Fetching current mid price
- Posting bid/ask quotes (GTC limit orders)
- Cancelling orders before match start
- Scanning reward markets across all categories
"""

import json
import os
import re
import sys
import requests
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions, RequestArgs
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.headers.headers import create_level_2_headers
from py_clob_client.http_helpers.helpers import get as _clob_get


def _order_options(condition_id: str) -> PartialCreateOrderOptions:
    """
    Fetch tick_size and neg_risk for a market from the CLOB.
    Both are baked into the EIP-712 order hash — if either is wrong
    the server's recomputed hash won't match and you get 'invalid signature'.
    """
    client = get_client()
    try:
        market = client.get_market(condition_id)
        tick         = str(market.get("minTickSize") or market.get("minimum_tick_size") or "0.01")
        neg_risk_raw = market.get("negRisk") or market.get("neg_risk")
        # Explicit check: don't use bool() — bool("false") == True in Python
        neg_risk     = neg_risk_raw is True or str(neg_risk_raw).lower() == "true"
    except Exception:
        tick, neg_risk = "0.01", False
    return PartialCreateOrderOptions(tick_size=tick, neg_risk=neg_risk)


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

    # signature_type controls how orders are signed:
    #   0 = EOA      — your private key IS your wallet (rare for Polymarket users)
    #   1 = Magic    — Magic.link / email login
    #   2 = Proxy    — Google login / most Polymarket accounts (most common)
    # Set POLY_SIGNATURE_TYPE in .env to override. Defaults to 2.
    sig_type = int(os.getenv("POLY_SIGNATURE_TYPE", "2"))

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        signature_type=sig_type,
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


@dataclass
class RewardMarket:
    """A Polymarket market with active liquidity rewards."""
    condition_id:      str
    question:          str
    daily_rate:        float   # USDC/day from native_daily_rate
    max_spread_c:      float   # max qualifying spread in cents
    min_shares:        int     # minimum shares per side to qualify
    competitiveness:   float   # lower = less competition (0 = sole maker)
    yes_price:         float   # current YES mid price
    capital_per_side:  float   # min_shares × yes_price (USDC needed per side)
    remaining_pool:    float   # remaining USDC in reward pool
    category:          str = ""  # tag slug from Polymarket (e.g. "esports", "politics")


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


def get_price_history(token_id: str, interval: str = "1w", fidelity: int = 100) -> list[dict]:
    """
    Fetch OHLC price history for a token from the CLOB.

    Parameters
    ----------
    token_id : YES token id
    interval : time window — "1d", "1w", "1m", "all"
    fidelity : number of data points (max ~100 for web display)

    Returns
    -------
    List of dicts with keys: t (unix timestamp), p (price 0–1)
    """
    client = get_client()
    try:
        history = client.get_prices_history(
            token_id=token_id,
            interval=interval,
            fidelity=fidelity,
        )
        # py_clob_client returns list of PricePoint objects or dicts
        result = []
        for pt in (history or []):
            if hasattr(pt, "t"):
                result.append({"t": pt.t, "p": float(pt.p)})
            elif isinstance(pt, dict):
                result.append({"t": pt.get("t") or pt.get("timestamp"), "p": float(pt.get("p") or pt.get("price", 0))})
        return result
    except Exception:
        return []


def get_market_volume(condition_id: str) -> dict:
    """
    Fetch total volume and liquidity for a market from the Gamma API.

    Returns dict with keys: volume (total USDC traded), liquidity (current)
    """
    try:
        resp = requests.get(
            f"https://gamma-api.polymarket.com/markets",
            params={"conditionId": condition_id},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        markets = data if isinstance(data, list) else data.get("markets", [])
        if markets:
            m = markets[0]
            return {
                "volume":    float(m.get("volume")    or m.get("volumeNum")    or 0),
                "liquidity": float(m.get("liquidity") or m.get("liquidityNum") or 0),
            }
    except Exception:
        pass
    return {"volume": 0.0, "liquidity": 0.0}


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
    token_id:     str,
    no_token_id:  str,
    condition_id: str,
    t1_price:     float,
    t2_price:     float,
    size:         float,
    dry_run:      bool = False,
) -> PostedQuote:
    """
    Post two standing BUY orders — one for team1's token, one for team2's token.

    Prices must already be clamped by the caller to sit below the current
    market ask for each token (so orders rest as makers, not cross as takers).

    Parameters
    ----------
    token_id     : YES token id (team1) from LoLMarket
    no_token_id  : NO token id  (team2) from LoLMarket
    condition_id : market condition_id (needed for EIP-712 order hash)
    t1_price     : price to BUY team1 token — must be < current YES ask
    t2_price     : price to BUY team2 token — must be < current NO ask (= 1 − YES bid)
    size         : number of shares for each side
    dry_run      : if True, print orders but don't submit
    """
    if t1_price + t2_price >= 1.0:
        raise ValueError(
            f"Crossed market: team1={t1_price} + team2={t2_price} >= 1.00. "
            "No guaranteed profit if both fill."
        )

    client = get_client()
    opts   = _order_options(condition_id)

    bid_id = None
    ask_id = None

    if dry_run:
        print(f"[DRY RUN] Would BUY team1 (YES) {size:.0f} shares @ {t1_price:.3f}")
        print(f"[DRY RUN] Would BUY team2 (NO)  {size:.0f} shares @ {t2_price:.3f}")
    else:
        bid_resp = client.create_and_post_order(
            OrderArgs(token_id=token_id, side=BUY, price=t1_price, size=size),
            opts,
        )
        ask_resp = client.create_and_post_order(
            OrderArgs(token_id=no_token_id, side=BUY, price=t2_price, size=size),
            opts,
        )

        bid_id = bid_resp.get("orderID") if isinstance(bid_resp, dict) else None
        ask_id = ask_resp.get("orderID") if isinstance(ask_resp, dict) else None

        print(f"Posted BUY team1 (YES) {size:.0f} @ {t1_price:.3f}  [id: {str(bid_id)[:8]}...]")
        print(f"Posted BUY team2 (NO)  {size:.0f} @ {t2_price:.3f}  [id: {str(ask_id)[:8]}...]")

    return PostedQuote(
        bid_order_id=bid_id,
        ask_order_id=ask_id,
        bid_price=t1_price,
        ask_price=t2_price,
        size=size,
        token_id=token_id,
    )


def post_directional_order(
    token_id:     str,
    condition_id: str,
    price:        float,
    size:         float,
    dry_run:      bool = False,
) -> str | None:
    """
    Post a single taker BUY order for the given token (YES or NO).

    To buy YES: pass yes_token_id and the current YES ask price.
    To buy NO:  pass no_token_id  and the current NO  ask price (= 1 − yes_bid).

    Posts at `price + 1 tick` to ensure the order crosses as a taker.
    Returns the order ID, or None on dry run.
    """
    TICK = 0.01
    taker_price = round(min(price + TICK, 0.99), 2)

    client = get_client()

    if dry_run:
        print(f"[DRY RUN] Would BUY  {size:.0f} shares @ {taker_price:.2f}")
        return None

    order = client.create_and_post_order(
        OrderArgs(token_id=token_id, side=BUY, price=taker_price, size=size),
        _order_options(condition_id),
    )
    order_id = order.get("orderID") if order else None
    print(f"Directional BUY  {size:.0f} @ {taker_price:.2f}  [id: {str(order_id)[:8]}...]")
    return order_id


def post_single_side_quote(
    token_id:     str,
    condition_id: str,
    price:        float,
    size:         float,
) -> str | None:
    """
    Post a single resting maker BUY order at exactly `price`.

    Used for Tier 2 rebalancing: when one side fills, re-post the unfilled
    side tighter (closer to current mid) to attract the complementary fill.
    Unlike post_directional_order, this does NOT add a tick — it rests passively.
    """
    client = get_client()
    order = client.create_and_post_order(
        OrderArgs(token_id=token_id, side=BUY, price=price, size=size),
        _order_options(condition_id),
    )
    order_id = order.get("orderID") if order else None
    print(f"Re-quoted BUY  {size:.0f} @ {price:.2f}  [id: {str(order_id)[:8]}...]")
    return order_id


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
    if isinstance(orders, list):
        return orders
    if isinstance(orders, dict):
        return orders.get("data", orders.get("orders", []))
    return []


def get_all_open_orders() -> list[dict]:
    """Return all open orders across all markets for this API key."""
    client = get_client()
    from py_clob_client.clob_types import OpenOrderParams
    END_CURSOR = "LTE="
    cursor = "MA=="
    results = []
    while True:
        page = client.get_orders(OpenOrderParams(), next_cursor=cursor)
        if isinstance(page, list):
            results.extend(page)
            break
        if isinstance(page, dict):
            results.extend(page.get("data", page.get("orders", [])))
            cursor = page.get("next_cursor", END_CURSOR)
            if cursor == END_CURSOR:
                break
        else:
            break
    return results


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


def check_order_scoring(order_ids: list[str]) -> None:
    """
    Print whether each order is currently scoring for liquidity rewards.

    An order scores when it rests within 3¢ of the market mid.  Both
    sides must score simultaneously for Q_min > 0 (full two-sided reward).
    """
    from py_clob_client.clob_types import OrdersScoringParams
    client = get_client()

    ids = [oid for oid in order_ids if oid]
    if not ids:
        print("No order IDs provided.")
        return

    result = client.are_orders_scoring(OrdersScoringParams(orderIds=ids))

    print(f"\n{'─'*50}")
    print(f"  Reward scoring check ({len(ids)} orders)")
    print(f"{'─'*50}")

    # Response is {orderID: bool}
    if not isinstance(result, dict):
        print(f"  Unexpected response format: {result}")
        return

    scoring_map: dict[str, bool] = result

    both_scoring = True
    for oid in ids:
        scoring = scoring_map.get(oid, False)
        icon = "✓" if scoring else "✗"
        label = "SCORING" if scoring else "NOT SCORING (outside 3¢ window)"
        print(f"  {icon}  {oid[:16]}...  {label}")
        if not scoring:
            both_scoring = False

    if len(ids) >= 2:
        print()
        if both_scoring:
            print("  Both orders scoring → two-sided Q_min > 0 → earning rewards ✓")
        else:
            print("  ⚠ One or both orders not scoring → Q_min = 0 → no rewards this period")
    print(f"{'─'*50}\n")


def _authed_get(path: str, params: dict | None = None) -> dict | list:
    """Make an authenticated L2 GET request to the CLOB API."""
    client = get_client()
    sig_type = int(os.environ.get("POLY_SIGNATURE_TYPE", "1"))
    query = f"?signature_type={sig_type}"
    if params:
        for k, v in params.items():
            query += f"&{k}={v}"
    full_path = path + query
    req = RequestArgs(method="GET", request_path=full_path)
    headers = create_level_2_headers(client.signer, client.creds, req)
    return _clob_get(f"{client.host}{full_path}", headers=headers)


def check_reward_percentages(condition_id: str | None = None) -> None:
    """
    Show your real-time liquidity reward % share per market.
    A non-zero % means your orders are currently scoring and earning rewards.
    """
    result = _authed_get("/rewards/user/percentages")

    print(f"\n{'─'*55}")
    print(f"  Real-time reward market share")
    print(f"{'─'*55}")

    if not isinstance(result, dict) or not result:
        print(f"  No active reward positions found.")
        print(f"  (This means 0% share — orders may not be scoring)")
        print(f"{'─'*55}\n")
        return

    for cid, pct in result.items():
        marker = " ← THIS MARKET" if condition_id and cid == condition_id else ""
        print(f"  {cid[:20]}...  {pct:.2f}%{marker}")

    if condition_id and condition_id not in result:
        print(f"\n  {condition_id[:20]}...  0.00%  ← THIS MARKET (not scoring)")

    print(f"{'─'*55}\n")


def check_reward_earnings(date: str | None = None) -> None:
    """
    Show actual USDC earnings from liquidity rewards for a given date (YYYY-MM-DD).
    Defaults to today.
    """
    from datetime import date as _date
    if date is None:
        date = _date.today().isoformat()

    result = _authed_get("/rewards/user", params={"date": date})

    print(f"\n{'─'*55}")
    print(f"  Reward earnings for {date}")
    print(f"{'─'*55}")

    rows = result if isinstance(result, list) else result.get("data", []) if isinstance(result, dict) else []

    if not rows:
        print(f"  No earnings recorded for {date}.")
        print(f"  (Rewards are updated once daily — check again tomorrow for today's earnings)")
        print(f"{'─'*55}\n")
        return

    total = 0.0
    for r in rows:
        cid      = r.get("condition_id", "?")[:20]
        earnings = float(r.get("earnings", 0))
        total   += earnings
        print(f"  {cid}...  ${earnings:.4f}")

    print(f"  {'─'*40}")
    print(f"  Total:  ${total:.4f}")
    print(f"{'─'*55}\n")


def check_market_rewards(condition_id: str) -> None:
    """
    Show the reward configuration for a specific market.
    Uses the public /rewards/markets/{condition_id} endpoint (no auth needed).
    """
    import requests
    client = get_client()
    url = f"{client.host}/rewards/markets/{condition_id}"
    resp = requests.get(url, timeout=10)
    data = resp.json() if resp.ok else {}

    # Response may be a list or dict with a 'data' key
    items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []

    print(f"\n{'─'*55}")
    print(f"  Reward config for market")
    print(f"  {condition_id[:40]}...")
    print(f"{'─'*55}")

    if not items:
        print(f"  ✗  NO liquidity rewards for this market")
        print(f"{'─'*55}\n")
        return

    item = items[0] if isinstance(items, list) else items
    max_spread   = item.get("rewards_max_spread", "?")
    min_size     = item.get("rewards_min_size", "?")
    daily_rate   = item.get("native_daily_rate") or item.get("total_daily_rate", "?")
    competitive  = item.get("market_competitiveness", "?")

    configs = item.get("rewards_config", [])
    total_pool = sum(float(c.get("total_rewards", 0)) for c in configs) if configs else None
    remaining  = sum(float(c.get("remaining_reward_amount", 0)) for c in configs) if configs else None

    print(f"  ✓  Rewards ACTIVE")
    print(f"     Max spread:      {max_spread}¢")
    print(f"     Min size:        {min_size} shares")
    print(f"     Daily rate:      ${daily_rate}")
    if total_pool is not None:
        print(f"     Total pool:      ${total_pool:,.0f}")
    if remaining is not None:
        print(f"     Remaining:       ${remaining:,.0f}")
    if competitive != "?":
        print(f"     Competitiveness: {competitive}")
    print(f"{'─'*55}\n")


def find_reward_markets(
    max_capital_per_side: float = 1000.0,
    min_daily_rate: float = 10.0,
    categories: list[str] | None = None,
) -> list[RewardMarket]:
    """
    Scan Polymarket for markets with active liquidity rewards that are
    affordable within the given capital budget.

    Returns a list of RewardMarket sorted by daily_rate descending
    (best opportunities first).

    Parameters
    ----------
    max_capital_per_side : max USDC per side (min_shares × price ≤ this)
    min_daily_rate       : minimum $/day reward pool to include
    categories           : list of tag slugs to filter by (e.g. ["esports", "politics"]).
                           None or empty = fetch ALL categories (no tag filter).
    """
    client = get_client()
    base_url = f"{client.host}/rewards/markets/multi"
    base_params = {
        "order_by": "rate_per_day",
        "position": "DESC",
        "page_size": 500,
    }

    # Build list of (slug, params) to fetch — one request per category, or one unfiltered
    if not categories:
        fetch_list = [(None, base_params)]
    else:
        fetch_list = [(slug, {**base_params, "tag_slug": slug}) for slug in categories]

    seen: set[str] = set()
    results: list[RewardMarket] = []

    for cat_slug, params in fetch_list:
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            if not resp.ok:
                continue
            data = resp.json()
        except Exception:
            continue

        items = data if isinstance(data, list) else data.get("data", [])

        for item in items:
            if not isinstance(item, dict):
                continue

            cid         = item.get("condition_id") or item.get("conditionId", "")
            if not cid or cid in seen:
                continue

            question    = item.get("question", "")
            max_spread  = float(item.get("rewards_max_spread", 3))       # cents
            min_shares  = int(item.get("rewards_min_size", 0))
            competitive = float(item.get("market_competitiveness", 0) or 0)
            daily_rate  = float(
                item.get("native_daily_rate")
                or item.get("total_daily_rate")
                or 0
            )

            # Extract remaining pool from rewards_config
            configs = item.get("rewards_config") or []
            remaining = sum(
                float(c.get("remaining_reward_amount", 0) or 0) for c in configs
            )

            # Get YES price from tokens list
            tokens = item.get("tokens") or []
            yes_price = 0.5  # fallback
            for tok in tokens:
                outcome = (tok.get("outcome") or "").lower()
                if outcome in ("yes", "1", "true", "win") or (
                    tokens.index(tok) == 0 and len(tokens) == 2
                ):
                    try:
                        yes_price = float(tok.get("price", 0.5) or 0.5)
                        break
                    except (ValueError, TypeError):
                        pass

            capital_per_side = min_shares * yes_price

            # Apply filters
            if daily_rate < min_daily_rate:
                continue
            if min_shares > 0 and capital_per_side > max_capital_per_side:
                continue

            # Determine category: prefer explicit tag_slug on item, fall back to the
            # slug we used when querying (cat_slug), or first tag in tags list.
            item_cat = (
                item.get("tag_slug")
                or (item.get("tags") or [{}])[0].get("slug", "")
                or cat_slug
                or ""
            )

            seen.add(cid)
            results.append(RewardMarket(
                condition_id=cid,
                question=question,
                daily_rate=daily_rate,
                max_spread_c=max_spread,
                min_shares=min_shares,
                competitiveness=competitive,
                yes_price=yes_price,
                capital_per_side=capital_per_side,
                remaining_pool=remaining,
                category=str(item_cat),
            ))

    results.sort(key=lambda m: m.daily_rate, reverse=True)
    return results



def print_reward_markets(markets: list[RewardMarket]) -> None:
    """Print a formatted table of reward market opportunities."""
    if not markets:
        print("No reward markets found matching your criteria.")
        return

    print(f"\n{'─'*108}")
    print(f"  {'MARKET':<42} {'CATEGORY':<12} {'$/DAY':>7}  {'SPR':>4}  {'MIN SH':>7}  {'CAP/SIDE':>9}  {'COMP':>5}  {'REMAIN':>8}")
    print(f"{'─'*108}")
    for m in markets:
        comp_str = f"{m.competitiveness:.1f}"
        q = m.question[:41]
        cat = (m.category or "—")[:11]
        print(
            f"  {q:<42} {cat:<12} ${m.daily_rate:>6.0f}  ±{m.max_spread_c:.0f}¢  "
            f"{m.min_shares:>7}  ${m.capital_per_side:>8.0f}  {comp_str:>5}  ${m.remaining_pool:>7.0f}"
        )
    print(f"{'─'*108}")
    print(f"  {len(markets)} markets  ·  Sorted by daily rate\n")


def merge_positions(condition_id: str, amount: float) -> None:
    """
    Merge YES + NO tokens back into USDC.e after both sides fill.

    This requires an on-chain transaction via the Polymarket CTF contract:
        Contract: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
        Function: mergePositions(collateral, parentCollectionId, conditionId, partition, amount)
        Partition: [1, 2]  (YES=1, NO=2)

    Currently requires web3.py or py_builder_relayer_client (neither installed).
    To merge manually: go to polymarket.com → Portfolio → find the market → Merge.

    This function is a placeholder — it will print instructions until web3 is available.
    """
    print(f"\n{'─'*55}")
    print(f"  Merge positions — {amount:.0f} YES + {amount:.0f} NO → ${amount:.2f} USDC.e")
    print(f"")
    print(f"  ⚠ Automated merge requires web3.py (not installed).")
    print(f"  Manual steps:")
    print(f"    1. Go to polymarket.com/portfolio")
    print(f"    2. Find this market: {condition_id[:20]}...")
    print(f"    3. Click 'Merge' to convert {amount:.0f} token pairs → USDC.e")
    print(f"    4. Capital is recycled and ready for the next market")
    print(f"{'─'*55}\n")
