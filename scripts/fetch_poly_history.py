"""
Fetch all closed/resolved LoL markets from Polymarket and save a historical
outcome dataset for backtesting.

No API credentials required — uses public Gamma API only.

Findings from API investigation (April 2026)
--------------------------------------------
- Polymarket uses TWO question formats for LoL markets:
    Old: "LoL: T1 vs Gen.G (BO3)"        (2023-2024 era)
    New: "LCK: T1 vs Gen.G"              (2024+ era, no "LoL:" prefix or "(BON)")
  Both are covered by the expanded _LOL_MATCH_RE pattern below.
- Resolution outcome is encoded in `outcomePrices` (NOT a `winner` field):
    Resolved team1 win:  outcomePrices = ["1", "0"]
    Resolved team2 win:  outcomePrices = ["0", "1"]
- CLOB price history API returns empty for resolved/old markets.
  Pre-match prices are not publicly archived.
  => p_market is set to None here; backtest.py uses elo_expected as proxy.

Output: cache/poly_historical_markets.csv

Columns
-------
condition_id, question, team1, team2, end_date, result (1=team1 won, 0=team2),
last_trade_price, volume, yes_token_id, n_outcomes

Usage
-----
    python -m scripts.fetch_poly_history [--max 5000] [--force]
"""

import json
import re
import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR  = ROOT / "cache"
OUTPUT_CSV = CACHE_DIR / "poly_historical_markets.csv"
GAMMA_HOST = "https://gamma-api.polymarket.com"

CACHE_DIR.mkdir(exist_ok=True)


# ─── LoL market detection ─────────────────────────────────────────────────────

# Match both old ("LoL: T1 vs Gen.G (BO3)") and new ("LCK: T1 vs Gen.G") formats.
# Group 1 = league/prefix, Group 2 = team1, Group 3 = team2.
_LOL_MATCH_RE = re.compile(
    r"^(LoL|LCK|LPL|LEC|LCS|MSI|Worlds?|LCK CL|LPL All-Star|LEC\s+\w+|LCS\s+\w+)"
    r"[\s:\-]+(.+?)\s+vs\.?\s+(.+?)(?:\s+\(BO\d+\))?$",
    re.I,
)

# Prop markets that should be excluded even if they match the above
_PROP_SKIP_RE = re.compile(
    r"\bGame [1-9]\b"
    r"|\bO/U\b"
    r"|\bover/under\b"
    r"|\bhandicap\b"
    r"|\bodd[/ ]?even\b"
    r"|\btotal kills\b"
    r"|\bGames Total\b"
    r"|\bwinner\b",        # "LCK Spring winner" = season-winner market, not head-to-head
    re.I,
)

# Phrases in the EVENT title that confirm it's a LoL event
_LOL_EVENT_RE = re.compile(
    r"\bleague of legends\b|\blck\b|\blpl\b|\blec\b|\blcs\b"
    r"|\bmsi\b|\bworlds?\b|\bvcs\b|\bpcs\b|\bcblol\b",
    re.I,
)

# Non-LoL esports that share league-like acronyms — blocklist
_NOT_LOL_RE = re.compile(
    r"\bcounter.strike\b|\bcs2?\b|\bcsg[o0]\b"
    r"|\bvalorant\b|\bvct\b"
    r"|\bdota\b|\boverwatch\b|\brainbow six\b"
    r"|\bstarcraft\b|\brocket league\b",
    re.I,
)


def _is_lol_market(question: str, event_title: str = "") -> bool:
    """Return True if the question/event is clearly a LoL H2H match."""
    combined = f"{question} {event_title}"
    if _NOT_LOL_RE.search(combined):
        return False
    return bool(_LOL_MATCH_RE.match(question)) and bool(_LOL_EVENT_RE.search(combined))


def _parse_teams(question: str) -> tuple[str, str] | None:
    """Extract (team1, team2) from the question string."""
    m = _LOL_MATCH_RE.match(question)
    if not m:
        return None
    t1 = m.group(2).strip().rstrip(".")
    t2 = m.group(3).strip().rstrip(".")
    return t1, t2


def _outcome_from_prices(outcome_prices: list | str) -> int | None:
    """
    Determine which team won from Gamma's outcomePrices field.

    outcomePrices is a JSON-encoded list of price strings, parallel to
    the outcomes/clobTokenIds arrays. For a resolved binary market:
        ["1", "0"]  → team1 (YES token, index 0) won  → result = 1
        ["0", "1"]  → team2 (NO  token, index 1) won  → result = 0

    For an unresolved market the prices are probabilistic floats, and
    we cannot determine a winner yet.
    """
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except Exception:
            return None
    if not isinstance(outcome_prices, list) or len(outcome_prices) < 2:
        return None
    try:
        p0 = float(outcome_prices[0])
        p1 = float(outcome_prices[1])
    except (TypeError, ValueError):
        return None
    if p0 >= 0.9 and p1 <= 0.1:   # team1 won (YES resolved to 1)
        return 1
    if p0 <= 0.1 and p1 >= 0.9:   # team2 won (NO resolved to 1)
        return 0
    return None                    # still in progress or ambiguous


def _fix_token_order(raw_ids: list, raw_outcomes: list, team1: str) -> list:
    """Ensure raw_ids[0] corresponds to team1 (YES token)."""
    if len(raw_outcomes) >= 2 and len(raw_ids) >= 2:
        t1l  = team1.lower()
        out0 = str(raw_outcomes[0]).strip().lower()
        out1 = str(raw_outcomes[1]).strip().lower()
        if out1 == t1l or (out0 != t1l and t1l in out1 and t1l not in out0):
            return [raw_ids[1], raw_ids[0]]
    return raw_ids


# ─── Main fetch ───────────────────────────────────────────────────────────────

def fetch_historical_markets(
    max_markets: int = 5000,
    force: bool = False,
) -> pd.DataFrame:
    """
    Paginate Gamma API events (closed, esports) and collect all resolved
    LoL head-to-head match markets.

    Parameters
    ----------
    max_markets : stop after this many NEW markets
    force       : if True, clear the existing CSV and re-fetch everything

    Returns a DataFrame and writes OUTPUT_CSV (incrementally, unless force=True).
    """
    if force and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        print("Cleared existing CSV (--force).")

    existing: pd.DataFrame | None = None
    existing_cids: set[str] = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, dtype={"yes_token_id": str})
        existing_cids = set(existing["condition_id"].dropna().tolist())
        print(f"Cache: {len(existing_cids)} markets already saved — will skip.")

    records: list[dict] = []
    offset    = 0
    page_size = 100
    total_new = 0
    empty_pages = 0

    print("Scanning Gamma events (closed, esports) …")

    while total_new < max_markets:
        try:
            resp = requests.get(
                f"{GAMMA_HOST}/events",
                params={
                    "tag_slug": "esports",
                    "active":   "false",
                    "closed":   "true",
                    "limit":    page_size,
                    "offset":   offset,
                },
                timeout=20,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"  Gamma error at offset {offset}: {e}")
            break

        data   = resp.json()
        events = data if isinstance(data, list) else data.get("events", [])

        if not events:
            empty_pages += 1
            if empty_pages >= 3:
                print(f"  {empty_pages} consecutive empty pages at offset {offset} — stopping.")
                break
            offset += page_size
            continue
        empty_pages = 0

        for ev in events:
            ev_title = ev.get("title", "")

            for m in ev.get("markets", []):
                q = m.get("question", "")

                # Must be a LoL H2H match-winner market
                if not _is_lol_market(q, ev_title):
                    continue
                if _PROP_SKIP_RE.search(q):
                    continue

                teams = _parse_teams(q)
                if not teams:
                    continue
                team1, team2 = teams

                cid = m.get("conditionId") or m.get("condition_id", "")
                if not cid or cid in existing_cids:
                    continue

                # ── outcomePrices → result ──
                op     = m.get("outcomePrices") or []
                result = _outcome_from_prices(op)

                # ── Token IDs ──
                raw_ids = m.get("clobTokenIds") or []
                if isinstance(raw_ids, str):
                    try:
                        raw_ids = json.loads(raw_ids)
                    except Exception:
                        raw_ids = []
                raw_outcomes = m.get("outcomes") or []
                if isinstance(raw_outcomes, str):
                    try:
                        raw_outcomes = json.loads(raw_outcomes)
                    except Exception:
                        raw_outcomes = []
                raw_ids = _fix_token_order(raw_ids, raw_outcomes, team1)
                yes_token_id = str(raw_ids[0]) if raw_ids else ""

                end_date_str = m.get("endDateIso") or m.get("endDate") or ev.get("endDate", "")
                volume       = float(m.get("volume") or m.get("volumeNum") or 0)
                last_price   = m.get("lastTradePrice")
                uma_status   = m.get("umaResolutionStatus", "")

                records.append({
                    "condition_id":      cid,
                    "question":          q,
                    "team1":             team1,
                    "team2":             team2,
                    "end_date":          end_date_str,
                    "result":            result,
                    "last_trade_price":  last_price,
                    "uma_status":        uma_status,
                    "volume":            round(volume, 2),
                    "yes_token_id":      yes_token_id,
                    "n_outcomes":        len(raw_outcomes),
                })

                existing_cids.add(cid)
                total_new += 1

                if total_new % 50 == 0 or total_new <= 10:
                    res_str = str(result) if result is not None else "?"
                    print(
                        f"  [{total_new:>4}] {q[:55]:<55}  "
                        f"vol=${volume:>8.0f}  result={res_str}"
                    )

                if total_new >= max_markets:
                    break

            if total_new >= max_markets:
                break

        offset += page_size
        time.sleep(0.05)   # polite pagination

    # ── Save ──────────────────────────────────────────────────────────────────
    new_df = pd.DataFrame(records)

    if len(new_df) == 0 and existing is not None:
        print("No new markets found. Existing CSV unchanged.")
        return existing
    if len(new_df) == 0:
        print("No markets found at all.")
        return pd.DataFrame()

    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)

    n_resolved = combined["result"].notna().sum()
    n_total    = len(combined)
    print(f"\nSaved {n_total} total markets ({total_new} new) → {OUTPUT_CSV}")
    print(f"  result known (resolved): {n_resolved} / {n_total}  "
          f"({n_resolved/n_total:.0%})")
    if n_resolved < n_total:
        statuses = combined.loc[combined["result"].isna(), "uma_status"].value_counts()
        print(f"  unresolved statuses: {dict(statuses)}")
    print(f"  avg volume: ${combined['volume'].mean():.0f}")
    print(f"  year range: {combined['end_date'].str[:4].min()} – "
          f"{combined['end_date'].str[:4].max()}")
    return combined


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical Polymarket LoL markets (public Gamma API only)"
    )
    parser.add_argument("--max",   type=int,  default=5000,
                        help="Max new markets to fetch (default: 5000)")
    parser.add_argument("--force", action="store_true",
                        help="Clear existing CSV and re-fetch from scratch")
    args = parser.parse_args()

    df = fetch_historical_markets(max_markets=args.max, force=args.force)
    if len(df):
        print("\nSample (10 rows with known result):")
        sample = df[df["result"].notna()].head(10)
        cols   = ["question", "team1", "team2", "end_date", "result", "volume"]
        print(sample[cols].to_string(index=False))
