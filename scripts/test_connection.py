"""
Test Polymarket API connection.

Run from project root:
    python -m scripts.test_connection
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from py_clob_client.client import ClobClient


def main():
    private_key = os.getenv("POLY_PRIVATE_KEY")
    funder      = os.getenv("POLY_FUNDER")

    if not private_key or not funder:
        print("ERROR: POLY_PRIVATE_KEY or POLY_FUNDER missing from .env")
        sys.exit(1)

    sig_type = int(os.getenv("POLY_SIGNATURE_TYPE", "2"))
    print(f"Connecting to Polymarket CLOB (signature_type={sig_type})...")

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        signature_type=sig_type,
        funder=funder,
    )

    # Test basic connectivity
    ok = client.get_ok()
    print(f"Server status:     {ok}")

    # Derive CLOB API credentials from private key
    print("Deriving CLOB API credentials...")
    try:
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print(f"CLOB API key:      {creds.api_key[:12]}...")
        print(f"Connected as:      {funder[:10]}...{funder[-6:]}")
    except Exception as e:
        print(f"ERROR deriving creds: {e}")
        print("Try changing signature_type to 1 in this script and retry.")
        sys.exit(1)

    # Fetch a sample LoL market to confirm data access
    print("\nSearching for LoL markets...")
    try:
        markets = client.get_markets()
        lol_markets = [
            m for m in markets.get("data", [])
            if "league of legends" in m.get("question", "").lower()
            or "lol" in m.get("question", "").lower()
            or "lck" in m.get("question", "").lower()
            or "lcs" in m.get("question", "").lower()
            or "lpl" in m.get("question", "").lower()
            or "lec" in m.get("question", "").lower()
        ]
        if lol_markets:
            print(f"Found {len(lol_markets)} LoL market(s):")
            for m in lol_markets[:5]:
                print(f"  [{m.get('condition_id','')[:8]}...]  {m.get('question','')[:80]}")
        else:
            print("No LoL markets found in first page — try browsing Esports on polymarket.com")
    except Exception as e:
        print(f"Market search error: {e}")

    print("\nConnection test complete.")


if __name__ == "__main__":
    main()
