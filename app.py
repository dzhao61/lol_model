"""
LoL Pre-Draft Market Maker — Terminal UI

Run with:
    streamlit run app.py
"""

import math
import re
import sys
import time
import warnings
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from pipeline.config import CACHE_DIR


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LoL MM Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Layout */
.block-container { padding-top: 0.5rem; padding-bottom: 1rem; }
section[data-testid="stSidebar"] { display: none; }

/* Section headers */
.s-hdr {
    font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
    color: #718096; text-transform: uppercase;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 4px; margin-bottom: 8px; margin-top: 14px;
}

/* Price display */
.price-block { text-align: center; padding: 8px 4px; }
.price-team  { font-size: 11px; color: #a0aec0; margin-bottom: 1px; font-weight: 600; letter-spacing: 0.05em; }
.price-mid   { font-size: 30px; font-weight: 700; line-height: 1.1; margin: 2px 0; }
.price-ba    { font-size: 11px; color: #718096; }

/* Prob bar */
.prob-bar-wrap { margin: 8px 0 4px; }
.prob-bar-outer { width:100%; height:6px; background:#2d3748; border-radius:3px; overflow:hidden; }
.prob-bar-inner { height:100%; background: linear-gradient(90deg,#4299e1,#2b6cb0); }
.prob-bar-labels { display:flex; justify-content:space-between; font-size:10px; color:#718096; margin-top:3px; }

/* Stats table */
.stats-tbl { width:100%; border-collapse:collapse; font-size:12px; }
.stats-tbl td { padding:3px 0; vertical-align:top; }
.stats-tbl .lbl { color:#718096; width:38%; }
.stats-tbl .val { font-weight:500; }

/* Model signals table */
.sig-tbl { width:100%; border-collapse:collapse; font-size:13px; }
.sig-tbl th { font-size:10px; font-weight:700; letter-spacing:0.08em; color:#718096;
              text-transform:uppercase; padding:4px 8px; border-bottom:1px solid #2d3748; text-align:right; }
.sig-tbl th:first-child { text-align:left; }
.sig-tbl td { padding:5px 8px; border-bottom:1px solid #1a202c; text-align:right; }
.sig-tbl td:first-child { text-align:left; font-weight:500; }
.sig-tbl tr.consensus { border-top:1px solid #2d3748; background:#0f1117; font-weight:600; }
.sig-tbl tr.consensus td { font-weight:600; }

/* Edge colors */
.pos { color:#48bb78; font-weight:600; }
.neg { color:#fc8181; font-weight:600; }
.neu { color:#f6ad55; font-weight:600; }

/* Quote card */
.q-card { border:1px solid #2d3748; border-radius:8px; padding:14px 16px; background:#0f1117; margin:8px 0; }
.q-side { text-align:center; }
.q-lbl  { font-size:10px; color:#718096; text-transform:uppercase; letter-spacing:0.1em; }
.q-price { font-size:32px; font-weight:700; line-height:1.2; }
.q-sub   { font-size:11px; color:#718096; margin-top:2px; }
.q-divider { border-left:1px solid #2d3748; margin:0 8px; }

/* Direction badge */
.dir-yes  { background:#1c4532; color:#48bb78; padding:3px 12px; border-radius:4px; font-weight:700; font-size:13px; display:inline-block; }
.dir-no   { background:#1a365d; color:#63b3ed; padding:3px 12px; border-radius:4px; font-weight:700; font-size:13px; display:inline-block; }
.dir-pass { background:#2d3748; color:#718096; padding:3px 12px; border-radius:4px; font-weight:600; font-size:13px; display:inline-block; }

/* Freshness */
.ts-fresh { color:#48bb78; font-size:11px; }
.ts-stale { color:#d69e2e; font-size:11px; }
.ts-none  { color:#718096; font-size:11px; }

/* Warn box */
.warn-box { border-left:3px solid #d69e2e; padding:4px 10px; font-size:12px; color:#d69e2e; margin:4px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Resource loading ─────────────────────────────────────────────────────────

MODEL_ORDER = ["xgboost", "ridge_lr", "lasso_lr", "elasticnet", "logistic"]
MODEL_LABEL = {
    "xgboost":    "XGBoost",
    "ridge_lr":   "Ridge",
    "lasso_lr":   "Lasso",
    "elasticnet": "ElasticNet",
    "logistic":   "Logistic",
}


@st.cache_resource
def load_all_bundles():
    import pickle
    bundles = {}
    for key in MODEL_ORDER:
        path = CACHE_DIR / f"model_bundle_{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                bundles[MODEL_LABEL[key]] = pickle.load(f)
    if not bundles:
        path = CACHE_DIR / "model_bundle.pkl"
        if path.exists():
            with open(path, "rb") as f:
                b = pickle.load(f)
            bundles[b["model_name"]] = b
    return bundles


@st.cache_resource
def load_feat_df():
    from scripts.predict_match import load_feat_df as _load
    return _load()


bundles = load_all_bundles()
feat_df = load_feat_df()

if not bundles:
    st.error("No model bundles found in cache/. Run notebook cell 31 first.")
    st.stop()
if feat_df is None:
    st.error("feat_df_for_inference.pkl not found. Run notebook cell 31 first.")
    st.stop()


# ─── Helper functions ─────────────────────────────────────────────────────────

@st.cache_data
def _oe_names(_df):
    names = sorted(_df["teamname"].dropna().unique().tolist())
    return names, {n.lower(): n for n in names}

oe_names, oe_lower_map = _oe_names(feat_df)

_STRIP_SUFFIX = re.compile(
    r"\s+(esports?|gaming|team|club|fc|gg|academy|challengers?)\s*$", re.I
)

def resolve_team(poly_name: str) -> str | None:
    key      = poly_name.strip().lower()
    stripped = _STRIP_SUFFIX.sub("", key).strip()
    for candidate in [key, stripped]:
        if candidate in oe_lower_map:
            return oe_lower_map[candidate]
    for candidate, cutoff in [(stripped, 0.72), (key, 0.72)]:
        if not candidate:
            continue
        m = get_close_matches(candidate, oe_lower_map.keys(), n=1, cutoff=cutoff)
        if m:
            return oe_lower_map[m[0]]
    return None


_VS_RE  = re.compile(r"LoL:\s*(.+?)\s+vs\s+(.+?)(?:\s+\(BO(\d)\)|\s+-|\s*$)", re.I)
_FMT_RE = re.compile(r"\(BO(\d)\)", re.I)

def parse_market_question(q: str) -> dict:
    m = _VS_RE.search(q)
    if not m:
        return {"team1": "", "team2": "", "series_fmt": "Bo3"}
    team1 = m.group(1).strip()
    team2 = re.sub(r"\s*\(BO\d\)\s*$", "", m.group(2).strip(), flags=re.I).strip()
    bo    = m.group(3) or ((_FMT_RE.search(q) or type("_", (), {"group": lambda s, n: "3"})()).group(1))
    return {"team1": team1, "team2": team2, "series_fmt": f"Bo{bo}"}


def _ctx(name: str) -> dict:
    rows = feat_df[feat_df["teamname"].str.lower() == name.lower()]
    if rows.empty:
        return {"league": "LCK", "split": "Spring", "playoffs": 0, "patch": "25.S1.3", "game": 1}
    r = rows.sort_values("date").iloc[-1]
    return {
        "league":   str(r.get("league",   "LCK")),
        "split":    str(r.get("split",    "Spring")),
        "playoffs": int(r.get("playoffs", 0) or 0),
        "patch":    str(r.get("patch",    "25.S1.3")),
        "game":     1,
    }


def _days_ago(ts) -> int:
    t   = pd.Timestamp(ts)
    now = pd.Timestamp.now(tz="UTC")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return (now - t).days


# ─── Cached data fetchers ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_markets(league: str = ""):
    from scripts.polymarket_client import find_lol_markets
    try:
        return find_lol_markets(league=league, limit=60), None
    except Exception as e:
        return [], str(e)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_prices(yes_token_id: str):
    from scripts.polymarket_client import get_orderbook_snapshot
    ts = time.time()
    snap = get_orderbook_snapshot(yes_token_id)
    return snap, ts


@st.cache_data(show_spinner=False)
def run_models(team1: str, team2: str, context_key: str, context: dict):
    """context_key is a hashable string for cache keying."""
    from scripts.predict_match import predict_match
    results = {}
    for label, bundle in bundles.items():
        try:
            r = predict_match(team1, team2, context, bundle, feat_df, verbose=False)
            results[label] = r
        except Exception as e:
            results[label] = {"error": str(e)}
    return results


# ─── TOP BAR ──────────────────────────────────────────────────────────────────

# Title row
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown(
        "<h2 style='margin:0;padding:4px 0 2px;font-size:22px;font-weight:700'>📊 LoL MM Terminal</h2>",
        unsafe_allow_html=True,
    )
with hdr_r:
    st.markdown(
        f"<div style='text-align:right;font-size:11px;color:#718096;padding-top:8px'>"
        f"Data through <b>{feat_df['date'].max().strftime('%Y-%m-%d')}</b>"
        f"&nbsp;·&nbsp;{len(bundles)} models</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='border-top:1px solid #2d3748;margin:4px 0 8px'></div>", unsafe_allow_html=True)

# Controls row: league | match (wide) | refresh
ctrl_league, ctrl_match, ctrl_refresh = st.columns([1, 6, 1], gap="small")

with ctrl_league:
    league_sel = st.selectbox(
        "League", ["LCK", "LPL", "LEC", "LCS", "All"],
        index=["LCK", "LPL", "LEC", "LCS", "All"].index(
            st.session_state.get("league_sel", "LCK")
        ),
        key="league_sel",
    )
    league_filter = "" if league_sel == "All" else league_sel

markets, market_err = load_markets(league=league_filter)

with ctrl_match:
    if market_err:
        st.error(f"Market load failed: {market_err}")
        st.stop()
    if not markets:
        st.info("No active LoL match markets found.")
        st.stop()

    def _label(m) -> str:
        p    = parse_market_question(m.question)
        date = m.end_date[:10] if m.end_date else ""
        return f"{p['team1']} vs {p['team2']}  [{p['series_fmt']}]  —  {date}"

    selected_idx = st.selectbox(
        "Match",
        range(len(markets)),
        format_func=lambda i: _label(markets[i]),
        label_visibility="collapsed",
    )

with ctrl_refresh:
    st.markdown("<div style='padding-top:4px'>", unsafe_allow_html=True)
    if st.button("⟳ Refresh", use_container_width=True, help="Refresh market list"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='border-top:1px solid #2d3748;margin:8px 0 14px'></div>", unsafe_allow_html=True)


# ─── Parse selected market ────────────────────────────────────────────────────

sel_market = markets[selected_idx]
parsed     = parse_market_question(sel_market.question)
poly_t1, poly_t2 = parsed["team1"], parsed["team2"]
series_fmt        = parsed["series_fmt"]

auto_t1 = resolve_team(poly_t1) or poly_t1
auto_t2 = resolve_team(poly_t2) or poly_t2

# Session state for overrides (reset on market change)
if st.session_state.get("_market_cid") != sel_market.condition_id:
    st.session_state["_market_cid"]  = sel_market.condition_id
    st.session_state["team1_ovr"]    = auto_t1
    st.session_state["team2_ovr"]    = auto_t2

team1   = st.session_state.get("team1_ovr", auto_t1)
team2   = st.session_state.get("team2_ovr", auto_t2)
context = _ctx(team1)


# ─── TWO-PANEL LAYOUT ─────────────────────────────────────────────────────────

left_col, right_col = st.columns([1.1, 2.0], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════

with left_col:

    # ── Market odds ──────────────────────────────────────────────────────────

    st.markdown('<div class="s-hdr">Market Odds</div>', unsafe_allow_html=True)

    # Auto-fetch (cached 60s); manual refresh with button
    price_refresh = st.button("↻ Refresh prices", key="price_refresh", use_container_width=False)
    if price_refresh:
        fetch_prices.clear()

    try:
        snap, fetched_at = fetch_prices(sel_market.yes_token_id)
        age = int(time.time() - fetched_at)
        if age < 30:
            ts_html = f'<span class="ts-fresh">● live  ({age}s ago)</span>'
        elif age < 90:
            ts_html = f'<span class="ts-fresh">● {age}s ago</span>'
        else:
            ts_html = f'<span class="ts-stale">⚠ stale ({age}s ago)</span>'
        st.markdown(ts_html, unsafe_allow_html=True)
    except Exception as e:
        snap, fetched_at = None, None
        st.markdown('<span class="ts-none">— price unavailable</span>', unsafe_allow_html=True)

    if snap and snap.get("mid") is not None:
        yes_mid = snap["mid"]
        yes_bid = snap["bids"][0][0] if snap.get("bids") else None
        yes_ask = snap["asks"][0][0] if snap.get("asks") else None
        no_mid  = 1.0 - yes_mid
        no_bid  = (1.0 - yes_ask) if yes_ask is not None else None
        no_ask  = (1.0 - yes_bid) if yes_bid is not None else None
        p_market_yes = yes_mid
    else:
        yes_mid = yes_bid = yes_ask = no_mid = no_bid = no_ask = None
        p_market_yes = None

    # Price grid
    def _fmt_cents(v) -> str:
        return f"{v*100:.1f}¢" if v is not None else "—"

    def _price_html(team, mid, bid, ask) -> str:
        mid_str = _fmt_cents(mid)
        ba_str  = f"bid {_fmt_cents(bid)} / ask {_fmt_cents(ask)}"
        return (
            f'<div class="price-block">'
            f'<div class="price-team">{team}</div>'
            f'<div class="price-mid">{mid_str}</div>'
            f'<div class="price-ba">{ba_str}</div>'
            f'</div>'
        )

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(_price_html(poly_t1, yes_mid, yes_bid, yes_ask), unsafe_allow_html=True)
    with pc2:
        st.markdown(_price_html(poly_t2, no_mid, no_bid, no_ask), unsafe_allow_html=True)

    # Implied probability bar
    if yes_mid is not None:
        pct = int(yes_mid * 100)
        st.markdown(
            f'<div class="prob-bar-wrap">'
            f'<div class="prob-bar-outer"><div class="prob-bar-inner" style="width:{pct}%"></div></div>'
            f'<div class="prob-bar-labels"><span>{poly_t1} {pct}%</span><span>{poly_t2} {100-pct}%</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p_market_yes is None:
        p_market_yes = st.number_input(
            f"Manual mid — P({poly_t1})", 0.01, 0.99, 0.50, 0.01, "%.2f",
            key="manual_mid",
        )

    # ── Team stats ───────────────────────────────────────────────────────────

    st.markdown('<div class="s-hdr">Team Stats</div>', unsafe_allow_html=True)

    first_r   = None
    stats_t1  = st.session_state.get("model_results_team1", team1)
    stats_t2  = st.session_state.get("model_results_team2", team2)
    _cached_res = st.session_state.get("model_results", {})
    if _cached_res and st.session_state.get("model_results_cid") == sel_market.condition_id:
        first_r = next((v for v in _cached_res.values() if "error" not in v), None)

    if first_r:
        t1_stats = {
            "Elo":        f"{first_r['elo_team1']:.0f}",
            "Form L10":   f"{first_r['wr_L10_team1']:.0%}",
            "H2H win%":   f"{first_r['h2h_wr_team1']:.0%}  ({first_r['h2h_record']})",
            "Games":      first_r["n_games_team1"],
            "Last game":  first_r["last_game_team1"],
        }
        t2_stats = {
            "Elo":        f"{first_r['elo_team2']:.0f}",
            "Form L10":   f"{first_r['wr_L10_team2']:.0%}",
            "H2H win%":   "—",
            "Games":      first_r["n_games_team2"],
            "Last game":  first_r["last_game_team2"],
        }

        rows_html = ""
        for k in t1_stats:
            rows_html += (
                f"<tr>"
                f"<td class='lbl'>{k}</td>"
                f"<td class='val'>{t1_stats[k]}</td>"
                f"<td class='val'>{t2_stats[k]}</td>"
                f"</tr>"
            )
        st.markdown(
            f'<table class="stats-tbl">'
            f"<thead><tr><td></td>"
            f"<td style='font-weight:600;font-size:11px'>{stats_t1}</td>"
            f"<td style='font-weight:600;font-size:11px'>{stats_t2}</td></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        # Data quality warnings
        warns = []
        if first_r["n_games_team1"] < 20 or first_r["n_games_team2"] < 20:
            warns.append("A team has < 20 games in training data — low confidence.")
        if _days_ago(first_r["last_game_team1"]) > 14:
            warns.append(f"{stats_t1} last played > 14 days ago.")
        if _days_ago(first_r["last_game_team2"]) > 14:
            warns.append(f"{stats_t2} last played > 14 days ago.")
        for w in warns:
            st.markdown(f'<div class="warn-box">⚠ {w}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:12px;color:#718096">Run models to see team stats.</div>',
            unsafe_allow_html=True,
        )

    # ── Settings ─────────────────────────────────────────────────────────────

    st.markdown('<div class="s-hdr">Settings</div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        bankroll    = st.number_input("Bankroll (USDC)", 10.0, value=200.0, step=10.0)
        half_spread = st.number_input("Half-spread (¢)", 1, value=3, step=1) / 100
    with s2:
        quote_size  = st.number_input("Shares / side", 1, value=100, step=10)
        min_edge    = st.number_input("Min edge (%)", 1, value=2, step=1) / 100

    # Team name overrides (collapsed)
    # Keys are scoped to condition_id so stale widget state never bleeds across markets.
    _cid_short = sel_market.condition_id[:8]
    with st.expander("Override team / context", expanded=False):
        oc1, oc2 = st.columns(2)
        with oc1:
            new_t1 = st.selectbox(
                f"Team 1  ({poly_t1})", oe_names,
                index=oe_names.index(team1) if team1 in oe_names else 0,
                key=f"t1_select_{_cid_short}",
            )
        with oc2:
            new_t2 = st.selectbox(
                f"Team 2  ({poly_t2})", oe_names,
                index=oe_names.index(team2) if team2 in oe_names else 1,
                key=f"t2_select_{_cid_short}",
            )
        if new_t1 != team1 or new_t2 != team2:
            st.session_state["team1_ovr"] = new_t1
            st.session_state["team2_ovr"] = new_t2
            team1, team2 = new_t1, new_t2

        ctx = _ctx(team1)
        cc1, cc2, cc3, cc4 = st.columns(4)
        ctx_league   = cc1.text_input("League",   ctx["league"],   key="ctx_lg")
        ctx_split    = cc2.text_input("Split",    ctx["split"],    key="ctx_sp")
        ctx_patch    = cc3.text_input("Patch",    ctx["patch"],    key="ctx_pa")
        ctx_playoffs = cc4.toggle("Playoffs",     bool(ctx["playoffs"]), key="ctx_po")
        context = {
            "league": ctx_league, "split": ctx_split,
            "playoffs": int(ctx_playoffs), "patch": ctx_patch, "game": 1,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════

with right_col:

    # ── Model signals ─────────────────────────────────────────────────────────

    st.markdown('<div class="s-hdr">Model Signals</div>', unsafe_allow_html=True)

    # Check for cached results for this market + team pair
    _res_key = f"{sel_market.condition_id}|{team1}|{team2}"
    if st.session_state.get("model_results_key") != _res_key:
        model_results = {}
    else:
        model_results = st.session_state.get("model_results", {})

    run_btn = st.button(
        "▶  Run all models",
        type="primary",
        use_container_width=False,
        key="run_models_btn",
    )

    if run_btn:
        if team1 == team2:
            st.error("Teams must be different.")
        else:
            with st.spinner(f"Running {len(bundles)} models..."):
                ctx_key = f"{context['league']}|{context['split']}|{context['patch']}|{context['playoffs']}"
                model_results = run_models(team1, team2, ctx_key, context)
            st.session_state["model_results"]       = model_results
            st.session_state["model_results_cid"]   = sel_market.condition_id
            st.session_state["model_results_key"]   = _res_key
            st.session_state["model_results_team1"] = team1
            st.session_state["model_results_team2"] = team2

    if model_results:
        from scripts.predict_match import series_win_prob

        def _edge_class(v: float, threshold: float) -> str:
            if v >= threshold:  return "pos"
            if v > 0:           return "neu"
            return "neg"

        def _edge_str(v: float) -> str:
            return f"{v*100:+.1f}%"

        rows_data = []
        p_series_list = []

        for label, r in model_results.items():
            if "error" in r:
                rows_data.append({"label": label, "error": r["error"]})
                continue
            p_game   = r["p_model"]
            p_ser    = series_win_prob(p_game, series_fmt) if series_fmt in ("Bo3", "Bo5") else p_game
            p_series_list.append(p_ser)
            edge_yes = p_ser - p_market_yes
            edge_no  = (1.0 - p_ser) - (1.0 - p_market_yes)
            rows_data.append({
                "label":    label,
                "p1":       p_ser,
                "p2":       1.0 - p_ser,
                "edge_yes": edge_yes,
                "edge_no":  edge_no,
            })

        # Build HTML table
        header = (
            f"<thead><tr>"
            f"<th>Model</th>"
            f"<th>P({poly_t1})</th>"
            f"<th>P({poly_t2})</th>"
            f"<th>Edge YES</th>"
            f"<th>Edge NO</th>"
            f"</tr></thead>"
        )

        body_rows = ""
        for rd in rows_data:
            if "error" in rd:
                body_rows += (
                    f"<tr><td>{rd['label']}</td>"
                    f"<td colspan='4' style='color:#fc8181;font-size:11px'>{rd['error']}</td></tr>"
                )
                continue
            ey_cls = _edge_class(rd["edge_yes"], min_edge)
            en_cls = _edge_class(rd["edge_no"],  min_edge)
            body_rows += (
                f"<tr>"
                f"<td>{rd['label']}</td>"
                f"<td>{rd['p1']:.1%}</td>"
                f"<td>{rd['p2']:.1%}</td>"
                f"<td class='{ey_cls}'>{_edge_str(rd['edge_yes'])}</td>"
                f"<td class='{en_cls}'>{_edge_str(rd['edge_no'])}</td>"
                f"</tr>"
            )

        # Consensus row
        if p_series_list:
            p_avg     = sum(p_series_list) / len(p_series_list)
            avg_ey    = p_avg - p_market_yes
            avg_en    = (1.0 - p_avg) - (1.0 - p_market_yes)
            ey_cls    = _edge_class(avg_ey, min_edge)
            en_cls    = _edge_class(avg_en, min_edge)
            n_pos_yes = sum(1 for rd in rows_data if "error" not in rd and rd["edge_yes"] >= min_edge)
            n_tot     = sum(1 for rd in rows_data if "error" not in rd)
            body_rows += (
                f"<tr class='consensus'>"
                f"<td>Consensus ({n_pos_yes}/{n_tot} YES)</td>"
                f"<td>{p_avg:.1%}</td>"
                f"<td>{1-p_avg:.1%}</td>"
                f"<td class='{ey_cls}'>{_edge_str(avg_ey)}</td>"
                f"<td class='{en_cls}'>{_edge_str(avg_en)}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<table class="sig-tbl">{header}<tbody>{body_rows}</tbody></table>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.markdown(
            f'<div style="font-size:13px;color:#718096;padding:12px 0">'
            f'Select a match, then press <b>Run all models</b> to generate predictions.</div>',
            unsafe_allow_html=True,
        )

    # ── Quote builder ─────────────────────────────────────────────────────────

    if model_results and any("error" not in r for r in model_results.values()):

        st.markdown('<div class="s-hdr">Quote Builder</div>', unsafe_allow_html=True)

        from pipeline.betting import taker_fee_per_share, kelly_fraction
        from scripts.predict_match import series_win_prob

        qb1, qb2, qb3 = st.columns([2, 1, 1])
        with qb1:
            valid_models = [lbl for lbl, r in model_results.items() if "error" not in r]
            model_choice = st.selectbox("Model", valid_models, key="model_choice")
        with qb2:
            prob_choice  = st.radio("Prob", ["Series", "Per-game"], horizontal=True, key="prob_choice")
        with qb3:
            kelly_frac   = st.number_input("Kelly frac", 0.05, 1.0, 0.25, 0.05, key="kelly_frac")

        r_chosen = model_results[model_choice]
        p_game   = r_chosen["p_model"]
        p_ser    = series_win_prob(p_game, series_fmt) if series_fmt in ("Bo3", "Bo5") else p_game
        p_fair   = p_ser if prob_choice == "Series" else p_game

        # Symmetric maker quotes: always fair ± half_spread, no skew.
        # The directional signal (Kelly bet) is shown separately.
        TICK = 0.01
        q_bid = max(0.01, round(math.floor((p_fair - half_spread) / TICK) * TICK, 4))
        q_ask = min(0.99, round(math.ceil( (p_fair + half_spread) / TICK) * TICK, 4))

        fee      = taker_fee_per_share(p_market_yes)
        edge_yes = p_fair - p_market_yes
        net_edge = abs(edge_yes) - fee

        # Kelly directional sizing
        k_size = kelly_fraction(p_fair, p_market_yes, kelly_frac, as_maker=False)
        k_size = min(k_size, 0.10)
        stake  = bankroll * k_size

        if abs(edge_yes) >= min_edge:
            if edge_yes > 0:
                dir_html = '<span class="dir-yes">▲ BUY YES</span>'
            else:
                dir_html = '<span class="dir-no">▼ BUY NO</span>'
        else:
            dir_html = '<span class="dir-pass">— PASS / MM only</span>'

        stake_html = (
            f'<b>${stake:.2f}</b> &nbsp;({k_size:.3f} Kelly)'
            if k_size > 0 else "—"
        )

        st.markdown(
            f'<div class="q-card">'
            f'<div style="display:flex; gap:0; align-items:stretch;">'

            f'<div class="q-side" style="flex:1">'
            f'<div class="q-lbl">BID (buy YES)</div>'
            f'<div class="q-price">{q_bid:.2f}</div>'
            f'<div class="q-sub">{q_bid*100:.0f}¢</div>'
            f'</div>'

            f'<div class="q-divider"></div>'

            f'<div class="q-side" style="flex:1">'
            f'<div class="q-lbl">ASK (sell YES)</div>'
            f'<div class="q-price">{q_ask:.2f}</div>'
            f'<div class="q-sub">{q_ask*100:.0f}¢</div>'
            f'</div>'

            f'<div class="q-divider"></div>'

            f'<div style="flex:1.4; padding:4px 12px; font-size:12px; line-height:1.9">'
            f'<div><span style="color:#718096">Fair value</span>  <b>{p_fair:.3f}</b></div>'
            f'<div><span style="color:#718096">Spread</span>  <b>{(q_ask-q_bid)*100:.0f}¢</b></div>'
            f'<div><span style="color:#718096">Edge YES</span>  <b>{edge_yes:+.3f}</b></div>'
            f'<div><span style="color:#718096">Net (−fee)</span>  '
            f'<b style="color:{"#48bb78" if net_edge>0 else "#fc8181"}">{net_edge:+.4f}</b></div>'
            f'</div>'

            f'<div class="q-divider"></div>'

            f'<div style="flex:1.3; padding:4px 12px; font-size:12px; line-height:1.9; display:flex; flex-direction:column; justify-content:center; gap:6px">'
            f'<div>{dir_html}</div>'
            f'<div style="color:#718096">Stake: {stake_html}</div>'
            f'<div style="color:#718096;font-size:10px">fee {fee:.4f}</div>'
            f'</div>'

            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Execute ───────────────────────────────────────────────────────────

        st.markdown('<div class="s-hdr">Execute</div>', unsafe_allow_html=True)

        ex1, ex2 = st.columns(2)

        with ex1:
            post_label = f"🚀  Post  {q_bid:.2f} / {q_ask:.2f}  ×{int(quote_size)}"
            if st.button(post_label, type="primary", use_container_width=True):
                with st.spinner("Submitting orders..."):
                    try:
                        from scripts.polymarket_client import post_two_sided_quote
                        posted = post_two_sided_quote(
                            token_id=sel_market.yes_token_id,
                            bid_price=q_bid,
                            ask_price=q_ask,
                            size=float(quote_size),
                            dry_run=False,
                        )
                        st.session_state["last_order"] = {
                            "bid_id": posted.bid_order_id,
                            "ask_id": posted.ask_order_id,
                            "bid":    q_bid,
                            "ask":    q_ask,
                            "cid":    sel_market.condition_id,
                            "ts":     time.time(),
                        }
                        st.success(
                            f"Orders posted  ·  Bid `{posted.bid_order_id[:12]}…`"
                            f"  Ask `{posted.ask_order_id[:12]}…`"
                        )
                    except Exception as e:
                        st.error(f"Post failed: {e}")

        with ex2:
            if st.button("❌  Cancel all orders", use_container_width=True):
                with st.spinner("Cancelling..."):
                    try:
                        from scripts.polymarket_client import cancel_all_orders
                        cancel_all_orders(sel_market.condition_id)
                        st.success("All orders cancelled.")
                        st.session_state.pop("last_order", None)
                    except Exception as e:
                        st.error(f"Cancel failed: {e}")

        # Last order status strip
        lo = st.session_state.get("last_order")
        if lo and lo.get("cid") == sel_market.condition_id:
            age_s = int(time.time() - lo["ts"])
            st.markdown(
                f'<div style="font-size:11px; color:#718096; margin-top:6px; padding:6px 10px; '
                f'border:1px solid #2d3748; border-radius:4px;">'
                f'Last posted  ·  BID <b>{lo["bid"]:.3f}</b>  ASK <b>{lo["ask"]:.3f}</b>'
                f'  ·  <span style="color:#f6ad55">{age_s}s ago</span>'
                f'  ·  Bid <code>{lo["bid_id"][:12]}…</code>'
                f'  ·  Ask <code>{lo["ask_id"][:12]}…</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
