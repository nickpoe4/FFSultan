"""
Sultan Model - hosted app (Streamlit), v9 parity.

Live pipeline: nflverse history + 2026 rosters + drafted rookies (+ your committed
analyst CSVs) -> projection (Level 2) + prospect model + dynasty + MASTER prep ->
the vaporwave UI (sultan_template.html) with the Redraft/Dynasty toggle.

Repo files: sultan_streamlit.py, flex_spine.py, flex_project.py, flex_score.py,
flex_prospect.py, flex_board.py, sultan_template.html, requirements.txt, and
(optional, for MASTER/analyst columns) an analyst_data/ folder with your UDK
position CSVs + Dynasty Startup CSV. See DEPLOY_sultan.md.
"""

import glob
import json
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import nflreadpy as nfl

import flex_spine
import flex_board
import flex_qb
import flex_market
import flex_project_stats

SEASONS = [2023, 2024, 2025]
st.set_page_config(page_title="Sultan — Fantasy Football Model", layout="wide")


def _pd(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def _pull(fn, season):
    for call in (lambda: fn(seasons=[season]), lambda: fn([season]), lambda: fn()):
        try:
            return _pd(call())
        except TypeError:
            continue
    return _pd(fn())


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Pulling nflverse history…")
def load_history():
    spine = flex_spine.build_player_spine(flex_spine.load_weekly(nfl, SEASONS))
    try:
        pbp = _pd(nfl.load_pbp(SEASONS))
        spine = spine.merge(flex_spine.build_team_context(pbp), on=["recent_team", "season"], how="left")
        spine = flex_spine.add_redzone(spine, pbp)   # red-zone usage, live
    except Exception as e:
        st.warning(f"Team context / red-zone skipped: {e}")
    try:
        ros = _pd(nfl.load_rosters(SEASONS))
        idc = "gsis_id" if "gsis_id" in ros.columns else "player_id"
        if "headshot_url" in ros.columns:
            hs = ros[[idc, "headshot_url"]].dropna().drop_duplicates(idc).rename(columns={idc: "player_id"})
            spine = spine.merge(hs, on="player_id", how="left")
    except Exception:
        pass
    return spine


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Pulling 2026 rosters…")
def load_rosters_2026():
    ros = _pd(nfl.load_rosters([2026]))
    idc = "gsis_id" if "gsis_id" in ros.columns else "player_id"
    ros = ros.rename(columns={idc: "player_id"})
    for tc in ["team", "recent_team", "club_code"]:
        if tc in ros.columns:
            ros = ros.rename(columns={tc: "team"})
            break
    ros["team"] = ros["team"].replace({"AZ": "ARI"})
    if "age" not in ros.columns and "birth_date" in ros.columns:
        bd = pd.to_datetime(ros["birth_date"], errors="coerce")
        ros["age"] = (pd.Timestamp("2026-09-01") - bd).dt.days / 365.25
    cols = [c for c in ["player_id", "team", "position", "age", "headshot_url"] if c in ros.columns]
    return ros[cols].dropna(subset=["player_id", "team"]).drop_duplicates("player_id")


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Pulling 2026 rookies…")
def load_rookies():
    draft = _pull(nfl.load_draft_picks, 2026)
    if "season" in draft.columns:
        draft = draft[draft["season"] == 2026]
    for a, b in [("pfr_player_name", "full_name"), ("player", "full_name"), ("pos", "position")]:
        if a in draft.columns and b not in draft.columns:
            draft = draft.rename(columns={a: b})
    # athleticism (optional)
    try:
        comb = _pull(nfl.load_combine, 2026)
        if "season" in comb.columns:
            comb = comb[comb["season"] == 2026]
        comb = comb.rename(columns={"pos": "position", "player_name": "full_name",
                                    "player": "full_name", "wt": "weight", "ht": "height"})
        for c in ["weight", "forty", "vertical", "broad_jump", "cone", "shuttle"]:
            if c in comb.columns:
                comb[c] = pd.to_numeric(comb[c], errors="coerce")
        if {"weight", "forty"}.issubset(comb.columns):
            comb["speed_score"] = (comb["weight"] * 200) / (comb["forty"] ** 4)
        if {"vertical", "broad_jump"}.issubset(comb.columns):
            comb["burst_score"] = comb["vertical"] + comb["broad_jump"]
        if {"cone", "shuttle"}.issubset(comb.columns):
            comb["agility_score"] = comb["cone"] + comb["shuttle"]
        key = next((k for k in ["pfr_player_id", "cfb_player_id"] if k in draft.columns and k in comb.columns), None)
        ath = [c for c in ["speed_score", "burst_score", "agility_score"] if c in comb.columns]
        if key and ath:
            draft = draft.merge(comb[[key] + ath], on=key, how="left")
        elif ath:
            draft["_k"] = draft["full_name"].str.lower().str.replace(r"[^a-z]", "", regex=True)
            comb["_k"] = comb["full_name"].str.lower().str.replace(r"[^a-z]", "", regex=True)
            draft = draft.merge(comb[["_k"] + ath], on="_k", how="left").drop(columns=["_k"])
    except Exception as e:
        st.info(f"Combine athleticism skipped ({e}); prospects use draft capital + age.")
    if "gsis_id" in draft.columns:
        draft["player_id"] = draft["gsis_id"]
    elif "pfr_player_id" in draft.columns:
        draft["player_id"] = draft["pfr_player_id"]
    else:
        draft["player_id"] = draft["full_name"]
    return draft


@st.cache_data(ttl=60 * 60 * 24)
def load_analysts():
    """Build redraft + dynasty analyst SOURCE lists from analyst_data/.
    Drop in any number of CSVs; each becomes one voice in the Analyst Composite.
      - UDK position files (Name/Position/Rank)         -> one redraft source
      - Dynasty Startup (Andy/Jason/Mike)               -> 3 dynasty sources
      - redraft_*.csv / dynasty_*.csv  (standard OR FantasyPros export) -> one each
    """
    redraft, dynasty = [], []
    udk = []
    for f in glob.glob("analyst_data/UDK*Position*.csv") + glob.glob("analyst_data/*Position Rankings*.csv"):
        d = pd.read_csv(f)
        if {"Name", "Position", "Rank"}.issubset(d.columns):
            udk.append(d[["Name", "Position", "Rank"]])
    if udk:
        redraft.append(flex_board.posrank_namepos(pd.concat(udk, ignore_index=True)))
    for f in glob.glob("analyst_data/*Dynasty Startup*.csv"):
        dynasty += flex_board.dynasty_startup_sources(pd.read_csv(f))
    for f in glob.glob("analyst_data/redraft_*.csv"):
        try:
            redraft.append(flex_board.parse_generic(pd.read_csv(f)))
        except Exception as e:
            st.info(f"Could not parse {f}: {e}")
    for f in glob.glob("analyst_data/dynasty_*.csv"):
        try:
            dynasty.append(flex_board.parse_generic(pd.read_csv(f)))
        except Exception as e:
            st.info(f"Could not parse {f}: {e}")
    return redraft, dynasty


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fetching FantasyPros consensus…")
def load_fantasypros():
    """Live FantasyPros ECR if a key is set in Streamlit secrets (FANTASYPROS_API_KEY)."""
    try:
        key = st.secrets["FANTASYPROS_API_KEY"]
    except Exception:
        key = None
    if not key:
        return None, None
    try:
        import flex_fantasypros as fp
        return (fp.fetch(key, year=2026, kind="draft", scoring="PPR"),
                fp.fetch(key, year=2026, kind="dynasty", scoring="PPR"))
    except Exception as e:
        st.info(f"FantasyPros fetch skipped ({e}). Using CSV/UDK analysts only.")
        return None, None


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Building QB projections…")
def load_qb():
    return flex_qb.build_qb_spine(flex_spine.load_weekly(nfl, SEASONS))


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Building the board…")
def build():
    redraft_sources, dynasty_sources = load_analysts()
    fp_rd, fp_dy = load_fantasypros()
    if fp_rd is not None and len(fp_rd):
        redraft_sources.append(fp_rd)
    if fp_dy is not None and len(fp_dy):
        dynasty_sources.append(fp_dy)
    board = flex_board.build_board(load_history(), load_rosters_2026(), load_rookies(),
                                   redraft_sources=redraft_sources, dynasty_sources=dynasty_sources,
                                   qb_hist=load_qb(), pass_td=4)
    return flex_board.board_records(board)


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner="Refreshing ADP…")
def adp_weekly():
    """Live FantasyPros ADP, refreshed at most weekly; CSV snapshot fallback."""
    return flex_market.get_adp("adp_2026.csv", live=True)


c1, c2 = st.columns([4, 1])
c1.markdown("#### Sultan — 2026 (Redraft + Dynasty)")
if c2.button("↻ Refresh data"):
    st.cache_data.clear()
    st.rerun()

adp, adp_src = adp_weekly()
records = flex_market.attach_adp(build(), adp)
try:
    records = flex_project_stats.attach(records, _pd(load_history()).to_dict("records"))
except Exception as e:
    st.info(f"Stat projections skipped ({e}).")
market = flex_market.both_markets(records)
template = pathlib.Path("sultan_template.html").read_text()
html = (template.replace("__DATA__", json.dumps(records, separators=(",", ":")))
                .replace("__MARKET__", json.dumps(market, separators=(",", ":"))))
components.html(html, height=940, scrolling=True)
st.caption("ADP: %s · auto-refreshes weekly (hit ↻ Refresh data to update now)." % adp_src)
