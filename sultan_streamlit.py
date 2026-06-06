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
        spine = spine.merge(flex_spine.build_team_context(_pd(nfl.load_pbp(SEASONS))),
                            on=["recent_team", "season"], how="left")
    except Exception as e:
        st.warning(f"Team context skipped: {e}")
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
    rd = glob.glob("analyst_data/UDK*Position*.csv") + glob.glob("analyst_data/*Position Rankings*.csv")
    udk = None
    if rd:
        frames = []
        for f in rd:
            d = pd.read_csv(f)
            if {"Name", "Position", "Rank"}.issubset(d.columns):
                frames.append(d[["Name", "Position", "Rank"]])
        if frames:
            udk = pd.concat(frames, ignore_index=True)
    dyn = None
    dl = glob.glob("analyst_data/*Dynasty Startup*.csv")
    if dl:
        dyn = pd.read_csv(dl[0])
    return udk, dyn


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Building the board…")
def build():
    udk, dyn = load_analysts()
    board = flex_board.build_board(load_history(), load_rosters_2026(), load_rookies(),
                                   udk_redraft=udk, dyn_startup=dyn)
    return flex_board.board_records(board)


c1, c2 = st.columns([4, 1])
c1.markdown("#### Sultan — 2026 (Redraft + Dynasty)")
if c2.button("↻ Refresh data"):
    st.cache_data.clear()
    st.rerun()

records = build()
template = pathlib.Path("sultan_template.html").read_text()
html = template.replace("__DATA__", json.dumps(records, separators=(",", ":")))
components.html(html, height=940, scrolling=True)
