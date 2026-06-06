"""
Sultan Model - board builder (shared by the local builds and the hosted app).

Given the historical spine + 2026 rosters + drafted rookies (+ any number of
analyst ranking sources), produce the full ranked board with redraft + dynasty
fields. Analysts roll into an Analyst Composite (unbiased avg across sources),
which is what feeds MASTER -> Model + My + Analyst Composite.
"""

import re
import numpy as np
import pandas as pd

import flex_project
import flex_score
import flex_prospect
import flex_qb


def norm(n):
    n = str(n).lower()
    n = re.sub(r"[^a-z ]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", "", n)


# ---------------------------------------------------------------------------
# Analyst source parsers — each returns a long table [nk, position, rank]
# ---------------------------------------------------------------------------
def posrank_namepos(df, rank_col="Rank"):
    """Standard format: columns Name, Position, Rank (positional rank)."""
    d = df.rename(columns={"Position": "position"}).copy()
    d["nk"] = d["Name"].map(norm)
    d["rank"] = pd.to_numeric(d[rank_col], errors="coerce")
    return d[["nk", "position", "rank"]].dropna()


def posrank_fantasypros(df):
    """FantasyPros export: a POS column like 'WR1' -> position WR, posrank 1."""
    poscol = next((c for c in df.columns if c.strip().upper() in ("POS", "POSITION")), None)
    namecol = next((c for c in df.columns if "PLAYER" in c.upper()
                    or c.strip().lower() in ("name", "player")), None)
    d = df.copy()
    d["position"] = d[poscol].astype(str).str.extract(r"([A-Za-z]+)")[0].str.upper()
    d["rank"] = pd.to_numeric(d[poscol].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    d["nk"] = d[namecol].map(norm)
    return d[["nk", "position", "rank"]].dropna()


def dynasty_startup_sources(df):
    """UDK Dynasty Startup with Andy/Jason/Mike -> one source per analyst."""
    out = []
    d = df.copy()
    d["nk"] = d["Name"].map(norm)
    d["position"] = d["Pos"] if "Pos" in d.columns else d.get("position")
    for col in ["Andy", "Jason", "Mike"]:
        if col in d.columns:
            t = d[["nk", "position"]].copy()
            t["val"] = pd.to_numeric(d[col], errors="coerce")
            t["rank"] = t.groupby("position")["val"].rank(method="min")
            out.append(t[["nk", "position", "rank"]].dropna())
    return out


def parse_generic(df):
    """Auto-detect standard vs FantasyPros format."""
    if {"Name", "Position", "Rank"}.issubset(df.columns):
        return posrank_namepos(df)
    return posrank_fantasypros(df)


def _combine_sources(ranked, sources):
    """Average positional rank across all sources per (nk, position)."""
    if not sources:
        return np.full(len(ranked), np.nan)
    base = ranked[["nk", "position"]].copy()
    cols = []
    for i, s in enumerate(sources):
        s2 = s.rename(columns={"rank": f"r{i}"}).dropna().drop_duplicates(["nk", "position"])
        base = base.merge(s2[["nk", "position", f"r{i}"]], on=["nk", "position"], how="left")
        cols.append(f"r{i}")
    return base[cols].mean(axis=1).values


# ---------------------------------------------------------------------------
def _long_mult(pos, age):
    if pd.isna(age):
        age = 26
    if pos == "RB":
        return float(np.clip(1 + (24 - age) * 0.07, 0.5, 1.5))
    if pos == "WR":
        return float(np.clip(1 + (26 - age) * 0.05, 0.6, 1.5))
    if pos == "QB":
        return float(np.clip(1 + (29 - age) * 0.03, 0.7, 1.3))   # QBs age well
    return float(np.clip(1 + (27 - age) * 0.04, 0.6, 1.4))  # TE


def build_board(hist, rosters, rookies, redraft_sources=None, dynasty_sources=None,
                season=2026, udk_redraft=None, dyn_startup=None, qb_hist=None, pass_td=4):
    """redraft_sources / dynasty_sources: lists of long-form [nk, position, rank]
    tables (use the parsers above). udk_redraft / dyn_startup kept for back-compat."""
    redraft_sources = list(redraft_sources or [])
    dynasty_sources = list(dynasty_sources or [])
    if udk_redraft is not None and len(udk_redraft):
        redraft_sources.append(posrank_namepos(udk_redraft))
    if dyn_startup is not None and len(dyn_startup):
        dynasty_sources += dynasty_startup_sources(dyn_startup)

    rb = flex_prospect.rookie_base(rookies, hist) if rookies is not None and len(rookies) else None
    spine = flex_project.project(hist, rosters, method="scheme", rookie_base=rb)
    ranked = flex_score.score_season(spine, season, min_games=1)

    extra_cols = [c for c in ["is_rookie", "prospect", "age"] if c in spine.columns]
    extra = spine[["player_display_name", "position", "recent_team"] + extra_cols].drop_duplicates()
    ranked = ranked.merge(extra, on=["player_display_name", "position", "recent_team"], how="left")
    ranked["is_rookie"] = ranked.get("is_rookie", 0)
    ranked["is_rookie"] = ranked["is_rookie"].fillna(0).astype(int)
    for c in ["prospect", "age"]:
        if c not in ranked.columns:
            ranked[c] = np.nan

    # QB track (superflex): project + fold QBs into the same board
    if qb_hist is not None and len(qb_hist):
        qrows = flex_qb.project_qb(qb_hist, rosters, hist, pass_td=pass_td, season=season)
        ranked = pd.concat([ranked, qrows], ignore_index=True, sort=False)
        ranked["is_rookie"] = ranked["is_rookie"].fillna(0).astype(int)
        for c in ["prospect", "age"]:
            if c not in ranked.columns:
                ranked[c] = np.nan

    ranked["nk"] = ranked["player_display_name"].map(norm)

    # analyst composites (unbiased avg across all sources)
    ranked["a_rd"] = _combine_sources(ranked, redraft_sources)
    ranked["a_dyn"] = _combine_sources(ranked, dynasty_sources)

    # dynasty score (age-aware; rookies prospect-led)
    ranked["lf"] = ranked.apply(lambda r: _long_mult(r["position"], r["age"]), axis=1)
    ranked["dyn_raw"] = ranked["overall_ppr"] * ranked["lf"]
    rk = ranked["is_rookie"] == 1
    ranked.loc[rk, "dyn_raw"] = (0.45 * ranked.loc[rk, "overall_ppr"]
                                 + 0.55 * ranked.loc[rk, "prospect"].fillna(50)) * 1.25
    ranked["dyn_score"] = (ranked.groupby("position")["dyn_raw"].rank(pct=True) * 100).round(1)
    ranked["dyn_pos_rank"] = ranked.groupby("position")["dyn_raw"].rank(ascending=False, method="min").astype(int)
    return ranked


def board_records(ranked):
    def i_or_none(v):
        return None if pd.isna(v) else int(round(v))

    recs = []
    for _, r in ranked.iterrows():
        recs.append({
            "name": r["player_display_name"], "pos": r["position"], "team": r["recent_team"],
            "games": int(r["games"]),
            "shot": (None if pd.isna(r.get("headshot_url")) else r.get("headshot_url")),
            "is_rookie": int(r["is_rookie"]),
            "prospect": (None if pd.isna(r["prospect"]) else round(float(r["prospect"]), 1)),
            "age": (None if pd.isna(r["age"]) else round(float(r["age"]), 1)),
            "overall_ppr": round(float(r["overall_ppr"]), 1), "overall_half": round(float(r["overall_half"]), 1),
            "grade_ppr": flex_score.letter(r["overall_ppr"]), "grade_half": flex_score.letter(r["overall_half"]),
            "pos_rank_ppr": int(r["pos_rank_ppr"]), "pos_rank_half": int(r["pos_rank_half"]),
            "a_rd": i_or_none(r["a_rd"]),
            "dyn_score": round(float(r["dyn_score"]), 1), "grade_dyn": flex_score.letter(r["dyn_score"]),
            "dyn_pos_rank": int(r["dyn_pos_rank"]), "a_dyn": i_or_none(r["a_dyn"]),
            "opportunity": round(float(r["opportunity"]), 1), "efficiency": round(float(r["efficiency"]), 1),
            "context": round(float(r["context"]), 1),
            "ppg_ppr": round(float(r["ppg_ppr"]), 1), "ppg_half": round(float(r["ppg_half"]), 1),
            "tgt_share_season": (None if pd.isna(r["tgt_share_season"]) else round(float(r["tgt_share_season"]), 3)),
            "rush_share": (None if pd.isna(r["rush_share"]) else round(float(r["rush_share"]), 3)),
            "wopr": (None if pd.isna(r["wopr"]) else round(float(r["wopr"]), 3)),
            "total_tds": round(float(r["total_tds"]), 1),
        })
    return recs
