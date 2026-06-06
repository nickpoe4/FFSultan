"""
Sultan Model - board builder (shared by the local builds and the hosted app).

Given the historical spine + 2026 rosters + drafted rookies (+ optional analyst
rankings), produce the full ranked board with redraft + dynasty fields, ready to
embed in the UI. One source of truth so local == hosted.
"""

import re
import numpy as np
import pandas as pd

import flex_project
import flex_score
import flex_prospect


def norm(n):
    n = str(n).lower()
    n = re.sub(r"[^a-z ]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", "", n)


def _long_mult(pos, age):
    if pd.isna(age):
        age = 26
    if pos == "RB":
        return float(np.clip(1 + (24 - age) * 0.07, 0.5, 1.5))
    if pos == "WR":
        return float(np.clip(1 + (26 - age) * 0.05, 0.6, 1.5))
    return float(np.clip(1 + (27 - age) * 0.04, 0.6, 1.4))  # TE


def build_board(hist, rosters, rookies, udk_redraft=None, dyn_startup=None, season=2026):
    """Returns a ranked DataFrame with redraft + dynasty fields.
    udk_redraft: DataFrame[Name, Position, Rank] (analyst redraft positional rank).
    dyn_startup: DataFrame[Name, Pos, Andy, Jason, Mike] (analyst dynasty ranks)."""
    rb = flex_prospect.rookie_base(rookies, hist) if rookies is not None and len(rookies) else None
    spine = flex_project.project(hist, rosters, method="scheme", rookie_base=rb)
    ranked = flex_score.score_season(spine, season, min_games=1)

    extra_cols = [c for c in ["is_rookie", "prospect", "age"] if c in spine.columns]
    extra = spine[["player_display_name", "position", "recent_team"] + extra_cols].drop_duplicates()
    ranked = ranked.merge(extra, on=["player_display_name", "position", "recent_team"], how="left")
    if "is_rookie" in ranked.columns:
        ranked["is_rookie"] = ranked["is_rookie"].fillna(0).astype(int)
    else:
        ranked["is_rookie"] = 0
    for c in ["prospect", "age"]:
        if c not in ranked.columns:
            ranked[c] = np.nan
    ranked["nk"] = ranked["player_display_name"].map(norm)

    # analyst redraft positional rank
    ranked["a_rd"] = np.nan
    if udk_redraft is not None and len(udk_redraft):
        u = udk_redraft.rename(columns={"Position": "position", "Rank": "a_rd"}).copy()
        u["nk"] = u["Name"].map(norm)
        u = u[["nk", "position", "a_rd"]].dropna().drop_duplicates(["nk", "position"])
        ranked = ranked.drop(columns=["a_rd"]).merge(u, on=["nk", "position"], how="left")

    # analyst dynasty composite -> positional rank
    ranked["a_dyn"] = np.nan
    if dyn_startup is not None and len(dyn_startup):
        d = dyn_startup.copy()
        for c in ["Andy", "Jason", "Mike"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        comp_cols = [c for c in ["Andy", "Jason", "Mike"] if c in d.columns]
        d["composite"] = d[comp_cols].mean(axis=1)
        d["nk"] = d["Name"].map(norm)
        d["position"] = d["Pos"] if "Pos" in d.columns else d.get("position")
        d["a_dyn"] = d.groupby("position")["composite"].rank(method="min")
        dm = d[["nk", "position", "a_dyn"]].dropna().drop_duplicates(["nk", "position"])
        ranked = ranked.drop(columns=["a_dyn"]).merge(dm, on=["nk", "position"], how="left")

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
    """List of dicts for the UI template."""
    def i_or_none(v):
        return None if pd.isna(v) else int(v)

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
            "tgt_share_season": round(float(r["tgt_share_season"]), 3), "rush_share": round(float(r["rush_share"]), 3),
            "wopr": round(float(r["wopr"]), 3), "total_tds": round(float(r["total_tds"]), 1),
        })
    return recs
