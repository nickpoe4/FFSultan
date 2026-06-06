"""
Sultan Model - QB track (superflex).

QBs get their own evaluation: passing volume is flat across starters, so the
differentiators are RUSHING production, passing efficiency, and the scoring
format (pass-TD value). Projects 2026 QB points (configurable pass-TD), grades
QBs vs QBs, and returns rows that slot into the same board as RB/WR/TE.
"""

import numpy as np
import pandas as pd

from flex_project import _recency_w, team_context_proj
from flex_spine import normalize_cols

PROJ_GAMES = 16.0
QB_W = (0.35, 0.35, 0.30)   # opportunity, efficiency, context
PROD_W = 0.25


def build_qb_spine(weekly):
    df = normalize_cols(weekly)
    df = df[df["position"] == "QB"].copy()
    need = ["attempts", "completions", "passing_yards", "passing_tds", "interceptions",
            "carries", "rushing_yards", "rushing_tds"]
    for c in need:
        if c not in df.columns:
            df[c] = 0
    return df.groupby(["player_id", "player_display_name", "position", "recent_team", "season"],
                      as_index=False).agg(
        games=("week", "nunique"),
        attempts=("attempts", "sum"), completions=("completions", "sum"),
        pass_yards=("passing_yards", "sum"), pass_tds=("passing_tds", "sum"),
        interceptions=("interceptions", "sum"),
        carries=("carries", "sum"), rush_yards=("rushing_yards", "sum"), rush_tds=("rushing_tds", "sum"))


def _pct(s):
    return s.rank(pct=True) * 100


def project_qb(qb_hist, rosters, hist_skill, pass_td=4, season=2026):
    """Return QB rows aligned to the scored board (overall/grades/ppg/etc.).
    pass_td: points per passing TD (4 standard, 6 for 6pt leagues)."""
    h = qb_hist.copy()
    w = _recency_w(sorted(h["season"].unique()))
    h["w"] = h["season"].map(w)
    stat = ["attempts", "completions", "pass_yards", "pass_tds", "interceptions",
            "carries", "rush_yards", "rush_tds"]
    for c in stat:
        h["W_" + c] = h["w"] * h[c]
    h["W_games"] = h["w"] * h["games"]
    g = h.groupby(["player_id", "player_display_name"], as_index=False).agg(
        **{("W_" + c): ("W_" + c, "sum") for c in stat}, W_games=("W_games", "sum"))

    def pg(c):
        return g["W_" + c] / g["W_games"].replace(0, np.nan)

    q = g[["player_id", "player_display_name"]].copy()
    for c in stat:
        q[c + "_pg"] = pg(c)

    # 2026 team + age from rosters (QBs only)
    ros = rosters.rename(columns={"team": "recent_team"})
    q = q.merge(ros[[c for c in ["player_id", "recent_team", "age", "headshot_url"] if c in ros.columns]],
                on="player_id", how="inner")

    # projected season
    for c in stat:
        q[c] = q[c + "_pg"] * PROJ_GAMES
    q["points"] = (q["pass_yards"] * 0.04 + q["pass_tds"] * pass_td
                   + q["rush_yards"] * 0.1 + q["rush_tds"] * 6 - q["interceptions"] * 1.0)
    q["ppg"] = q["points"] / PROJ_GAMES

    # efficiency rates
    q["ypa"] = q["pass_yards"] / q["attempts"].replace(0, np.nan)
    q["td_rate"] = q["pass_tds"] / q["attempts"].replace(0, np.nan)
    q["int_rate"] = q["interceptions"] / q["attempts"].replace(0, np.nan)
    q["rush_ypc"] = q["rush_yards"] / q["carries"].replace(0, np.nan)

    # team context
    ctx = team_context_proj(hist_skill)
    q = q.merge(ctx, on="recent_team", how="left")

    # grades (percentile within the QB pool)
    opp = pd.concat([_pct(q["attempts_pg"]), _pct(q["carries_pg"])], axis=1).mean(axis=1)
    eff = pd.concat([_pct(q["ypa"]), _pct(q["td_rate"]), 100 - _pct(q["int_rate"]),
                     _pct(q["rush_ypc"])], axis=1).mean(axis=1)
    ctxg = pd.concat([_pct(q["plays_per_game"]), _pct(q["proe"])], axis=1).mean(axis=1)
    q["opportunity"] = opp.round(1)
    q["efficiency"] = eff.round(1)
    q["context"] = ctxg.round(1)
    wo, we, wc = QB_W
    q["role_grade"] = (wo * q["opportunity"] + we * q["efficiency"] + wc * q["context"]).round(1)
    prod = _pct(q["ppg"])
    q["overall_ppr"] = ((1 - PROD_W) * q["role_grade"] + PROD_W * prod).round(1)
    q["overall_half"] = q["overall_ppr"]   # PPR doesn't affect QB scoring
    q = q.dropna(subset=["overall_ppr"])
    q["pos_rank_ppr"] = q["overall_ppr"].rank(ascending=False, method="min").astype(int)
    q["pos_rank_half"] = q["pos_rank_ppr"]

    out = pd.DataFrame({
        "player_display_name": q["player_display_name"], "position": "QB",
        "recent_team": q["recent_team"], "season": season, "games": PROJ_GAMES,
        "overall_ppr": q["overall_ppr"], "overall_half": q["overall_half"],
        "role_grade": q["role_grade"], "opportunity": q["opportunity"],
        "efficiency": q["efficiency"], "context": q["context"],
        "tgt_share_season": np.nan, "rush_share": np.nan, "wopr": np.nan,
        "ppg_ppr": q["ppg"].round(1), "ppg_half": q["ppg"].round(1),
        "total_tds": (q["pass_tds"] + q["rush_tds"]).round(1),
        "pos_rank_ppr": q["pos_rank_ppr"], "pos_rank_half": q["pos_rank_half"],
        "age": q.get("age", np.nan),
    })
    if "headshot_url" in q.columns:
        out["headshot_url"] = q["headshot_url"]
    out["is_rookie"] = 0
    out["prospect"] = np.nan
    return out
