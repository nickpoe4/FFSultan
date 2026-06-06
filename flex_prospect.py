"""
Sultan Model - rookie PROSPECT model.

Rookies have no NFL history, so we can't project them like veterans. Instead:
  1. prospect_grade  = draft capital (dominant) + athleticism + youth.
  2. a pick-based pseudo-baseline (projected per-game targets/carries) so the
     rookie can be dropped into the SAME projection engine and compete with
     veterans for his 2026 team's volume.

Output rows match what flex_project.project() expects as `rookie_base`.
"""

import numpy as np
import pandas as pd
from flex_project import weighted_baseline, SKILL

# draft-source (PFR-style) -> nflverse abbreviations
TEAM_NORM = {
    "LVR": "LV", "OAK": "LV", "NOR": "NO", "SFO": "SF", "GNB": "GB", "KAN": "KC",
    "TAM": "TB", "NWE": "NE", "SDG": "LAC", "SD": "LAC", "JAC": "JAX", "AZ": "ARI",
    "ARZ": "ARI", "LAR": "LA", "STL": "LA", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}


def _pct(s):
    return s.rank(pct=True) * 100


def prospect_grade(df):
    d = df.copy()
    d["pick"] = pd.to_numeric(d["pick"], errors="coerce")
    d["age"] = pd.to_numeric(d.get("age", np.nan), errors="coerce")
    # draft capital: earlier picks much better
    d["dc"] = 100 * np.exp(-(d["pick"] - 1) / 55.0)

    # athleticism: position-relative percentile of available scores (impute 50)
    d["ath"] = np.nan
    for pos, g in d.groupby("position"):
        idx = g.index
        parts = []
        for col, inv in [("speed_score", False), ("burst_score", False), ("agility_score", True)]:
            if col in d.columns and g[col].notna().sum() >= 3:
                p = _pct(g[col])
                parts.append(100 - p if inv else p)
        if parts:
            a = pd.concat(parts, axis=1).mean(axis=1)
            d.loc[idx, "ath"] = a
    d["ath"] = d["ath"].fillna(50.0)

    # youth: younger = better (breakout-age proxy)
    d["youth_s"] = ((24 - d["age"]).clip(-3, 4) + 3) / 7 * 100
    d["youth_s"] = d["youth_s"].fillna(50.0)

    d["prospect"] = (0.55 * d["dc"] + 0.30 * d["ath"] + 0.15 * d["youth_s"]).clip(0, 100).round(1)
    return d


def rookie_base(rookies, hist):
    """Build projection-ready baseline rows for drafted skill rookies."""
    d = prospect_grade(rookies)
    d["team"] = d["team"].map(lambda t: TEAM_NORM.get(t, t))
    d = d[d["position"].isin(SKILL)].copy()

    # positional efficiency priors = veteran positional medians (rookies ~ league avg)
    vb = weighted_baseline(hist)
    med = {pos: vb[vb["position"] == pos][["ypt", "catch_rate", "rec_td_rate", "adot",
                                            "ypc", "rush_td_rate"]].median() for pos in SKILL}

    def opp(row):
        pick, pos = row["pick"], row["position"]
        pm = 0.8 + 0.4 * (row["prospect"] / 100.0)  # prospect bump/haircut
        if pos in ("WR", "TE"):
            return pd.Series({"tgt_pg": max(1.5, 7.5 * np.exp(-(pick - 1) / 80.0)) * pm, "car_pg": 0.0})
        # RB
        return pd.Series({"tgt_pg": max(1.0, 3.0 * np.exp(-(pick - 1) / 90.0)) * pm,
                          "car_pg": max(3.0, 14.0 * np.exp(-(pick - 1) / 60.0)) * pm})

    d = pd.concat([d, d.apply(opp, axis=1)], axis=1)
    for pos in SKILL:
        idx = d[d["position"] == pos].index
        for c in ["ypt", "catch_rate", "rec_td_rate", "adot", "ypc", "rush_td_rate"]:
            d.loc[idx, c] = med[pos][c]

    out = pd.DataFrame({
        "player_id": d["player_id"], "player_display_name": d["full_name"],
        "position": d["position"], "team2026": d["team"], "age": d["age"],
        "tgt_pg": d["tgt_pg"], "car_pg": d["car_pg"],
        "ypt": d["ypt"], "catch_rate": d["catch_rate"], "rec_td_rate": d["rec_td_rate"],
        "adot": d["adot"], "ypc": d["ypc"], "rush_td_rate": d["rush_td_rate"],
        "prospect": d["prospect"], "is_rookie": 1,
    })
    return out
