"""
FLEX Model - Phase 2: the scoring engine.

Turns the Phase-1 data spine into a player PROFILE:
  - Opportunity grade   (0-100)
  - Efficiency grade    (0-100)
  - Context grade       (0-100)
  - Overall FLEX score  (weighted blend, per position)

HOW GRADES ARE BUILT
  Each grade is the average PERCENTILE of its component metrics, computed
  WITHIN a position-season group (so a WR is graded against other WRs that
  year, not against RBs). 0 = worst in class, 100 = best in class.

WEIGHTS are per-position and fully tunable (blueprint section 6).

NOTE ON SCORING FORMAT
  The three grades describe role + talent and are FORMAT-AGNOSTIC. PPR vs
  Half-PPR points are carried alongside (ppg_ppr / ppg_half) so the app can
  display either and so we can later fold production into the Overall if we
  choose. This v1 keeps Overall as a pure grade blend - a calibration knob
  we will tune together.
"""

import argparse
import numpy as np
import pandas as pd

# (opportunity, efficiency, context) weights per position.
# Context intentionally down-weighted so team environment is a tiebreaker,
# not a driver (calibration decision).
WEIGHTS = {
    "RB": (0.60, 0.25, 0.15),
    "WR": (0.55, 0.30, 0.15),
    "TE": (0.50, 0.25, 0.25),
}

# Overall = (1 - PROD_W) * role-grade blend  +  PROD_W * production percentile.
# Production is format-specific, so PPR and Half-PPR produce different rankings.
PROD_W = 0.25

# component metrics feeding each grade, per position
GRADE_INPUTS = {
    # 'rz_tgt_share' / 'rz_rush_share' are used only when present (red-zone data on)
    "WR": {
        "opportunity": ["tgt_share_season", "wopr", "air_yards_share", "tgts_per_game", "rz_tgt_share"],
        "efficiency": ["yards_per_target", "catch_rate", "td_per_target"],
        "context": ["plays_per_game", "proe", "pass_rate"],
    },
    "TE": {
        "opportunity": ["tgt_share_season", "wopr", "air_yards_share", "tgts_per_game", "rz_tgt_share"],
        "efficiency": ["yards_per_target", "catch_rate", "td_per_target"],
        "context": ["plays_per_game", "proe", "pass_rate"],
    },
    "RB": {
        "opportunity": ["rush_share", "carries_per_game", "touch_share", "tgts_per_game", "rz_rush_share"],
        "efficiency": ["yards_per_carry", "yards_per_touch", "catch_rate", "td_per_touch"],
        "context": ["plays_per_game", "run_rate"],
    },
}


def _derive(df):
    df = df.copy()
    df["td_per_target"] = df["total_tds"] / df["targets"].replace(0, np.nan)
    touches = (df["targets"] + df["carries"]).replace(0, np.nan)
    df["yards_per_touch"] = df["total_yards"] / touches
    df["td_per_touch"] = df["total_tds"] / touches
    df["run_rate"] = 1 - df["pass_rate"]
    return df


def _pct(s):
    """Percentile rank 0-100 (NaNs stay NaN and are ignored in the mean)."""
    return s.rank(pct=True) * 100


def score_season(df, season, min_games=6):
    d = _derive(df[(df["season"] == season) & (df["games"] >= min_games)])
    rows = []
    for pos, grp in d.groupby("position"):
        if pos not in WEIGHTS:
            continue
        grp = grp.copy()
        grades = {}
        for grade, cols in GRADE_INPUTS[pos].items():
            pcts = pd.DataFrame({c: _pct(grp[c]) for c in cols if c in grp.columns})
            grades[grade] = pcts.mean(axis=1, skipna=True)
        grp["opportunity"] = grades["opportunity"].round(1)
        grp["efficiency"] = grades["efficiency"].round(1)
        grp["context"] = grades["context"].round(1)
        wo, we, wc = WEIGHTS[pos]
        grp["role_grade"] = (
            wo * grp["opportunity"] + we * grp["efficiency"] + wc * grp["context"]
        ).round(1)
        # drop players with too little data to grade (e.g. 0 targets & 0 carries)
        grp = grp.dropna(subset=["role_grade"])
        # production percentile (within position-season), per scoring format
        prod_ppr = _pct(grp["ppg_ppr"])
        prod_half = _pct(grp["ppg_half"])
        grp["overall_ppr"] = ((1 - PROD_W) * grp["role_grade"] + PROD_W * prod_ppr).round(1)
        grp["overall_half"] = ((1 - PROD_W) * grp["role_grade"] + PROD_W * prod_half).round(1)
        grp["pos_rank_ppr"] = grp["overall_ppr"].rank(ascending=False, method="min").astype(int)
        grp["pos_rank_half"] = grp["overall_half"].rank(ascending=False, method="min").astype(int)
        rows.append(grp)

    out = pd.concat(rows, ignore_index=True)
    out["overall_rank_ppr"] = out["overall_ppr"].rank(ascending=False, method="min").astype(int)
    out["overall_rank_half"] = out["overall_half"].rank(ascending=False, method="min").astype(int)
    keep = [
        "overall_rank_ppr", "pos_rank_ppr", "overall_rank_half", "pos_rank_half",
        "player_display_name", "position", "recent_team", "season", "games",
        "overall_ppr", "overall_half", "role_grade",
        "opportunity", "efficiency", "context",
        "tgt_share_season", "rush_share", "wopr", "ppg_ppr", "ppg_half", "total_tds",
    ]
    if "headshot_url" in out.columns:   # player faces (private use), if the spine carries them
        keep.append("headshot_url")
    return out[keep].sort_values("overall_ppr", ascending=False).reset_index(drop=True)


def letter(score):
    bins = [(90, "A"), (80, "A-"), (73, "B+"), (66, "B"), (60, "B-"),
            (53, "C+"), (46, "C"), (40, "C-"), (33, "D+"), (26, "D"), (0, "F")]
    return next(g for t, g in bins if score >= t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default="flex_spine.csv")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default="flex_rankings.csv")
    ap.add_argument("--min-games", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.spine)
    ranked = score_season(df, args.season, args.min_games)
    ranked["grade_ppr"] = ranked["overall_ppr"].map(letter)
    ranked["grade_half"] = ranked["overall_half"].map(letter)
    ranked.to_csv(args.out, index=False)
    print(f"Scored {len(ranked)} players for {args.season} -> {args.out}")


if __name__ == "__main__":
    main()
