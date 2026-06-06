"""
FLEX Model - Phase 1: the data spine.

Builds a per-player-season table of Opportunity / Efficiency / Context inputs
for RB / WR / TE from free nflverse data, using the official, maintained
`nflreadpy` library (the successor to nfl_data_py).

The transform functions take plain DataFrames, so the same code runs against
real nflverse data OR a synthetic test frame (unit-testable without a network).

RUN (open-internet environment - laptop, Google Colab, or Streamlit Cloud):
    pip install nflreadpy pyarrow pandas numpy
    python flex_spine.py --seasons 2023 2024 2025 --out flex_spine.csv

The nflverse data CDN is blocked inside the Claude dev sandbox, so the real
pull happens in an open environment; the metric logic is tested on synthetic data.
"""

import argparse
import sys
import numpy as np
import pandas as pd

SKILL_POS = ["RB", "WR", "TE"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _safe_div(num, den):
    return num / den.replace(0, np.nan)


def normalize_cols(df):
    """nflverse player_stats sometimes names the team column 'team' and the
    name 'player_name'. Standardize to what the pipeline expects."""
    if "recent_team" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "recent_team"})
    if "player_display_name" not in df.columns and "player_name" in df.columns:
        df = df.rename(columns={"player_name": "player_display_name"})
    return df


def _ensure(df, cols):
    """Add any missing numeric column as 0 so groupby never KeyErrors."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df


def _to_pd(x):
    """nflreadpy returns polars; convert to pandas (needs pyarrow)."""
    return x.to_pandas() if hasattr(x, "to_pandas") else x


# ----------------------------------------------------------------------------
# Player spine - Opportunity + Efficiency inputs (one row per player-season)
# ----------------------------------------------------------------------------
def build_player_spine(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = normalize_cols(weekly)
    need = [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_air_yards", "carries", "rushing_yards", "rushing_tds",
        "fantasy_points_ppr", "target_share", "air_yards_share", "wopr",
    ]
    weekly = _ensure(weekly, need)
    df = weekly[weekly["position"].isin(SKILL_POS)].copy()

    agg = df.groupby(
        ["player_id", "player_display_name", "position", "recent_team", "season"],
        as_index=False,
    ).agg(
        games=("week", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("receiving_tds", "sum"),
        air_yards=("receiving_air_yards", "sum"),
        carries=("carries", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"),
        ppr=("fantasy_points_ppr", "sum"),
        target_share=("target_share", "mean"),
        air_yards_share=("air_yards_share", "mean"),
        wopr=("wopr", "mean"),
    )

    team = df.groupby(["recent_team", "season"], as_index=False).agg(
        team_targets=("targets", "sum"),
        team_carries=("carries", "sum"),
    )
    out = agg.merge(team, on=["recent_team", "season"], how="left")

    # ===================== OPPORTUNITY =====================
    out["tgt_share_season"] = _safe_div(out["targets"], out["team_targets"])
    out["rush_share"] = _safe_div(out["carries"], out["team_carries"])
    out["tgts_per_game"] = _safe_div(out["targets"], out["games"])
    out["carries_per_game"] = _safe_div(out["carries"], out["games"])
    out["touch_share"] = _safe_div(
        out["targets"] + out["carries"], out["team_targets"] + out["team_carries"]
    )

    # ============ SCORING FORMATS (redraft & dynasty toggle) ============
    # nflverse gives full PPR; Half and Standard are exact derivations.
    out["pts_ppr"] = out["ppr"]
    out["pts_half"] = out["ppr"] - 0.5 * out["receptions"]
    out["pts_std"] = out["ppr"] - 1.0 * out["receptions"]
    out["ppg_ppr"] = _safe_div(out["pts_ppr"], out["games"])
    out["ppg_half"] = _safe_div(out["pts_half"], out["games"])
    out["ppg_std"] = _safe_div(out["pts_std"], out["games"])

    # ===================== EFFICIENCY ======================
    out["yards_per_target"] = _safe_div(out["rec_yards"], out["targets"])
    out["yards_per_carry"] = _safe_div(out["rush_yards"], out["carries"])
    out["catch_rate"] = _safe_div(out["receptions"], out["targets"])
    out["total_tds"] = out["rec_tds"] + out["rush_tds"]
    out["total_yards"] = out["rec_yards"] + out["rush_yards"]

    return out


# ----------------------------------------------------------------------------
# Team context - pace / pass tendency / PROE (one row per team-season)
# ----------------------------------------------------------------------------
def build_team_context(pbp: pd.DataFrame) -> pd.DataFrame:
    for c in ["pass", "rush", "pass_oe"]:
        if c not in pbp.columns:
            pbp[c] = 0
    plays = pbp[(pbp["pass"].fillna(0) + pbp["rush"].fillna(0)) > 0].copy()
    g = plays.groupby(["posteam", "season"])["game_id"].nunique().rename("games")
    ctx = plays.groupby(["posteam", "season"]).agg(
        plays=("pass", "size"),
        pass_rate=("pass", "mean"),
        proe=("pass_oe", "mean"),
    )
    ctx = ctx.join(g)
    ctx["plays_per_game"] = _safe_div(ctx["plays"], ctx["games"])
    ctx = ctx.reset_index().rename(columns={"posteam": "recent_team"})
    return ctx[["recent_team", "season", "plays_per_game", "pass_rate", "proe"]]


# ----------------------------------------------------------------------------
# Real data pull (open-internet only)
# ----------------------------------------------------------------------------
def load_weekly(nfl, seasons):
    try:
        return _to_pd(nfl.load_player_stats(seasons=seasons, summary_level="week"))
    except TypeError:
        try:
            return _to_pd(nfl.load_player_stats(seasons=seasons))
        except TypeError:
            return _to_pd(nfl.load_player_stats(seasons))


def pull_and_build(seasons):
    import nflreadpy as nfl

    print(f"Pulling weekly data for {seasons} ...")
    spine = build_player_spine(load_weekly(nfl, seasons))

    try:
        print("Pulling play-by-play for team context ...")
        pbp = _to_pd(nfl.load_pbp(seasons))
        spine = spine.merge(build_team_context(pbp), on=["recent_team", "season"], how="left")
    except Exception as e:
        print(f"WARN: team context skipped ({e})")

    return spine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--out", default="flex_spine.csv")
    args = ap.parse_args()

    try:
        spine = pull_and_build(args.seasons)
    except Exception as e:
        print(f"ERROR pulling nflverse data: {e}", file=sys.stderr)
        print("Are you on an open-internet machine? (the dev sandbox blocks the data CDN)")
        sys.exit(1)

    spine = spine.sort_values(["season", "ppr"], ascending=[False, False])
    spine.to_csv(args.out, index=False)
    print(f"Wrote {len(spine)} player-seasons to {args.out}")


if __name__ == "__main__":
    main()
