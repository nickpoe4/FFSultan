"""
Sultan Model - 2026 PROJECTION engine (Level 1: situation-anchored baseline).

Turns the historical spine (2023-2025 actuals) into a projected 2026 "spine"
with the SAME column names, so the existing scoring engine (flex_score.py)
grades the projection exactly like it grades a real season.

LEVEL 1 LOGIC
  1. Per-player baseline = recency-weighted, games-weighted per-game rates,
     with efficiency regressed toward the positional mean (small-sample shrink).
  2. Optional age adjustment via a simple positional age curve.
  3. Project each 2026 team's per-game pass/rush volume from franchise history.
  4. DISTRIBUTE that volume across the players on the 2026 roster by normalizing
     their baseline shares within the team  <-- this is the swappable module that
     Level 2 (scheme/coordinator-aware allocation) will replace.
  5. Convert projected volume x regressed efficiency -> projected points/shares.

The distribution function is intentionally isolated (distribute_team) so Level 2
can drop in without touching anything else.
"""

import argparse
import numpy as np
import pandas as pd

SKILL = ["RB", "WR", "TE"]
PROJ_GAMES = 16.0
# how much of a team's targets / carries go to tracked skill players
TARGET_CAP = 0.92
RUSH_CAP = 0.90
# efficiency shrink constants (toward positional mean)
K_TGT, K_CAR = 30.0, 40.0

# simple positional age multipliers applied to projected volume/points
AGE_CURVE = {
    "RB": {21: 1.02, 22: 1.04, 23: 1.05, 24: 1.04, 25: 1.0, 26: 0.96, 27: 0.9, 28: 0.83, 29: 0.76, 30: 0.68},
    "WR": {21: 0.95, 22: 1.0, 23: 1.03, 24: 1.05, 25: 1.05, 26: 1.04, 27: 1.02, 28: 1.0, 29: 0.96, 30: 0.9, 31: 0.84, 32: 0.78},
    "TE": {22: 0.9, 23: 0.96, 24: 1.0, 25: 1.03, 26: 1.04, 27: 1.04, 28: 1.02, 29: 1.0, 30: 0.95, 31: 0.9, 32: 0.84},
}


def _age_mult(pos, age):
    if age is None or np.isnan(age):
        return 1.0
    curve = AGE_CURVE.get(pos, {})
    if not curve:
        return 1.0
    a = int(round(age))
    ks = sorted(curve)
    if a <= ks[0]:
        return curve[ks[0]]
    if a >= ks[-1]:
        return curve[ks[-1]]
    return curve.get(a, np.interp(a, ks, [curve[k] for k in ks]))


def _recency_w(seasons):
    mx = max(seasons)
    return {s: 0.55 ** (mx - s) for s in seasons}


def weighted_baseline(hist):
    """One row per player: recency/games-weighted per-game rates + regressed eff."""
    h = hist[hist["position"].isin(SKILL)].copy()
    w = _recency_w(sorted(h["season"].unique()))
    h["w"] = h["season"].map(w)
    stat = ["targets", "receptions", "rec_yards", "rec_tds", "air_yards",
            "carries", "rush_yards", "rush_tds"]
    stat += [c for c in ["rz_targets", "rz_carries"] if c in h.columns]  # red-zone (optional)
    for c in stat:
        h["W_" + c] = h["w"] * h[c]
    h["W_games"] = h["w"] * h["games"]
    g = h.groupby(["player_id", "player_display_name", "position"], as_index=False).agg(
        **{("W_" + c): ("W_" + c, "sum") for c in stat}, W_games=("W_games", "sum"))

    def pg(c):
        return g["W_" + c] / g["W_games"].replace(0, np.nan)

    out = g[["player_id", "player_display_name", "position"]].copy()
    out["tgt_pg"] = pg("targets")
    out["car_pg"] = pg("carries")
    # raw efficiency
    Wt = g["W_targets"].replace(0, np.nan)
    Wc = g["W_carries"].replace(0, np.nan)
    out["ypt"] = g["W_rec_yards"] / Wt
    out["catch_rate"] = g["W_receptions"] / Wt
    out["rec_td_rate"] = g["W_rec_tds"] / Wt
    out["adot"] = g["W_air_yards"] / Wt
    out["ypc"] = g["W_rush_yards"] / Wc
    out["rush_td_rate"] = g["W_rush_tds"] / Wc
    out["_Wt"] = g["W_targets"]
    out["_Wc"] = g["W_carries"]

    # regress efficiency toward positional mean (shrink small samples)
    for pos, grp in out.groupby("position"):
        m_ypt = np.nanmedian(grp["ypt"]); m_cr = np.nanmedian(grp["catch_rate"])
        m_rtd = np.nanmedian(grp["rec_td_rate"]); m_ypc = np.nanmedian(grp["ypc"])
        m_rrtd = np.nanmedian(grp["rush_td_rate"]); m_adot = np.nanmedian(grp["adot"])
        idx = grp.index
        wt = out.loc[idx, "_Wt"].fillna(0); wc = out.loc[idx, "_Wc"].fillna(0)
        out.loc[idx, "ypt"] = (out.loc[idx, "ypt"].fillna(m_ypt) * wt + m_ypt * K_TGT) / (wt + K_TGT)
        out.loc[idx, "catch_rate"] = (out.loc[idx, "catch_rate"].fillna(m_cr) * wt + m_cr * K_TGT) / (wt + K_TGT)
        out.loc[idx, "rec_td_rate"] = (out.loc[idx, "rec_td_rate"].fillna(m_rtd) * wt + m_rtd * K_TGT) / (wt + K_TGT)
        out.loc[idx, "adot"] = out.loc[idx, "adot"].fillna(m_adot)
        out.loc[idx, "ypc"] = (out.loc[idx, "ypc"].fillna(m_ypc) * wc + m_ypc * K_CAR) / (wc + K_CAR)
        out.loc[idx, "rush_td_rate"] = (out.loc[idx, "rush_td_rate"].fillna(m_rrtd) * wc + m_rrtd * K_CAR) / (wc + K_CAR)

    # red-zone propensity (share of a player's targets/carries that come in the RZ)
    if "W_rz_targets" in g.columns:
        out["rz_tgt_rate"] = (g["W_rz_targets"] / g["W_targets"].replace(0, np.nan)).clip(0, 1)
    if "W_rz_carries" in g.columns:
        out["rz_car_rate"] = (g["W_rz_carries"] / g["W_carries"].replace(0, np.nan)).clip(0, 1)
    return out.drop(columns=["_Wt", "_Wc"])


def team_volume(hist):
    """Projected per-game team pass (targets) and rush attempts, by franchise."""
    w = _recency_w(sorted(hist["season"].unique()))
    ts = hist.groupby(["recent_team", "season"], as_index=False).agg(
        team_targets=("team_targets", "first"), team_carries=("team_carries", "first"))
    ts["w"] = ts["season"].map(w)
    g = ts.groupby("recent_team").apply(
        lambda d: pd.Series({
            "tgt_pg": (d["w"] * d["team_targets"]).sum() / (d["w"] * 17).sum(),
            "car_pg": (d["w"] * d["team_carries"]).sum() / (d["w"] * 17).sum(),
        })).reset_index()
    lg_t, lg_c = g["tgt_pg"].mean(), g["car_pg"].mean()
    return g, lg_t, lg_c


def team_context_proj(hist):
    """Projected 2026 team context (pace / pass rate / PROE) from franchise history."""
    w = _recency_w(sorted(hist["season"].unique()))
    t = hist.groupby(["recent_team", "season"], as_index=False).agg(
        plays_per_game=("plays_per_game", "first"),
        pass_rate=("pass_rate", "first"), proe=("proe", "first"))
    t["w"] = t["season"].map(w)

    def wm(d, c):
        m = d[c].notna()
        denom = (d["w"] * m).sum()
        return (d["w"] * d[c].fillna(0)).sum() / denom if denom > 0 else np.nan

    return t.groupby("recent_team").apply(lambda d: pd.Series({
        "plays_per_game": wm(d, "plays_per_game"),
        "pass_rate": wm(d, "pass_rate"), "proe": wm(d, "proe")})).reset_index()


def team_tendencies(hist):
    """Level 2 scheme inputs from franchise history (recency-weighted):
    positional target split (RB/WR/TE) and within-position concentration (HHI)."""
    w = _recency_w(sorted(hist["season"].unique()))
    h = hist[hist["position"].isin(SKILL)].copy()
    h["w"] = h["season"].map(w)
    h["wt"] = h["w"] * h["targets"]
    team_tot = h.groupby("recent_team")["wt"].sum()
    pos_tot = h.groupby(["recent_team", "position"])["wt"].sum()
    lg = h.groupby("position")["wt"].sum()
    lg = lg / lg.sum()
    lg_split = {p: float(lg.get(p, 0.0)) for p in SKILL}
    tend = {}
    for team in team_tot.index:
        tt = team_tot[team]
        split = {p: (float(pos_tot.get((team, p), 0.0)) / tt if tt > 0 else lg_split[p]) for p in SKILL}
        hhi = {}
        for p in SKILL:
            sub = h[(h["recent_team"] == team) & (h["position"] == p)]
            ptot = sub["wt"].sum()
            if ptot > 0:
                sh = sub.groupby("player_id")["wt"].sum() / ptot
                hhi[p] = float((sh ** 2).sum())
        tend[team] = {"split": split, "hhi": hhi}
    return tend, lg_split


def distribute_team_scheme(d, team_tgt_pg, team_car_pg, tend, lg_split):
    """Level 2: allocate team volume by the team's positional target split, then
    within each position by baseline share with a concentration skew (HHI). RB
    carries split by committee structure."""
    d = d.copy()
    split = tend.get("split", lg_split)
    hhi = tend.get("hhi", {})
    d["proj_tgt_pg"] = 0.0
    for pos in SKILL:
        sub = d[d["position"] == pos]
        base = sub["tgt_pg"].fillna(0.0)
        if len(sub) == 0 or base.sum() <= 0:
            continue
        pool = team_tgt_pg * split.get(pos, lg_split[pos]) * TARGET_CAP
        gamma = 1.0
        if pos in hhi and hhi[pos] == hhi[pos]:
            gamma = min(1.5, max(0.85, 1 + (hhi[pos] - 0.22) * 1.5))  # concentrated -> skew to top
        sk = base ** gamma
        sk = sk / sk.sum()
        d.loc[sub.index, "proj_tgt_pg"] = (pool * sk).values
    d["proj_tgt_share"] = d["proj_tgt_pg"] / team_tgt_pg if team_tgt_pg > 0 else 0.0
    # rushing: committee split among RBs
    d["proj_car_pg"] = 0.0
    rbs = d[d["position"] == "RB"]
    basec = rbs["car_pg"].fillna(0.0)
    if basec.sum() > 0:
        sk = basec / basec.sum()
        d.loc[rbs.index, "proj_car_pg"] = (team_car_pg * RUSH_CAP * sk).values
    d["proj_rush_share"] = d["proj_car_pg"] / team_car_pg if team_car_pg > 0 else 0.0
    return d


def distribute_team(team_df, team_tgt_pg, team_car_pg):
    """Level 1 fallback. Normalize baseline shares within a team
    so tracked players consume up to the cap of team volume."""
    d = team_df.copy()
    raw_t = d["tgt_pg"].fillna(0)
    raw_c = d["car_pg"].fillna(0)
    st = raw_t.sum(); sc = raw_c.sum()
    # scale so the tracked group sums to <= cap of the team's projected volume
    tgt_share = raw_t / st * TARGET_CAP if st > 0 else raw_t
    rush_share = raw_c / sc * RUSH_CAP if sc > 0 else raw_c
    d["proj_tgt_share"] = tgt_share
    d["proj_rush_share"] = rush_share
    d["proj_tgt_pg"] = tgt_share * team_tgt_pg
    d["proj_car_pg"] = rush_share * team_car_pg
    return d


def _displace_incumbents(base):
    """Rookie-overtakes-incumbent: a high-capital rookie reduces the incumbent
    veterans' projected share at his team+position. Steeper for RB (one lead back),
    lighter for WR (rooms are deeper)."""
    if "pick" not in base.columns or "is_rookie" not in base.columns:
        return base
    base = base.copy()
    for team, g in base.groupby("team2026"):
        # RB lead-back displacement
        rr = g[(g["position"] == "RB") & (g["is_rookie"] == 1)]
        if len(rr):
            mp = rr["pick"].min()
            f = 0.50 if mp <= 15 else (0.68 if mp <= 45 else 0.85)
            idx = g[(g["position"] == "RB") & (g["is_rookie"] == 0)].index
            base.loc[idx, "car_pg"] = base.loc[idx, "car_pg"] * f
            base.loc[idx, "tgt_pg"] = base.loc[idx, "tgt_pg"] * (0.5 + 0.5 * f)
        # WR displacement (only very early picks, mild)
        wr = g[(g["position"] == "WR") & (g["is_rookie"] == 1)]
        if len(wr):
            mp = wr["pick"].min()
            if mp <= 32:
                fw = 0.85 if mp <= 15 else 0.93
                idx = g[(g["position"] == "WR") & (g["is_rookie"] == 0)].index
                base.loc[idx, "tgt_pg"] = base.loc[idx, "tgt_pg"] * fw
    return base


def project(hist, rosters, method="scheme", rookie_base=None):
    """rosters: player_id, team (2026), position, optional age.
    method='scheme' (Level 2, positional-split + concentration aware) or 'flat' (Level 1).
    rookie_base: optional DataFrame of drafted-rookie baseline rows (from flex_prospect)."""
    base = weighted_baseline(hist)
    base = base.drop(columns=["position"]).merge(
        rosters.rename(columns={"team": "team2026"}), on="player_id", how="inner")
    base = base[base["position"].isin(SKILL)]
    if rookie_base is not None and len(rookie_base):
        base = pd.concat([base, rookie_base], ignore_index=True, sort=False)
    if "is_rookie" in base.columns:
        base["is_rookie"] = base["is_rookie"].fillna(0).astype(int)
        base = _displace_incumbents(base)
    tv, lg_t, lg_c = team_volume(hist)
    tvmap = tv.set_index("recent_team")
    tend, lg_split = team_tendencies(hist)

    rows = []
    for team, grp in base.groupby("team2026"):
        ttg = tvmap["tgt_pg"].get(team, lg_t)
        tcg = tvmap["car_pg"].get(team, lg_c)
        if method == "scheme":
            rows.append(distribute_team_scheme(
                grp.copy(), ttg, tcg, tend.get(team, {"split": lg_split, "hhi": {}}), lg_split))
        else:
            rows.append(distribute_team(grp.copy(), ttg, tcg))
    p = pd.concat(rows, ignore_index=True)

    # projected season volume
    p["targets"] = p["proj_tgt_pg"] * PROJ_GAMES
    p["carries"] = p["proj_car_pg"] * PROJ_GAMES
    p["receptions"] = p["targets"] * p["catch_rate"]
    p["rec_yards"] = p["targets"] * p["ypt"]
    p["rush_yards"] = p["carries"] * p["ypc"]
    p["air_yards"] = p["targets"] * p["adot"]

    # TD projection: red-zone-driven when RZ data is present, else flat historical rate
    if "rz_tgt_rate" in p.columns or "rz_car_rate" in p.columns:
        RZ_T_TD, NON_T_TD = 0.19, 0.018   # TD per RZ target / per non-RZ target
        RZ_C_TD, NON_C_TD = 0.11, 0.006   # TD per RZ carry / per non-RZ carry
        p["proj_rz_targets"] = p["targets"] * p.get("rz_tgt_rate", pd.Series(0, index=p.index)).fillna(0)
        p["proj_rz_carries"] = p["carries"] * p.get("rz_car_rate", pd.Series(0, index=p.index)).fillna(0)
        p["rec_tds"] = p["proj_rz_targets"] * RZ_T_TD + (p["targets"] - p["proj_rz_targets"]) * NON_T_TD
        p["rush_tds"] = p["proj_rz_carries"] * RZ_C_TD + (p["carries"] - p["proj_rz_carries"]) * NON_C_TD
    else:
        p["rec_tds"] = p["targets"] * p["rec_td_rate"]
        p["rush_tds"] = p["carries"] * p["rush_td_rate"]

    # age adjustment (scales volume-driven production)
    if "age" in p.columns:
        mult = p.apply(lambda r: _age_mult(r["position"], r.get("age", np.nan)), axis=1)
        for c in ["targets", "receptions", "rec_yards", "rec_tds", "carries",
                  "rush_yards", "rush_tds", "air_yards"]:
            p[c] = p[c] * mult

    # team air-yards share (proper) for WOPR
    p["_tay"] = p.groupby("team2026")["air_yards"].transform("sum")
    p["air_yards_share"] = p["air_yards"] / p["_tay"].replace(0, np.nan)
    p["wopr"] = 1.5 * p["proj_tgt_share"] + 0.7 * p["air_yards_share"].fillna(0)

    # projected team totals for shares
    p["team_targets"] = p.groupby("team2026")["targets"].transform("sum")
    p["team_carries"] = p.groupby("team2026")["carries"].transform("sum")

    # projected red-zone shares (for the Opportunity grade)
    if "proj_rz_targets" in p.columns:
        p["rz_tgt_share"] = p["proj_rz_targets"] / p.groupby("team2026")["proj_rz_targets"].transform("sum").replace(0, np.nan)
        p["rz_rush_share"] = p["proj_rz_carries"] / p.groupby("team2026")["proj_rz_carries"].transform("sum").replace(0, np.nan)

    # full flex_spine-schema derivations so the scoring engine consumes it directly
    p["ppr"] = (p["receptions"] + 0.1 * p["rec_yards"] + 6 * p["rec_tds"]
                + 0.1 * p["rush_yards"] + 6 * p["rush_tds"])
    p["pts_ppr"] = p["ppr"]
    p["pts_half"] = p["ppr"] - 0.5 * p["receptions"]
    p["pts_std"] = p["ppr"] - 1.0 * p["receptions"]
    p["ppg_ppr"] = p["pts_ppr"] / PROJ_GAMES
    p["ppg_half"] = p["pts_half"] / PROJ_GAMES
    p["ppg_std"] = p["pts_std"] / PROJ_GAMES
    p["tgt_share_season"] = p["proj_tgt_share"]
    p["rush_share"] = p["proj_rush_share"]
    p["target_share"] = p["proj_tgt_share"]
    p["tgts_per_game"] = p["targets"] / PROJ_GAMES
    p["carries_per_game"] = p["carries"] / PROJ_GAMES
    p["touch_share"] = (p["targets"] + p["carries"]) / (p["team_targets"] + p["team_carries"]).replace(0, np.nan)
    p["yards_per_target"] = p["rec_yards"] / p["targets"].replace(0, np.nan)
    p["yards_per_carry"] = p["rush_yards"] / p["carries"].replace(0, np.nan)
    p["catch_rate"] = p["receptions"] / p["targets"].replace(0, np.nan)
    p["total_tds"] = p["rec_tds"] + p["rush_tds"]
    p["total_yards"] = p["rec_yards"] + p["rush_yards"]
    p["season"] = 2026
    p["games"] = PROJ_GAMES
    p = p.rename(columns={"team2026": "recent_team"})

    # projected 2026 team context from franchise history
    p = p.merge(team_context_proj(hist), on="recent_team", how="left")

    cols = ["player_id", "player_display_name", "position", "recent_team", "season", "games",
            "targets", "receptions", "rec_yards", "rec_tds", "air_yards",
            "carries", "rush_yards", "rush_tds", "ppr", "target_share", "air_yards_share", "wopr",
            "team_targets", "team_carries", "tgt_share_season", "rush_share",
            "tgts_per_game", "carries_per_game", "touch_share",
            "pts_ppr", "pts_half", "pts_std", "ppg_ppr", "ppg_half", "ppg_std",
            "yards_per_target", "yards_per_carry", "catch_rate", "total_tds", "total_yards",
            "plays_per_game", "pass_rate", "proe"]
    for c in ["rz_tgt_share", "rz_rush_share"]:   # red-zone shares (optional)
        if c in p.columns:
            cols.append(c)
    if "headshot_url" in p.columns:   # already carried through from the roster merge
        cols.append("headshot_url")
    if "age" in p.columns:
        cols.append("age")
    if "is_rookie" in p.columns:
        p["is_rookie"] = p["is_rookie"].fillna(0).astype(int)
        cols.append("is_rookie")
    if "prospect" in p.columns:
        cols.append("prospect")
    return p[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spine", default="flex_spine.csv")
    ap.add_argument("--rosters", default="rosters_2026.csv")
    ap.add_argument("--out", default="flex_proj_spine_2026.csv")
    args = ap.parse_args()
    hist = pd.read_csv(args.spine)
    ros = pd.read_csv(args.rosters)
    spine26 = project(hist, ros)
    spine26.to_csv(args.out, index=False)
    print(f"Projected {len(spine26)} players for 2026 -> {args.out}")


if __name__ == "__main__":
    main()
