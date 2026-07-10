#!/usr/bin/env python3
"""
flex_project_stats.py -- Sultan 2026 bottom-up statistical projections.

Projects an actual STAT LINE per WR/RB/TE, then derives fantasy points:
  targets, receptions, rec_yards, rec_tds, carries, rush_yards, rush_tds,
  + Full-PPR / Half-PPR points.

Used two ways:
  * In the hosted pipeline (sultan_streamlit.py):
        import flex_project_stats
        records = flex_project_stats.attach(records, _pd(load_history()).to_dict("records"))
    attach() adds these fields to each record (matched by name|pos):
        p_tgt p_rec p_ryd p_rtd p_car p_ruy p_rutd p_pts_ppr p_pts_half
    Returning players -> bottom-up (volume x efficiency, red-zone-driven TDs).
    Rookies          -> archetype allocation from the model's projected PPG.
  * Standalone (python flex_project_stats.py): validates against data.json and
    writes projections_2026.csv (same numbers as the module path).

Design notes: volume = recency-weighted per-game rate, reliability-regressed to
positional mean, scaled by the 2026 team pass/run environment; efficiency
(catch rate, Y/tgt, Y/carry) regressed to positional means (YPC regressed most);
touchdowns are a BALANCED blend of projected red-zone opportunity x positional
RZ conversion and the player's own TD-per-opportunity history. All knobs below.
"""

import math, re, statistics as st
from collections import defaultdict

# ----------------------------- knobs -----------------------------
REC_W   = {2025: 0.55, 2024: 0.30, 2023: 0.15}   # recency weights (renormalized)
K_VOL   = 12      # games; anchor for per-game volume rates
K_EFF   = 40      # targets; anchor for catch_rate & yards/target
K_YPC   = 70      # carries; anchor for yards/carry (noisiest -> largest K)
K_TD    = 45      # opportunities; anchor for historical TD-per-opp rates
TD_RZ_W = 0.50    # weight on red-zone-opportunity TD estimate (balanced)
ENV_CLAMP = (0.85, 1.15)
POSSET  = ("WR", "RB", "TE")

# positional fallbacks if the spine can't supply an anchor
FALLBACK = {
    "WR": dict(catch_rate=0.62, ypt=8.1, ypc=6.5, tpg=5.2, cpg=0.4, rz_tgt_td=0.20, rz_car_td=0.15),
    "RB": dict(catch_rate=0.75, ypt=6.3, ypc=4.3, tpg=2.6, cpg=9.5, rz_tgt_td=0.22, rz_car_td=0.11),
    "TE": dict(catch_rate=0.66, ypt=7.2, ypc=5.0, tpg=3.8, cpg=0.1, rz_tgt_td=0.24, rz_car_td=0.12),
}


def _norm(n):
    return re.sub(r"[^a-z0-9]", "", str(n or "").lower())


def _f(x):
    try:
        v = float(x)
        return 0.0 if (v != v or math.isinf(v)) else v   # NaN/inf -> 0
    except (TypeError, ValueError):
        return 0.0


def _get(row, *names):
    """first present, finite column among names (dict-like row)"""
    for n in names:
        if n in row and row[n] is not None:
            v = _f(row[n])
            if v or v == 0.0:
                return v
    return 0.0


def _has(row, name):
    return name in row and row[name] is not None and _f(row[name]) == _f(row[name])


def _scoring(rec, ry, rt, rushy, rutd):
    tds = rt + rutd
    ppr  = 0.1 * (ry + rushy) + 6 * tds + 1.0 * rec
    half = 0.1 * (ry + rushy) + 6 * tds + 0.5 * rec
    return ppr, half


def _reg(pv, n, K, anchor):
    w = n / (n + K) if (n + K) > 0 else 0.0
    return w * pv + (1 - w) * anchor


# --------------------- team environment multipliers ---------------------
def _load_team_env(path):
    env = {}
    try:
        import csv
        rows = list(csv.DictReader(open(path)))
        lg = st.mean(_f(r["proj_pass_rate"]) for r in rows if r.get("proj_pass_rate"))
        for r in rows:
            pr = _f(r["proj_pass_rate"])
            pm = max(ENV_CLAMP[0], min(ENV_CLAMP[1], pr / lg)) if lg else 1.0
            rm = max(ENV_CLAMP[0], min(ENV_CLAMP[1], (1 - pr) / (1 - lg))) if lg else 1.0
            env[r["team"]] = (pm, rm)
    except Exception:
        pass
    return env


# --------------------- per-player historical aggregation ---------------------
def _rate(row, per_game_col, total_col, games):
    """per-game value, using the per_game column if present else total/games"""
    if _has(row, per_game_col):
        return _f(row[per_game_col])
    g = games or _f(row.get("games")) or 1
    return _f(row.get(total_col, 0)) / (g or 1)


def _eff(row, col, num_col, den_col):
    if _has(row, col):
        return _f(row[col])
    den = _f(row.get(den_col, 0))
    return _f(row.get(num_col, 0)) / den if den else 0.0


def _build_positional(hist):
    acc = defaultdict(lambda: defaultdict(float))
    for r in hist:
        p = r.get("position") or r.get("pos")
        if p not in POSSET:
            continue
        g = _f(r.get("games")) or 1
        cr  = _eff(r, "catch_rate", "receptions", "targets")
        ypt = _eff(r, "yards_per_target", "rec_yards", "targets")
        acc[p]["cr_n"]  += cr * g;  acc[p]["g"] += g
        acc[p]["ypt_n"] += ypt * g
        car = _f(r.get("carries"))
        if car > 0:
            ypc = _eff(r, "yards_per_carry", "rush_yards", "carries")
            acc[p]["ypc_n"] += ypc * car;  acc[p]["car"] += car
        acc[p]["tpg_n"] += _rate(r, "tgts_per_game", "targets", g) * g
        acc[p]["cpg_n"] += _rate(r, "carries_per_game", "carries", g) * g
        acc[p]["rec_td"] += _f(r.get("rec_tds"));  acc[p]["rz_tgt"] += _f(r.get("rz_targets"))
        acc[p]["rush_td"] += _f(r.get("rush_tds")); acc[p]["rz_car"] += _f(r.get("rz_carries"))
    POS = {}
    for p in POSSET:
        a = acc.get(p); fb = FALLBACK[p]
        if not a or not a["g"]:
            POS[p] = dict(fb); continue
        POS[p] = dict(
            catch_rate=a["cr_n"]/a["g"] if a["g"] else fb["catch_rate"],
            ypt=a["ypt_n"]/a["g"] if a["g"] else fb["ypt"],
            ypc=a["ypc_n"]/a["car"] if a["car"] else fb["ypc"],
            tpg=a["tpg_n"]/a["g"] if a["g"] else fb["tpg"],
            cpg=a["cpg_n"]/a["g"] if a["g"] else fb["cpg"],
            rz_tgt_td=a["rec_td"]/a["rz_tgt"] if a["rz_tgt"] else fb["rz_tgt_td"],
            rz_car_td=a["rush_td"]/a["rz_car"] if a["rz_car"] else fb["rz_car_td"],
        )
    return POS


def _project_returning(hist, POS, env):
    """returns {(norm_name, pos): line_dict} for players with history"""
    byp = defaultdict(dict)
    meta = {}
    for r in hist:
        p = r.get("position") or r.get("pos")
        if p not in POSSET:
            continue
        pid = r.get("player_id") or r.get("player_display_name") or r.get("name")
        seas = int(_f(r.get("season")) or 0)
        byp[pid][seas] = r
        meta[pid] = (r.get("player_display_name") or r.get("name"), p, r.get("recent_team") or r.get("team"))

    def rw(seasons, field, per_game=False, num_col=None, den_col=None, eff=None):
        num = den = 0.0
        for yr, w in REC_W.items():
            if yr not in seasons:
                continue
            row = seasons[yr]
            if eff:
                v = _eff(row, eff[0], eff[1], eff[2])
            elif per_game:
                g = _f(row.get("games")) or 1
                v = _f(row.get(field)) / g
            else:
                v = _f(row.get(field))
            num += w * v; den += w
        return num / den if den else 0.0

    def wsum(seasons, field):
        return sum(_f(seasons[yr].get(field)) for yr in seasons if yr in REC_W)

    out = {}
    for pid, seasons in byp.items():
        name, pos, team = meta[pid]
        pm, rm = env.get(team, (1.0, 1.0))
        g_samp   = wsum(seasons, "games")
        tgt_samp = wsum(seasons, "targets")
        car_samp = wsum(seasons, "carries")
        games = min(17.0, rw(seasons, "games")) or 16.0     # provisional; overridden by record games in attach()

        tpg = _reg(rw(seasons, "tgts_per_game") or rw(seasons, "targets", per_game=True), g_samp, K_VOL, POS[pos]["tpg"])
        cpg = _reg(rw(seasons, "carries_per_game") or rw(seasons, "carries", per_game=True), g_samp, K_VOL, POS[pos]["cpg"])
        cr  = _reg(rw(seasons, None, eff=("catch_rate", "receptions", "targets")), tgt_samp, K_EFF, POS[pos]["catch_rate"])
        ypt = _reg(rw(seasons, None, eff=("yards_per_target", "rec_yards", "targets")), tgt_samp, K_EFF, POS[pos]["ypt"])
        ypc = _reg(rw(seasons, None, eff=("yards_per_carry", "rush_yards", "carries")), car_samp, K_YPC, POS[pos]["ypc"])
        cr  = max(0.35, min(0.95, cr))
        rz_tpg = rw(seasons, "rz_targets", per_game=True)
        rz_cpg = rw(seasons, "rz_carries", per_game=True)
        hist_rec_td = rw(seasons, "rec_tds") / max(1e-9, rw(seasons, "targets"))
        hist_rush_td = rw(seasons, "rush_tds") / max(1e-9, rw(seasons, "carries"))
        out[(_norm(name), pos)] = dict(
            pos=pos, pm=pm, rm=rm, tpg=tpg, cpg=cpg, cr=cr, ypt=ypt, ypc=ypc,
            rz_tpg=max(0.0, rz_tpg), rz_cpg=max(0.0, rz_cpg),
            htd_t=_reg(hist_rec_td, tgt_samp, K_TD, POS[pos]["rz_tgt_td"] * 0.10),
            htd_c=_reg(hist_rush_td, car_samp, K_TD, POS[pos]["rz_car_td"] * 0.10),
        )
    return out


def _compose_returning(pr, games, POS):
    pos = pr["pos"]
    targets = pr["tpg"] * games * pr["pm"]
    carries = pr["cpg"] * games * pr["rm"]
    rec        = targets * pr["cr"]
    rec_yards  = targets * pr["ypt"]
    rush_yards = carries * pr["ypc"]
    rz_t = pr["rz_tpg"] * games
    rz_c = pr["rz_cpg"] * games
    rec_td  = max(0.0, TD_RZ_W * (rz_t * POS[pos]["rz_tgt_td"]) + (1 - TD_RZ_W) * (pr["htd_t"] * targets))
    rush_td = max(0.0, TD_RZ_W * (rz_c * POS[pos]["rz_car_td"]) + (1 - TD_RZ_W) * (pr["htd_c"] * carries))
    ppr, half = _scoring(rec, rec_yards, rec_td, rush_yards, rush_td)
    return dict(p_tgt=targets, p_rec=rec, p_ryd=rec_yards, p_rtd=rec_td,
                p_car=carries, p_ruy=rush_yards, p_rutd=rush_td,
                p_pts_ppr=ppr, p_pts_half=half)


# --------------------- public entry point ---------------------
def attach(records, hist_rows, team_ctx_path="team_context_2026.csv"):
    """Mutate `records` in place, adding p_* projection fields. Returns records.

    records   : list of player dicts (need name, pos, games, is_rookie, ppg_ppr).
    hist_rows : iterable of dict-like historical player-season rows (the spine).
    """
    hist = list(hist_rows)
    POS = _build_positional(hist)
    env = _load_team_env(team_ctx_path)
    vet = _project_returning(hist, POS, env)

    # returning players
    for r in records:
        pos = r.get("pos")
        if pos not in POSSET or r.get("is_rookie"):
            continue
        pr = vet.get((_norm(r.get("name")), pos))
        if not pr:
            continue
        pm, rm = env.get(r.get("team"), (pr["pm"], pr["rm"]))
        pr = dict(pr); pr["pm"], pr["rm"] = pm, rm
        games = _f(r.get("games")) or 16.0
        r.update({k: round(v, 3) for k, v in _compose_returning(pr, max(1.0, min(17.0, games)), POS).items()})

    # rookie archetype allocation from projected PPG (veteran stat-per-point medians)
    ratio = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("is_rookie") or "p_pts_ppr" not in r:
            continue
        pts = r["p_pts_ppr"] or 1
        for s in ("p_tgt", "p_rec", "p_ryd", "p_rtd", "p_car", "p_ruy", "p_rutd"):
            ratio[r["pos"]][s].append(r[s] / pts)

    def med(P, s):
        xs = ratio[P][s]; return st.median(xs) if xs else 0.0

    for r in records:
        if not r.get("is_rookie") or r.get("pos") not in POSSET:
            continue
        games = _f(r.get("games")) or 15.0
        season_ppr = _f(r.get("ppg_ppr")) * games
        line = {s: med(r["pos"], s) * season_ppr for s in
                ("p_tgt", "p_rec", "p_ryd", "p_rtd", "p_car", "p_ruy", "p_rutd")}
        ppr, half = _scoring(line["p_rec"], line["p_ryd"], line["p_rtd"], line["p_ruy"], line["p_rutd"])
        if ppr > 0:
            k = season_ppr / ppr
            for s in line:
                line[s] *= k
            ppr, half = _scoring(line["p_rec"], line["p_ryd"], line["p_rtd"], line["p_ruy"], line["p_rutd"])
        line["p_pts_ppr"] = ppr; line["p_pts_half"] = half
        r.update({k: round(v, 3) for k, v in line.items()})

    return records


# ----------------------------- standalone self-test -----------------------------
if __name__ == "__main__":
    import csv, json
    SPINE  = "/mnt/user-data/uploads/flex_spine__4_.csv"
    DATAJS = "/mnt/user-data/outputs/data.json"
    TEAMCTX = "/mnt/user-data/outputs/team_context_2026.csv"
    OUT     = "/mnt/user-data/outputs/projections_2026.csv"

    hist = list(csv.DictReader(open(SPINE)))
    records = json.load(open(DATAJS))
    attach(records, hist, team_ctx_path=TEAMCTX)

    got = [r for r in records if "p_pts_ppr" in r]
    print("records:", len(records), "with projections:", len(got))

    # validation vs existing model PPG
    def spearman(xy):
        xy = [(a, b) for a, b in xy if a is not None and b is not None]
        n = len(xy)
        if n < 3: return float("nan")
        def rank(v):
            o = sorted(range(n), key=lambda i: v[i]); rk = [0]*n
            for p, i in enumerate(o): rk[i] = p
            return rk
        rx = rank([a for a, _ in xy]); ry = rank([b for _, b in xy])
        d2 = sum((rx[i]-ry[i])**2 for i in range(n))
        return 1 - 6*d2/(n*(n*n-1))

    print("\n=== composed PPG vs existing model PPG (returning) ===")
    for pos in ("RB", "WR", "TE"):
        xy = []
        for r in records:
            if r.get("is_rookie") or r.get("pos") != pos or "p_pts_ppr" not in r:
                continue
            g = _f(r.get("games")) or 1
            xy.append((r["p_pts_ppr"]/g, _f(r.get("ppg_ppr"))))
        if xy:
            print(f"  {pos}: n={len(xy):3d}  Spearman={spearman(xy):.3f}")

    COLS = ["name","pos","team","is_rookie","p_tgt","p_rec","p_ryd","p_rtd",
            "p_car","p_ruy","p_rutd","p_pts_ppr","p_pts_half"]
    def rnd(k, v):
        if v is None: return ""
        if k in ("p_ryd","p_ruy","p_pts_ppr","p_pts_half"): return round(v)
        if k in ("p_tgt","p_rec","p_car"): return round(v, 1)
        if k in ("p_rtd","p_rutd"): return round(v, 1)
        return v
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in sorted(got, key=lambda x: -x["p_pts_ppr"]):
            w.writerow({c: (rnd(c, r.get(c)) if c.startswith("p_") else r.get(c)) for c in COLS})
    print("\nwrote", OUT)
