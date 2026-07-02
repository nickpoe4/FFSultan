#!/usr/bin/env python3
"""
flex_project_stats.py  -- Sultan 2026 bottom-up statistical projections.

Projects an actual STAT LINE for each WR/RB/TE, then derives fantasy points:
  targets, receptions, rec_yards, rec_tds, carries, rush_yards, rush_tds,
  total_tds, total_yards, pts_ppr, pts_half, ppg_ppr, ppg_half.

Approach
--------
Returning players (have 2023-25 history in the spine): bottom-up.
  volume  = recency-weighted per-game rate, reliability-regressed to positional
            mean, then scaled by the 2026 team pass/run environment.
  efficiency = catch_rate, yards/target, yards/carry -> recency-weighted,
            regressed to positional means (YPC hardest, regressed most).
  touchdowns = BALANCED: 50% from projected red-zone opportunity * positional
            RZ conversion rate, 50% from the player's own recency-weighted
            TD-per-opportunity history.  (Tames TD variance.)
  games   = taken from data.json (the model's already-projected availability,
            which bakes in the known injury cases) for one source of truth.

Rookies (no NFL history): allocate a positional-archetype stat line from the
  model's existing projected PPG (data.json), calibrated to veteran
  stat-per-point ratios within position, then rescaled so recomputed points
  match the projection.  Flagged confidence="rookie" (lower).

All knobs are named constants at the top.  Pure local computation (no network).
"""

import csv, json, re, math, statistics as st
from collections import defaultdict

# ----------------------------- paths -----------------------------
SPINE   = "/mnt/user-data/uploads/flex_spine__4_.csv"
DATAJS  = "/mnt/user-data/outputs/data.json"
TEAMCTX = "/mnt/user-data/outputs/team_context_2026.csv"
INJ     = "/mnt/user-data/uploads/injuries.csv"
OUT     = "/mnt/user-data/outputs/projections_2026.csv"

# ----------------------------- knobs -----------------------------
REC_W = {2025: 0.55, 2024: 0.30, 2023: 0.15}   # recency weights (renormalized)
K_VOL = 12     # games; reliability anchor for per-game volume rates
K_EFF = 40     # targets; anchor for catch_rate & yards/target
K_YPC = 70     # carries; anchor for yards/carry (noisiest -> largest K)
K_TD  = 45     # opportunities; anchor for historical TD-per-opp rates
TD_RZ_W = 0.50            # weight on red-zone-opportunity TD estimate (balanced)
ENV_CLAMP = (0.85, 1.15)  # cap team-environment volume multiplier
GAMES_FALLBACK = 16       # if a returning player is not in data.json

def norm(n):
    return re.sub(r'[^a-z0-9]', '', (n or '').lower())

# ----------------------------- load -----------------------------
spine = list(csv.DictReader(open(SPINE)))
data  = json.load(open(DATAJS))

def f(x):
    try: return float(x)
    except: return 0.0

# team environment multipliers
team_pass = {}
try:
    tc = list(csv.DictReader(open(TEAMCTX)))
    lg_pass = st.mean(f(r['proj_pass_rate']) for r in tc if r.get('proj_pass_rate'))
    for r in tc:
        pr = f(r['proj_pass_rate'])
        pass_mult = max(ENV_CLAMP[0], min(ENV_CLAMP[1], pr / lg_pass)) if lg_pass else 1.0
        rush_mult = max(ENV_CLAMP[0], min(ENV_CLAMP[1], (1-pr)/(1-lg_pass))) if lg_pass else 1.0
        team_pass[r['team']] = (pass_mult, rush_mult)
except FileNotFoundError:
    pass
def env(team): return team_pass.get(team, (1.0, 1.0))

# injury severity haircut (only used if a player is missing from data.json games)
sev = {}
try:
    for r in csv.DictReader(open(INJ)):
        blob = (r.get('detail','')+' '+r.get('timeline','')).lower()
        if any(k in blob for k in ['acl','achilles','jeopardy','pup','out for','season-ending']):
            sev[norm(r['name'])] = 0.75
        elif any(k in blob for k in ['surgery','fracture','torn','miss']):
            sev[norm(r['name'])] = 0.9
except FileNotFoundError:
    pass

# data.json lookups by name|pos
dj = {(norm(r['name']), r['pos']): r for r in data}

# ----------------------------- positional anchors -----------------------------
# group spine seasons by player
byplayer = defaultdict(dict)
for r in spine:
    byplayer[(r['player_id'])][int(r['season'])] = r
# also index a display name + pos for each player id (use most recent season)
pid_meta = {}
for pid, seasons in byplayer.items():
    latest = seasons[max(seasons)]
    pid_meta[pid] = (latest['player_display_name'], latest['position'], latest['recent_team'])

# positional means (games-weighted) + RZ conversion rates
pos_acc = defaultdict(lambda: defaultdict(float))
pos_rz  = defaultdict(lambda: {'rec_td':0.0,'rz_tgt':0.0,'rush_td':0.0,'rz_car':0.0})
for r in spine:
    p = r['position']; g = f(r['games']) or 1
    pos_acc[p]['cr_n']  += f(r['catch_rate'])*g;  pos_acc[p]['cr_d']  += g
    pos_acc[p]['ypt_n'] += f(r['yards_per_target'])*g
    if f(r['carries'])>0:
        pos_acc[p]['ypc_n'] += f(r['yards_per_carry'])*f(r['carries']); pos_acc[p]['ypc_d'] += f(r['carries'])
    pos_acc[p]['tpg_n'] += f(r['tgts_per_game'])*g
    pos_acc[p]['cpg_n'] += f(r['carries_per_game'])*g
    pos_rz[p]['rec_td'] += f(r['rec_tds']);  pos_rz[p]['rz_tgt'] += f(r['rz_targets'])
    pos_rz[p]['rush_td']+= f(r['rush_tds']); pos_rz[p]['rz_car'] += f(r['rz_carries'])
POS = {}
for p, a in pos_acc.items():
    POS[p] = dict(
        catch_rate = a['cr_n']/a['cr_d'] if a['cr_d'] else 0.62,
        ypt        = a['ypt_n']/a['cr_d'] if a['cr_d'] else 7.0,
        ypc        = a['ypc_n']/a['ypc_d'] if a['ypc_d'] else 4.2,
        tpg        = a['tpg_n']/a['cr_d'] if a['cr_d'] else 2.0,
        cpg        = a['cpg_n']/a['cr_d'] if a['cr_d'] else 1.0,
        rz_tgt_td  = pos_rz[p]['rec_td']/pos_rz[p]['rz_tgt'] if pos_rz[p]['rz_tgt'] else 0.28,
        rz_car_td  = pos_rz[p]['rush_td']/pos_rz[p]['rz_car'] if pos_rz[p]['rz_car'] else 0.12,
    )

def reg(player_val, sample_n, K, anchor):
    """reliability regression toward positional anchor"""
    w = sample_n/(sample_n+K) if (sample_n+K)>0 else 0.0
    return w*player_val + (1-w)*anchor

def rec_weighted(seasons, field, per_game=False):
    """recency-weighted value of `field` over available seasons"""
    num=den=0.0
    for yr, w in REC_W.items():
        if yr in seasons:
            r = seasons[yr]
            v = f(r[field])
            if per_game:
                g = f(r['games']) or 1; v = v/g
            num += w*v; den += w
    return num/den if den else 0.0

def wsum(seasons, field):
    return sum(f(seasons[yr][field]) for yr in seasons if yr in REC_W)

# ----------------------------- project returning players -----------------------------
rows_out = []
def scoring(rec, ry, rt, rushy, rutd):
    tds = rt+rutd
    ppr  = 0.1*(ry+rushy) + 6*tds + 1.0*rec
    half = 0.1*(ry+rushy) + 6*tds + 0.5*rec
    return ppr, half

projected_pids = set()
for pid, seasons in byplayer.items():
    name, pos, team = pid_meta[pid]
    if pos not in ('WR','RB','TE'): continue
    key = (norm(name), pos)
    dr = dj.get(key)
    if dr and dr.get('is_rookie'): continue   # handled in rookie pass
    # games
    if dr: games = f(dr.get('games')) or GAMES_FALLBACK
    else:
        games = min(17, rec_weighted(seasons,'games'))
        if games<=0: games = GAMES_FALLBACK
        games *= sev.get(norm(name),1.0)
    games = max(1.0, min(17.0, games))
    team = (dr.get('team') if dr else team) or team
    pmult, rmult = env(team)

    # samples for reliability
    g_samp   = wsum(seasons,'games')
    tgt_samp = wsum(seasons,'targets')
    car_samp = wsum(seasons,'carries')

    # volume (per game -> season, env-scaled, regressed)
    tpg = reg(rec_weighted(seasons,'tgts_per_game'),    g_samp, K_VOL, POS[pos]['tpg'])
    cpg = reg(rec_weighted(seasons,'carries_per_game'), g_samp, K_VOL, POS[pos]['cpg'])
    targets = tpg*games*pmult
    carries = cpg*games*rmult

    # efficiency (regressed)
    cr  = reg(rec_weighted(seasons,'catch_rate'),      tgt_samp, K_EFF, POS[pos]['catch_rate'])
    ypt = reg(rec_weighted(seasons,'yards_per_target'),tgt_samp, K_EFF, POS[pos]['ypt'])
    ypc = reg(rec_weighted(seasons,'yards_per_carry'), car_samp, K_YPC, POS[pos]['ypc'])
    cr  = max(0.35, min(0.95, cr))
    rec       = targets*cr
    rec_yards = targets*ypt
    rush_yards= carries*ypc

    # touchdowns: balanced blend of RZ-opportunity and player history
    rz_tpg = reg(rec_weighted(seasons,'rz_targets',per_game=True), g_samp, K_VOL, POS[pos]['rz_tgt_td']*0)  # anchor 0 -> pure own, then league rate applied
    rz_cpg = reg(rec_weighted(seasons,'rz_carries',per_game=True), g_samp, K_VOL, 0)
    rz_tgt_season = max(0.0, rz_tpg)*games
    rz_car_season = max(0.0, rz_cpg)*games
    rec_td_rz  = rz_tgt_season*POS[pos]['rz_tgt_td']
    rush_td_rz = rz_car_season*POS[pos]['rz_car_td']
    # historical TD-per-opportunity
    hist_rec_td_rate = reg(rec_weighted(seasons,'rec_tds')/max(1e-9,rec_weighted(seasons,'targets')),
                           tgt_samp, K_TD, POS[pos]['rz_tgt_td']*0.10)  # league ~ TD per target
    hist_rush_td_rate= reg(rec_weighted(seasons,'rush_tds')/max(1e-9,rec_weighted(seasons,'carries')),
                           car_samp, K_TD, POS[pos]['rz_car_td']*0.10)
    rec_td_hist  = hist_rec_td_rate*targets
    rush_td_hist = hist_rush_td_rate*carries
    rec_tds  = max(0.0, TD_RZ_W*rec_td_rz  + (1-TD_RZ_W)*rec_td_hist)
    rush_tds = max(0.0, TD_RZ_W*rush_td_rz + (1-TD_RZ_W)*rush_td_hist)

    ppr, half = scoring(rec, rec_yards, rec_tds, rush_yards, rush_tds)
    rows_out.append(dict(name=name,pos=pos,team=team,is_rookie=0,confidence="returning",
        games=games,targets=targets,receptions=rec,rec_yards=rec_yards,rec_tds=rec_tds,
        carries=carries,rush_yards=rush_yards,rush_tds=rush_tds,
        total_tds=rec_tds+rush_tds,total_yards=rec_yards+rush_yards,
        pts_ppr=ppr,pts_half=half,ppg_ppr=ppr/games,ppg_half=half/games))
    projected_pids.add(key)

# ----------------------------- rookie allocation -----------------------------
# veteran stat-per-PPR-point ratios within position (for archetype allocation)
vet_ratio = defaultdict(lambda: defaultdict(list))
for r in rows_out:
    P=r['pos']; pts=r['pts_ppr'] or 1
    for s in ('targets','receptions','rec_yards','rec_tds','carries','rush_yards','rush_tds'):
        vet_ratio[P][s].append(r[s]/pts)
def ratio(P,s): 
    xs=vet_ratio[P][s]; return st.median(xs) if xs else 0.0

for dr in data:
    if not dr.get('is_rookie'): continue
    pos=dr['pos']
    if pos not in ('WR','RB','TE'): continue
    games=f(dr.get('games')) or 15
    season_ppr = f(dr.get('ppg_ppr'))*games
    line={s: ratio(pos,s)*season_ppr for s in
          ('targets','receptions','rec_yards','rec_tds','carries','rush_yards','rush_tds')}
    # rescale so recomputed PPR matches the model's projected points
    ppr,half = scoring(line['receptions'],line['rec_yards'],line['rec_tds'],line['rush_yards'],line['rush_tds'])
    if ppr>0:
        k=season_ppr/ppr
        for s in line: line[s]*=k
        ppr,half = scoring(line['receptions'],line['rec_yards'],line['rec_tds'],line['rush_yards'],line['rush_tds'])
    rows_out.append(dict(name=dr['name'],pos=pos,team=dr.get('team',''),is_rookie=1,confidence="rookie",
        games=games,targets=line['targets'],receptions=line['receptions'],rec_yards=line['rec_yards'],
        rec_tds=line['rec_tds'],carries=line['carries'],rush_yards=line['rush_yards'],rush_tds=line['rush_tds'],
        total_tds=line['rec_tds']+line['rush_tds'],total_yards=line['rec_yards']+line['rush_yards'],
        pts_ppr=ppr,pts_half=half,ppg_ppr=ppr/games,ppg_half=half/games))

# ----------------------------- write -----------------------------
COLS=['name','pos','team','is_rookie','confidence','games','targets','receptions','rec_yards',
      'rec_tds','carries','rush_yards','rush_tds','total_tds','total_yards','pts_ppr','pts_half','ppg_ppr','ppg_half']
def rnd(v,s):
    if s in ('targets','receptions','carries'): return round(v,1)
    if s in ('rec_yards','rush_yards','total_yards','pts_ppr','pts_half'): return round(v)
    if s in ('rec_tds','rush_tds','total_tds'): return round(v,1)
    if s in ('ppg_ppr','ppg_half'): return round(v,2)
    if s=='games': return round(v,1)
    return v
with open(OUT,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=COLS); w.writeheader()
    for r in sorted(rows_out,key=lambda x:-x['pts_ppr']):
        w.writerow({c:(rnd(r[c],c) if isinstance(r[c],(int,float)) and c not in('is_rookie',) else r[c]) for c in COLS})
print("wrote",OUT,"rows:",len(rows_out))

# ----------------------------- validation -----------------------------
def spearman(xy):
    xy=[(a,b) for a,b in xy if a is not None and b is not None]
    n=len(xy)
    if n<3: return float('nan')
    def rank(vals):
        order=sorted(range(n),key=lambda i:vals[i]); rk=[0]*n
        for pos_,i in enumerate(order): rk[i]=pos_
        return rk
    xs=[a for a,_ in xy]; ys=[b for _,b in xy]
    rx=rank(xs); ry=rank(ys)
    d2=sum((rx[i]-ry[i])**2 for i in range(n))
    return 1-6*d2/(n*(n*n-1))

print("\n=== VALIDATION: composed PPG vs existing model PPG (returning players) ===")
for pos in ('RB','WR','TE'):
    xy=[]; ae=[]
    for r in rows_out:
        if r['is_rookie'] or r['pos']!=pos: continue
        dr=dj.get((norm(r['name']),pos))
        if dr and dr.get('ppg_ppr') is not None:
            xy.append((r['ppg_ppr'], f(dr['ppg_ppr']))); ae.append(abs(r['ppg_ppr']-f(dr['ppg_ppr'])))
    if xy:
        print(f"  {pos}: n={len(xy):3d}  Spearman={spearman(xy):.3f}  MAE={st.mean(ae):.2f} ppg")

print("\n=== SAMPLE: top projected lines by position (PPR) ===")
for pos in ('RB','WR','TE'):
    print(f"\n-- {pos} --")
    top=[r for r in rows_out if r['pos']==pos and not r['is_rookie']]
    top=sorted(top,key=lambda x:-x['pts_ppr'])[:8]
    print(f"  {'player':22s}{'g':>4}{'tgt':>5}{'rec':>5}{'ryd':>6}{'rtd':>5}{'car':>5}{'rush':>6}{'rutd':>5}{'PPR':>6}{'HALF':>6}")
    for r in top:
        print(f"  {r['name'][:21]:22s}{r['games']:4.0f}{r['targets']:5.0f}{r['receptions']:5.0f}"
              f"{r['rec_yards']:6.0f}{r['rec_tds']:5.1f}{r['carries']:5.0f}{r['rush_yards']:6.0f}"
              f"{r['rush_tds']:5.1f}{r['pts_ppr']:6.0f}{r['pts_half']:6.0f}")

print("\n=== SAMPLE: top rookies (allocation-based, lower confidence) ===")
rk=sorted([r for r in rows_out if r['is_rookie']],key=lambda x:-x['pts_ppr'])[:6]
for r in rk:
    print(f"  {r['name'][:21]:22s} {r['pos']}  g{r['games']:.0f}  {r['receptions']:.0f}/{r['rec_yards']:.0f}/{r['rec_tds']:.1f} rec"
          f"  {r['carries']:.0f}/{r['rush_yards']:.0f}/{r['rush_tds']:.1f} rush  PPR {r['pts_ppr']:.0f}")
