"""
FantasyPros consensus (ECR) fetch -> a Sultan analyst source.

Returns a long table [nk, position, rank] (positional ECR rank) for RB/WR/TE,
ready to drop into flex_board's redraft_sources / dynasty_sources lists.

Needs a free FantasyPros API key (https://developers.fantasypros.com or request
one from FantasyPros). The exact endpoint/params can vary by API tier, so this is
written defensively and logs clearly if the response shape differs — first run may
need a small tweak once we see a real payload.
"""

import re
import requests
import pandas as pd

from flex_board import norm

BASE = "https://api.fantasypros.com/v2/json/nfl/{year}/consensus-rankings"


def _players_from(payload):
    if isinstance(payload, dict):
        for k in ("players", "rankings", "data"):
            if isinstance(payload.get(k), list):
                return payload[k]
    if isinstance(payload, list):
        return payload
    return []


def _to_long(players):
    rows = []
    for p in players:
        pos = str(p.get("player_position_id") or p.get("position") or p.get("pos") or "").upper()
        name = p.get("player_name") or p.get("name") or p.get("player")
        pr = str(p.get("pos_rank") or "")
        m = re.search(r"(\d+)", pr)
        ecr = p.get("rank_ecr") or p.get("rank") or p.get("rank_ave")
        rows.append({"name": name, "position": pos,
                     "posrank": (int(m.group(1)) if m else None),
                     "ecr": pd.to_numeric(ecr, errors="coerce")})
    df = pd.DataFrame(rows).dropna(subset=["name", "position"])
    df = df[df["position"].isin(["RB", "WR", "TE"])].copy()
    need = df["posrank"].isna()
    if need.any():  # derive positional rank from overall ECR if pos_rank missing
        df.loc[need, "posrank"] = df[need].groupby("position")["ecr"].rank(method="min")
    df["nk"] = df["name"].map(norm)
    df = df.rename(columns={"posrank": "rank"})
    return df[["nk", "position", "rank"]].dropna()


def fetch(api_key, year=2026, kind="draft", scoring="PPR", positions=("RB", "WR", "TE")):
    """kind: 'draft' (redraft) or 'dynasty'. Returns long [nk, position, rank]."""
    headers = {"x-api-key": api_key}
    frames = []
    for pos in positions:
        params = {"position": pos, "type": kind, "scoring": scoring, "week": 0}
        r = requests.get(BASE.format(year=year), params=params, headers=headers, timeout=25)
        r.raise_for_status()
        frames.append(_to_long(_players_from(r.json())))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["nk", "position", "rank"])
    return out.drop_duplicates(["nk", "position"])
