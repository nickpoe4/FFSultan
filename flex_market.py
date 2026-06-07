"""
Sultan — ADP + market analysis (Undervalued / Overvalued / Breakouts).

Single source of truth shared by the offline builder (build_sultan90.py) and the
hosted app (sultan_streamlit.py):

  - Hosted app     -> get_adp() fetches LIVE 2026 ADP from FantasyPros (cached
                      weekly) and falls back to the committed CSV snapshot.
  - Offline build  -> load_adp_csv() reads the committed adp_2026.csv snapshot
                      (the sandbox can't reach the web, so the openable file is a
                      snapshot; the hosted app is the live, self-updating one).

FantasyPros' ADP "overall" pages are SERVER-rendered for 2026 — the player rows
(fp-player-name="..." + an AVG/ADP column) are in the raw HTML, so a plain
requests.get + regex parse works with no key and no browser. The ?year context
is 2026 because we hit the 2026 ADP pages directly.

(Note: Fantasy Football Calculator's API was rejected as a source — its
year=2026 endpoint still returns September-2025-dated drafts, i.e. last season's
numbers, until real 2026 drafts accumulate. FantasyPros is genuine 2026 now.)
"""

import os
import re
import pandas as pd

# 2026 ADP pages (server-rendered; the rows live in the raw HTML).
ADP_PAGES = {
    "ppr": "https://www.fantasypros.com/nfl/adp/ppr-overall.php",
    "half": "https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php",
}
_UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def norm(s):
    s = "".join(c for c in str(s).lower() if c.isalpha() or c == " ")
    return "".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def _fmt_adp(a):
    return "%g" % a


# ---------------------------------------------------------------- ADP sources
def parse_adp_html(html):
    """{norm_name: adp} from a FantasyPros 2026 ADP overall page (the AVG column,
    which is the last <td> in each player row)."""
    out = {}
    for row in html.split("<tr")[1:]:
        nm = re.search(r'fp-player-name="([^"]+)"', row)
        if not nm:
            continue
        tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", row)
        if not tds:
            continue
        try:
            out[norm(nm.group(1))] = float(re.sub(r"<[^>]+>", "", tds[-1]).strip())
        except ValueError:
            continue
    return out


def fetch_adp_live(timeout=20):
    """{'ppr': {...}, 'half': {...}} from FantasyPros' 2026 pages. Raises on
    failure so get_adp() can fall back to the committed snapshot."""
    import requests
    res = {}
    for kind, url in ADP_PAGES.items():
        html = requests.get(url, headers=_UA, timeout=timeout).text
        d = parse_adp_html(html)
        if len(d) < 50:
            raise RuntimeError("ADP parse looks wrong for %s (got %d rows)" % (kind, len(d)))
        res[kind] = d
    return res


def load_adp_csv(path="adp_2026.csv"):
    """{'ppr': {...}, 'half': {...}} from the committed snapshot CSV."""
    if not os.path.exists(path):
        return {"ppr": {}, "half": {}}
    d = pd.read_csv(path)

    def col(c):
        if c not in d.columns:
            return {}
        return {norm(n): float(v) for n, v in zip(d["name"], d[c]) if pd.notna(v) and v != ""}
    return {"ppr": col("adp_ppr"), "half": col("adp_half")}


def get_adp(csv_path="adp_2026.csv", live=True):
    """Live FantasyPros 2026 ADP if reachable, else the committed CSV snapshot.
    Returns ({'ppr':..,'half':..}, source_label)."""
    if live:
        try:
            return fetch_adp_live(), "FantasyPros 2026 (live)"
        except Exception as e:
            print("ADP live fetch failed (%s); using committed snapshot." % e)
    return load_adp_csv(csv_path), "committed 2026 snapshot"


def attach_adp(records, adp):
    for r in records:
        n = norm(r["name"])
        r["adp_ppr"] = adp.get("ppr", {}).get(n)
        r["adp_half"] = adp.get("half", {}).get(n)
    return records


# ---------------------------------------------------------------- market lists
def market_lists(records, scoring="half"):
    """Sultan's default Undervalued / Overvalued / Breakout picks for one scoring,
    comparing the model's positional rank to ADP's positional rank."""
    adpk, rankk, ppgk = "adp_" + scoring, "pos_rank_" + scoring, "ppg_" + scoring
    pool = [r for r in records if r.get(adpk) is not None and r.get(rankk) is not None]
    bypos = {}
    for r in pool:
        bypos.setdefault(r["pos"], []).append(r)
    adp_pos, gap = {}, {}
    for _, lst in bypos.items():
        for i, r in enumerate(sorted(lst, key=lambda r: r[adpk])):
            adp_pos[id(r)] = i + 1
    for r in pool:
        gap[id(r)] = adp_pos[id(r)] - r[rankk]

    def card(r, note):
        return {"name": r["name"], "pos": r["pos"], "team": r["team"],
                "adp": r[adpk], "ppg": r[ppgk], "note": note}

    und = sorted([r for r in pool if r[adpk] <= 170 and r[rankk] <= 40],
                 key=lambda r: gap[id(r)], reverse=True)
    UND = [card(r, "Model has him %s%d, the market drafts him %s%d (ADP %s) — %d spots of value."
                % (r["pos"], r[rankk], r["pos"], adp_pos[id(r)], _fmt_adp(r[adpk]), gap[id(r)]))
           for r in und[:6]]

    ovr = sorted([r for r in pool if r[adpk] <= 120], key=lambda r: gap[id(r)])
    OVR = [card(r, "Market drafts him %s%d (ADP %s) but the model only has him %s%d — paying up %d spots."
                % (r["pos"], adp_pos[id(r)], _fmt_adp(r[adpk]), r["pos"], r[rankk], -gap[id(r)]))
           for r in ovr[:6]]

    def rscore(r):
        opp = r.get("opportunity") or 0
        ctx = r.get("context") or 0
        age = r.get("age")
        youth = max(0.0, (26.0 - age)) * 4 if age else 0
        rook = 8 if r.get("is_rookie") else 0
        return opp * 0.5 + ctx * 0.3 + youth + rook + max(0, gap[id(r)]) * 0.6

    def rnote(r):
        bits = []
        if r.get("age"):
            bits.append("Age %.0f" % r["age"])
        if r.get("is_rookie"):
            bits.append("rookie")
        if r.get("opportunity") is not None:
            bits.append("%.0f opportunity grade" % r["opportunity"])
        bits.append("ADP %s" % _fmt_adp(r[adpk]))
        return "Breakout profile: " + ", ".join(bits) + "."

    roc = sorted([r for r in pool if 28 <= r[adpk] <= 180
                  and (r.get("age") is None or r.get("age") <= 25 or r.get("is_rookie"))],
                 key=rscore, reverse=True)
    ROC = [card(r, rnote(r)) for r in roc[:6]]
    return {"under": UND, "over": OVR, "rocket": ROC}


def both_markets(records):
    return {"half": market_lists(records, "half"), "ppr": market_lists(records, "ppr")}
