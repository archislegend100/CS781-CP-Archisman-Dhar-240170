"""Manual Codabench score log. Nothing here is computed; values are typed in from the leaderboard.

python scores.py add G1 bert=0.8412 rouge=0.2011 bleu=0.0432 date=2026-10-03 notes="best ckpt"
python scores.py show
"""
import json
import sys
from datetime import date

import yaml

from config import path
from data import writejson

FILE = "scores.json"


def load(p=None):
    p = path(p or FILE)
    return json.load(open(p)) if p.exists() else {}


def add(argv, p=None):
    sid = argv[0]
    kv = dict(a.split("=", 1) for a in argv[1:])
    opt = lambda k: float(kv[k]) if k in kv else None
    rec = {"bert": float(kv["bert"]), "rouge": opt("rouge"), "bleu": opt("bleu"),
           "date": kv.get("date", date.today().isoformat()), "notes": kv.get("notes", "")}
    for k in ("bert", "rouge", "bleu"):
        if rec[k] is not None and not 0.0 <= rec[k] <= 1.0:
            raise SystemExit(f"{k} must be in [0, 1]")
    s = load(p)
    s.setdefault(sid, []).append(rec)
    writejson(path(p or FILE), s, 1)
    return s


def table(s, reg=None):
    if reg is None:
        try:
            reg = {e["id"]: e for e in yaml.safe_load(open(path("phase1/experiments.yaml")))["experiments"]}
        except FileNotFoundError:
            reg = {}
    rows = []
    for sid, recs in s.items():
        r = max(recs, key=lambda x: x["bert"])
        e = reg.get(sid, {})
        rows.append((sid, e.get("name", "?"), e.get("category", "?"), r["bert"], r["rouge"], r["bleu"], r["date"],
                     len(recs), r["notes"]))
    rows.sort(key=lambda x: -x[3])
    lines = [f"{'id':4s} {'name':20s} {'category':16s} {'BERTScore':>9s} {'ROUGE-L':>8s} {'BLEU-4':>7s} {'date':10s} n  notes"]
    f = lambda v, w: f"{v:{w}.4f}" if v is not None else f"{'-':>{w}s}"
    for sid, name, cat, b, r, bl, d, n, notes in rows:
        lines.append(f"{sid:4s} {name[:20]:20s} {cat[:16]:16s} {b:9.4f} {f(r, 8)} {f(bl, 7)} {d:10s} {n}  {notes}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("add", "show"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "add":
        add(sys.argv[2:])
    print(table(load()))


if __name__ == "__main__":
    main()
