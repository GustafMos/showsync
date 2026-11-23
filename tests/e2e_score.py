"""End-to-end scorecard: run the full pipeline on synthetic scenarios and compare with truth.

    .venv/Scripts/python -m tests.e2e_score [fixtures_dir] [--only scenario,...]

Prints one block per scenario and returns a dict of metrics (used by tests/test_e2e.py).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

from showsync.pipeline import analyze
from tests.synth import map_eval, map_inverse_eval

HERE = os.path.dirname(__file__)
DEFAULT = os.path.join(HERE, "_out", "fixtures")


def _pairs(groups):
    s = set()
    for g in groups:
        for a, b in itertools.combinations(sorted(g), 2):
            s.add((a, b))
    return s


def score_scenario(sdir: str) -> dict:
    truth = json.load(open(os.path.join(sdir, "truth.json"), encoding="utf-8"))
    files = sorted(truth["files"])
    refs_dir = os.path.join(sdir, "refs")
    t0 = time.time()
    res = analyze([os.path.join(sdir, f) for f in files],
                  refs=[refs_dir] if os.path.isdir(refs_dir) else None)
    secs = time.time() - t0

    rel = {cid: os.path.relpath(mi.path, sdir).replace("\\", "/") for cid, mi in res.clips.items()}
    cid_of = {v: k for k, v in rel.items()}

    # ---- exact grouping
    is_ref = {c for c, mi in res.clips.items() if getattr(mi, "role", "clip") == "reference"}
    pred_groups = [[rel[c] for c in g.maps if c not in is_ref] for g in res.groups]
    tp_pairs, exp_pairs = _pairs(pred_groups), _pairs(truth["expected_exact_groups"])
    true_pos = tp_pairs & exp_pairs
    grp_prec = len(true_pos) / len(tp_pairs) if tp_pairs else 1.0
    grp_rec = len(true_pos) / len(exp_pairs) if exp_pairs else 1.0
    wrong_pairs = sorted(tp_pairs - exp_pairs)
    missed_pairs = sorted(exp_pairs - tp_pairs)

    # ---- sync error inside correct groups (ms), sampled along the true overlap
    errs = []
    worst = {}
    ambiguous_pairs = []
    for g in res.groups:
        for a, b in itertools.combinations([c for c in g.maps if c not in is_ref], 2):
            ra, rb = rel[a], rel[b]
            if (min(ra, rb), max(ra, rb)) not in exp_pairs:
                continue
            ta, tb = truth["files"][ra], truth["files"][rb]
            lo = max(ta["exact_span"][0], tb["exact_span"][0])
            hi = min(ta["exact_span"][1], tb["exact_span"][1])
            if hi - lo < 1:
                continue
            ev = np.linspace(lo + 0.5, hi - 0.5, 50)
            dev_a = np.array([map_inverse_eval(ta["map"], e) for e in ev])
            dev_b_true = np.array([map_inverse_eval(tb["map"], e) for e in ev])
            # identical repeats: a placement at any recorded alternative counts as correct
            sa = [0.0] + list(g.alt_shifts.get(a, []))
            sb = [0.0] + list(g.alt_shifts.get(b, []))
            e = None
            for xa in sa:
                for xb in sb:
                    pred = g.maps[b].inverse()(g.maps[a](dev_a) + xa - xb)
                    cand = np.abs(pred - dev_b_true) * 1000
                    if e is None or cand.max() < e.max():
                        e = cand
            if len(sa) > 1 or len(sb) > 1:
                ambiguous_pairs.append(f"{ra}~{rb}")
            errs.extend(e.tolist())
            worst[f"{ra}~{rb}"] = round(float(e.max()), 2)
    errs = np.array(errs) if errs else np.array([0.0])

    # ---- song membership: (file, song) items
    truth_items = {(f, s) for s, fl in truth["expected_songs"].items() for f in fl}
    pred_items = set()
    seg_by_id = {s.id: s for s in res.segments}
    for s in res.segments:
        if not s.song_id:
            continue
        if s.owner_kind == "clip":
            pred_items.add((rel[s.owner], s.song_id))
        else:
            g = next(g for g in res.groups if g.id == s.owner)
            for c, (lo, hi) in g.spans.items():
                if c in is_ref:
                    continue
                if min(hi, s.end) - max(lo, s.start) > 3.0:
                    pred_items.add((rel[c], s.song_id))
    # map predicted song ids to truth songs greedily by overlap
    tsongs = sorted({s for _, s in truth_items})
    psongs = sorted({s for _, s in pred_items})
    overlap = {(p, t): len({f for f, s in pred_items if s == p} & {f for f, s in truth_items if s == t})
               for p in psongs for t in tsongs}
    assign = {}
    for (p, t), n in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if n and p not in assign and t not in assign.values():
            assign[p] = t
    mapped = {(f, assign.get(p, "?" + p)) for f, p in pred_items}
    hit = mapped & truth_items
    song_prec = len(hit) / len(mapped) if mapped else (1.0 if not truth_items else 0.0)
    song_rec = len(hit) / len(truth_items) if truth_items else 1.0

    names = {s.id: s.name for s in res.songs}
    return {
        "scenario": truth["scenario"], "seconds": round(secs, 1),
        "groups_pred": len(res.groups), "groups_true": len(truth["expected_exact_groups"]),
        "group_precision": round(grp_prec, 3), "group_recall": round(grp_rec, 3),
        "wrong_pairs": wrong_pairs, "missed_pairs": missed_pairs,
        "sync_ms_median": round(float(np.median(errs)), 3),
        "sync_ms_p95": round(float(np.percentile(errs, 95)), 3),
        "sync_ms_max": round(float(errs.max()), 3), "worst_pairs": worst,
        "songs_pred": len(res.songs), "songs_true": len(tsongs),
        "song_precision": round(song_prec, 3), "song_recall": round(song_rec, 3),
        "song_assign": {names.get(p, p): t for p, t in assign.items()},
        "warnings": res.warnings,
        "ambiguous_clips": sorted(rel[c] for g in res.groups for c in g.alt_shifts),
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only = set(argv[i + 1].split(","))
        argv = argv[:i] + argv[i + 2:]
    root = argv[0] if argv else DEFAULT
    out = {}
    for name in sorted(os.listdir(root)):
        sdir = os.path.join(root, name)
        if not os.path.isfile(os.path.join(sdir, "truth.json")) or (only and name not in only):
            continue
        r = score_scenario(sdir)
        out[name] = r
        print(f"\n== {name}  ({r['seconds']} s)")
        print(f"  exact groups  pred {r['groups_pred']} / true {r['groups_true']}   "
              f"pair precision {r['group_precision']}  recall {r['group_recall']}")
        if r["wrong_pairs"]:
            print(f"  WRONG pairs: {r['wrong_pairs']}")
        if r["missed_pairs"]:
            print(f"  missed pairs: {r['missed_pairs']}")
        print(f"  sync error ms  median {r['sync_ms_median']}  p95 {r['sync_ms_p95']}  max {r['sync_ms_max']}")
        print(f"  songs  pred {r['songs_pred']} / true {r['songs_true']}   "
              f"precision {r['song_precision']}  recall {r['song_recall']}   {r['song_assign']}")
        if r["ambiguous_clips"]:
            print(f"  repeat-ambiguous (fit at several identical repeats): {r['ambiguous_clips']}")
        for w in r["warnings"]:
            print("  ! " + w[:150])
    with open(os.path.join(root, "scorecard.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    main()
