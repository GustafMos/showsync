"""Pairwise exact sync over all clips, and grouping of the accepted matches.

match_all  : runs dsp.align.match_pair for every plausible clip pair (thread pool).
build_groups: connected components -> EventGroup with composed maps on one event-time line.

Also holds timecode arithmetic (tc_to_seconds / seconds_to_tc, drop-frame aware). The Resolve
plugin copies those two functions, so they depend on nothing but the standard library.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

from dataclasses import replace

import numpy as np

from showsync.config import MATCH_ACCEPT
from showsync.models import EventGroup, Features, MediaInfo, PairMatch, TimeMap

# edge whose placement differs from the tree's by more than this is a wrong match, not drift
GROSS_DISAGREE_S = 1.0
# same-device clips overlapping by more than this in event time are a conflict
SAME_DEVICE_OVERLAP_S = 0.5
# wall-clock slack when deciding two same-device clips cannot overlap
WALL_MARGIN_S = 1.0


# =========================================================================== timecode

_TC_RE = re.compile(r"^\s*(\d+)[:;.](\d+)[:;.](\d+)([:;.])(\d+)\s*$")


def _real_fps(fps: float) -> float:
    """Snap 23.976/29.97/59.94 style rates to the exact N*1000/1001."""
    for nom in (24, 30, 48, 60, 120):
        if abs(fps - nom * 1000.0 / 1001.0) < 0.01:
            return nom * 1000.0 / 1001.0
    return float(fps)


def _drop_frames(fps: float) -> int:
    """Frames dropped per minute for a drop-frame rate, 0 if drop-frame is not defined for fps."""
    r = _real_fps(fps)
    for nom, d in ((30, 2), (60, 4)):
        if abs(r - nom * 1000.0 / 1001.0) < 1e-6:
            return d
    return 0


def tc_is_drop(tc: str, fps: float) -> bool:
    """True when the string uses ';' before the frames and fps is 29.97 / 59.94."""
    m = _TC_RE.match(tc)
    return bool(m and m.group(4) == ";" and _drop_frames(fps) > 0)


def tc_to_seconds(tc: str, fps: float) -> float:
    """'HH:MM:SS:FF' (or ';' before FF for drop-frame) -> seconds of real time."""
    m = _TC_RE.match(tc)
    if not m:
        raise ValueError(f"bad timecode {tc!r}")
    h, mi, s, sep, f = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    nom = int(round(fps))
    frames = (h * 3600 + mi * 60 + s) * nom + f
    d = _drop_frames(fps) if sep == ";" else 0
    if d:
        total_min = h * 60 + mi
        frames -= d * (total_min - total_min // 10)
    return frames / _real_fps(fps)


def seconds_to_tc(sec: float, fps: float, drop: bool = False) -> str:
    """Seconds of real time -> 'HH:MM:SS:FF' ('HH:MM:SS;FF' when drop and fps is 29.97/59.94).
    Wraps at 24 hours."""
    nom = int(round(fps))
    n = int(round(sec * _real_fps(fps)))
    d = _drop_frames(fps) if drop else 0
    if d:
        per_min = nom * 60 - d
        per_10 = nom * 600 - 9 * d
        n %= 24 * 6 * per_10
        blocks, rem = divmod(n, per_10)
        n += 9 * d * blocks
        if rem > d:
            n += d * ((rem - d) // per_min)
    else:
        n %= 24 * 3600 * nom
    f = n % nom
    t = n // nom
    sep = ";" if d else ":"
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}{sep}{f:02d}"


def _shift_tc(tc: str, fps: float, delta_s: float) -> str:
    return seconds_to_tc(tc_to_seconds(tc, fps) + delta_s, fps, drop=tc_is_drop(tc, fps))


def _real_tc(info: Optional[MediaInfo]) -> bool:
    """Embedded TC that carries information (many cameras write 00:00:00:00 on every clip)."""
    if info is None or not info.start_tc or info.fps <= 0:
        return False
    m = _TC_RE.match(info.start_tc)
    return bool(m) and any(int(m[i]) for i in (1, 2, 3, 5))


# =========================================================================== matching

def _epoch(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _disjoint(lo1, hi1, lo2, hi2) -> bool:
    return hi1 + WALL_MARGIN_S < lo2 or hi2 + WALL_MARGIN_S < lo1


def can_skip_pair(ia: Optional[MediaInfo], ib: Optional[MediaInfo]) -> bool:
    """True only when two clips of the same device clearly were recorded one after the other.

    Clocks are never trusted across devices (phone / Hi8 clocks are often wrong). Within one
    device a wrong clock is at least consistent. creation_time may mark the start or the end of
    a recording, so the pair is skipped only if both readings say 'no overlap'."""
    if ia is None or ib is None or not ia.device or ia.device != ib.device:
        return False
    ca, cb = _epoch(ia.creation_time), _epoch(ib.creation_time)
    if ca is not None and cb is not None:
        as_start = _disjoint(ca, ca + ia.duration, cb, cb + ib.duration)
        as_end = _disjoint(ca - ia.duration, ca, cb - ib.duration, cb)
        return as_start and as_end
    if _real_tc(ia) and _real_tc(ib):
        try:
            ta = tc_to_seconds(ia.start_tc, ia.fps)
            tb = tc_to_seconds(ib.start_tc, ib.fps)
        except ValueError:
            return False
        return _disjoint(ta, ta + ia.duration, tb, tb + ib.duration)
    return False


WEAK_KEEP = 0.1          # weak matches kept for corroboration (see corroborate)
STEADY_PPM = 500.0       # clock within this of the group's majority = steady crystal clock (not a tape)
STEADY_MAX_WIGGLE_S = 0.120   # larger wiggle on a steady clock is kept (real dropout, not fit noise)
CORROBORATE_S = 0.040    # two weak placements agreeing this closely corroborate each other
                         # (same limit as the conflict check; 20 ms dropped a real phone clip
                         # whose tape and 4K matches agree within 16 ms at 30 s, more mid-clip)


_MISS = object()


def _matcher_signature(align_module) -> str:
    """Changes whenever the matcher's code or settings change, so cached pairs never go stale."""
    import hashlib
    from showsync import config
    from showsync.dsp import features as feat_mod
    h = hashlib.sha1()
    for mod in (align_module, feat_mod):
        try:
            with open(mod.__file__, "rb") as fh:
                h.update(fh.read())
        except (OSError, AttributeError, TypeError):
            return ""                       # unknown code (patched in tests): do not cache
    h.update(repr(sorted((k, v) for k, v in vars(config).items() if k.isupper())).encode())
    return h.hexdigest()


def _pair_cache_key(ia, ib, code_sig):
    if not code_sig or ia is None or ib is None or not ia.path or not ib.path:
        return None
    from showsync.media.cache import cache_key
    import hashlib
    raw = "|".join([cache_key(ia.path), cache_key(ib.path), code_sig])
    return "pair_" + hashlib.sha1(raw.encode()).hexdigest()


def _pair_cache_load(key, a, b):
    import json
    from showsync.media.cache import cache_dir
    p = cache_dir() / (key + ".json")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _MISS
    if d.get("none"):
        return None
    tm = lambda x: TimeMap.from_dict(x) if x is not None else None
    return PairMatch(**{**d, "a": a, "b": b, "map": tm(d["map"]),
                        "alternatives": [tm(x) for x in d.get("alternatives", [])]})


def _pair_cache_save(key, m):
    import json
    from dataclasses import asdict
    from showsync.media.cache import cache_dir
    if m is None:
        d = {"none": True}
    else:
        d = asdict(m)
        d["map"] = m.map.to_dict()
        d["alternatives"] = [x.to_dict() for x in (m.alternatives or [])]
    p = cache_dir() / (key + ".json")
    tmp = p.with_suffix(".tmp%d" % os.getpid())
    try:
        tmp.write_text(json.dumps(d), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def match_all(feats: dict[str, Features], audio: dict[str, np.ndarray],
              infos: dict[str, MediaInfo], progress=None,
              weak_out: list | None = None) -> list[PairMatch]:
    """Exact-sync every unordered pair of clips that have audio; keep confidence >= MATCH_ACCEPT.

    Threads, not processes: the heavy work is numpy/scipy FFT which releases the GIL, and threads
    avoid pickling hours of audio per task. (Not benchmarked against a ProcessPool yet.)"""
    from showsync.dsp import align  # late import: lets tests patch align.match_pair

    ids = sorted(c for c in feats if c in audio and (c not in infos or infos[c].has_audio))
    pairs = []
    skipped = 0
    for a, b in combinations(ids, 2):
        if can_skip_pair(infos.get(a), infos.get(b)):
            skipped += 1
            continue
        pairs.append((a, b))
    total = len(pairs)
    out: list[PairMatch] = []
    if progress:
        progress(0, total, f"matching {total} pairs" + (f" ({skipped} sequential same-device pairs skipped)" if skipped else ""))
    if not total:
        return out

    code_sig = _matcher_signature(align)

    def work(pair):
        a, b = pair
        key = _pair_cache_key(infos.get(a), infos.get(b), code_sig)
        hit = _pair_cache_load(key, a, b) if key else _MISS
        if hit is not _MISS:
            return hit
        m = align.match_pair(feats[a], feats[b], audio[a], audio[b])
        if key:
            _pair_cache_save(key, m)
        return m

    workers = max(1, min(total, os.cpu_count() or 2))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, p): p for p in pairs}
        for fut in as_completed(futs):
            a, b = futs[fut]
            done += 1
            msg = f"{a} / {b}"
            try:
                m = fut.result()
            except Exception as e:  # one bad pair must not kill the run
                m = None
                msg += f" failed: {e!r}"
            if m is not None and m.confidence >= MATCH_ACCEPT:
                out.append(m)
            elif (m is not None and weak_out is not None and m.confidence >= WEAK_KEEP
                  and not m.ambiguous):
                weak_out.append(m)
            if progress:
                progress(done, total, msg)
    out.sort(key=lambda m: (-m.confidence, m.a, m.b))
    return out


# =========================================================================== grouping

def _name(infos, cid):
    i = infos.get(cid)
    return os.path.basename(i.path) if i is not None and i.path else cid


def _dur(infos, cid):
    i = infos.get(cid)
    return float(i.duration) if i is not None and i.duration > 0 else 0.0


def _shift_map(tm: TimeMap, dt: float) -> TimeMap:
    return TimeMap(tm.knots_src.copy(), tm.knots_dst + dt)


def _simplify(tm: TimeMap) -> TimeMap:
    """Collapse collinear knots of a composed linear map back to two knots."""
    if len(tm.knots_src) > 2 and tm.is_linear:
        return TimeMap.linear(tm.ratio, tm.offset)
    return tm


def _edge_samples(m: PairMatch, infos, n: int = 41) -> np.ndarray:
    """n clip-b times spread evenly over the overlap of a match (where both clips exist).

    Spread over the overlap itself, not over clip b: a 3-minute phone clip against a 70-minute
    tape otherwise got 1-2 samples when the tape happened to be `b` and 41 when it was `a`, so the
    refinement and the conflict check depended on which way round match_all stored the pair (i.e.
    on clip ids, which change when files move). Measured on real concert footage: 20-25 ms."""
    db = _dur(infos, m.b) or 1.0
    da = _dur(infos, m.a)
    lo, hi = 0.0, db
    if da > 0:
        inv = m.map.inverse()
        x0, x1 = sorted((float(inv(0.0)), float(inv(da))))
        if min(hi, x1) > max(lo, x0):
            lo, hi = max(lo, x0), min(hi, x1)
    return np.linspace(lo, hi, n)


def _edge_residual(m: PairMatch, maps, infos) -> np.ndarray:
    """Signed disagreement in event seconds between the edge and the group maps, over the overlap."""
    t = _edge_samples(m, infos)
    return maps[m.a](m.map(t)) - maps[m.b](t)


def _near_linear(m: TimeMap, dur: float, tol_s: float = 0.010) -> bool:
    """Within tol of a straight line over the clip (a steady clock seen through a wobbly tape
    map picks up a few ms of wobble but is still a steady clock)."""
    if m.is_linear or dur <= 0:
        return True
    t = np.linspace(0.0, dur, 200)
    y = m(t)
    fit = np.polyval(np.polyfit(t, y, 1), t)
    return bool(np.max(np.abs(y - fit)) < tol_s)


def _global_tails(m: TimeMap, reach: float = 1e6) -> TimeMap:
    """Extrapolate a piecewise map beyond its knots with its AVERAGE slope, not its end
    segment's. A tape map measured over one camera's overlap ends in local wobble; extending
    that wobble's slope over the rest of a 40-minute tape bends every clip placed there (seen on
    real concert footage: +2660 ppm on clips from the anchor's own camera)."""
    if len(m.knots_src) <= 2:
        return m
    r = m.ratio
    s0, s1 = m.knots_src[0], m.knots_src[-1]
    d0, d1 = m.knots_dst[0], m.knots_dst[-1]
    src = np.concatenate([[s0 - reach], m.knots_src, [s1 + reach]])
    dst = np.concatenate([[d0 - r * reach], m.knots_dst, [d1 + r * reach]])
    return TimeMap(src, dst)


def _clip_ratio(m: TimeMap, dur: float) -> float:
    """Average slope of a map over the clip itself."""
    if dur <= 0:
        return m.ratio
    return float((m(dur) - m(0.0)) / dur)


def _resolve_against(e: PairMatch, maps, infos) -> PairMatch:
    """Ambiguous edge -> copy whose map is the placement that best agrees with the group maps."""
    best, err = e.map, None
    for pm in [e.map] + list(e.alternatives or []):
        r = float(np.max(np.abs(_edge_residual(replace(e, map=pm), maps, infos))))
        if err is None or r < err:
            best, err = pm, r
    return replace(e, map=best)


def _pick_anchor(members, infos, edges=()):
    """The clip whose clock defines event time.

    1. a reference song (playback takes then live on song time);
    2. else the longest clip with a steady clock. A tape capture (most of its matches need a
       nonlinear map) wanders in speed: anchoring on it, or on a short phone clip placed through
       it, bends every other clip (seen on real tour footage). Time-of-day TC does not decide the
       anchor any more; anchor_tc is derived from any member that has it.
    """
    nonlin, total = {}, {}
    for e in edges:
        for c in (e.a, e.b):
            total[c] = total.get(c, 0) + 1
            nonlin[c] = nonlin.get(c, 0) + (1 if e.nonlinear else 0)

    def steady(c):
        return 0.3 if total.get(c) and nonlin.get(c, 0) / total[c] > 0.5 else 1.0

    return max(members, key=lambda c: (getattr(infos.get(c), "role", "clip") == "reference",
                                       _dur(infos, c) * steady(c), _real_tc(infos.get(c)), c))


def _frame_tol(infos, a, b) -> float:
    fps = [infos[c].fps for c in (a, b) if c in infos and infos[c].fps > 0]
    return max(0.020, 1.0 / min(fps)) if fps else 0.020


def edge_quality(m: PairMatch, extent_s: float = 0.0) -> float:
    """Tree preference: confident AND precise over the clip being placed.

    A map measured over a short overlap has an uncertain slope; placing a 44-minute camera
    through a 29-second phone clip stretches that uncertainty over the whole camera (measured on
    real tour footage: 69 ms). Expected error ~ residual * (1 + extent / overlap). A Hi8/tape
    match can be fully confident yet wobble by ms, so chaining through it is discouraged too."""
    res = max(m.residual_ms, 0.05)
    ov = m.overlap_s if m.overlap_s and m.overlap_s > 0 else max(extent_s, 1.0)
    # slope from a short overlap is uncertain whatever its residual says: assume >= 1 ms of
    # timing uncertainty across the overlap, extrapolated over the part of the clip beyond it
    beyond = max(0.0, extent_s - ov)
    err_ms = res + max(res, 1.0) * beyond / max(ov, 1.0)
    q = m.confidence / (1.0 + err_ms)
    return q * (0.7 if m.nonlinear else 1.0)


def _build_one(members, edges, infos, gid, force_anchor=None) -> EventGroup:
    members = sorted(members)
    anchor = force_anchor or _pick_anchor(members, infos, edges)
    conflicts: list[str] = []

    # Grow from the anchor (Prim): always take the best edge to an unplaced clip, unambiguous
    # edges first. An ambiguous edge (identical repeats) offers several exact placements; pick
    # the one the other already-placed clips agree with; on a tie keep the primary and record
    # the others in alt_shifts (the take genuinely fits at each repeat).
    maps: dict[str, TimeMap] = {anchor: TimeMap.identity()}
    alt_shifts: dict[str, list[float]] = {}
    tree, used = [], set()

    def placements(e):
        return [_global_tails(pm) for pm in [e.map] + list(e.alternatives or [])]

    def via(e, frm, pm):
        """Map of the other end of e, given placed `frm` and placement pm (t_a = pm(t_b))."""
        return _simplify(maps[frm].compose(pm if e.a == frm else pm.inverse()))

    def mid(c):
        return 0.5 * _dur(infos, c)

    while len(maps) < len(members):
        frontier = [e for e in edges if (e.a in maps) != (e.b in maps)]
        if not frontier:
            break
        # the user's sync points outrank everything, then unambiguous audio, then ambiguous
        def new_end(m):
            return m.b if m.a in maps else m.a
        e = min(frontier, key=lambda m: (m.kind not in ("hinted", "manual"), m.ambiguous,
                                         -edge_quality(m, _dur(infos, new_end(m))), m.a, m.b))
        frm = e.a if e.a in maps else e.b
        new = e.b if frm == e.a else e.a
        cands = [via(e, frm, pm) for pm in placements(e)]
        if len(cands) > 1:
            others = [f for f in edges if f is not e and new in (f.a, f.b)
                      and (f.a if f.b == new else f.b) in maps]
            tm = mid(new)
            scores = []
            for cm in cands:
                sc = 0.0
                for f in others:
                    q = f.a if f.b == new else f.b
                    if any(abs(float(via(f, q, pm)(tm)) - float(cm(tm))) < 0.05 for pm in placements(f)):
                        sc += edge_quality(f)
                scores.append(sc)
            best = int(np.argmax(scores))
            maps[new] = cands[best]
            ties = [k for k, v in enumerate(scores) if k != best and abs(v - scores[best]) < 1e-9]
            if ties:
                alt_shifts[new] = sorted(float(cands[k](tm) - cands[best](tm)) for k in ties)
            if best:
                e = replace(e, map=placements(e)[best])
        else:
            maps[new] = cands[0]
        tree.append(e)
        used.add(id(e))
    rest = [e for e in edges if id(e) not in used and not any(t.a == e.a and t.b == e.b for t in tree)]
    # an ambiguous non-tree edge is judged by whichever of its placements fits the tree best
    rest = [_resolve_against(e, maps, infos) if e.ambiguous else e for e in rest]

    # non-tree edges: wrong place (drop) or just inexact (keep for refinement)
    kept = list(tree)
    for e in rest:
        res = _edge_residual(e, maps, infos)
        worst = float(np.max(np.abs(res)))
        if worst > GROSS_DISAGREE_S:
            conflicts.append(
                f"{_name(infos, e.a)} and {_name(infos, e.b)}: a match (confidence {e.confidence:.2f}) "
                f"disagrees by {worst:.1f} s with where their other matches place them; "
                f"kept the better-supported placement and dropped this match")
        else:
            kept.append(e)

    # least-squares refinement of offsets (ratios kept) using all edges
    if len(kept) > len(tree):
        idx = {c: i for i, c in enumerate(c for c in members if c != anchor)}
        rows, rhs, w = [], [], []
        for e in kept:
            r = float(np.mean(_edge_residual(e, maps, infos)))   # E_a(M(t)) - E_b(t)
            row = np.zeros(len(idx))
            if e.a in idx:
                row[idx[e.a]] += 1.0
            if e.b in idx:
                row[idx[e.b]] -= 1.0
            rows.append(row)
            rhs.append(-r)
            w.append(np.sqrt(max(edge_quality(e), 1e-4)))
        A = np.array(rows) * np.array(w)[:, None]
        y = np.array(rhs) * np.array(w)
        delta, *_ = np.linalg.lstsq(A, y, rcond=None)
        before = float(np.sum(y ** 2))
        after = float(np.sum((A @ delta - y) ** 2))
        if after < before:
            for c, i in idx.items():
                maps[c] = _shift_map(maps[c], float(delta[i]))

    # Re-anchor on the majority clock: if the anchor runs at a speed far from most members'
    # (a drifting tape capture), rebuild the group from the longest clip that runs with the
    # majority. Then only the odd clip needs speed correction in the edit, and steady clips are
    # placed through their own links instead of through the tape's wander.
    if force_anchor is None and len(maps) >= 3:
        rat = {c: _clip_ratio(maps[c], _dur(infos, c)) for c in maps}
        med = float(np.median(list(rat.values())))
        if abs(rat[anchor] - med) > 500e-6:
            steady = [c for c in maps if abs(rat[c] - med) < 500e-6]
            if steady:
                new = max(steady, key=lambda c: (_dur(infos, c), c))
                return _build_one(members, edges, infos, gid, force_anchor=new)

    # Steady clocks are straight lines. Phones and cameras run on crystals: their only real deviation
    # from a line is a constant rate offset. Piecewise wiggle in their maps is measurement noise from
    # piecewise fits, often inherited through a tape they were placed by. Measured on real footage
    # (2026-11-23): Canon/phone maps with 25-73 ms of wiggle; the least-squares line fits the audio
    # better in every case (rms 9-14 ms vs 13-18 ms), and timelines no longer chase the noise.
    # Clocks far from the group's majority (tape captures, real wow) keep their measured shape; so
    # does a wiggle too large to be noise (likely a real dropout).
    if len(maps) >= 2:
        rat = {c: _clip_ratio(maps[c], _dur(infos, c)) for c in maps}
        med = float(np.median(list(rat.values())))
        for c in list(maps):
            d = _dur(infos, c)
            if c == anchor or maps[c].is_linear or d <= 0 or abs(rat[c] - med) > STEADY_PPM * 1e-6:
                continue
            t = np.linspace(0.0, d, 400)
            y = maps[c](t)
            k, b = np.polyfit(t, y, 1)
            if float(np.max(np.abs(y - (k * t + b)))) <= STEADY_MAX_WIGGLE_S:
                maps[c] = TimeMap.linear(float(k), float(b))

    # each map describes its own clip only (drops tape knots far outside it)
    maps = {c: _simplify(m.cropped(0.0, _dur(infos, c))) if _dur(infos, c) > 0 else m
            for c, m in maps.items()}

    # event time 0 = earliest covered moment
    spans = {c: (float(maps[c](0.0)), float(maps[c](_dur(infos, c)))) for c in members}
    t0 = min(s for s, _ in spans.values())
    maps = {c: _shift_map(m, -t0) for c, m in maps.items()}
    spans = {c: (s - t0, e - t0) for c, (s, e) in spans.items()}

    # residual disagreement report
    for e in kept:
        worst = float(np.max(np.abs(_edge_residual(e, maps, infos))))
        tol = _frame_tol(infos, e.a, e.b)
        if worst > tol:
            conflicts.append(
                f"{_name(infos, e.a)} and {_name(infos, e.b)}: sync disagrees by {worst * 1000:.0f} ms "
                f"between the direct match and the other matches (limit {tol * 1000:.0f} ms)")

    # same-device clips cannot overlap
    for a, b in combinations(members, 2):
        ia, ib = infos.get(a), infos.get(b)
        if ia is None or ib is None or not ia.device or ia.device != ib.device:
            continue
        ov = min(spans[a][1], spans[b][1]) - max(spans[a][0], spans[b][0])
        if ov > SAME_DEVICE_OVERLAP_S:
            conflicts.append(
                f"{_name(infos, a)} and {_name(infos, b)} come from the same device ({ia.device}) "
                f"but overlap by {ov:.1f} s in event time; one camera cannot record twice at once")

    anchor_tc = None
    ai = infos.get(anchor)
    if _real_tc(ai):
        anchor_tc = _shift_tc(ai.start_tc, ai.fps, -spans[anchor][0])
    else:
        # borrow time of day from the longest member that has real timecode and a linear map
        tcs = [c for c in members if _real_tc(infos.get(c)) and maps[c].is_linear]
        if tcs:
            c = max(tcs, key=lambda c: (_dur(infos, c), c))
            # expressed at the anchor's frame rate when it has one (audio-only anchors: the TC
            # source's rate), so consumers can read it with infos[anchor].fps or that rate
            fps = infos[anchor].fps if infos.get(anchor) and infos[anchor].fps > 0 else infos[c].fps
            secs = tc_to_seconds(infos[c].start_tc, infos[c].fps) - spans[c][0]
            anchor_tc = seconds_to_tc(secs % 86400.0, fps, drop=tc_is_drop(infos[c].start_tc, infos[c].fps))

    return EventGroup(id=gid, anchor=anchor, maps=maps, spans=spans, edges=kept,
                      conflicts=conflicts, anchor_tc=anchor_tc,
                      alt_shifts={c: v for c, v in alt_shifts.items() if c in maps})


def build_groups(matches: list[PairMatch], infos: dict[str, MediaInfo]) -> tuple[list[EventGroup], list[str]]:
    """Accepted matches -> (groups, loners). Group ids G01.. ordered by time, else by size."""
    best: dict[frozenset, PairMatch] = {}
    for m in matches:
        if m.a == m.b:
            continue
        k = frozenset((m.a, m.b))
        if k not in best or m.confidence > best[k].confidence:
            best[k] = m

    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in best.values():
        parent[find(m.a)] = find(m.b)
    comps: dict[str, list[str]] = {}
    for c in list(parent):
        comps.setdefault(find(c), []).append(c)

    built = []
    for members in comps.values():
        mset = set(members)
        edges = [m for m in best.values() if m.a in mset]
        built.append(_build_one(members, edges, infos, ""))

    def ctime(g):
        ts = [_epoch(infos[c].creation_time) for c in g.maps if c in infos]
        ts = [t for t in ts if t is not None]
        return min(ts) if ts else None

    def tcs(g):
        return tc_to_seconds(g.anchor_tc, infos[g.anchor].fps or 25.0) if g.anchor_tc else None

    by_size = lambda g: (-len(g.maps), -g.duration, g.anchor)
    for keyf in (ctime, tcs):
        keys = [keyf(g) for g in built]
        if built and all(k is not None for k in keys):
            built.sort(key=lambda g: (keyf(g),) + by_size(g))
            break
    else:
        built.sort(key=by_size)
    for i, g in enumerate(built, 1):
        g.id = f"G{i:02d}"

    grouped = {c for g in built for c in g.maps}
    loners = [c for c in infos if c not in grouped]
    return built, loners


DRIFT_MIN_PPM = 1000.0      # the member's clock must be clearly off (tape captures: ~2200 ppm)
DRIFT_AGREE_PPM = 150.0     # ... and the weak match must restore the group's majority clock this closely
DRIFT_MIN_CONF = 0.2
DRIFT_MAX_RES_MS = 8.0
DRIFT_MIN_OVERLAP_S = 120.0


def _drift_corroborated(x, g, cands, infos):
    """One weak match can stand on its own when it reproduces a known clock error: x is placed through
    member y whose clock runs far off the group's majority (a tape capture, +-2000 ppm), and the
    match's slope turns that back into exactly the majority clock for x. A wrong match lands on an
    arbitrary slope in +-MAX_DRIFT; hitting y's measured error within DRIFT_AGREE_PPM is ~2% chance,
    on top of passing the coarse+fine stages with a small residual over a long overlap.
    Real case (drum-side phone vs tape, conf 0.27): implied phone clock 1.000003."""
    rat = {c: _clip_ratio(g.maps[c], _dur(infos, c)) for c in g.maps}
    majority = float(np.median(list(rat.values())))
    for _ev, y, m in cands:
        if (m.confidence < DRIFT_MIN_CONF or m.residual_ms > DRIFT_MAX_RES_MS or m.ambiguous
                or (m.overlap_s or 0.0) < DRIFT_MIN_OVERLAP_S):
            continue
        if abs(rat[y] - majority) * 1e6 < DRIFT_MIN_PPM:
            continue
        to_y = m.map if m.a == y else m.map.inverse()          # x clip time -> y clip time
        dx = max(_dur(infos, x), 1.0)
        implied = float((g.maps[y](to_y(dx)) - g.maps[y](to_y(0.0))) / dx)   # x clip -> event slope
        if abs(implied - majority) * 1e6 <= DRIFT_AGREE_PPM:
            return [(_ev, y, m)]
    return None


def corroborate(matches: list[PairMatch], weak: list[PairMatch], infos) -> tuple[list[PairMatch], list[str]]:
    """Accept weak matches of an ungrouped clip when two or more of them, to different members
    of one group, place it at the same moment (within CORROBORATE_S). Real short clips (tape vs
    phone) often score below MATCH_ACCEPT on their own while being exactly right; independent
    agreement is strong evidence a wrong placement would almost never produce."""
    notes: list[str] = []
    accepted = list(matches)
    for _round in range(3):
        groups, loners = build_groups(accepted, infos)
        added = False
        for x in loners:
            for g in groups:
                cands = []          # (event time of x's clip-time 0, edge)
                for m in weak:
                    if x not in (m.a, m.b):
                        continue
                    y = m.b if m.a == x else m.a
                    if y not in g.maps:
                        continue
                    # x clip time -> y clip time -> event time
                    to_y = m.map if m.a == y else m.map.inverse()
                    t_mid = 0.5 * (infos[x].duration if x in infos else 0.0)
                    cands.append((float(g.maps[y](to_y(t_mid))), y, m))
                best = None
                for ev, y, m in cands:
                    agree = [(e2, y2, m2) for e2, y2, m2 in cands if abs(e2 - ev) < CORROBORATE_S]
                    if len({yy for _, yy, _ in agree}) >= 2 and (best is None or len(agree) > len(best)):
                        best = agree
                if not best:
                    best = _drift_corroborated(x, g, cands, infos)
                    if best:
                        m0 = best[0][2]
                        notes.append(f"{_name(infos, x)}: its weak match to {_name(infos, best[0][1])} reproduces "
                                     f"that clip's measured speed error exactly, so it was synced")
                        accepted.append(replace(m0, kind="corroborated", confidence=max(m0.confidence, MATCH_ACCEPT),
                                                note=(m0.note + "; " if m0.note else "") + "corroborated by clock drift"))
                        weak = [m for m in weak if m is not m0]
                        added = True
                        break
                if best:
                    for _e, _y, m in best:
                        accepted.append(replace(m, kind="corroborated", confidence=max(m.confidence, MATCH_ACCEPT),
                                                note=(m.note + "; " if m.note else "") +
                                                f"corroborated by {len(best)} independent matches"))
                    notes.append(f"{_name(infos, x)}: weak matches to {len(best)} clips agree within "
                                 f"{CORROBORATE_S * 1000:.0f} ms, so it was synced")
                    weak = [m for m in weak if all(m is not b[2] for b in best)]
                    added = True
                    break
        if not added:
            break
    return accepted, notes
