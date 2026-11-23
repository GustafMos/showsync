"""Apply user hints (models.Hints) inside the pipeline. See docs/ARCHITECTURE.md "User hints"."""
from __future__ import annotations

import inspect
import os
from dataclasses import replace
from typing import Optional

import numpy as np

from .models import EventGroup, Features, Hints, MediaInfo, PairMatch, SongSegment, TimeMap

PICK_TOL_S = 0.25          # a placement within this of the user's point is "the one they meant"
MANUAL_RESIDUAL_MS = 20.0  # markers are frame-accurate, not sample-accurate


def load(path: Optional[str]) -> Hints:
    if not path or not os.path.isfile(path):
        return Hints()
    with open(path, encoding="utf-8") as fh:
        return Hints.from_json(fh.read())


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def id_lookup(infos: dict[str, MediaInfo]):
    by_path = {_norm(m.path): c for c, m in infos.items()}
    return lambda p: by_path.get(_norm(p))


def ignored(hints: Hints) -> set[str]:
    return {_norm(p) for p in hints.ignore}


# --------------------------------------------------------------------------- exact sync

def drop_different_events(matches: list[PairMatch], hints: Hints, idof) -> list[PairMatch]:
    bad = set()
    for pair in hints.different_event:
        ids = [idof(p) for p in pair]
        ids = [i for i in ids if i]
        for i in ids:
            for j in ids:
                if i != j:
                    bad.add((i, j))
    return [m for m in matches if (m.a, m.b) not in bad]


def _points_by_pair(hints: Hints, idof) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """(a, b) -> [(t_a, t_b), ...], keyed in a canonical order (a < b)."""
    out: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for sp in hints.sync_points:
        a, b = idof(sp.a), idof(sp.b)
        if not a or not b or a == b:
            continue
        if a < b:
            out.setdefault((a, b), []).append((float(sp.ta), float(sp.tb)))
        else:
            out.setdefault((b, a), []).append((float(sp.tb), float(sp.ta)))
    return out


def _manual_map(points: list[tuple[float, float]]) -> TimeMap:
    """t_a = map(t_b) from user points: ratio 1 unless 2+ points far enough apart to fit drift."""
    pts = np.array(points, dtype=np.float64)
    ta, tb = pts[:, 0], pts[:, 1]
    if len(pts) >= 2 and np.ptp(tb) > 30.0:
        r, o = np.polyfit(tb, ta, 1)
        if abs(r - 1.0) < 0.003:
            return TimeMap.linear(float(r), float(o))
    return TimeMap.linear(1.0, float(np.median(ta - tb)))


def apply_sync_points(matches: list[PairMatch], hints: Hints, idof,
                      feats: dict[str, Features], audio: dict[str, np.ndarray],
                      warnings: list[str], names, musical: Optional[list] = None) -> list[PairMatch]:
    """Exact/manual edges from the user's sync points. Points between two clips that both have
    audio but do NOT share sound at the marked moment (different nights) are not exact sync:
    they are appended to `musical` as (clip_a, t_a, clip_b, t_b) musical anchors for songs."""
    pts = _points_by_pair(hints, idof)
    if not pts:
        return matches
    by_pair = {}
    for m in matches:
        by_pair[(m.a, m.b) if m.a < m.b else (m.b, m.a)] = m
    out = [m for k, m in by_pair.items() if k not in pts]

    from .dsp import align
    has_hint_arg = "hint" in inspect.signature(align.match_pair).parameters
    # pairs whose points are all typed/verified exact positions ("manual"): same event even when the
    # audio matcher cannot confirm it (a drum-side camera vs a room tape), never a cross-night anchor
    exact_pairs = set()
    for k in pts:
        srcs = {sp.source for sp in hints.sync_points if {idof(sp.a), idof(sp.b)} == set(k)}
        if srcs == {"manual"}:
            exact_pairs.add(k)

    for (a, b), plist in pts.items():
        m = by_pair.get((a, b))
        chosen = None
        if m is not None:
            # express the user's points in this match's orientation: t_m.a = map(t_m.b)
            p_ab = plist if m.a == a else [(tb, ta) for ta, tb in plist]
            for pm in [m.map] + list(m.alternatives or []):
                err = max(abs(float(pm(tb)) - ta) for ta, tb in p_ab)
                if err < PICK_TOL_S and (chosen is None or err < chosen[1]):
                    chosen = (pm, err)
            if chosen is not None:
                out.append(replace(m, map=chosen[0], ambiguous=False, alternatives=[],
                                   confidence=1.0, kind="hinted",
                                   note=(m.note + "; " if m.note else "") + "placement chosen by your sync point"))
                continue
        # no audio match agrees: refine at the hinted spot if both have audio, else manual edge
        refined = None
        if has_hint_arg and a in feats and b in feats:
            try:
                refined = align.match_pair(feats[a], feats[b], audio[a], audio[b], hint=plist)
            except Exception:
                refined = None
        if refined is not None:
            out.append(replace(refined, confidence=1.0, kind="hinted", ambiguous=False, alternatives=[]))
        elif a in feats and b in feats and m is None and (a, b) not in exact_pairs:
            # both heard something, but not the same sound: same music on different nights
            if musical is not None:
                musical.extend((a, ta, b, tb) for ta, tb in plist)
        else:
            out.append(PairMatch(a=a, b=b, map=_manual_map(plist), confidence=1.0, overlap_s=0.0,
                                 residual_ms=MANUAL_RESIDUAL_MS, n_windows=len(plist),
                                 n_inliers=len(plist), kind="manual",
                                 note="synced by your markers (frame accurate)"))
            if m is not None:
                warnings.append(f"{names(a)} and {names(b)}: your sync point disagrees with the "
                                f"audio match; your sync point was used")
    return out


# --------------------------------------------------------------------------- songs

def _clip_event_range(cid: str, start, end, groups: list[EventGroup], infos):
    """(owner, owner_kind, lo, hi) of a clip time range, in the owner's time base."""
    dur = infos[cid].duration if cid in infos else 0.0
    s = 0.0 if start is None else float(start)
    e = dur if end is None else float(end)
    for g in groups:
        if cid in g.maps:
            return g.id, "group", float(g.maps[cid](s)), float(g.maps[cid](e))
    return cid, "clip", s, e


def _overlapping(segments, owner, lo, hi, min_s=3.0):
    return [sg for sg in segments if sg.owner == owner and min(sg.end, hi) - max(sg.start, lo) >= min_s]


def song_constraints(hints: Hints, idof, groups, infos, segments: list[SongSegment],
                     seg_feats: dict[str, Features], owner_feats: dict[str, Features], crop):
    """Clip-level hints -> (tags {seg_id: title}, must_link, cannot_link). A tagged range with no
    segment (no music detected, e.g. a crowd-drowned phone) gets a forced partial segment."""
    tags: dict[str, str] = {}
    for t in hints.song_tags:
        cid = idof(t.clip)
        if not cid:
            continue
        owner, kind, lo, hi = _clip_event_range(cid, t.start, t.end, groups, infos)
        hits = _overlapping(segments, owner, lo, hi)
        if not hits and owner in owner_feats and hi - lo > 1.0:
            n = sum(1 for sg in segments if sg.owner == owner)
            sg = SongSegment(id=f"{owner}:t{n:02d}", owner=owner, owner_kind=kind,
                             start=max(lo, 0.0), end=hi, is_partial=True)
            segments.append(sg)
            seg_feats[sg.id] = crop(owner_feats[owner], sg.start, sg.end)
            hits = [sg]
        for sg in hits:
            tags[sg.id] = t.song.strip()

    def main_seg(path):
        cid = idof(path)
        if not cid:
            return None
        owner, _k, lo, hi = _clip_event_range(cid, None, None, groups, infos)
        hits = _overlapping(segments, owner, lo, hi, 0.5)
        return max(hits, key=lambda sg: min(sg.end, hi) - max(sg.start, lo)).id if hits else None

    def links(pairs):
        out = []
        for grp in pairs:
            ids = [main_seg(p) for p in grp]
            ids = [i for i in ids if i]
            out += [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))
                    if ids[i] != ids[j]]
        return out

    return tags, links(hints.same_song), links(hints.different_song)


def musical_anchors(musical, groups, infos, segments):
    """(clip_a, t_a, clip_b, t_b) -> (seg_a, owner_time_a, seg_b, owner_time_b): the user says
    these two moments are the same musical moment (e.g. the first hit of chorus 2 on two nights)."""
    out = []
    for a, ta, b, tb in musical:
        ends = []
        for c, t in ((a, ta), (b, tb)):
            owner, _k, lo, _hi = _clip_event_range(c, t, t, groups, infos)
            seg = next((sg for sg in segments if sg.owner == owner and sg.start <= lo <= sg.end), None)
            ends.append((seg.id if seg else None, lo))
        if ends[0][0] and ends[1][0] and ends[0][0] != ends[1][0]:
            out.append((ends[0][0], ends[0][1], ends[1][0], ends[1][1]))
    return out


def call_assign_songs(songs_mod, segments, seg_feats, refs, tags, must, cannot, warnings,
                      anchors=None):
    params = inspect.signature(songs_mod.assign_songs).parameters
    kw = {}
    if anchors:
        if "anchors" in params:
            kw["anchors"] = anchors
        else:
            warnings.append("Cross-show sync points were given but this engine version cannot apply them yet.")
    conflicts: list[str] = []
    if tags or must or cannot:
        if "tags" in params:
            kw.update(tags=tags or None, must_link=must or None, cannot_link=cannot or None)
            if "conflicts" in params:
                kw["conflicts"] = conflicts
        else:
            warnings.append("Song hints were given but this engine version cannot apply them yet.")
    report: dict = {}
    if "beat_report" in params:
        kw["beat_report"] = report
    out = songs_mod.assign_songs(segments, seg_feats, refs, **kw)
    warnings.extend("Song hint ignored: " + c for c in conflicts)
    for sg in segments:                      # expose beat lock per segment (contract fields)
        r = report.get(sg.id)
        if r:
            sg.beat_report = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                              for k, v in r.items()}
            sg.beat_synced = bool(r.get("beat_synced"))
    return out
