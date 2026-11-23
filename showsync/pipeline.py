"""End-to-end analysis: files in, AnalysisResult out. See docs/ARCHITECTURE.md "Pipeline"."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from . import config
from .models import AnalysisResult, Features, MediaInfo, SongSegment

MEDIA_EXTS = {
    ".mov", ".mp4", ".m4v", ".mxf", ".avi", ".mkv", ".mts", ".m2ts", ".mpg", ".mpeg", ".dv",
    ".3gp", ".webm", ".braw", ".r3d", ".crm",
    ".wav", ".aif", ".aiff", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".bwf",
}

Progress = Callable[[str, int, int, str], None]


def expand_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if Path(f).suffix.lower() in MEDIA_EXTS:
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# --------------------------------------------------------------------------- features with cache

def _features_for(path: str, cid: str) -> tuple[Features, np.ndarray]:
    from .media.decode import decode_audio
    from .media.cache import cache_key, load_npz, save_npz
    from .dsp.features import compute_features

    audio = decode_audio(path, config.ANALYSIS_SR, mmap=True)
    key = cache_key(path, "features", config.ANALYSIS_VERSION, config.ANALYSIS_SR)
    d = load_npz(key)
    if d is not None:
        f = Features(cid, float(d["duration"]), d["onset"], d["energy_db"], d["chroma"], d["musicness"],
                     d["perc"] if "perc" in d and d["perc"].size else None)
    else:
        f = compute_features(cid, np.asarray(audio), config.ANALYSIS_SR)
        save_npz(key, duration=np.float64(f.duration), onset=f.onset, energy_db=f.energy_db,
                 chroma=f.chroma, musicness=f.musicness,
                 perc=f.perc if f.perc is not None else np.zeros(0, np.float32))
    return f, audio


# --------------------------------------------------------------------------- main entry

def analyze(paths: Iterable[str], refs: Optional[Iterable[str]] = None,
            progress: Optional[Progress] = None, workers: int = 0,
            hints_path: Optional[str] = None) -> AnalysisResult:
    from .media.probe import probe, clip_id
    from .graph.groups import match_all, build_groups, corroborate
    from .music import songs as songs_mod

    from . import hints as H

    say = progress or (lambda *a: None)
    hints = H.load(hints_path)
    skip = H.ignored(hints)
    files = [f for f in expand_paths(paths) if os.path.normcase(os.path.abspath(f)) not in skip]
    warnings: list[str] = []
    workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))

    # 1. probe
    infos: dict[str, MediaInfo] = {}
    for i, p in enumerate(files):
        say("probe", i, len(files), os.path.basename(p))
        try:
            mi = probe(p)
        except Exception as e:  # unreadable file: report, continue
            warnings.append(f"Could not read {p}: {e}")
            continue
        infos[mi.clip_id] = mi
        if mi.is_vfr:
            warnings.append(f"{os.path.basename(p)} has variable frame rate ({mi.vfr_detail}). "
                            f"Audio sync is still valid, but conform it to constant frame rate "
                            f"(showsync conform) before editing or picture will drift.")
        if not mi.has_audio:
            warnings.append(f"{os.path.basename(p)} has no audio: it can only be synced by "
                            f"same-named # markers (e.g. #kick) shared with another clip.")
    say("probe", len(files), len(files), "")

    # 2. decode + features
    feats: dict[str, Features] = {}
    audio: dict[str, np.ndarray] = {}
    todo = [c for c, m in infos.items() if m.has_audio]

    def work(cid):
        return cid, _features_for(infos[cid].path, cid)

    with ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
        for n, fut in enumerate([ex.submit(work, c) for c in todo]):
            try:
                cid, (f, a) = fut.result()
                feats[cid], audio[cid] = f, a
            except Exception as e:
                warnings.append(f"Audio analysis failed: {e}")
            say("features", n + 1, len(todo), "")

    # references: they also take part in exact sync (a playback take hears the very same
    # recording), so a reference can anchor a group and give sample-accurate song time
    ref_feats: dict[str, Features] = {}
    ref_ids: dict[str, str] = {}            # clip id -> reference path
    ref_files = expand_paths(refs or [])
    for i, rp in enumerate(ref_files):
        say("references", i, len(ref_files), os.path.basename(rp))
        try:
            mi = probe(rp)
            mi.role = "reference"
            f, a = _features_for(rp, mi.clip_id)
        except Exception as e:
            warnings.append(f"Reference {rp} failed: {e}")
            continue
        ref_feats[rp] = f
        if mi.clip_id not in infos:         # same file given as clip and reference: keep it a clip
            infos[mi.clip_id], feats[mi.clip_id], audio[mi.clip_id] = mi, f, a
            ref_ids[mi.clip_id] = rp

    # 3. exact sync
    weak: list = []
    matches = match_all(feats, audio, infos,
                        progress=lambda d, t, m="": say("sync", d, t, m), weak_out=weak)
    idof = H.id_lookup(infos)
    names = lambda c: os.path.basename(infos[c].path) if c in infos else c
    matches = H.drop_different_events(matches, hints, idof)
    weak = H.drop_different_events(weak, hints, idof)
    musical: list = []
    matches = H.apply_sync_points(matches, hints, idof, feats, audio, warnings, names, musical)
    matches, notes = corroborate(matches, weak, infos)
    warnings.extend(notes)
    groups, loners = build_groups(matches, infos)
    loners = [c for c in loners if c not in ref_ids]
    for g in groups:
        warnings.extend(f"{g.id}: {c}" for c in g.conflicts)

    # 4. songs
    say("songs", 0, 1, "finding songs")
    segments: list[SongSegment] = []
    seg_feats: dict[str, Features] = {}
    owners: list[tuple[str, str, Features]] = []
    for g in groups:
        heard = {c: m for c, m in g.maps.items() if c in feats}   # clips without audio add nothing
        gview = replace(g, maps=heard, spans={c: g.spans[c] for c in heard})
        owners.append((g.id, "group", songs_mod.group_features(gview, feats)))
    for cid in loners:
        if cid in feats:
            owners.append((cid, "clip", feats[cid]))
    for owner, kind, f in owners:
        segs = songs_mod.find_segments(owner, kind, f, references=ref_feats or None)
        for s in segs:
            seg_feats[s.id] = songs_mod.crop_features(f, s.start, s.end)
        segments.extend(segs)
    tags, must, cannot = H.song_constraints(hints, idof, groups, infos, segments, seg_feats,
                                            {o: f for o, _k, f in owners}, songs_mod.crop_features)
    anchors = H.musical_anchors(musical, groups, infos, segments)
    songs = H.call_assign_songs(songs_mod, segments, seg_feats, ref_feats or None,
                                tags, must, cannot, warnings, anchors)
    _exact_song_maps(groups, segments, songs, ref_ids)
    warnings.extend(_continuous_takes(segments, songs))
    songs = _drop_lonely_nonmusic(songs, segments, seg_feats)
    say("songs", 1, 1, f"{len(songs)} songs")

    for cid in ref_ids:
        if not any(cid in g.maps for g in groups):
            pass  # a reference that heard nothing exactly is normal (live takes); no warning
    for cid, mi in infos.items():
        if mi.has_audio and cid in loners:
            in_song = any(s.owner == cid and s.song_id for s in segments)
            if not in_song:
                warnings.append(f"{os.path.basename(mi.path)} matched no other clip and no song.")

    return AnalysisResult(
        version=config.ANALYSIS_VERSION, clips=infos, groups=groups, loners=loners,
        segments=segments, songs=songs, matches=matches, warnings=warnings,
    )


def _exact_song_maps(groups, segments, songs, ref_ids):
    """Where a group contains the reference recording of the song a segment was assigned to,
    replace the musical (DTW) song map with the exact one: song time = reference clip time."""
    song_ref = {so.id: so.reference[5:] for so in songs if so.reference.startswith("file:")}
    by_path = {os.path.normcase(p): cid for cid, p in ref_ids.items()}
    for seg in segments:
        if seg.owner_kind != "group" or seg.song_id not in song_ref:
            continue
        rcid = by_path.get(os.path.normcase(song_ref[seg.song_id]))
        g = next((g for g in groups if g.id == seg.owner), None)
        if g is None or rcid not in g.maps:
            continue
        seg.song_map = g.maps[rcid].inverse()      # event time -> reference (song) time
        seg.match_score = max(seg.match_score, 0.99)


TAKE_GAP_MAX_S = 60.0      # pieces of one recording this close belong to one continuous performance
TAKE_JUMP_TOL_S = 2.0      # song-time jump between them tolerated before re-placing the later piece


def _continuous_takes(segments, songs) -> list:
    """One recording plays one song front to back: two pieces of the same song from the same owner,
    separated by a short gap, must continue each other in song time at real-time pace. The musical
    (DTW) map of a later piece can land on an earlier repeat of the same section (a chorus), which
    would put one camera at two places at once. Real case (2026-11-23): a tape's second Auto piece,
    32 s after the first, was mapped 49 s BACK into the first piece's song time; continuing at real
    time put it exactly where the other nights' takes end. Such a piece is shifted to continue the
    previous one, and the gap is filled at real-time pace. Returns notes for the warnings list."""
    from .models import TimeMap
    import numpy as np
    notes = []
    by_id = {s.id: s for s in segments}
    for so in songs:
        own = {}
        for sid in so.segments:
            sg = by_id.get(sid)
            if sg is not None and sg.song_map is not None:
                own.setdefault(sg.owner, []).append(sg)
        for owner, segs in own.items():
            segs.sort(key=lambda s: s.start)
            for a, b in zip(segs, segs[1:]):
                gap = b.start - a.end
                if gap < 0 or gap > TAKE_GAP_MAX_S:
                    continue
                expect = float(a.song_map(a.end)) + gap
                got = float(b.song_map(b.start))
                if abs(got - expect) > TAKE_JUMP_TOL_S:
                    m = b.song_map
                    b.song_map = TimeMap(np.asarray(m.knots_src, float), np.asarray(m.knots_dst, float) + (expect - got))
                    b.beat_synced = False
                    notes.append(f"{so.name}: a later piece of the same recording ({owner}) was mapped {got - expect:+.1f} s "
                                 f"away from where it continues the earlier piece; placed it in sequence instead")
                if gap > 0:                       # fill the gap at real-time pace, as one continuous take
                    src = np.append(np.asarray(a.song_map.knots_src, float), b.start)
                    dst = np.append(np.asarray(a.song_map.knots_dst, float), expect)
                    a.song_map = TimeMap(src, dst)
                    a.end = b.start
    return notes


LONELY_MIN_MUSICNESS = 0.3


def _drop_lonely_nonmusic(songs, segments, seg_feats):
    """A song made of one segment with weak music evidence (applause, noise, a crowd-drowned
    phone that matched nothing) is not a song: unassign it rather than invent "Song 07"."""
    by_id = {s.id: s for s in segments}
    keep = []
    for so in songs:
        if len(so.segments) == 1 and not so.reference.startswith("file:"):
            f = seg_feats.get(so.segments[0])
            mus = float(np.mean(f.musicness)) if f is not None and f.musicness.size else 0.0
            if mus < LONELY_MIN_MUSICNESS:
                seg = by_id.get(so.segments[0])
                if seg is not None:
                    seg.song_id, seg.song_map, seg.match_score = None, None, 0.0
                continue
        keep.append(so)
    import re
    k = 0
    for so in keep:                       # renumber "Song NN" after removals
        if re.fullmatch(r"Song \d+", so.name):
            k += 1
            so.name = f"Song {k:02d}"
    return keep
