"""ShowSync Resolve plugin: executes plans from showsync_core against Resolve.

`apply_plan` / `undo_backup` only talk to a *layer* object (duck-typed, see LayerBase), so tests
use a fake. `ResolveLayer` is the real implementation over the Resolve scripting API.

API facts verified read-only against Resolve Studio 21.2 (see README): timeline item start frames are
ABSOLUTE (start frame 86400 for a 01:00:00:00 / 24 fps timeline), so record frames are offset by
Timeline.GetStartFrame(). Verified live (Resolve 21.2, Nov 2026): AppendToTimeline startFrame/endFrame are
clip frames at the clip's own fps relative to its first frame, endFrame is exclusive, recordFrame is
absolute; media-pool marker frames are clip-relative; TimelineItem.SetSpeed keeps full float precision and
the item's timeline duration fixed (the consumed source range changes). Still unverified: GetSourceEndFrame
being inclusive and audio-item source frames.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Optional

from showsync_core import (Backup, ApplyPlan, ORIG_TC_PREFIX, TAG, norm_path)

END_FRAME_INCLUSIVE = False   # verified live in Resolve 21.2 (Nov 2026): endFrame is exclusive
METADATA_KEY = "Comments"
AUDIO_TRACK_SUBTYPE = "stereo"
MARKER_FRAMES_ABSOLUTE = False   # verified live: media-pool marker frames are clip-relative


class LayerBase:
    """Interface the executor needs. ResolveLayer implements it; tests provide a fake."""
    def timeline_start_frame(self, tl) -> int: ...
    def clip_for_path(self, path: str): ...                       # handle or None
    def timeline_names(self) -> list: ...
    def create_timeline(self, name: str, start_tc: str, fps: float): ...   # handle or None
    def set_track_count(self, tl, n_video: int, n_audio: int) -> None: ...
    def audio_channels(self, clip) -> int: return 2                 # clip's audio channel count
    def set_track_name(self, tl, ttype: str, index: int, name: str) -> None: ...
    def append_item(self, tl, info: dict): ...                     # AppendToTimeline for one clip info -> item or None
    def set_speed(self, item, pct: float) -> bool: ...             # SetSpeed(Percentage, PitchCorrection False)
    def current_timeline(self): ...                                 # the user's current timeline (or None)
    def set_current_timeline(self, tl) -> bool: ...                 # track-level calls need the timeline current
    def set_track_enable(self, tl, ttype: str, index: int, enabled: bool) -> bool: ...
    def is_track_enabled(self, tl, ttype: str, index: int) -> bool: ...
    def delete_item(self, tl, item) -> bool: ...                   # Timeline.DeleteClips([item])
    def item_extent(self, item, media_type: int): ...              # (duration_frames, src_start, src_end_exclusive|None)
    def add_timeline_marker(self, tl, frame, color, name, note, duration, custom) -> bool: ...
    def delete_timeline(self, name: str) -> bool: ...
    def get_start_tc(self, clip) -> str: ...
    def set_start_tc(self, clip, tc: str) -> bool: ...
    def get_metadata(self, clip, key: str) -> str: ...
    def set_metadata(self, clip, key: str, value: str) -> bool: ...
    def add_clip_marker(self, clip, frame, color, name, note, duration, custom) -> bool: ...
    def delete_clip_marker(self, clip, frame: int, custom: str) -> bool: ...


def unique_name(name: str, existing) -> str:
    ex = set(existing)
    if name not in ex:
        return name
    i = 2
    while "%s (%d)" % (name, i) in ex:
        i += 1
    return "%s (%d)" % (name, i)


MAX_TRIES = 4
SOURCE_JOIN_TOL = 2          # source frames: a re-derived start this close to the consumed end is replaced by it


def slot_of(p, tl_fps: float, base: int):
    """Absolute timeline slot [start, end) of a placement."""
    start = base + p.record_frame
    if p.record_end > p.record_frame:
        return start, base + p.record_end
    cfps = p.clip_fps or tl_fps
    return start, start + max(1, int(round((p.end_frame - p.start_frame) / cfps * 100.0 / p.speed_pct * tl_fps)))


def append_piece(layer, tl, clip, p, tl_fps: float, base: int, log=print, occ=None, followed: bool = True,
                 state: Optional[dict] = None) -> str:
    """Fill the placement's timeline slot [rec, rec_end) exactly, never writing past slot_end.

    Verified live: SetSpeed keeps the item's timeline duration fixed and changes the consumed source
    range. So the main item appends source of length slot_len (in clip frames at 100 %), is sped up,
    and its read-back end is compared with the slot end; if short (the clip ran out of source before
    SetSpeed, or rounding) ONE remainder item continues from the consumed source end with a timeline
    duration of exactly slot_end - end. An item whose duration comes out longer than allowed is
    deleted and re-appended one source frame shorter. `occ` collects (start, end) of created items.
    A remainder of <= 1 timeline frame is not created (the slot ends 1 frame short = "tail short") unless
    another slot of the same clip follows (`followed`), where that frame keeps the clip continuous.
    `state` (per clip and track, shared by the consecutive pieces) carries the consumed source end and slot
    end of the previous piece: when the next slot starts where the previous one ended and its planned
    source start is within SOURCE_JOIN_TOL frames of that consumed end, the piece starts exactly at the
    consumed end, so source steps at slot joins are monotonic (no re-read frames, no tiny backwards steps).
    Returns an error text or ''."""
    cfps = p.clip_fps or tl_fps
    tfps = float(tl_fps)
    nclip = p.clip_frames or p.end_frame
    slot0, slot1 = slot_of(p, tl_fps, base)
    plain = abs(p.speed_pct - 100.0) < 1e-9
    mt = p.media_type
    sf0 = p.start_frame
    if state and state.get("slot_end") is not None and state["slot_end"] >= slot0 - 1 and \
            0 < state["src_end"] - sf0 <= SOURCE_JOIN_TOL:
        # only a BACKWARDS step is corrected; a forward gap keeps the planned start (sync never drifts: piece
        # speeds are derived from their slots, so consumed end and planned start normally coincide)
        sf0 = state["src_end"]

    def info(sf, n, rec):
        return {"mediaPoolItem": clip, "startFrame": sf, "endFrame": sf + n - 1 if END_FRAME_INCLUSIVE else sf + n,
                "mediaType": mt, "trackIndex": p.track, "recordFrame": rec}

    def put(sf, rec, allowed):
        """Append so that the item lasts exactly `allowed` timeline frames (or fewer if the source ends).
        Returns (item, duration, consumed_source_frames) or None / an error string.
        (Verified live 2026-11-23: GetSourceStartFrame can report a frame below the one appended
        (asked 1303, reports 1302) while GetSourceStartTime shows the item starts exactly where asked.
        So the reported frame is never used to 'correct' a placement.)"""
        return place(sf, rec, allowed)

    def place(sf, rec, allowed, keep_occ=True):
        n = min(max(1, int(round(allowed / tfps * cfps))), nclip - sf)
        item, tries, last = None, 0, None
        while n >= 1 and tries < 8:
            tries += 1
            item = layer.append_item(tl, info(sf, n, rec))
            if item is None:
                return None
            dur, _s0, _s1 = layer.item_extent(item, mt)
            if dur <= allowed:
                break
            layer.delete_item(tl, item)          # longer than the slot: shorten the source and retry
            last = (n, dur, allowed)
            # jump by the measured excess (one frame at a time took hundreds of appends and timed
            # out when Resolve counts a clip's frames differently, e.g. AVCHD audio)
            excess = int(math.ceil((dur - allowed) * n / max(dur, 1)))
            item, n = None, n - (excess if excess > 2 else 1)   # exact single steps near the fit (no stubs)
        if item is None:
            return "append never fit its slot (source frames %s -> %s timeline frames, slot %s)" % last if last else None
        if not plain and not layer.set_speed(item, p.speed_pct):
            return "SetSpeed(%.4f) failed" % p.speed_pct
        dur, s0, s1 = layer.item_extent(item, mt)
        if plain:
            consumed = n
        elif s0 is None or s1 is None:            # audio items: derive from the duration
            consumed = int(round(dur / tfps * p.speed_pct / 100.0 * cfps))
        else:
            consumed = s1 - s0
        if not keep_occ:
            return item, dur, max(0, consumed), s0
        if occ is not None:
            occ.append((rec, rec + dur))
        return item, dur, max(0, consumed)

    r = put(sf0, slot0, slot1 - slot0)
    if r is None:
        return "append failed at source frame %d (slot %d-%d)" % (sf0, slot0, slot1)
    if isinstance(r, str):
        return r
    _item, dur, consumed = r
    end, cur_sf = slot0 + dur, sf0 + consumed
    if end < slot1 and cur_sf < nclip and (slot1 - end > 1 or followed):
        r2 = put(cur_sf, end, slot1 - end)         # ONE remainder item, exactly the missing frames
        if isinstance(r2, str):
            return r2
        if r2 is None:
            return "remainder append failed at source frame %d (slot end %d)" % (cur_sf, slot1)
        cur_sf += r2[2]
        end += r2[1]
    if state is not None:
        state["src_end"], state["slot_end"] = cur_sf, max(end, slot0 + dur)
    return ""


def occupancy_summary(slots, items) -> dict:
    """slots: [(start, end)], items: [(start, end)] for one clip/track. max_gap counts the uncovered
    frames inside every slot that is followed by an adjacent slot; a slot followed by a gap (structure
    cut) or the last slot may end short (tail_short: the source ran out, a frame of rounding). overlaps
    counts items that start before the previous one ended."""
    items = sorted(items)
    slots = sorted(slots)
    gap = overlap = tail_short = 0
    for (a0, a1), (b0, b1) in zip(items, items[1:]):
        if b0 < a1:
            overlap += 1
    for k, (s0, s1) in enumerate(slots):
        inside = sum(max(0, min(e, s1) - max(b, s0)) for b, e in items)
        short = max(0, (s1 - s0) - inside)
        if k + 1 < len(slots) and slots[k + 1][0] <= s1:
            gap = max(gap, short)
        else:
            tail_short = max(tail_short, short)
    return {"slots": len(slots), "items": len(items), "max_gap": gap, "overlaps": overlap, "tail_short": tail_short}


def apply_plan(layer, plan: ApplyPlan, backup: Backup, log=print) -> list:
    """Execute the plan. Returns a list of error strings (empty on full success). The backup is
    saved after every step so a crash mid-way can still be undone."""
    errors = []

    def err(msg):
        errors.append(msg)
        backup.errors.append(msg)
        log("ERROR: " + msg)

    backup.save()
    for w in plan.warnings:
        log(w)
    previous = layer.current_timeline()           # restored at the end: the user keeps their view

    # 1. timelines
    for tp in plan.timelines:
        name = unique_name(tp.name, layer.timeline_names())
        tl = layer.create_timeline(name, tp.start_tc, tp.fps)
        if tl is None:
            err("could not create timeline %r" % name)
            continue
        backup.timelines.append(name)
        backup.save()
        log("timeline %s" % name)
        for w in tp.warnings:
            log("  " + w)
        missing, found, multich = set(), [], set()
        for p in tp.placements:
            clip = layer.clip_for_path(p.path)
            if clip is None:
                if not p.optional:           # reference songs are normally not imported
                    missing.add(p.path)
            elif p.media_type == 2 and layer.audio_channels(clip) > 2:
                # 5.1 camcorder audio (AVCHD) is appended as six mono channels on six tracks whatever
                # the track type, colliding with the next angle; its picture is still placed
                multich.add(os.path.basename(p.path))
            else:
                found.append((p, clip))
        for n in sorted(multich):
            log("  %s: picture only (its %s-channel audio cannot share a track)" % (n, "multi"))
            backup.summary.append("%s: %s placed picture only (multichannel audio)" % (name, n))
        # no empty tracks: renumber so that only track slots with a placeable clip exist
        remap = {old: new for new, old in enumerate(sorted({p.track for p, _ in found}), 1)}
        n_video = max([remap[p.track] for p, _ in found if p.media_type == 1] or [0])
        n_audio = max([remap[p.track] for p, _ in found if p.media_type == 2] or [0])
        layer.set_current_timeline(tl)                # track-level operations need this timeline current
        layer.set_track_count(tl, n_video, n_audio)
        for ttype, idx, tname in tp.track_names:
            if idx in remap and remap[idx] <= (n_video if ttype == "video" else n_audio):
                layer.set_track_name(tl, ttype, remap[idx], tname)
        base = layer.timeline_start_frame(tl)
        occ_by, slots_by, states = {}, {}, {}
        ordered = sorted(((dataclasses.replace(p, track=remap[p.track]), clip) for p, clip in found),
                         key=lambda pc: (pc[0].media_type, pc[0].track, pc[0].record_frame))
        for n, (p, clip) in enumerate(ordered):
            key = (p.path, p.media_type, p.track)
            nxt = ordered[n + 1][0] if n + 1 < len(ordered) else None
            followed = nxt is not None and (nxt.path, nxt.media_type, nxt.track) == key and \
                nxt.record_frame <= (p.record_end or p.record_frame)
            occ = occ_by.setdefault(key, [])
            slots_by.setdefault(key, []).append(slot_of(p, tp.fps, base))
            msg = append_piece(layer, tl, clip, p, tp.fps, base, log, occ, followed, states.setdefault(key, {}))
            if msg:
                err("%s track %s%d: %s" % (name, "V" if p.media_type == 1 else "A", p.track, msg))
        # Audio track states, only now (every item exists) and only on the CURRENT timeline (verified live:
        # SetTrackEnable has no effect otherwise). Rule: every placed angle of the master night keeps its audio
        # track enabled; only other nights/takes are disabled; never disable everything.
        layer.set_current_timeline(tl)
        audio_slots = {p.track for p, _ in found if p.media_type == 2}
        master_idx = sorted({remap[sl] for sl in tp.master_audio_slots if sl in audio_slots})
        muted_idx = sorted({remap[sl] for sl in tp.muted_audio_slots if sl in audio_slots} - set(master_idx))
        if muted_idx and not master_idx:
            err("%s: no master audio track would remain enabled (master clips missing from the media pool?); "
                "other nights' audio left ENABLED" % name)
            muted_idx = []
        for idx in muted_idx:
            ok = False
            for _try in range(2):
                layer.set_track_enable(tl, "audio", idx, False)
                if not layer.is_track_enabled(tl, "audio", idx):
                    ok = True
                    break
            if ok:
                log("  audio track A%d disabled" % idx)
            else:
                err("%s: audio track A%d is still enabled after SetTrackEnable(False)" % (name, idx))
        for idx in master_idx:
            if not layer.is_track_enabled(tl, "audio", idx):
                layer.set_track_enable(tl, "audio", idx, True)
                if not layer.is_track_enabled(tl, "audio", idx):
                    err("%s: master audio track A%d could not be enabled" % (name, idx))
        if n_audio and not any(layer.is_track_enabled(tl, "audio", i) for i in range(1, n_audio + 1)):
            err("%s: all audio tracks are disabled" % name)
        for key, occ in occ_by.items():
            sm = occupancy_summary(slots_by[key], occ)
            line = "%s: %s%d %s: %d slot(s), %d item(s), max gap %d frame(s)%s" % (
                name, "V" if key[1] == 1 else "A", key[2], os.path.basename(key[0]), sm["slots"], sm["items"],
                sm["max_gap"], ((", %d overlap(s)" % sm["overlaps"]) if sm["overlaps"] else "") +
                ((", tail short %d frame(s) (source ended)" % sm["tail_short"]) if sm["tail_short"] else ""))
            backup.summary.append(line)
            log(line)
            if sm["max_gap"] > 0 or sm["overlaps"] or sm["tail_short"] > 2:
                err("gap/overlap on " + line)
        for pth in sorted(missing):
            err("clip not in media pool: %s" % pth)
        for mk in tp.markers:
            if layer.add_timeline_marker(tl, mk.frame, mk.color, mk.name, mk.note, mk.duration, mk.custom):
                backup.timeline_markers.append({"timeline": name, "frame": mk.frame, "custom": mk.custom})
            else:
                err("marker %r on %s failed" % (mk.name, name))
        backup.save()

    # 2. media pool markers
    for mk in plan.clip_markers:
        clip = layer.clip_for_path(mk.path)
        if clip is None:
            err("clip not in media pool: %s" % mk.path)
            continue
        if layer.add_clip_marker(clip, mk.frame, mk.color, mk.name, mk.note, mk.duration, mk.custom):
            backup.clip_markers.append({"path": mk.path, "frame": mk.frame, "custom": mk.custom})
        else:
            err("marker %r at frame %d on %s failed" % (mk.name, mk.frame, os.path.basename(mk.path)))
    backup.save()

    # 3. timecodes (+ original TC kept in metadata)
    for op in plan.tc_ops:
        clip = layer.clip_for_path(op.path)
        if clip is None:
            err("clip not in media pool: %s" % op.path)
            continue
        old = layer.get_start_tc(clip) or op.old_tc or ""
        old_meta = layer.get_metadata(clip, METADATA_KEY) or ""
        if not layer.set_start_tc(clip, op.new_tc):
            err("set Start TC failed: %s" % os.path.basename(op.path))
            continue
        backup.tc_changes.append({"path": op.path, "old_tc": old, "new_tc": op.new_tc})
        if ORIG_TC_PREFIX not in old_meta:    # keep the first original, never overwrite existing text
            new_meta = (old_meta + "\n" if old_meta else "") + ORIG_TC_PREFIX + old
            if layer.set_metadata(clip, METADATA_KEY, new_meta):
                backup.metadata.append({"path": op.path, "key": METADATA_KEY, "old": old_meta, "new": new_meta})
            else:
                err("set metadata failed: %s" % os.path.basename(op.path))
        backup.save()
    if previous is not None:
        layer.set_current_timeline(previous)          # leave the user's view as it was
    return errors


def undo_backup(layer, backup: Backup, log=print) -> list:
    """Reverse an apply: delete created timelines and our markers, restore TC and metadata."""
    errors = []
    for m in reversed(backup.metadata):
        clip = layer.clip_for_path(m["path"])
        if clip is None or not layer.set_metadata(clip, m["key"], m["old"]):
            errors.append("restore metadata failed: %s" % m["path"])
    for t in reversed(backup.tc_changes):
        clip = layer.clip_for_path(t["path"])
        if clip is None or not layer.set_start_tc(clip, t["old_tc"]):
            errors.append("restore Start TC failed: %s" % t["path"])
    for m in reversed(backup.clip_markers):
        clip = layer.clip_for_path(m["path"])
        if clip is None or not layer.delete_clip_marker(clip, m["frame"], m["custom"]):
            errors.append("remove marker failed: %s @%s" % (m["path"], m["frame"]))
    for name in reversed(backup.timelines):     # timeline markers vanish with their timeline
        if not layer.delete_timeline(name):
            errors.append("delete timeline failed: %s" % name)
    backup.undone = True
    backup.errors += errors
    backup.save()
    for e in errors:
        log("ERROR: " + e)
    return errors


# --------------------------------------------------------------------------- real Resolve layer

class ResolveLayer(LayerBase):
    """Thin wrapper over the Resolve scripting API. Not exercised outside Resolve."""

    def __init__(self, resolve, project):
        self.resolve = resolve
        self.project = project
        self.pool = project.GetMediaPool()
        self._index: Optional[dict] = None

    # -- media pool
    def _walk(self, folder):
        for c in folder.GetClipList() or []:
            yield c
        for sub in folder.GetSubFolderList() or []:
            yield from self._walk(sub)

    def all_clips(self):
        return list(self._walk(self.pool.GetRootFolder()))

    def clip_for_path(self, path):
        if self._index is None:
            self.refresh_index()
        return self._index.get(norm_path(path))

    def refresh_index(self):
        self._index = {}
        for c in self.all_clips():
            p = c.GetClipProperty("File Path")
            if p:
                self._index.setdefault(norm_path(p), c)

    # -- project
    def project_fps(self):
        """(fps float, drop bool) of the current timeline settings."""
        fps = float(self.project.GetSetting("timelineFrameRate") or 24)
        drop = str(self.project.GetSetting("timelineDropFrameTimecodeMode") or "") in ("1", "DF", "True")
        return fps, drop

    # -- timelines
    def timeline_names(self):
        return [self.project.GetTimelineByIndex(i).GetName()
                for i in range(1, self.project.GetTimelineCount() + 1)]

    def _find_timeline(self, name):
        for i in range(1, self.project.GetTimelineCount() + 1):
            t = self.project.GetTimelineByIndex(i)
            if t.GetName() == name:
                return t
        return None

    def create_timeline(self, name, start_tc, fps):
        tl = self.pool.CreateEmptyTimeline(name, start_tc)   # None if the name is taken/invalid
        if tl is not None:
            self.project.SetCurrentTimeline(tl)
        return tl

    def timeline_start_frame(self, tl):
        return int(tl.GetStartFrame())

    def audio_channels(self, clip):
        try:
            return int(clip.GetClipProperty("Audio Ch") or 2)
        except (TypeError, ValueError):
            return 2

    def set_track_count(self, tl, n_video, n_audio):
        while tl.GetTrackCount("video") < n_video:
            if not tl.AddTrack("video"):
                break
        while tl.GetTrackCount("audio") < n_audio:
            if not (tl.AddTrack("audio", AUDIO_TRACK_SUBTYPE) or tl.AddTrack("audio")):
                break

    def set_track_name(self, tl, ttype, index, name):
        try:
            tl.SetTrackName(ttype, index, name)
        except Exception:
            pass

    def append_item(self, tl, info):
        self.project.SetCurrentTimeline(tl)
        items = self.pool.AppendToTimeline([info]) or []
        item = items[0] if items else None
        # Resolve can hand back a dead item (GetDuration() None) for a range past the end of the media,
        # e.g. the last frames of interlaced AVCHD; treat that as a failed append, not a crash
        if item is not None and item.GetDuration() is None:
            return None
        return item

    def delete_item(self, tl, item):
        return bool(tl.DeleteClips([item]))

    def current_timeline(self):
        return self.project.GetCurrentTimeline()

    def set_current_timeline(self, tl):
        return bool(self.project.SetCurrentTimeline(tl))

    def set_track_enable(self, tl, ttype, index, enabled):
        # verified live: SetTrackEnable only takes effect on the CURRENT timeline
        self.project.SetCurrentTimeline(tl)
        return bool(tl.SetTrackEnable(ttype, index, enabled))

    def is_track_enabled(self, tl, ttype, index):
        self.project.SetCurrentTimeline(tl)
        return bool(tl.GetIsTrackEnabled(ttype, index))

    def set_speed(self, item, pct):
        # verified live: full float precision kept, duration stays fixed (RippleTimeline default False)
        return bool(item.SetSpeed({"Percentage": float(pct), "PitchCorrection": False}))

    def item_extent(self, item, media_type):
        dur = int(round(item.GetDuration()))
        if media_type != 1:                 # audio items: caller derives the consumed range from the duration
            return dur, None, None
        # GetSourceEndFrame is inclusive (14399 for a 14400-frame clip)
        return dur, int(item.GetSourceStartFrame()), int(item.GetSourceEndFrame()) + 1

    def add_timeline_marker(self, tl, frame, color, name, note, duration, custom):
        return bool(tl.AddMarker(frame, color, name, note, duration, custom))

    def delete_timeline(self, name):
        tl = self._find_timeline(name)
        return tl is not None and bool(self.pool.DeleteTimelines([tl]))

    # -- clips
    def get_start_tc(self, clip):
        return clip.GetClipProperty("Start TC")

    def set_start_tc(self, clip, tc):
        return bool(clip.SetClipProperty("Start TC", tc)) and clip.GetClipProperty("Start TC") == tc

    def get_metadata(self, clip, key):
        return clip.GetMetadata(key) or ""

    def set_metadata(self, clip, key, value):
        return bool(clip.SetMetadata({key: value}))

    def add_clip_marker(self, clip, frame, color, name, note, duration, custom):
        return bool(clip.AddMarker(frame, color, name, note, duration, custom))

    def delete_clip_marker(self, clip, frame, custom):
        mk = clip.GetMarkers() or {}
        info = mk.get(frame)
        if info is None or not str(info.get("customData", "")).startswith(TAG):
            return False
        return bool(clip.DeleteMarkerAtFrame(frame))

    # -- helpers for the UI
    def clip_markers(self, clip) -> list:
        """User markers on a clip as dicts (excluding ShowSync's own)."""
        out = []
        for fr, info in (clip.GetMarkers() or {}).items():
            if str(info.get("customData", "")).startswith(TAG):
                continue
            out.append({"frame": int(fr), "name": info.get("name", ""), "color": info.get("color", "Blue"),
                        "note": info.get("note", ""), "duration": int(info.get("duration", 1))})
        return sorted(out, key=lambda m: m["frame"])

    def clip_marker_fps(self, clip, fallback: float) -> float:
        """Rate for converting a clip's marker frames to seconds: its FPS property, else the timeline
        rate (audio-only clips report no FPS; UNVERIFIED which rate Resolve uses for their markers)."""
        try:
            f = float(clip.GetClipProperty("FPS"))
            return f if f > 0 else fallback
        except (TypeError, ValueError):
            return fallback

    def clip_marker_dump(self, clip, fallback_fps: float) -> dict:
        """{path, fps, markers} with ALL markers (including ShowSync's own, flagged by the caller)."""
        fps = self.clip_marker_fps(clip, fallback_fps)
        origin = 0
        if MARKER_FRAMES_ABSOLUTE:
            try:
                origin = int(clip.GetClipProperty("Start"))
            except (TypeError, ValueError):
                origin = 0
        mk = []
        for fr, info in (clip.GetMarkers() or {}).items():
            if str(info.get("customData", "")).startswith(TAG):
                continue
            mk.append({"frame": int(fr) - origin, "name": info.get("name", ""), "color": info.get("color", ""),
                       "note": info.get("note", ""), "duration": int(info.get("duration", 1))})
        return {"path": clip.GetClipProperty("File Path"), "fps": fps, "markers": mk}

    def open_timeline_at(self, name_prefix: str, tc: str) -> str:
        """Make the newest timeline whose name equals/starts with name_prefix current and move the
        playhead to tc. Returns "" on success or an error text."""
        found = None
        for i in range(1, self.project.GetTimelineCount() + 1):
            t = self.project.GetTimelineByIndex(i)
            n = t.GetName()
            if n == name_prefix or n.startswith(name_prefix + " ("):
                found = t
        if found is None:
            return "timeline %r not found (apply event timelines first)" % name_prefix
        if not self.project.SetCurrentTimeline(found):
            return "could not open timeline"
        return "" if found.SetCurrentTimecode(tc) else "could not set playhead to " + tc

