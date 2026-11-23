"""ShowSync Resolve plugin: pure-stdlib logic (no Resolve, no numpy).

Everything that decides *what* to do lives here: reading the AnalysisResult JSON, time-map
evaluation, timecode maths, plan building, marker propagation and the backup file format.
resolve_ops.py executes the plans against Resolve (or a fake in tests).

Timecode helpers are written here (showsync/graph/groups.py had none when this was written).
"""
from __future__ import annotations

import bisect
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

DRIFT_RESYNC_FRAC = 0.5          # mirror of showsync.config.DRIFT_RESYNC_FRAC
TL_START_TC = "01:00:00:00"
SONG_COLORS = ["Blue", "Green", "Yellow", "Red", "Purple", "Cyan", "Pink", "Sand",
               "Fuchsia", "Mint", "Lemon", "Rose", "Sky", "Lavender", "Cocoa", "Cream"]
TAG = "showsync"                  # marker customData prefix
ORIG_TC_PREFIX = "ShowSync orig TC="
EXPORT_VERSION = 1


# --------------------------------------------------------------------------- paths

def norm_path(p: str) -> str:
    """Normalised key for matching result paths against media-pool 'File Path' values."""
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


# --------------------------------------------------------------------------- time maps

class TimeMap:
    """Piecewise-linear monotonic map; extrapolates with the end slopes (same as showsync.models)."""

    def __init__(self, src, dst):
        self.src = [float(x) for x in src]
        self.dst = [float(x) for x in dst]
        if len(self.src) < 2 or len(self.src) != len(self.dst):
            raise ValueError("TimeMap needs >= 2 equal-length knot lists")

    @staticmethod
    def from_dict(d) -> Optional["TimeMap"]:
        return None if d is None else TimeMap(d["src"], d["dst"])

    def to_dict(self):
        return {"src": list(self.src), "dst": list(self.dst)}

    @staticmethod
    def _eval(t, xs, ys):
        if t <= xs[0]:
            return ys[0] + (t - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        if t >= xs[-1]:
            return ys[-1] + (t - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        i = bisect.bisect_right(xs, t) - 1
        return ys[i] + (t - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])

    def __call__(self, t: float) -> float:
        return self._eval(float(t), self.src, self.dst)

    def inverse(self) -> "TimeMap":
        return TimeMap(self.dst, self.src)

    @property
    def is_linear(self) -> bool:
        if len(self.src) <= 2:
            return True
        sl = [(self.dst[i + 1] - self.dst[i]) / (self.src[i + 1] - self.src[i]) for i in range(len(self.src) - 1)]
        return all(abs(s - sl[0]) < 1e-9 for s in sl)

    @property
    def ratio(self) -> float:
        return (self.dst[-1] - self.dst[0]) / (self.src[-1] - self.src[0])

    @property
    def offset(self) -> float:
        return self(0.0)


# --------------------------------------------------------------------------- timecode

_NTSC = {23.976: 24000 / 1001, 24000 / 1001: 24000 / 1001, 29.97: 30000 / 1001, 30000 / 1001: 30000 / 1001,
         47.952: 48000 / 1001, 59.94: 60000 / 1001, 60000 / 1001: 60000 / 1001, 119.88: 120000 / 1001}


def exact_fps(fps: float) -> float:
    """23.976 -> 24000/1001 etc.; other rates unchanged."""
    for k, v in _NTSC.items():
        if abs(fps - k) < 0.005:
            return v
    return float(fps)


def nominal_fps(fps: float) -> int:
    return int(round(fps))


def supports_drop(fps: float) -> bool:
    return abs(exact_fps(fps) - nominal_fps(fps)) > 1e-6 and nominal_fps(fps) in (30, 60)


def _drop_frames(fps: float) -> int:
    return 2 * (nominal_fps(fps) // 30)


def tc_is_drop(tc: str) -> bool:
    return bool(tc) and (";" in tc or "." in tc)


def tc_to_frames(tc: str, fps: float, drop: Optional[bool] = None) -> int:
    """'HH:MM:SS:FF' (or ';' before FF for drop frame) -> frame count since 00:00:00:00."""
    m = re.fullmatch(r"\s*(\d+)[:;.](\d+)[:;.](\d+)([:;.])(\d+)\s*", tc)
    if not m:
        raise ValueError("bad timecode %r" % tc)
    h, mi, s, sep, f = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    if drop is None:
        drop = sep in ";."
    nom = nominal_fps(fps)
    n = (h * 3600 + mi * 60 + s) * nom + f
    if drop and supports_drop(fps):
        tm = h * 60 + mi
        n -= _drop_frames(fps) * (tm - tm // 10)
    return n


def frames_to_tc(frames: int, fps: float, drop: bool = False) -> str:
    frames = int(frames)
    if frames < 0:
        raise ValueError("negative frame count")
    nom = nominal_fps(fps)
    use_drop = drop and supports_drop(fps)
    if use_drop:
        d = _drop_frames(fps)
        per10 = nom * 600 - 9 * d
        per1 = nom * 60 - d
        blocks, rem = divmod(frames, per10)
        frames += 9 * d * blocks
        if rem >= d:
            frames += d * ((rem - d) // per1)
    ff = frames % nom
    ss = (frames // nom) % 60
    mm = (frames // (nom * 60)) % 60
    hh = (frames // (nom * 3600)) % 24
    return "%02d:%02d:%02d%s%02d" % (hh, mm, ss, ";" if use_drop else ":", ff)


def tc_to_seconds(tc: str, fps: float, drop: Optional[bool] = None) -> float:
    return tc_to_frames(tc, fps, drop) / exact_fps(fps)


def seconds_to_tc(sec: float, fps: float, drop: bool = False) -> str:
    return frames_to_tc(int(round(sec * exact_fps(fps))), fps, drop)


def sec_to_frames(sec: float, fps: float) -> int:
    return int(round(sec * exact_fps(fps)))


# --------------------------------------------------------------------------- result wrapper

class Result:
    """Read-only view of the AnalysisResult JSON (dict form) with TimeMaps parsed."""

    def __init__(self, d: dict, path: str = ""):
        self.d = d
        self.path = path
        self.clips: dict = d.get("clips", {})
        self.groups: list = d.get("groups", [])
        self.loners: list = d.get("loners", [])
        self.segments: list = d.get("segments", [])
        self.songs: list = d.get("songs", [])
        self.matches: list = d.get("matches", [])
        self.warnings: list = d.get("warnings", [])
        self._gmaps = {g["id"]: {c: TimeMap.from_dict(m) for c, m in g["maps"].items()} for g in self.groups}
        self._smaps = {s["id"]: TimeMap.from_dict(s.get("song_map")) for s in self.segments}
        self._song_index = {s["id"]: i + 1 for i, s in enumerate(self.songs)}
        self._group_index = {g["id"]: i + 1 for i, g in enumerate(self.groups)}
        self._clip_group = {}
        for g in self.groups:
            for c in g["maps"]:
                self._clip_group[c] = g
        self._by_path = {norm_path(c["path"]): cid for cid, c in self.clips.items()}

    # loading
    @staticmethod
    def load(path: str) -> "Result":
        with open(path, "r", encoding="utf-8") as f:
            return Result(json.load(f), path)

    # clips
    def clip_ids_by_path(self) -> dict:
        return dict(self._by_path)

    def clip_path(self, cid): return self.clips[cid]["path"]
    def clip_name(self, cid): return os.path.basename(self.clips[cid]["path"])
    def clip_duration(self, cid): return float(self.clips[cid]["duration"])
    def has_video(self, cid): return bool(self.clips[cid].get("has_video"))
    def has_audio(self, cid): return bool(self.clips[cid].get("has_audio", True))
    def is_reference(self, cid): return self.clips[cid].get("role", "clip") == "reference"

    def clip_fps(self, cid, fallback: float) -> float:
        f = float(self.clips[cid].get("fps") or 0.0)
        return f if f > 0 else float(fallback)

    # groups
    def group_of(self, cid):
        return self._clip_group.get(cid)

    def group_index(self, gid) -> int:
        return self._group_index[gid]

    def event_map(self, cid) -> Optional[TimeMap]:
        g = self._clip_group.get(cid)
        return self._gmaps[g["id"]][cid] if g else None

    def group_map(self, gid, cid) -> TimeMap:
        return self._gmaps[gid][cid]

    # songs
    def song_map(self, seg) -> Optional[TimeMap]:
        return self._smaps[seg["id"]]

    def song_name(self, song_id) -> str:
        for s in self.songs:
            if s["id"] == song_id:
                return s["name"]
        return str(song_id)

    def song_number(self, song_id) -> int:
        return self._song_index[song_id]

    def song_segments(self, song_id) -> list:
        return [s for s in self.segments if s.get("song_id") == song_id and self._smaps[s["id"]] is not None]

    def segments_of_group(self, gid) -> list:
        return sorted([s for s in self.segments if s["owner"] == gid], key=lambda s: s["start"])


def _safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "-", s).strip()


_UNNAMED_SONG = re.compile(r"^\s*song\s*\d+\s*$", re.I)


def group_date(res: Result, g: dict) -> str:
    """Date (YYYY-MM-DD) of a group from its members' creation_time (most common, earliest on ties)."""
    days = {}
    for cid in [g["anchor"]] + list(g["maps"]):
        if cid in res.clips and not res.is_reference(cid):
            ct = (res.clips[cid].get("creation_time") or "")[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ct):
                days[ct] = days.get(ct, 0) + 1
    return min(days, key=lambda d: (-days[d], d)) if days else ""


def group_title(res: Result, gi: int) -> str:
    """'ShowSync G01 <first named song or date>'. Unnamed songs ("Song 07") are never used as a label."""
    g = res.groups[gi]
    segs = [s for s in res.segments if s["owner"] == g["id"] and s.get("song_id")]
    if not segs:
        segs = [s for s in res.segments if s["owner"] in g["maps"] and s.get("song_id")]
    label = ""
    for sg in sorted(segs, key=lambda x: x["start"]):
        nm = res.song_name(sg["song_id"])
        if not _UNNAMED_SONG.match(nm):
            label = nm
            break
    if not label:
        label = group_date(res, g)
    name = "ShowSync G%02d" % (gi + 1)
    return _safe_name((name + " " + label).strip())


# --------------------------------------------------------------------------- drift splitting

@dataclass
class Piece:
    sf: int                  # first source frame (clip frame rate)
    ef: int                  # one past the last source frame
    rec: int                 # record frame, timeline frames, relative to timeline start
    drift_err_frames: float  # max |accumulated timing error| inside the piece, in timeline frames
    quant_err_frames: float  # rounding of the record position, timeline frames (<= 0.5)
    speed_pct: float = 100.0  # playback speed of the piece (100 = plain 1:1)
    rec_end: int = 0          # end of the piece's timeline slot [rec, rec_end), relative frames


def split_for_drift(m: TimeMap, clip_fps: float, n_frames: int, tl_fps: float,
                    frac: float = DRIFT_RESYNC_FRAC, first_frame: int = 0) -> list:
    """Cut clip frames [first_frame, n_frames) into pieces played 1:1 so that within every piece
    |m(t) - (m(a) + (t - a))| stays <= frac timeline frames (a = piece start). Boundaries lie on
    source frame boundaries. Each piece's record frame comes from m at its start."""
    cfps, tfps = exact_fps(clip_fps), exact_fps(tl_fps)
    tol = frac / tfps * (1 - 1e-9)
    pieces = []
    sf = first_frame
    guard = 0
    while sf < n_frames:
        guard += 1
        if guard > 100000:
            raise RuntimeError("drift splitting did not converge")
        a = sf / cfps
        T = n_frames / cfps
        fa = m(a)
        brk = [k for k in m.src if a < k < T] + [T]
        ef = n_frames
        p, ep = a, 0.0
        for q in brk:
            eq = m(q) - fa - (q - a)
            if abs(eq) <= tol:
                p, ep = q, eq
                continue
            target = tol if eq > 0 else -tol
            t = p + (target - ep) / (eq - ep) * (q - p)
            ef = max(sf + 1, int(math.floor(t * cfps + 1e-9)))
            break
        ef = min(ef, n_frames)
        b = ef / cfps
        pts = [a, b] + [k for k in m.src if a < k < b]
        err = max(abs(m(t) - fa - (t - a)) for t in pts) * tfps
        rec_exact = fa * tfps
        rec = int(round(rec_exact))
        pieces.append(Piece(sf, ef, rec, err, abs(rec - rec_exact)))
        sf = ef
    return pieces


def _pts(m: TimeMap, a: float, b: float, n: int = 64) -> list:
    pts = [a + (b - a) * i / n for i in range(n + 1)]
    pts += [k for k in m.src if a < k < b]
    return pts


def _fit(m: TimeMap, a: float, b: float, plain: bool):
    """Line through the map over [a, b]. plain: slope fixed to 1 (offset minimax). Returns
    (max_abs_error_seconds, slope, line value at a)."""
    ts = _pts(m, a, b)
    ys = [m(t) for t in ts]
    if plain:
        d = [y - t for t, y in zip(ts, ys)]
        hi, lo = max(d), min(d)
        return (hi - lo) / 2.0, 1.0, a + (hi + lo) / 2.0
    n = float(len(ts))
    mx, my = sum(ts) / n, sum(ys) / n
    sxx = sum((t - mx) ** 2 for t in ts)
    slope = (sum((t - mx) * (y - my) for t, y in zip(ts, ys)) / sxx) if sxx > 0 else 1.0
    y0 = my + slope * (a - mx)
    err = max(abs(y - (y0 + slope * (t - a))) for t, y in zip(ts, ys))
    return err, slope, y0


def _extent(m: TimeMap, a: float, T: float, tol: float, plain: bool) -> float:
    """Largest b in (a, T] whose line fit over [a, b] stays within tol."""
    feas = lambda b: _fit(m, a, b, plain)[0] <= tol
    if feas(T):
        return T
    cand = [k for k in m.src if a < k < T] + [T]
    lo_i, hi_i = -1, len(cand) - 1            # cand[hi_i] is infeasible
    while hi_i - lo_i > 1:
        mid = (lo_i + hi_i) // 2
        if feas(cand[mid]):
            lo_i = mid
        else:
            hi_i = mid
    lo = a if lo_i < 0 else cand[lo_i]
    hi = cand[hi_i]
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if feas(mid):
            lo = mid
        else:
            hi = mid
    return lo


def fit_pieces(m: TimeMap, clip_fps: float, n_frames: int, tl_fps: float,
               frac: float = DRIFT_RESYNC_FRAC, first_frame: int = 0, snap_knots: bool = False,
               merge_pct: float = 0.0, min_piece_s: float = 0.0) -> list:
    """Place a clip with as few pieces as possible. Each piece is either plain (speed 100 %, offset
    only) or a speed change: speed_pct = 100 / slope of the line fitted to the map over the piece
    (slope = event seconds per clip second). Pieces are the longest ones whose line stays within
    `frac` timeline frames of the map; boundaries lie on source-frame boundaries. A plain piece is
    chosen whenever it reaches at least as far as the speed-changed one.
    snap_knots: end a piece at the last map knot (bar/beat boundary) of its final 40 % when that
    still fits. merge_pct: merge neighbouring pieces whose speeds differ by less than that many
    percent when one line still fits both. min_piece_s: prefer pieces at least that long (and no short
    tail piece) by accepting up to twice the tolerance."""
    cfps, tfps = exact_fps(clip_fps), exact_fps(tl_fps)
    tol = frac / tfps * (1 - 1e-9)
    T = n_frames / cfps

    def make(sf, ef, plain):
        a = sf / cfps
        err, slope, y0 = _fit(m, a, ef / cfps, plain)
        rec_exact = y0 * tfps
        rec = int(round(rec_exact))
        pc = Piece(sf, ef, rec, err * tfps, abs(rec - rec_exact), 100.0 if plain else 100.0 / slope)
        pc._end_exact = (y0 + slope * (ef / cfps - a)) * tfps
        return pc

    pieces, sf = [], first_frame
    while sf < n_frames:
        a = sf / cfps

        def end_frame(plain):
            b = _extent(m, a, T, tol, plain)
            return n_frames if b >= T else min(n_frames, max(sf + 1, int(math.floor(b * cfps + 1e-9))))

        ef_p, ef_f = end_frame(True), end_frame(False)
        plain = ef_p >= ef_f
        ef = ef_p if plain else ef_f
        if snap_knots and ef < n_frames:
            b = ef / cfps
            ks = [k for k in m.src if a + 0.6 * (b - a) < k <= b]
            if ks:
                ef2 = int(math.floor(max(ks) * cfps + 1e-9))
                if ef2 > sf and _fit(m, a, ef2 / cfps, plain)[0] <= tol:
                    ef = ef2
        if min_piece_s > 0 and ef < n_frames:
            min_f = int(math.ceil(min_piece_s * cfps))
            ef_min = min(n_frames, sf + min_f)
            if ef < ef_min and _fit(m, a, ef_min / cfps, plain)[0] <= 2 * tol:
                ef = ef_min
            if 0 < n_frames - ef < min_f and _fit(m, a, T, plain)[0] <= 2 * tol:
                ef = n_frames
        pieces.append(make(sf, ef, plain))
        sf = ef
    if merge_pct > 0:
        k = 0
        while k + 1 < len(pieces):
            x, y = pieces[k], pieces[k + 1]
            if abs(x.speed_pct - y.speed_pct) < merge_pct:
                done = False
                for plain in (True, False):
                    if _fit(m, x.sf / cfps, y.ef / cfps, plain)[0] <= tol:
                        pieces[k:k + 2] = [make(x.sf, y.ef, plain)]
                        done = True
                        break
                if done:
                    continue
            k += 1
    # timeline slots: contiguous by construction, slot_k = [rec_k, rec_k+1), the last ends at line(end)
    for k in range(1, len(pieces)):
        if pieces[k].rec <= pieces[k - 1].rec:
            pieces[k].rec = pieces[k - 1].rec + 1
    for k, pc in enumerate(pieces):
        end = pieces[k + 1].rec if k + 1 < len(pieces) else int(round(pc._end_exact))
        pc.rec_end = max(end, pc.rec + 1)
        if k + 1 < len(pieces):
            # speed from the SLOT, not from the fitted line (every piece that is followed by another): it then
            # consumes exactly (ef - sf) source frames in rec..rec_end, so the next piece starts exactly where
            # this one ends (no source-time drift or re-read frames at joins; < 1 frame/piece change)
            pc.speed_pct = (pc.ef - pc.sf) / cfps / ((pc.rec_end - pc.rec) / tfps) * 100.0
            if abs(pc.speed_pct - 100.0) < 1e-6:
                pc.speed_pct = 100.0
    return pieces


# --------------------------------------------------------------------------- plan types

@dataclass
class Placement:
    clip_id: str
    path: str
    track: int               # 1-based track index (video index for mediaType 1, audio index for 2)
    media_type: int          # 1 video, 2 audio
    start_frame: int         # source frame (clip fps), inclusive
    end_frame: int           # source frame, EXCLUSIVE (ops converts if the API wants inclusive)
    record_frame: int        # timeline frames relative to timeline start
    drift_err_frames: float = 0.0
    quant_err_frames: float = 0.0
    optional: bool = False   # reference songs: quietly skipped if not in the media pool
    speed_pct: float = 100.0  # clip speed (no pitch correction); timeline length = source length * 100 / speed
    record_end: int = 0      # end of the timeline slot [record_frame, record_end); 0 = derive from the speed
    clip_fps: float = 0.0    # source frame rate of start/end frames (0 = timeline rate)
    clip_frames: int = 0     # total source frames of the clip (0 = unknown)


@dataclass
class MarkerOp:
    frame: int
    name: str
    color: str
    note: str = ""
    duration: int = 1
    custom: str = TAG
    clip_id: str = ""        # set for media-pool markers
    path: str = ""
    timeline: str = ""       # set for timeline markers
    via: str = ""            # "event"/"song" for propagated markers


@dataclass
class TimelinePlan:
    name: str
    kind: str                # "event" | "song"
    ref_id: str              # group id or song id
    fps: float
    start_tc: str = TL_START_TC
    n_video: int = 0
    n_audio: int = 0
    track_names: list = field(default_factory=list)   # (type, index, name)
    placements: list = field(default_factory=list)
    markers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    muted_audio_slots: list = field(default_factory=list)   # track slots whose audio track gets disabled
    master_audio_slots: list = field(default_factory=list)  # master-night audio slots: must stay enabled

    @property
    def max_drift_err_frames(self) -> float:
        return max([p.drift_err_frames for p in self.placements] + [0.0])


@dataclass
class TcOp:
    clip_id: str
    path: str
    mode: str
    old_tc: Optional[str]
    new_tc: str
    fps: float
    drop: bool


@dataclass
class ApplyPlan:
    fps: float
    timelines: list = field(default_factory=list)
    clip_markers: list = field(default_factory=list)
    tc_ops: list = field(default_factory=list)
    tc_untouched: list = field(default_factory=list)   # dicts {clip_id, path, reason}
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ApplyOptions:
    event_timelines: bool = True
    song_timelines: bool = True
    song_align: str = "warp"           # songs with takes from several nights/owners: "warp" (speed-matched
                                       # to the longest take, others muted) | "offset" (one offset per take)
    song_markers: bool = True
    tc_mode: str = "none"              # none | event | song
    propagate_source: str = ""         # clip id whose markers are propagated
    propagate_markers: list = field(default_factory=list)   # dicts frame,name,color,note,duration
    propagate_fps: float = 0.0
    frac: float = DRIFT_RESYNC_FRAC
    master_mode: bool = False          # master-night song timelines (cross-show)
    master_group: Optional[str] = None  # group id or loner clip id whose take is the master (default: longest)
    max_warp_pct: float = 5.0          # master mode: speed changes allowed within +-this percent
    multicam: bool = False             # master mode: ask for a multicam clip (not possible via the API)
    other_audio: str = "off"           # master mode: other nights' audio: "off" | "muted"
    master_tol_frames: float = 1.5     # master mode: residual tolerance vs the musical map (picture only)
    handles_s: float = 0.0             # song timelines: seconds of handle each side of a segment
    whole_clips: bool = False          # song timelines: whole clips instead of segments


# --------------------------------------------------------------------------- timelines

def build_event_timeline(res: Result, gi: int, tl_fps: float, frac: float = DRIFT_RESYNC_FRAC) -> TimelinePlan:
    g = res.groups[gi]
    tp = TimelinePlan(group_title(res, gi), "event", g["id"], tl_fps)
    allm = sorted(g["maps"], key=lambda c: (g["spans"][c][0], c))
    members = [c for c in allm if not res.is_reference(c)]
    refs = [c for c in allm if res.is_reference(c)]
    tp.n_video = len(members)
    tp.n_audio = len(members) + len(refs)
    for i, cid in enumerate(members, 1):
        m = res.group_map(g["id"], cid)
        cfps = res.clip_fps(cid, tl_fps) if res.has_video(cid) else tl_fps
        nfr = int(math.floor(res.clip_duration(cid) * exact_fps(cfps) + 1e-6))
        if nfr < 1:
            tp.warnings.append("%s has no frames" % res.clip_name(cid))
            continue
        pieces = fit_pieces(m, cfps, nfr, tl_fps, frac)
        name = res.clip_name(cid)
        tp.track_names.append(("video", i, name))
        tp.track_names.append(("audio", i, name))
        for pc in pieces:
            for mt, ok in ((1, res.has_video(cid)), (2, res.has_audio(cid))):
                if ok:
                    tp.placements.append(Placement(cid, res.clip_path(cid), i, mt, pc.sf, pc.ef, pc.rec,
                                                   pc.drift_err_frames, pc.quant_err_frames,
                                                   speed_pct=pc.speed_pct, clip_fps=cfps, clip_frames=nfr,
                                                   record_end=pc.rec_end))
        sp = [pc.speed_pct for pc in pieces]
        tp.warnings.append("%s: speed %s, %d piece(s), max residual %.3f frames" % (
            name, ("%.4f %%" % sp[0]) if max(sp) - min(sp) < 1e-9 else "%.4f-%.4f %%" % (min(sp), max(sp)),
            len(pieces), max(pc.drift_err_frames for pc in pieces)))
    # reference songs: master playback track(s) on the top audio tracks, optional (usually not imported)
    for j, cid in enumerate(refs, len(members) + 1):
        m = res.group_map(g["id"], cid)
        nfr = int(math.floor(res.clip_duration(cid) * exact_fps(tl_fps) + 1e-6))
        if nfr < 1 or not res.has_audio(cid):
            continue
        tp.track_names.append(("audio", j, "REF " + res.clip_name(cid)))
        for pc in fit_pieces(m, tl_fps, nfr, tl_fps, frac):
            tp.placements.append(Placement(cid, res.clip_path(cid), j, 2, pc.sf, pc.ef, pc.rec,
                                           pc.drift_err_frames, pc.quant_err_frames, optional=True,
                                           speed_pct=pc.speed_pct, clip_fps=tl_fps, clip_frames=nfr,
                                           record_end=pc.rec_end))
    # repeat-ambiguous clips: mark every alternative position
    for cid, shifts in (g.get("alt_shifts") or {}).items():
        if cid not in g["spans"]:
            continue
        for sh in shifts:
            t = g["spans"][cid][0] + float(sh)
            if t < 0:
                tp.warnings.append("%s: alternative at %.1f s is before the group start" % (res.clip_name(cid), t))
                continue
            tc = seconds_to_tc(3600.0 + t, tl_fps)
            tp.markers.append(MarkerOp(frame=sec_to_frames(t, tl_fps), name="Also fits", color="Cyan",
                                       note="%s: this take also fits at %s" % (res.clip_name(cid), tc),
                                       duration=1, custom="%s:alt:%s" % (TAG, cid), timeline=tp.name))
    # song markers on the event timeline
    for s in res.segments_of_group(g["id"]):
        if s.get("song_id"):
            tp.markers.append(song_marker(res, s, tl_fps, timeline=tp.name, event_time=s["start"]))
    return tp


def song_marker(res: Result, seg: dict, fps: float, timeline: str = "", event_time: Optional[float] = None,
                clip_id: str = "", path: str = "", clip_time: Optional[float] = None,
                clip_end: Optional[float] = None) -> MarkerOp:
    sid = seg["song_id"]
    color = SONG_COLORS[(res.song_number(sid) - 1) % len(SONG_COLORS)]
    t0 = event_time if event_time is not None else clip_time
    t1 = seg["end"] if clip_end is None else clip_end
    dur = max(1, sec_to_frames(max(0.0, (t1 - t0)), fps)) if t1 is not None else 1
    note = "match %.2f%s" % (seg.get("match_score", 0.0), ", partial" if seg.get("is_partial") else "")
    return MarkerOp(frame=max(0, sec_to_frames(t0, fps)), name=res.song_name(sid), color=color, note=note,
                    duration=dur, custom="%s:song:%s" % (TAG, sid), clip_id=clip_id, path=path, timeline=timeline)


def build_song_timeline(res: Result, song: dict, tl_fps: float, handles: float = 0.0,
                        whole_clips: bool = False, frac: float = DRIFT_RESYNC_FRAC) -> Optional[TimelinePlan]:
    """One timeline per song. Group-owned segments place EVERY member covering the segment, each on its
    own V/A track pair with the event-timeline speed/slot logic; members keep their exact relative
    offsets (song time = event time + delta, delta from the song map at the segment start). Clip-owned
    segments (loner takes) are one plain 1:1 piece at song_map(segment start). Track order: groups by
    index (anchor first, then by coverage), then loner takes (full takes first, then snippets)."""
    segs = res.song_segments(song["id"])
    if not segs:
        return None
    tp = TimelinePlan(_safe_name("ShowSync Song - %s" % song["name"]), "song", song["id"], tl_fps)
    entries = []   # (sort key, seg, cid, comp map, c0, c1, label)
    for s in segs:
        sm = res.song_map(s)
        delta = sm(s["start"]) - s["start"]
        if s["owner_kind"] == "clip":
            cid = s["owner"]
            if cid not in res.clips or res.is_reference(cid):
                continue
            dur = res.clip_duration(cid)
            comp = TimeMap([0.0, max(dur, 1.0)], [delta, max(dur, 1.0) + delta])
            full = not s.get("is_partial")
            entries.append(((1, not full, -(s["end"] - s["start"])), s, cid, comp, s["start"], s["end"],
                            "%s %s" % ("take" if full else "snippet", res.clip_name(cid))))
            continue
        gi = next((k for k, x in enumerate(res.groups) if x["id"] == s["owner"]), None)
        if gi is None:
            continue
        g = res.groups[gi]
        found = False
        for cid, (s0, s1) in g["spans"].items():
            o0, o1 = max(s["start"], s0), min(s["end"], s1)
            if o1 <= o0 or res.is_reference(cid):
                continue
            found = True
            em = res.group_map(g["id"], cid)
            inv = em.inverse()
            comp = TimeMap(em.src, [v + delta for v in em.dst])
            entries.append(((0, gi, s["start"], cid != g["anchor"], -(o1 - o0), cid), s, cid, comp,
                            inv(o0), inv(o1), "G%02d %s" % (gi + 1, res.clip_name(cid))))
        if not found:
            tp.warnings.append("segment %s: no covering clip" % s["id"])
    entries.sort(key=lambda e: e[0])
    i = 0
    for _k, s, cid, comp, c0, c1, label in entries:
        cfps = res.clip_fps(cid, tl_fps) if res.has_video(cid) else tl_fps
        ef_ = exact_fps(cfps)
        dur = res.clip_duration(cid)
        if whole_clips:
            c0, c1 = 0.0, dur
        else:
            c0, c1 = max(0.0, c0 - handles), min(dur, c1 + handles)      # clamp to the media extent
        if comp(c0) < 0:                                                  # cut would start before song time 0
            c0 = comp.inverse()(0.0)
            if handles > 0 or whole_clips:
                tp.warnings.append("%s: head handle trimmed at song start" % res.clip_name(cid))
        nclip = int(math.floor(dur * ef_ + 1e-6))
        sf = max(0, int(math.ceil(c0 * ef_ - 1e-9)))
        ef = min(nclip, int(round(c1 * ef_)))
        if ef <= sf:
            tp.warnings.append("segment %s empty for %s" % (s["id"], res.clip_name(cid)))
            continue
        i += 1
        pieces = fit_pieces(comp, cfps, ef, tl_fps, frac, first_frame=sf)
        tp.track_names += [("video", i, label), ("audio", i, label)]
        for pc in pieces:
            for mt, ok in ((1, res.has_video(cid)), (2, res.has_audio(cid))):
                if ok:
                    tp.placements.append(Placement(cid, res.clip_path(cid), i, mt, pc.sf, pc.ef, pc.rec,
                                                   pc.drift_err_frames, pc.quant_err_frames,
                                                   speed_pct=pc.speed_pct, clip_fps=cfps, clip_frames=nclip,
                                                   record_end=pc.rec_end))
        sp = [pc.speed_pct for pc in pieces]
        if max(sp) - min(sp) > 1e-9 or abs(sp[0] - 100.0) > 1e-9:
            tp.warnings.append("%s: speed %.4f %%, %d piece(s)" % (res.clip_name(cid), sp[0], len(pieces)))
    tp.n_video = tp.n_audio = i
    ref = song.get("reference") or ""
    if ref.startswith("file:"):
        rpath = ref[5:]
        nfr = int(math.floor(float(song.get("duration", 0)) * exact_fps(tl_fps) + 1e-6))
        if nfr > 0:
            j = tp.n_audio + 1
            tp.n_audio = j
            tp.track_names.append(("audio", j, "REF " + os.path.basename(rpath)))
            tp.placements.append(Placement("", rpath, j, 2, 0, nfr, 0, optional=True, clip_fps=tl_fps,
                                           clip_frames=nfr, record_end=nfr))
    color = SONG_COLORS[(res.song_number(song["id"]) - 1) % len(SONG_COLORS)]
    tp.markers.append(MarkerOp(frame=0, name=song["name"], color=color,
                               note="song start; %d track pair(s), %.0f s" % (i, float(song.get("duration", 0))),
                               duration=1, custom="%s:song:%s" % (TAG, song["id"]), timeline=tp.name))
    tp.warnings.append("song placement is musical (approximate); group members keep their exact relative offsets")
    return tp


# --------------------------------------------------------------------------- master-night timelines

MULTICAM_NOTE = ("multicam clip skipped: MediaPool.CreateMulticamClip takes media-pool clips (not a timeline) "
                 "and would lose the speed warps and cuts built here; use the master timeline's tracks as angles")
MIN_PIECE_FRAMES = 2


def _safe_inv(m: TimeMap) -> TimeMap:
    """Inverse that tolerates flat parts (keeps the first knot of each flat run)."""
    src, dst = [], []
    for x, y in zip(m.src, m.dst):
        if dst and y <= dst[-1] + 1e-9:
            continue
        src.append(x)
        dst.append(y)
    if len(dst) < 2:
        return TimeMap([m.dst[0], m.dst[0] + 1.0], [m.src[0], m.src[0] + 1.0])
    return TimeMap(dst, src)


def _solve(m: TimeMap, y: float, lo: float, hi: float) -> float:
    """t in [lo, hi] with m(t) = y for a nondecreasing m (bisection)."""
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if m(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compose_master_map(em: Optional[TimeMap], sm_other: TimeMap, sm_master: TimeMap, cut0: float,
                       t_lo: float, t_hi: float) -> TimeMap:
    """clip time -> master-timeline time: (event map) -> owner time -> song time (this take's song map)
    -> master owner time (inverse of the master's song map) minus the master cut start. Piecewise
    linear with knots at every knot of any of the maps, restricted to [t_lo, t_hi]."""
    inv_em = _safe_inv(em) if em is not None else None
    inv_other = _safe_inv(sm_other)
    inv_master = _safe_inv(sm_master)
    to_owner = (lambda t: em(t)) if em is not None else (lambda t: t)
    from_owner = (lambda o: inv_em(o)) if em is not None else (lambda o: o)
    ks = {t_lo, t_hi}
    if em is not None:
        ks |= set(em.src)
    owner_knots = list(sm_other.src) + [inv_other(sig) for sig in sm_master.dst]
    ks |= {from_owner(o) for o in owner_knots}
    ts = []
    for k in sorted(k for k in ks if t_lo - 1e-12 <= k <= t_hi + 1e-12):
        if not ts or k - ts[-1] > 1e-9:
            ts.append(min(max(k, t_lo), t_hi))
    if len(ts) < 2:
        ts = [t_lo, t_lo + max(t_hi - t_lo, 1e-6)]
    return TimeMap(ts, [inv_master(sm_other(to_owner(t))) - cut0 for t in ts])


def _theil_sen(ts, ys):
    """Robust line (slope, intercept) through points: median of the pairwise slopes, median intercept."""
    sl = sorted((ys[b] - ys[a]) / (ts[b] - ts[a]) for a in range(len(ts)) for b in range(a + 1, len(ts))
                if ts[b] - ts[a] > 1e-9)
    if not sl:
        return 1.0, (ys[0] - ts[0] if ys else 0.0)
    slope = sl[len(sl) // 2] if len(sl) % 2 else 0.5 * (sl[len(sl) // 2 - 1] + sl[len(sl) // 2])
    ic = sorted(y - slope * t for t, y in zip(ts, ys))
    return slope, ic[len(ic) // 2]


def warp_runs(comp: TimeMap, max_warp_pct: float, window_s: float = 8.0, step: float = 0.5,
              dev_s: float = 0.25, persist_s: float = 4.0, stable_resid_s: float = 0.2):
    """Split the composite map into usable regions and structure breaks, robust against jittery
    (per-beat) knots. The map is sampled every `step` s and judged over ~`window_s` windows with a
    median-based line fit (jitter and one-beat outliers do not move it):
      * a region starts where a window has a speed (100 / slope) within +-max_warp_pct and a tight fit;
      * it ends at a sample that deviates more than `dev_s` from the trailing-window prediction AND
        whose deviation persists (>= 75 % of the following `persist_s` seconds) = a structure break
        (extra bars, skipped section, tempo beyond the limit); isolated deviations are ignored;
      * after a break the next region starts at the first stable window.
    Region edges snap to the map's own knots. Returns (regions [(t0, t1)], breaks [(t0, t1)])."""
    lo_s, hi_s = 100.0 * (1 - max_warp_pct / 100.0), 100.0 * (1 + max_warp_pct / 100.0)
    t0, t1 = comp.src[0], comp.src[-1]
    n = max(2, int(math.ceil((t1 - t0) / step)) + 1)
    ts = [min(t0 + k * step, t1) for k in range(n)]
    ts[-1] = t1
    ys = [comp(t) for t in ts]
    W = window_s

    def window(a, b):                         # sample indices with ts in [a, b]
        return [k for k in range(len(ts)) if a - 1e-9 <= ts[k] <= b + 1e-9]

    def stable(s):
        idx = window(ts[s], ts[s] + W)
        if len(idx) < 6:
            return None
        sl, ic = _theil_sen([ts[k] for k in idx], [ys[k] for k in idx])
        if sl <= 1e-9 or not (lo_s <= 100.0 / sl <= hi_s):
            return None
        if max(abs(ys[k] - (sl * ts[k] + ic)) for k in idx) > stable_resid_s:
            return None
        return sl, ic

    def snap_lo(t):                           # smallest knot >= t
        c = [k for k in comp.src if k >= t - 1e-9]
        return min(c) if c else t

    def snap_hi(t):                           # largest knot <= t
        c = [k for k in comp.src if k <= t + 1e-9]
        return max(c) if c else t

    regions, breaks = [], []
    k = 0
    pend = None                               # start time of an unusable stretch
    while k < n:
        st, fit0 = k, None
        while st < n:
            fit0 = stable(st)
            if fit0:
                break
            st += 1
        if st >= n:                           # nothing usable until the end
            if pend is None:
                pend = ts[k]
            breaks.append((pend, t1))
            break
        if st > k or pend is not None:
            breaks.append((ts[k] if pend is None else pend, snap_lo(ts[st])))
        pend = None
        good = [st]
        j, dev = st, None
        while j < n:
            if ts[j] - ts[st] <= W or len(good) < 3:
                sl, ic = fit0
            else:
                g = [q for q in good if ts[q] >= ts[j] - W]
                sl, ic = _theil_sen([ts[q] for q in g], [ys[q] for q in g]) if len(g) >= 3 else fit0
            if abs(ys[j] - (sl * ts[j] + ic)) > dev_s:
                fut = window(ts[j], ts[j] + persist_s)
                bad = sum(1 for q in fut if abs(ys[q] - (sl * ts[q] + ic)) > dev_s)
                if bad >= 0.75 * len(fut) and (len(fut) >= 3 or ts[j] + persist_s >= t1 - 1e-9):
                    dev = j
                    break
            else:
                good.append(j)
            j += 1
        e = (dev - 1) if dev is not None else n - 1
        r0, r1 = snap_lo(ts[st]), snap_hi(ts[e])
        if r1 > r0:
            regions.append((r0, r1))
        if dev is None:
            break
        pend = r1
        k = dev
    # merge breaks that touch
    out = []
    for b in sorted(breaks):
        if out and b[0] <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b[1]))
        else:
            out.append(b)
    return regions, [b for b in out if b[1] > b[0]]


def smooth_region_map(comp: TimeMap, r0: float, r1: float, window_s: float = 8.0, grid: float = 1.0,
                      step: float = 0.5) -> TimeMap:
    """Jitter-free version of the composite map over [r0, r1]: a median-based local line (+-window/2)
    evaluated on a coarse grid. Real tempo trends (slower than the window) survive; per-beat jitter and
    one-beat outlier knots do not."""
    n = max(2, int(math.ceil((r1 - r0) / step)) + 1)
    ts = [min(r0 + k * step, r1) for k in range(n)]
    ts[-1] = r1
    ys = [comp(t) for t in ts]
    gn = max(1, int(math.ceil((r1 - r0) / grid)))
    gts = [min(r0 + k * grid, r1) for k in range(gn + 1)]
    gts[-1] = r1
    out = []
    for g in gts:
        # full-width window, shifted inwards at the region edges (a one-sided half window gives a noisy slope)
        w0 = min(max(g - window_s / 2.0, r0), max(r0, r1 - window_s))
        idx = [k for k in range(n) if w0 - 1e-9 <= ts[k] <= w0 + window_s + 1e-9]
        if len(idx) >= 3:
            sl, ic = _theil_sen([ts[k] for k in idx], [ys[k] for k in idx])
            out.append(sl * g + ic)
        else:
            out.append(comp(g))
    for k in range(1, len(out)):             # keep it monotone
        out[k] = max(out[k], out[k - 1])
    return TimeMap(gts, out)


def _chord_err(m: TimeMap, a: float, b: float) -> float:
    """Largest deviation of the map from the straight chord between its values at a and b (seconds)."""
    ya, yb = m(a), m(b)
    if b <= a:
        return 0.0
    sl = (yb - ya) / (b - a)
    return max(abs(m(t) - (ya + sl * (t - a))) for t in _pts(m, a, b))


def fit_anchored_pieces(m: TimeMap, clip_fps: float, n_frames: int, tl_fps: float, tol_frames: float,
                        first_frame: int = 0, forced_frames=(), anchor_tau: Optional[dict] = None) -> list:
    """Pieces for an angle that must agree with its sisters. Each piece is the CHORD of the map between its
    boundary frames (continuous at joins, no fitting jumps), boundaries include `forced_frames` (the shared
    warp's boundaries, so every angle of the night starts a piece at the same master time and rounds the
    record position identically), and record positions come from the map value at the boundary. Speeds are
    derived from the slots (rec..rec_end), so a piece consumes exactly (ef - sf) source frames.
    anchor_tau {frame: master time}: for pieces starting at such a frame the record position is rounded from
    that shared master time instead of the angle's own map value, so sister angles round identically and
    only their (small) source-frame quantisation differs."""
    cfps, tfps = exact_fps(clip_fps), exact_fps(tl_fps)
    tol = tol_frames / tfps * (1 - 1e-9)
    forced = sorted(f for f in set(forced_frames) if first_frame < f < n_frames)
    bounds = []
    sf = first_frame
    while sf < n_frames:
        limit = next((f for f in forced if f > sf), n_frames)
        a = sf / cfps
        feas = lambda b: _chord_err(m, a, b) <= tol
        top = limit / cfps
        if feas(top):
            ef = limit
        else:
            cand = [k for k in m.src if a < k < top] + [top]
            lo_i, hi_i = -1, len(cand) - 1
            while hi_i - lo_i > 1:
                mid = (lo_i + hi_i) // 2
                if feas(cand[mid]):
                    lo_i = mid
                else:
                    hi_i = mid
            lo = a if lo_i < 0 else cand[lo_i]
            hi = cand[hi_i]
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if feas(mid):
                    lo = mid
                else:
                    hi = mid
            ef = min(limit, max(sf + 1, int(math.floor(lo * cfps + 1e-9))))
        bounds.append((sf, ef))
        sf = ef
    # merge tiny pieces (< 1.5 s) into a neighbour across a non-forced boundary when one chord still fits
    min_f = int(math.ceil(1.5 * cfps))
    k = 0
    while k < len(bounds) and len(bounds) > 1:
        b0, b1 = bounds[k]
        if b1 - b0 < min_f:
            for nb in (k + 1, k - 1):
                if 0 <= nb < len(bounds):
                    lo_b, hi_b = (bounds[k][0], bounds[nb][1]) if nb > k else (bounds[nb][0], bounds[k][1])
                    joint = bounds[k][1] if nb > k else bounds[k][0]
                    if joint in forced:
                        continue
                    if _chord_err(m, lo_b / cfps, hi_b / cfps) <= 2 * tol:
                        bounds[min(k, nb):max(k, nb) + 1] = [(lo_b, hi_b)]
                        k = max(0, min(k, nb) - 1)
                        break
            else:
                k += 1
            continue
        k += 1
    anchor_tau = anchor_tau or {}
    recs = [int(round((anchor_tau[b0] if b0 in anchor_tau else m(b0 / cfps)) * tfps)) for b0, _ in bounds]
    last_end = bounds[-1][1]
    end_rec = int(round((anchor_tau[last_end] if last_end in anchor_tau else m(last_end / cfps)) * tfps))
    for k in range(1, len(recs)):
        if recs[k] <= recs[k - 1]:
            recs[k] = recs[k - 1] + 1
    out = []
    for k, (b0, b1) in enumerate(bounds):
        rec = recs[k]
        rec_end = recs[k + 1] if k + 1 < len(recs) else max(end_rec, rec + 1)
        err = _chord_err(m, b0 / cfps, b1 / cfps) * tfps
        if k + 1 < len(bounds):                # followed by another piece: exact consumption from the slot
            speed = (b1 - b0) / cfps / ((rec_end - rec) / tfps) * 100.0
        else:                                  # last piece: the chord's own speed
            ya, yb = m(b0 / cfps), m(b1 / cfps)
            speed = 100.0 * (b1 - b0) / cfps / max(yb - ya, 1e-9)
        if abs(speed - 100.0) < 1e-6:
            speed = 100.0
        pc = Piece(b0, b1, rec, err, abs(rec - m(b0 / cfps) * tfps), speed)
        pc.rec_end = rec_end
        out.append(pc)
    return out


def _warp_from_pieces(m: TimeMap, pieces: list, fps: float) -> TimeMap:
    """Continuous map event time -> master time: the (de-jittered) map sampled at the piece boundaries, linear
    in between. No jumps, so every angle of the night composes with the same smooth warp."""
    f = exact_fps(fps)
    ks = [pieces[0].sf / f] + [pc.ef / f for pc in pieces]
    vs = [m(k) for k in ks]
    for k in range(1, len(vs)):
        vs[k] = max(vs[k], vs[k - 1])
    return TimeMap(ks, vs)


def build_master_timeline(res: Result, song: dict, tl_fps: float, handles: float = 0.0, whole_clips: bool = False,
                          master_id: Optional[str] = None, max_warp_pct: float = 5.0, other_audio: str = "off",
                          tol_frames: float = 1.5, frac: float = DRIFT_RESYNC_FRAC, member_frac: float = 0.25,
                          shared_warp: bool = True) -> Optional[TimelinePlan]:
    """Timeline time = the master take's time. The master night's angles are placed like in event
    timelines (video + audio, on the top tracks); every other night/take is mapped master time ->
    song time -> its own time and only its PICTURE is placed (audio off or on muted tracks), with speed
    changes within +-max_warp_pct, cut where the structure differs."""
    segs = res.song_segments(song["id"])
    if not segs:
        return None
    tp = TimelinePlan(_safe_name("ShowSync Master - %s" % song["name"]), "master", song["id"], tl_fps)
    length = lambda sg: sg["end"] - sg["start"]
    def is_master(owner):
        if not master_id:
            return False
        key = str(master_id).strip().lower()
        gi_ = next((k for k, x in enumerate(res.groups) if x["id"] == owner), None)
        names = {str(owner).lower()}
        if gi_ is not None:
            names.add("g%02d" % (gi_ + 1))
            names.add("g%d" % (gi_ + 1))
        elif owner in res.clips:
            names.add(res.clip_name(owner).lower())
        return key in names

    cands = [sg for sg in segs if is_master(sg["owner"])]
    if master_id and not cands:
        tp.warnings.append("master %r has no take of this song: using the longest" % master_id)
    M = max(cands or segs, key=length)
    sm_m = res.song_map(M)
    gM = next((x for x in res.groups if x["id"] == M["owner"]), None)
    if gM is not None:
        members = [c for c in gM["spans"] if not res.is_reference(c)]
        lo_all = min(gM["spans"][c][0] for c in members)
        hi_all = max(gM["spans"][c][1] for c in members)
    else:
        members = [M["owner"]]
        lo_all, hi_all = 0.0, res.clip_duration(M["owner"])
    if whole_clips:
        cut0, cut1 = lo_all, hi_all
    else:
        cut0, cut1 = max(lo_all, M["start"] - handles), min(hi_all, M["end"] + handles)
    L = cut1 - cut0
    master_label = ("G%02d" % (res.groups.index(gM) + 1)) if gM is not None else res.clip_name(M["owner"])

    # ---- master night: all angles, exactly as in the event timelines
    master_entries = []
    for cid in members:
        if gM is not None:
            s0, s1 = gM["spans"][cid]
            o0, o1 = max(cut0, s0), min(cut1, s1)
            if o1 <= o0:
                continue
            em = res.group_map(gM["id"], cid)
            inv = em.inverse()
            comp = TimeMap(em.src, [v - cut0 for v in em.dst])
            c0, c1 = inv(o0), inv(o1)
            key = (cid != gM["anchor"], -(o1 - o0), cid)
        else:
            comp = TimeMap([0.0, max(res.clip_duration(cid), 1.0)], [-cut0, max(res.clip_duration(cid), 1.0) - cut0])
            c0, c1, key = cut0, cut1, (False, 0.0, cid)
        master_entries.append((key, cid, comp, c0, c1))
    master_entries.sort(key=lambda e: e[0])

    # ---- other takes
    other = []     # (sort key, seg, cid, comp, t_lo, t_hi, label)
    for sg in segs:
        if sg is M:
            continue
        if sg["owner"] == M["owner"]:
            tp.warnings.append("second take of this song inside the master night skipped (its angles are already "
                               "the master tracks)")
            continue
        sm2 = res.song_map(sg)
        lo_o, hi_o = (0.0, None)
        if sg["owner_kind"] == "clip":
            cid = sg["owner"]
            if cid not in res.clips or res.is_reference(cid) or not res.has_video(cid):
                continue
            dur = res.clip_duration(cid)
            a0, a1 = (0.0, dur) if whole_clips else (max(0.0, sg["start"] - handles), min(dur, sg["end"] + handles))
            other.append(((1, -length(sg)), sg, cid, None, a0, a1, "take %s" % res.clip_name(cid)))
            continue
        gi = next((k for k, x in enumerate(res.groups) if x["id"] == sg["owner"]), None)
        if gi is None:
            continue
        g = res.groups[gi]
        for cid, (s0, s1) in g["spans"].items():
            if res.is_reference(cid) or not res.has_video(cid):
                continue
            w0 = s0 if whole_clips else max(s0, sg["start"] - handles)
            w1 = s1 if whole_clips else min(s1, sg["end"] + handles)
            if w1 <= w0:
                continue
            em = res.group_map(g["id"], cid)
            inv = em.inverse()
            other.append(((0, gi, cid != g["anchor"], -(w1 - w0), cid), sg, cid, em, inv(w0), inv(w1),
                          "G%02d %s" % (gi + 1, res.clip_name(cid))))
    other.sort(key=lambda e: e[0])

    slot = 0
    tfps = exact_fps(tl_fps)
    gw_cache = {}

    def group_warp(sg, g):
        """ONE warp per non-master group segment: master time <- the group's EVENT time (fitted once, with
        smoothing, windows and structure breaks). Every angle of the night is then placed by composing this
        single warp with its own exact event map, so same-night angles stay together."""
        if sg["id"] in gw_cache:
            return gw_cache[sg["id"]]
        gw_cache[sg["id"]] = None
        vids = [c for c in g["spans"] if not res.is_reference(c) and res.has_video(c)]
        if not vids:
            return None
        lo_e = min(g["spans"][c][0] for c in vids)
        hi_e = max(g["spans"][c][1] for c in vids)
        w0, w1 = (lo_e, hi_e) if whole_clips else (max(lo_e, sg["start"] - handles), min(hi_e, sg["end"] + handles))
        if w1 <= w0:
            return None
        sm2 = res.song_map(sg)
        full = compose_master_map(None, sm2, sm_m, cut0, w0, w1)
        e_lo = w0 if full(w0) >= 0 else _solve(full, 0.0, w0, w1)
        e_hi = w1 if full(w1) <= L else _solve(full, L, w0, w1)
        if e_hi - e_lo < 0.1:
            return None
        comp = compose_master_map(None, sm2, sm_m, cut0, e_lo, e_hi)
        regions, breaks = warp_runs(comp, max_warp_pct)
        wregions = []
        for r0, r1 in regions:
            sf = int(math.ceil(r0 * tfps - 1e-9))
            nf = int(math.floor(r1 * tfps + 1e-9))
            if nf - sf < MIN_PIECE_FRAMES:
                continue
            smooth = smooth_region_map(comp, r0, r1)
            gp = fit_pieces(smooth, tl_fps, nf, tl_fps, tol_frames, first_frame=sf, snap_knots=True,
                            merge_pct=0.2, min_piece_s=8.0)
            F = _warp_from_pieces(smooth, gp, tl_fps)
            wregions.append((sf / tfps, nf / tfps, F, gp))
        resid = 0.0
        for _r0, _r1, F, _gp in wregions:
            for q in range(201):
                e = F.src[0] + (F.src[-1] - F.src[0]) * q / 200.0
                resid = max(resid, abs(F(e) - comp(e)) * tfps)
        gp_all = [pc for w in wregions for pc in w[3]]
        gw_cache[sg["id"]] = {"comp": comp, "regions": wregions, "breaks": breaks, "resid": resid, "gp": gp_all}
        if gp_all:
            sp = [pc.speed_pct for pc in gp_all]
            tp.warnings.append("G%02d warp (shared by all angles): %d piece(s), speed %.2f-%.2f %%, %d structure "
                               "cut(s), max residual vs the song-map composite %.3f frames" % (
                                   res.groups.index(g) + 1, len(gp_all), min(sp), max(sp), len(breaks), resid))
        return gw_cache[sg["id"]]

    for _key, sg, cid, em, a0, a1, label in other:
        cfps = res.clip_fps(cid, tl_fps)
        ef_ = exact_fps(cfps)
        nclip = int(math.floor(res.clip_duration(cid) * ef_ + 1e-6))
        extra_note = ""
        if em is None or not shared_warp:
            # loner take (or shared_warp=False, the old per-angle fit kept for comparison): warp fitted
            # straight onto the clip
            sm2 = res.song_map(sg)
            full = compose_master_map(em, sm2, sm_m, cut0, a0, a1)
            t_lo = a0 if full(a0) >= 0 else _solve(full, 0.0, a0, a1)
            t_hi = a1 if full(a1) <= L else _solve(full, L, a0, a1)
            if t_hi - t_lo < MIN_PIECE_FRAMES / ef_:
                tp.warnings.append("%s: outside the master take, not placed" % label)
                continue
            comp = compose_master_map(em, sm2, sm_m, cut0, t_lo, t_hi)
            regions, breaks = warp_runs(comp, max_warp_pct)
            pieces = []
            for r0, r1 in regions:
                sf = max(0, int(math.ceil(r0 * ef_ - 1e-9)))
                nf = min(nclip, int(math.floor(r1 * ef_ + 1e-9)))
                if nf - sf < MIN_PIECE_FRAMES:
                    continue
                smooth = smooth_region_map(comp, r0, r1)          # fit against the de-jittered map
                pieces += fit_pieces(smooth, cfps, nf, tl_fps, tol_frames, first_frame=sf, snap_knots=True,
                                     merge_pct=0.2, min_piece_s=8.0)
            ncuts = len(breaks)
        else:
            g = next(x for x in res.groups if sg["owner"] == x["id"])
            gw = group_warp(sg, g)
            if gw is None or not gw["regions"]:
                tp.warnings.append("%s: not placed (outside the master take or needs more than +-%.1f %% speed "
                                   "change)" % (label, max_warp_pct))
                continue
            inv_em = _safe_inv(em)
            s0, s1 = g["spans"][cid]
            pieces, n_group = [], 0
            for r0, r1, F, gp in gw["regions"]:
                e0, e1 = max(r0, s0), min(r1, s1)
                if e1 - e0 < MIN_PIECE_FRAMES / ef_:
                    continue
                c_lo, c_hi = inv_em(e0), inv_em(e1)
                ks = {c_lo, c_hi} | {k for k in em.src if c_lo < k < c_hi}
                ks |= {inv_em(k) for k in F.src if e0 < k < e1}
                ts_ = []
                for k in sorted(ks):
                    if not ts_ or k - ts_[-1] > 1e-9:
                        ts_.append(k)
                if len(ts_) < 2:
                    continue
                mm = TimeMap(ts_, [F(min(max(em(k), e0), e1)) for k in ts_])      # clip time -> master time
                sf = max(0, int(math.ceil(c_lo * ef_ - 1e-9)))
                nf = min(nclip, int(math.floor(c_hi * ef_ + 1e-9)))
                if nf - sf < MIN_PIECE_FRAMES:
                    continue
                anchors = {int(round(inv_em(k) * ef_)): F(k) for k in F.src if e0 + 1e-6 < k < e1 - 1e-6}
                if abs(e0 - r0) < 1e-6:                # the member covers the region start: shared start too
                    anchors[sf] = F(e0)
                if abs(e1 - r1) < 1e-6:
                    anchors[nf] = F(e1)
                pieces += fit_anchored_pieces(mm, cfps, nf, tl_fps, member_frac, first_frame=sf,
                                              forced_frames=[f for f in anchors if sf < f < nf], anchor_tau=anchors)
                n_group += sum(1 for pc in gp if pc.ef / tfps > e0 and pc.sf / tfps < e1)
            comp = gw["comp"]
            ncuts = len(gw["breaks"])
            extra_note = ", %d extra piece(s) beyond the group warp" % max(0, len(pieces) - n_group)
        for pc in pieces:                                      # report the residual against the raw composite map
            a = pc.sf / ef_
            ts_ = [a + (pc.ef / ef_ - a) * q / 40.0 for q in range(41)]
            if em is None or not shared_warp:
                pc.drift_err_frames = max(abs(pc.rec / tfps + (t - a) * 100.0 / pc.speed_pct - comp(t))
                                          for t in ts_) * tfps
            else:
                pc.drift_err_frames = max(abs(pc.rec / tfps + (t - a) * 100.0 / pc.speed_pct -
                                              comp(min(max(em(t), comp.src[0]), comp.src[-1]))) for t in ts_) * tfps
        if not pieces:
            tp.warnings.append("%s: not placed (needs more than +-%.1f %% speed change, %d structure cut(s))" % (
                label, max_warp_pct, len(breaks) if (em is None or not shared_warp) else len(gw["breaks"])))
            continue
        slot += 1
        tp.track_names.append(("video", slot, label))
        for pc in pieces:
            tp.placements.append(Placement(cid, res.clip_path(cid), slot, 1, pc.sf, pc.ef, pc.rec,
                                           pc.drift_err_frames, pc.quant_err_frames, speed_pct=pc.speed_pct,
                                           clip_fps=cfps, clip_frames=nclip, record_end=pc.rec_end))
            if other_audio == "muted" and res.has_audio(cid):
                tp.placements.append(Placement(cid, res.clip_path(cid), slot, 2, pc.sf, pc.ef, pc.rec,
                                               pc.drift_err_frames, pc.quant_err_frames, speed_pct=pc.speed_pct,
                                               clip_fps=cfps, clip_frames=nclip, record_end=pc.rec_end))
        if other_audio == "muted" and res.has_audio(cid):
            tp.track_names.append(("audio", slot, label + " (muted)"))
            tp.muted_audio_slots.append(slot)
        sp = [pc.speed_pct for pc in pieces]
        tp.warnings.append("%s: %d piece(s)%s, speed %.2f-%.2f %%, %d structure cut(s), max residual %.3f frames" % (
            label, len(pieces), extra_note, min(sp), max(sp), ncuts, max(pc.drift_err_frames for pc in pieces)))

    # ---- master night on the top tracks
    for _key, cid, comp, c0, c1 in master_entries:
        cfps = res.clip_fps(cid, tl_fps) if res.has_video(cid) else tl_fps
        ef_ = exact_fps(cfps)
        nclip = int(math.floor(res.clip_duration(cid) * ef_ + 1e-6))
        sf = max(0, int(math.ceil(c0 * ef_ - 1e-9)))
        ef = min(nclip, int(round(c1 * ef_)))
        if ef - sf < MIN_PIECE_FRAMES:
            continue
        slot += 1
        label = "MASTER %s %s" % (master_label, res.clip_name(cid))
        tp.track_names += [("video", slot, label), ("audio", slot, label)]
        if res.has_audio(cid):
            tp.master_audio_slots.append(slot)
        pieces = fit_anchored_pieces(comp, cfps, ef, tl_fps, member_frac, first_frame=sf)
        for pc in pieces:
            for mt, ok in ((1, res.has_video(cid)), (2, res.has_audio(cid))):
                if ok:
                    tp.placements.append(Placement(cid, res.clip_path(cid), slot, mt, pc.sf, pc.ef, pc.rec,
                                                   pc.drift_err_frames, pc.quant_err_frames, speed_pct=pc.speed_pct,
                                                   clip_fps=cfps, clip_frames=nclip, record_end=pc.rec_end))
        sp = [pc.speed_pct for pc in pieces]
        tp.warnings.append("%s: master angle, %d piece(s), speed %.4f-%.4f %%, max residual %.3f frames" % (
            label, len(pieces), min(sp), max(sp), max(pc.drift_err_frames for pc in pieces)))
    tp.n_video = max([p.track for p in tp.placements if p.media_type == 1] or [0])
    tp.n_audio = max([p.track for p in tp.placements if p.media_type == 2] or [0])
    color = SONG_COLORS[(res.song_number(song["id"]) - 1) % len(SONG_COLORS)]
    tp.markers.append(MarkerOp(frame=0, name=song["name"], color=color,
                               note="master: %s, %.0f s" % (master_label, L), duration=1,
                               custom="%s:song:%s" % (TAG, song["id"]), timeline=tp.name))
    tp.warnings.append("master timeline time = %s's time; other takes follow the musical song maps (approximate)" % master_label)
    return tp


# --------------------------------------------------------------------------- media pool markers

def build_song_clip_markers(res: Result, tl_fps: float) -> list:
    out = []
    for s in res.segments:
        if not s.get("song_id") or res.song_map(s) is None:
            continue
        if s["owner_kind"] == "clip":
            cid = s["owner"]
            if cid in res.clips and not res.is_reference(cid):
                f = res.clip_fps(cid, tl_fps)
                out.append(song_marker(res, s, f, clip_id=cid, path=res.clip_path(cid), clip_time=s["start"]))
        else:
            g = next((x for x in res.groups if x["id"] == s["owner"]), None)
            if not g:
                continue
            for cid, (s0, s1) in g["spans"].items():
                o0, o1 = max(s["start"], s0), min(s["end"], s1)
                if o1 <= o0 or res.is_reference(cid):
                    continue
                em = res.group_map(g["id"], cid).inverse()
                f = res.clip_fps(cid, tl_fps)
                out.append(song_marker(res, s, f, clip_id=cid, path=res.clip_path(cid),
                                       clip_time=em(o0), clip_end=em(o1)))
    return out


# --------------------------------------------------------------------------- timecode plan

def _tc_context(res: Result, cid: str, tl_fps: float):
    fps = res.clip_fps(cid, tl_fps) if res.has_video(cid) else tl_fps
    old = res.clips[cid].get("start_tc")
    drop = bool(old) and tc_is_drop(old) and supports_drop(fps)
    return fps, old, drop


def build_tc_plan(res: Result, mode: str, tl_fps: float):
    """Returns (ops, untouched). mode: 'event' | 'song'."""
    ops, untouched = [], []
    if mode == "event":
        for gi, g in enumerate(res.groups):
            if g.get("anchor_tc"):
                afps = res.clip_fps(g["anchor"], tl_fps) if res.has_video(g["anchor"]) else tl_fps
                base = tc_to_seconds(g["anchor_tc"], afps)
            else:
                base = ((gi + 1) % 24) * 3600.0
            for cid in g["maps"]:
                if res.is_reference(cid):
                    continue
                fps, old, drop = _tc_context(res, cid, tl_fps)
                t0 = res.group_map(g["id"], cid)(0.0)
                new = seconds_to_tc(base + t0, fps, drop)
                ops.append(TcOp(cid, res.clip_path(cid), "event", old, new, fps, drop))
        grouped = {c for g in res.groups for c in g["maps"]}
        for cid in res.clips:
            if cid not in grouped and not res.is_reference(cid):
                untouched.append({"clip_id": cid, "path": res.clip_path(cid), "reason": "not in an event group"})
    elif mode == "song":
        eps = 0.25
        hit = {}
        for s in res.segments:
            if not s.get("song_id") or res.song_map(s) is None:
                continue
            sm = res.song_map(s)
            if s["owner_kind"] == "clip":
                cid = s["owner"]
                if cid in res.clips and not res.is_reference(cid) and s["start"] <= eps and s["end"] >= res.clip_duration(cid) - eps:
                    hit.setdefault(cid, []).append((s, sm(0.0)))
            else:
                g = next((x for x in res.groups if x["id"] == s["owner"]), None)
                if not g:
                    continue
                for cid, (s0, s1) in g["spans"].items():
                    if s0 >= s["start"] - eps and s1 <= s["end"] + eps:
                        hit.setdefault(cid, []).append((s, sm(s0)))
        for cid in res.clips:
            if res.is_reference(cid):
                continue
            fps, old, drop = _tc_context(res, cid, tl_fps)
            h = hit.get(cid, [])
            songs = {s["song_id"] for s, _ in h}
            if len(songs) != 1:
                untouched.append({"clip_id": cid, "path": res.clip_path(cid),
                                  "reason": "no single song covers the whole clip" if not songs else "spans several songs"})
                continue
            s, st = h[0]
            if st < 0:
                untouched.append({"clip_id": cid, "path": res.clip_path(cid), "reason": "negative song time"})
                continue
            base = (res.song_number(s["song_id"]) % 24) * 3600.0
            ops.append(TcOp(cid, res.clip_path(cid), "song", old, seconds_to_tc(base + st, fps, drop), fps, drop))
    else:
        raise ValueError("tc mode must be 'event' or 'song'")
    return ops, untouched


# --------------------------------------------------------------------------- marker propagation

def propagate_markers(res: Result, source: str, markers: list, src_fps: float, tl_fps: float = 24.0) -> list:
    """Copy markers (dicts: frame, name, color, note, duration; frames at src_fps, clip-relative) from
    clip `source` to every other clip: exact via event maps in the same group, else via song maps.
    Returns MarkerOp list (frames at each target clip's own fps)."""
    out = []
    sg = res.group_of(source)
    sfps = exact_fps(src_fps)
    for mk in markers:
        t = mk["frame"] / sfps
        dur_s = max(1, int(mk.get("duration", 1))) / sfps
        done = set()

        def emit(cid, tt, via):
            if cid == source or cid in done or res.is_reference(cid):
                return
            if not (-1e-6 <= tt <= res.clip_duration(cid) + 1e-6):
                return
            f = res.clip_fps(cid, tl_fps) if res.has_video(cid) else tl_fps
            out.append(MarkerOp(frame=max(0, sec_to_frames(tt, f)), name=mk.get("name", ""),
                                color=mk.get("color", "Blue"), note=mk.get("note", ""),
                                duration=max(1, sec_to_frames(dur_s, f)),
                                custom="%s:prop:%s" % (TAG, source), clip_id=cid,
                                path=res.clip_path(cid), via=via))
            done.add(cid)

        te = None
        if sg:
            te = res.group_map(sg["id"], source)(t)
            for cid in sg["maps"]:
                emit(cid, res.group_map(sg["id"], cid).inverse()(te), "event")
        # song route
        song_time = None
        for s in res.segments:
            if not s.get("song_id") or res.song_map(s) is None:
                continue
            if s["owner_kind"] == "clip" and s["owner"] == source and s["start"] <= t <= s["end"]:
                song_time, sid = res.song_map(s)(t), s["song_id"]
                break
            if s["owner_kind"] == "group" and sg and s["owner"] == sg["id"] and s["start"] <= te <= s["end"]:
                song_time, sid = res.song_map(s)(te), s["song_id"]
                break
        if song_time is None:
            continue
        for s in res.song_segments(sid):
            ot = res.song_map(s).inverse()(song_time)
            if not (s["start"] - 1e-6 <= ot <= s["end"] + 1e-6):
                continue
            if s["owner_kind"] == "clip":
                emit(s["owner"], ot, "song")
            else:
                g = next((x for x in res.groups if x["id"] == s["owner"]), None)
                if g:
                    for cid, (s0, s1) in g["spans"].items():
                        if s0 - 1e-6 <= ot <= s1 + 1e-6:
                            emit(cid, res.group_map(g["id"], cid).inverse()(ot), "song")
    return out


# --------------------------------------------------------------------------- top-level plan

def build_apply_plan(res: Result, tl_fps: float, opts: ApplyOptions) -> ApplyPlan:
    plan = ApplyPlan(fps=tl_fps)
    if opts.event_timelines:
        for gi in range(len(res.groups)):
            plan.timelines.append(build_event_timeline(res, gi, tl_fps, opts.frac))
    if opts.song_timelines:
        for song in res.songs:
            owners = {s["owner"] for s in res.song_segments(song["id"])}
            if opts.song_align == "warp" and len(owners) > 1:
                # Takes from different nights/owners: one constant offset per take (aligned at the song
                # start) drifts with the tempo difference. Measured on real footage (2026-11-23): one
                # night +600 ms, another wandering 0.2-1.5 s. So stack them like a master timeline:
                # the longest take runs at its own time, every other take follows the song map,
                # speed-matched within max_warp_pct, its audio muted (warped audio would be pitched).
                tp = build_master_timeline(res, song, tl_fps, opts.handles_s, opts.whole_clips, None,
                                           opts.max_warp_pct, "muted", opts.master_tol_frames)
                if tp:
                    old = tp.name
                    tp.name = _safe_name("ShowSync Song - %s" % song["name"])
                    tp.kind = "song"
                    for mk in tp.markers:
                        if mk.timeline == old:
                            mk.timeline = tp.name
            else:
                tp = build_song_timeline(res, song, tl_fps, opts.handles_s, opts.whole_clips)
            if tp:
                plan.timelines.append(tp)
    if opts.master_mode:
        for song in res.songs:
            tp = build_master_timeline(res, song, tl_fps, opts.handles_s, opts.whole_clips, opts.master_group,
                                       opts.max_warp_pct, opts.other_audio, opts.master_tol_frames)
            if tp:
                plan.timelines.append(tp)
        if opts.multicam:
            plan.warnings.append(MULTICAM_NOTE)
    if opts.song_markers:
        plan.clip_markers += build_song_clip_markers(res, tl_fps)
    if opts.tc_mode in ("event", "song"):
        plan.tc_ops, plan.tc_untouched = build_tc_plan(res, opts.tc_mode, tl_fps)
    if opts.propagate_source and opts.propagate_markers:
        plan.clip_markers += propagate_markers(res, opts.propagate_source, opts.propagate_markers,
                                               opts.propagate_fps or tl_fps, tl_fps)
    return plan


# --------------------------------------------------------------------------- result summary rows

def footage_fps(res: Result, fallback: float = 25.0) -> float:
    """Timeline rate that suits the footage: the rate most of the video (by duration) plays at
    cleanly. A clip at twice a rate plays cleanly at it (50p tape on a 25p timeline), so 25 collects
    the 50 fps clips too. Real tour footage: phones/Canon/camcorders 25, tapes 50 -> 25."""
    weight: dict = {}
    for cid, c in res.clips.items():
        if res.is_reference(cid) or not res.has_video(cid):
            continue
        f = float(c.get("fps") or 0)
        if f <= 0:
            continue
        weight[round(f, 3)] = weight.get(round(f, 3), 0.0) + float(c.get("duration") or 0.0)
    if not weight:
        return fallback
    score = {r: w + weight.get(round(2 * r, 3), 0.0) for r, w in weight.items()}
    return max(score, key=lambda r: (score[r], -r))


def summarize_result(res: Result, tl_fps: float = 24.0) -> list:
    """One dict per clip for the review table."""
    rows = []
    best_conf = {}
    for m in res.matches:
        for c in (m["a"], m["b"]):
            best_conf[c] = max(best_conf.get(c, 0.0), float(m.get("confidence", 0.0)))
    conflicts_by_group = {g["id"]: g.get("conflicts", []) for g in res.groups}
    for cid, info in res.clips.items():
        g = res.group_of(cid)
        fps = res.clip_fps(cid, tl_fps) if info.get("has_video") else tl_fps
        row = {"clip_id": cid, "clip": res.clip_name(cid), "group": "", "offset_tc": "", "drift_ppm": "",
               "songs": "", "confidence": "", "repeat_ambiguous": "", "warnings": []}
        if g:
            m = res.group_map(g["id"], cid)
            row["group"] = "G%02d" % res.group_index(g["id"])
            row["offset_tc"] = seconds_to_tc(max(0.0, m(0.0)), fps)
            row["drift_ppm"] = "%.1f" % ((m.ratio - 1.0) * 1e6)
            if not m.is_linear:
                row["warnings"].append("nonlinear")
            row["warnings"] += conflicts_by_group[g["id"]]
            alts = (g.get("alt_shifts") or {}).get(cid)
            if alts:
                row["repeat_ambiguous"] = "yes (%d)" % len(alts)
                row["warnings"].append("repeat-ambiguous: also fits at %s s" % ", ".join("%+.1f" % a for a in alts))
            if res.is_reference(cid):
                row["clip"] += " [reference]"
        else:
            row["warnings"].append("no exact match")
        names = []
        for s in res.segments:
            if not s.get("song_id"):
                continue
            if s["owner"] == cid or (g and s["owner"] == g["id"]):
                n = res.song_name(s["song_id"])
                if n not in names:
                    names.append(n)
        row["songs"] = ", ".join(names)
        if cid in best_conf:
            row["confidence"] = "%.2f" % best_conf[cid]
        if info.get("is_vfr"):
            row["warnings"].append("VFR: " + (info.get("vfr_detail") or ""))
        row["warnings"] = "; ".join(w for w in row["warnings"] if w)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- backup / undo

@dataclass
class Backup:
    run_id: str
    created: str
    result_path: str = ""
    timelines: list = field(default_factory=list)       # created timeline names
    tc_changes: list = field(default_factory=list)      # {path, old_tc, new_tc}
    metadata: list = field(default_factory=list)        # {path, key, old, new}
    clip_markers: list = field(default_factory=list)    # {path, frame, custom}
    timeline_markers: list = field(default_factory=list)  # {timeline, frame, custom}
    undone: bool = False
    errors: list = field(default_factory=list)
    summary: list = field(default_factory=list)         # per-clip apply summary lines
    file: str = ""

    @staticmethod
    def new(result_path: str = "") -> "Backup":
        rid = time.strftime("%Y%m%d-%H%M%S")
        return Backup(rid, time.strftime("%Y-%m-%d %H:%M:%S"), result_path)

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.file or backup_path_for(self.result_path, self.run_id)
        self.file = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": EXPORT_VERSION, **asdict(self)}, f, indent=1)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str) -> "Backup":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d.pop("version", None)
        b = Backup(**d)
        b.file = path
        return b


def backup_path_for(result_path: str, run_id: str) -> str:
    base, _ = os.path.splitext(result_path or os.path.join(os.getcwd(), "showsync"))
    return "%s.backup-%s.json" % (base, run_id)


def find_latest_backup(result_path: str) -> Optional[str]:
    """Newest not-yet-undone backup next to the result JSON."""
    base, _ = os.path.splitext(result_path)
    d, stem = os.path.dirname(os.path.abspath(base)), os.path.basename(base)
    cands = sorted(f for f in os.listdir(d) if f.startswith(stem + ".backup-") and f.endswith(".json"))
    for f in reversed(cands):
        try:
            if not Backup.load(os.path.join(d, f)).undone:
                return os.path.join(d, f)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- settings

def appdata_dir() -> str:
    return os.environ.get("SHOWSYNC_APPDATA") or os.path.join(
        os.environ.get("APPDATA") or os.path.expanduser("~"), "ShowSync")


def settings_path() -> str:
    return os.path.join(appdata_dir(), "settings.json")


def default_settings(repo_root: Optional[str] = None) -> dict:
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = os.path.join(root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = os.path.join(root, ".venv", "bin", "python")
    return {"engine_python": py, "engine_root": root,
            "plugin_dir": os.path.join(appdata_dir(), "plugin"), "last_refs_dir": ""}


def load_settings() -> dict:
    s = default_settings()
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            s.update(json.load(f))
    except (OSError, ValueError):
        pass
    return s


def save_settings(s: dict) -> None:
    os.makedirs(appdata_dir(), exist_ok=True)
    with open(settings_path(), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)


# --------------------------------------------------------------------------- progress lines

def parse_progress(line: str) -> Optional[dict]:
    """One --jsonl-progress line -> dict(stage, done, total, msg) or None for non-JSON output."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d if isinstance(d, dict) and "stage" in d else None


# --------------------------------------------------------------------------- user hints (hints.json)

_SONG_RE = re.compile(r"^\s*song\s*:\s*(.+?)\s*$", re.I)


def marker_sync_points(clips: list) -> list:
    """clips: [{path, fps, markers:[{frame,name,...}]}], frames clip-relative at that clip's `fps`
    (for audio-only clips the caller passes the rate Resolve uses for their markers). Markers named
    '#...' with the same (case-insensitive) name on 2+ clips give a SyncPoint dict for every pair."""
    by_name = {}
    for c in clips:
        seen = set()
        fps = exact_fps(float(c["fps"]))
        for m in c.get("markers", []):
            nm = (m.get("name") or "").strip()
            if not nm.startswith("#") or len(nm) < 2:
                continue
            key = nm.lower()
            if key in seen:          # same name twice on one clip is ambiguous: first one wins
                continue
            seen.add(key)
            by_name.setdefault(key, []).append((c["path"], m["frame"] / fps, nm))
    out = []
    for key, lst in by_name.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (pa, ta, nm), (pb, tb, _) = lst[i], lst[j]
                out.append({"a": pa, "ta": ta, "b": pb, "tb": tb, "label": nm, "source": "marker"})
    return out


def song_tags_from_markers(path: str, fps: float, markers: list) -> list:
    """'song: Title' markers -> SongTag dicts. duration > 1 frame: marker range; else from the marker
    to the clip end (end None)."""
    f = exact_fps(float(fps))
    out = []
    for m in markers:
        mt = _SONG_RE.match(m.get("name") or "")
        if not mt:
            continue
        start = m["frame"] / f
        dur = int(m.get("duration", 1) or 1)
        end = (m["frame"] + dur) / f if dur > 1 else None
        out.append({"clip": path, "song": mt.group(1), "start": start, "end": end})
    return out


def hints_path_for(paths, directory: str) -> str:
    """One hints file per analysed source set (stable hash of the sorted normalised paths)."""
    import hashlib
    h = hashlib.sha1("\n".join(sorted(norm_path(p) for p in paths)).encode("utf-8")).hexdigest()[:10]
    return os.path.join(directory, "hints-%s.json" % h)


def hints_path_for_key(key: str, directory: str) -> str:
    """Hints file for an arbitrary key (the plugin uses the project name, so hints survive changing the
    selection between analyses)."""
    import hashlib
    return os.path.join(directory, "hints-%s.json" % hashlib.sha1(key.encode("utf-8")).hexdigest()[:10])


class Hints:
    """hints.json (showsync.models.Hints JSON shape). Marker-derived entries are refreshed on every
    Analyze; entries the user made (buttons, picks) are kept. Marker-derived song tags are remembered
    in the private top-level key 'plugin_marker_tags' (the engine ignores unknown keys)."""

    def __init__(self, d: Optional[dict] = None):
        d = d or {}
        self.sync_points = list(d.get("sync_points", []))
        self.song_tags = list(d.get("song_tags", []))
        self.same_song = [list(x) for x in d.get("same_song", [])]
        self.different_song = [list(x) for x in d.get("different_song", [])]
        self.different_event = [list(x) for x in d.get("different_event", [])]
        self.ignore = list(d.get("ignore", []))
        self.marker_tags = list(d.get("plugin_marker_tags", []))

    @staticmethod
    def load(path: str) -> "Hints":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Hints(json.load(f))
        except (OSError, ValueError):
            return Hints()

    def to_dict(self) -> dict:
        return {"version": 1, "sync_points": self.sync_points, "song_tags": self.song_tags,
                "same_song": self.same_song, "different_song": self.different_song,
                "different_event": self.different_event, "ignore": self.ignore,
                "plugin_marker_tags": self.marker_tags}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    def count(self) -> int:
        return (len(self.sync_points) + len(self.song_tags) + len(self.same_song)
                + len(self.different_song) + len(self.different_event) + len(self.ignore))

    def clear(self) -> None:
        self.__init__()

    @staticmethod
    def _add_pair(lst, a, b):
        key = frozenset((norm_path(a), norm_path(b)))
        if a != b and not any(frozenset(norm_path(x) for x in p) == key for p in lst):
            lst.append([a, b])

    def relate(self, kind: str, paths: list) -> int:
        """kind: same_song | different_song | different_event | ignore (ignore takes single clips).
        Returns the number of new entries."""
        n = 0
        if kind == "ignore":
            for p in paths:
                if norm_path(p) not in {norm_path(x) for x in self.ignore}:
                    self.ignore.append(p)
                    n += 1
            return n
        lst = getattr(self, kind)
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                before = len(lst)
                self._add_pair(lst, paths[i], paths[j])
                n += len(lst) - before
        return n

    def tag_song(self, paths: list, title: str) -> None:
        for p in paths:
            self.song_tags = [t for t in self.song_tags if not (
                norm_path(t["clip"]) == norm_path(p) and t.get("start") is None and t.get("end") is None
                and t["song"].lower() == title.lower())]
            self.song_tags.append({"clip": p, "song": title, "start": None, "end": None})

    def add_pick(self, point: dict) -> None:
        """A user pick replaces earlier picks for the same clip pair."""
        key = frozenset((norm_path(point["a"]), norm_path(point["b"])))
        self.sync_points = [s for s in self.sync_points if not (
            s.get("source") == "pick" and frozenset((norm_path(s["a"]), norm_path(s["b"]))) == key)]
        self.sync_points.append(point)

    def merge_marker_derived(self, points: list, tags: list) -> None:
        """Replace everything derived from markers by the fresh set; keep the user's own hints."""
        self.sync_points = [s for s in self.sync_points if s.get("source") != "marker"] + list(points)
        old = {json.dumps(t, sort_keys=True) for t in self.marker_tags}
        self.song_tags = [t for t in self.song_tags if json.dumps(t, sort_keys=True) not in old] + list(tags)
        self.marker_tags = list(tags)


# --------------------------------------------------------------------------- repeat-ambiguity picker

def alt_options(res: Result, cid: str, tl_fps: float) -> list:
    """Placement options of a repeat-ambiguous clip: the current placement plus every alternative.
    Each: {label, shift, event_start, tc} with tc = position on the group timeline (01:00:00:00 base)."""
    g = res.group_of(cid)
    if not g:
        return []
    shifts = (g.get("alt_shifts") or {}).get(cid) or []
    if not shifts:
        return []
    start = g["spans"][cid][0]
    out = []
    for k, sh in enumerate([0.0] + [float(x) for x in shifts]):
        t = start + sh
        if t < 0:
            continue
        tc = seconds_to_tc(3600.0 + t, tl_fps)
        out.append({"label": ("current" if k == 0 else "alternative %d" % k) + " @ " + tc,
                    "shift": sh, "event_start": t, "tc": tc})
    return out


def pick_sync_point(res: Result, cid: str, shift: float) -> dict:
    """SyncPoint (source 'pick') between the group anchor and `cid` that selects placement `shift`
    (0 = current): at the clip's middle, anchor time = maps[anchor]^-1(maps[clip](t) + shift)."""
    g = res.group_of(cid)
    if not g or cid == g["anchor"]:
        raise ValueError("clip has no group anchor to pin against")
    t_clip = res.clip_duration(cid) / 2.0
    ev = res.group_map(g["id"], cid)(t_clip) + float(shift)
    t_anchor = res.group_map(g["id"], g["anchor"]).inverse()(ev)
    return {"a": res.clip_path(g["anchor"]), "ta": t_anchor, "b": res.clip_path(cid), "tb": t_clip,
            "label": "pick %+.1fs" % shift, "source": "pick"}


def group_timeline_tc(res: Result, cid: str, shift: float, tl_fps: float) -> str:
    g = res.group_of(cid)
    return seconds_to_tc(3600.0 + g["spans"][cid][0] + shift, tl_fps)
