"""Shared data model. Every module speaks these types; nothing else crosses module seams.

Conventions (read docs/ARCHITECTURE.md before changing anything here):
  * All times are float seconds (float64). Never frames, never samples, outside a module.
  * A clip's "clip time" is seconds from the first decoded audio sample of that file.
  * TimeMap maps one time base to another. A linear TimeMap is  dst = ratio * src + offset.
  * Clip ids are stable strings (see media.probe.clip_id) derived from absolute path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

import numpy as np


# --------------------------------------------------------------------------- media

@dataclass
class MediaInfo:
    clip_id: str
    path: str
    duration: float                 # seconds, container duration
    has_audio: bool
    has_video: bool
    audio_sr: int = 0
    audio_channels: int = 0
    fps: float = 0.0                # nominal frame rate (r_frame_rate), 0 for audio-only
    fps_num: int = 0
    fps_den: int = 1
    is_vfr: bool = False            # frame timing is not constant (phones, screen recordings)
    vfr_detail: str = ""            # human readable reason, e.g. "avg 29.71 vs nominal 30, jitter 4.2 ms"
    start_tc: Optional[str] = None  # embedded timecode "HH:MM:SS:FF" if any
    creation_time: Optional[str] = None  # ISO 8601 from container tags if any
    width: int = 0
    height: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    device: str = ""                # make/model/serial when available, used to detect same-camera clips
    role: str = "clip"              # "clip" | "reference" (a reference song file the user supplied; it can
                                    # anchor an event group so event time = song time, sample-accurate)


# --------------------------------------------------------------------------- features

@dataclass
class Features:
    """Analysis features of one clip's audio. Arrays are float32.

    onset:      onset-strength envelope at config.ENV_RATE (100 Hz), zero-mean-ish, robust to EQ
    energy_db:  RMS level in dBFS at config.ENV_RATE
    chroma:     shape (12, n) at config.CHROMA_RATE (10 Hz), each column L2-normalised (zeros in silence)
    musicness:  shape (n,) at config.CHROMA_RATE, 0..1, high while music plays, low for talk/applause/silence
    """
    clip_id: str
    duration: float
    onset: np.ndarray
    energy_db: np.ndarray
    chroma: np.ndarray
    musicness: np.ndarray
    # percussive envelope at ENV_RATE: kick band (40-150 Hz) + snare/hat band (2-5 kHz at 8 kHz
    # decode: 2-3.9 kHz) spectral flux, whitened. Survives crowd/tape noise better than `onset`
    # for cross-night beat alignment. Optional: None for caches made before it existed.
    perc: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- time maps

@dataclass
class TimeMap:
    """Monotonic piecewise-linear map src -> dst.

    knots_src/knots_dst are equal-length increasing arrays (len >= 2). Outside the knots the
    map extrapolates with the first/last segment's slope. A pure linear map has 2 knots.
    """
    knots_src: np.ndarray
    knots_dst: np.ndarray

    @staticmethod
    def linear(ratio: float, offset: float, src_lo: float = 0.0, src_hi: float = 1.0) -> "TimeMap":
        s = np.array([src_lo, src_hi], dtype=np.float64)
        return TimeMap(s, ratio * s + offset)

    @staticmethod
    def identity() -> "TimeMap":
        return TimeMap.linear(1.0, 0.0)

    def __call__(self, t):
        return _pwl(np.asarray(t, dtype=np.float64), self.knots_src, self.knots_dst)

    def inverse(self) -> "TimeMap":
        return TimeMap(self.knots_dst.copy(), self.knots_src.copy())

    def compose(self, inner: "TimeMap") -> "TimeMap":
        """Return self(inner(t)). Knots are the union of inner's knots and inner^-1(self's knots)."""
        s = np.union1d(inner.knots_src, inner.inverse()(self.knots_src))
        return TimeMap(s, self(inner(s)))

    def cropped(self, lo: float, hi: float) -> "TimeMap":
        """Same map restricted to src [lo, hi]: knots inside plus exact end points. Used so a
        clip's map (and its ratio) describes the clip itself, not knots inherited from a long
        tape map it was composed through."""
        if hi <= lo:
            return self
        inner = self.knots_src[(self.knots_src > lo) & (self.knots_src < hi)]
        s = np.concatenate([[lo], inner, [hi]])
        return TimeMap(s, np.asarray(self(s), dtype=np.float64))

    @property
    def is_linear(self) -> bool:
        if len(self.knots_src) <= 2:
            return True
        slopes = np.diff(self.knots_dst) / np.diff(self.knots_src)
        return bool(np.allclose(slopes, slopes[0], rtol=0, atol=1e-9))

    @property
    def ratio(self) -> float:
        """Average slope over the knot span (dst seconds per src second)."""
        return float((self.knots_dst[-1] - self.knots_dst[0]) / (self.knots_src[-1] - self.knots_src[0]))

    @property
    def offset(self) -> float:
        """dst value at src = 0 (by extrapolation)."""
        return float(self(0.0))

    def to_dict(self) -> dict:
        return {"src": self.knots_src.tolist(), "dst": self.knots_dst.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "TimeMap":
        return TimeMap(np.asarray(d["src"], dtype=np.float64), np.asarray(d["dst"], dtype=np.float64))


def _pwl(t, xs, ys):
    t = np.asarray(t, dtype=np.float64)
    y = np.interp(t, xs, ys)
    lo_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    hi_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(t < xs[0], ys[0] + (t - xs[0]) * lo_slope, y)
    y = np.where(t > xs[-1], ys[-1] + (t - xs[-1]) * hi_slope, y)
    return y if y.ndim else float(y)


# --------------------------------------------------------------------------- sync results

@dataclass
class PairMatch:
    """Sample-level sync between clip a and clip b:  t_a = map(t_b).

    kind:
      "exact"  - same sound event (same show, or same playback track). map is sample-accurate.
    confidence 0..1 (see config.MATCH_ACCEPT). overlap_s is how much of the two clips overlaps.
    residual_ms is the RMS of fine-sync measurements around the fitted map.
    n_windows / n_inliers count the fine-sync windows used.
    """
    a: str
    b: str
    map: TimeMap
    confidence: float
    overlap_s: float
    residual_ms: float
    n_windows: int
    n_inliers: int
    nonlinear: bool = False          # True if a piecewise map was needed (Hi8 wow, VFR, dropouts)
    kind: str = "exact"
    note: str = ""
    # Identical repeats (e.g. copy-pasted choruses on a playback track) can make several
    # placements equally exact. Then ambiguous=True, `map` is one of them and `alternatives`
    # holds the others; each is sample-accurate. The group builder chooses between them using
    # the rest of the graph. `confidence` is the confidence that ONE OF them is right.
    ambiguous: bool = False
    alternatives: list = field(default_factory=list)   # list[TimeMap]


@dataclass
class EventGroup:
    """Clips that heard the same sound, placed on one shared "event time" line.

    maps[clip_id] maps clip time -> event time. Event time 0 is the earliest covered moment.
    anchor is the clip whose time base defines event time (ratio 1). If anchor_tc is set the
    anchor had real time-of-day timecode and event time 0 corresponds to it.
    """
    id: str
    anchor: str
    maps: dict[str, TimeMap]
    spans: dict[str, tuple[float, float]]   # clip_id -> (event_start, event_end)
    edges: list[PairMatch]                  # accepted matches used to build the group
    conflicts: list[str] = field(default_factory=list)  # human readable problems
    anchor_tc: Optional[str] = None
    # clip_id -> other event-time shifts (seconds) where that clip fits equally well
    # (identical repeated material). The plugin can mark them; maps[] holds the chosen one.
    alt_shifts: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(e for _, e in self.spans.values()) if self.spans else 0.0


# --------------------------------------------------------------------------- music

@dataclass
class SongSegment:
    """A stretch of one clip (or of a group's event time) that holds one song.

    owner is a clip_id or a group id; start/end are in that owner's time base.
    song_map maps owner time -> song time (seconds into the song's reference), monotonic,
    approximate (musical alignment, typically within ~0.1-0.3 s for live takes).
    """
    id: str
    owner: str
    owner_kind: str                 # "clip" | "group"
    start: float
    end: float
    song_id: Optional[str] = None
    song_map: Optional[TimeMap] = None
    match_score: float = 0.0        # 0..1 similarity to the song's reference
    is_partial: bool = False        # snippet that does not cover the whole song
    beat_synced: bool = False       # song_map is snapped beat-to-beat (cross-night cuts land on the beat)
    beat_report: Optional[dict] = None  # {beat_synced, snapped_frac, bpm, reliable, base_reliable}
    breaks: list = field(default_factory=list)  # owner times where the song structure differs from the
                                                # base (extra/skipped bars): cut here, stretch elsewhere


@dataclass
class Song:
    id: str
    name: str                       # reference file stem, or "Song 01"...
    reference: str                  # "file:<path>" or "segment:<segment id>" (the master take)
    duration: float
    segments: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- whole result

@dataclass
class AnalysisResult:
    version: int
    clips: dict[str, MediaInfo]
    groups: list[EventGroup]
    loners: list[str]               # clips that matched nothing exactly
    segments: list[SongSegment]
    songs: list[Song]
    matches: list[PairMatch]        # every accepted pair, including ones not used in groups
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(self), indent=1)

    @staticmethod
    def from_json(text: str) -> "AnalysisResult":
        d = json.loads(text)
        tm = lambda x: TimeMap.from_dict(x) if x is not None else None
        pm = lambda m: PairMatch(**{**m, "map": tm(m["map"]),
                                    "alternatives": [tm(x) for x in m.get("alternatives", [])]})
        return AnalysisResult(
            version=d["version"],
            clips={k: MediaInfo(**v) for k, v in d["clips"].items()},
            groups=[EventGroup(**{**g,
                                  "maps": {k: tm(v) for k, v in g["maps"].items()},
                                  "spans": {k: tuple(v) for k, v in g["spans"].items()},
                                  "edges": [pm(e) for e in g["edges"]]}) for g in d["groups"]],
            loners=d["loners"],
            segments=[SongSegment(**{**s, "song_map": tm(s["song_map"])}) for s in d["segments"]],
            songs=[Song(**s) for s in d["songs"]],
            matches=[pm(m) for m in d["matches"]],
            warnings=d.get("warnings", []),
        )


def _to_jsonable(o):
    if isinstance(o, TimeMap):
        return o.to_dict()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return {k: _to_jsonable(getattr(o, k)) for k in o.__dataclass_fields__}
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    return o


# --------------------------------------------------------------------------- user hints
# Written by the Resolve plugin (or by hand) as hints.json next to the result, passed to
# `showsync analyze --hints`. Clips are referenced by FILE PATH (stable across runs); the
# pipeline converts to clip ids. Hints are hard rules: they override what the audio says.

@dataclass
class SyncPoint:
    """Moment ta (seconds, clip time) in clip a is the same moment as tb in clip b.

    From same-named markers (an iconic drum hit marked on two clips), from the user picking one
    of several identical-repeat placements, or typed by hand. Clips without audio can be synced
    only this way. Two or more points on one pair also give the drift between them.
    """
    a: str
    ta: float
    b: str
    tb: float
    label: str = ""
    source: str = "marker"          # "marker" | "pick" | "manual"


@dataclass
class SongTag:
    """Clip `clip` (optionally only clip time [start, end]) contains the song titled `song`."""
    clip: str
    song: str
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class Hints:
    version: int = 1
    sync_points: list[SyncPoint] = field(default_factory=list)
    song_tags: list[SongTag] = field(default_factory=list)
    same_song: list[list[str]] = field(default_factory=list)       # [[clip, clip], ...] contain the same song
    different_song: list[list[str]] = field(default_factory=list)  # never put these in one song
    different_event: list[list[str]] = field(default_factory=list) # never exact-sync these (e.g. two nights)
    ignore: list[str] = field(default_factory=list)                # leave these clips out entirely

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(self), indent=1)

    @staticmethod
    def from_json(text: str) -> "Hints":
        d = json.loads(text) if text.strip() else {}
        return Hints(
            version=d.get("version", 1),
            sync_points=[SyncPoint(**x) for x in d.get("sync_points", [])],
            song_tags=[SongTag(**x) for x in d.get("song_tags", [])],
            same_song=d.get("same_song", []), different_song=d.get("different_song", []),
            different_event=d.get("different_event", []), ignore=d.get("ignore", []),
        )
