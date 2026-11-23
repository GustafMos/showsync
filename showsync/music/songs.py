"""Music understanding: which song is this, where in the song are we, which clips share a song.

Public API (contract in docs/ARCHITECTURE.md):

    find_segments(owner, owner_kind, feats, start_offset=0.0, references=None) -> list[SongSegment]
    locate_references(feats, references, start_offset=0.0) -> [(name, start, end, score, hits, core0, core1)]
    song_novelty(chroma, onset) -> (times, novelty)          # tempo + key novelty at 1 Hz
    group_features(group, feats) -> Features            # fused onto event time
    crop_features(feats, start, end, start_offset=0.0) -> Features
    assign_songs(segments, seg_feats, references, allow_transpose=False,
                 similar=None, ref_accept=None, tags=None, must_link=None,
                 cannot_link=None, conflicts=None) -> list[Song]
    cluster_segments(segments, S, thresh, pre=None, cannot=None, labels=None) -> list[list[int]]
    propagate_points(times, from_owner, to_owner, segments, groups) -> list[float | None]
    propagate_points_json(result, times, from_owner, to_owner) -> list[float | None]   # stdlib only

Time bases
    * A segment's start/end are in its owner's time (clip time for loners, event time for groups).
    * ``feats`` passed to find_segments start at owner time ``start_offset`` (frame 0 of the
      10 Hz chroma is owner time start_offset).
    * ``seg_feats[seg.id]`` must be the owner features cropped to [seg.start, seg.end]
      (use crop_features), so frame 0 is owner time seg.start.
    * Song time: seconds into the reference file (with references) or into the master take
      (master segment start = song time 0).

Group features
    group_features marks event-time frames no clip covers with energy_db = NaN (musicness 0,
    chroma 0). find_segments treats such holes like the owner's start/end: music running into
    a hole is cut off, so the segment is partial.
"""
from __future__ import annotations

import bisect
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, label
from scipy.optimize import lsq_linear

from showsync.config import CHROMA_RATE, ENV_RATE, SONG_GAP_S, SONG_MIN_S, SONG_REF_ACCEPT, SONG_SIMILAR
from showsync.models import EventGroup, Features, Song, SongSegment, TimeMap
from showsync.music import dtw
from showsync.music import beats as beats_mod

# ---------------------------------------------------------------- module tunables
MUSIC_ON = 0.25            # hysteresis thresholds on (median-smoothed) musicness
MUSIC_OFF = 0.12
MUSIC_MEDIAN = 11          # median smoothing of musicness (frames at CHROMA_RATE)
MUSIC_MIN_S = 6.0          # less music than this is never a segment
SNIPPET_OWNER_S = 60.0     # owners shorter than this keep a short music stretch even if it touches nothing
FALLBACK_OWNER_S = 900.0   # clips up to this long with no detected music become one partial segment
EDGE_TOL_S = 1.5           # music within this of an owner edge / coverage hole counts as "touching"

SEGUE_WIN_S = 40.0         # context each side when testing a segue
SEGUE_MIN_PIECE_S = SONG_MIN_S
SEGUE_KEY_STRONG = 0.80    # centred key-profile distance (40 s each side) that means "another song"
SEGUE_TEMPO = 0.06         # folded |log tempo ratio| (~6 %) that blocks merging two runs
SEGUE_TEMPO_WIN_S = 20.0   # onset window each side for the tempo estimate

BG_WIN_S = 20.0            # per-pitch-class moving average removed before matching
MERGE_GAP_S = 12.0         # music runs this close may be re-joined (see _merge_runs)
MERGE_KEY = 0.30
PULSE_WIN_S = 6.0          # beat-clarity window
PULSE_GAP = 0.22           # beat clarity below this ...
PULSE_GAP_S = 5.0          # ... for this long is a candidate talk / applause gap
PULSE_INFORMATIVE = 0.35   # owner's 75th-percentile pulse inside music needed to trust the pulse
STRUCT_Q_S = 30.0          # piece length for the structural same-song test
STRUCT_SAME = 0.35         # DTW score above which two pieces count as one song
STRUCT_EDGE = 0.15         # an edge gap is trimmed only below this (drum-less intros often score 0.2-0.3)
CORE_MIN_S = 2 * SONG_MIN_S # segments this long cluster in stage A even when partial
MASTER_PARTIAL_WEIGHT = 0.8  # partial segments count 80 % of their length when picking the master
SNIP_RELAXED = 0.75        # see cluster_segments
SNIP_MARGIN = 2.5
NOV_TEMPO_WIN_S = 12.0     # tempogram window
NOV_L_S = 16               # checkerboard half-width (s, at 1 Hz)
NOV_STRONG = 1.40          # novelty that splits anywhere (synthetic within-song max 1.03)
NOV_PRIOR = 0.60           # novelty enough to split a piece longer than LONG_PIECE_S
LONG_PIECE_S = 330.0       # song-length prior: longer continuous music is split
NOV_MIN_PIECE_S = 60.0
CHAIN_CHUNK_S = 30.0       # reference chunk chains (live versions of studio songs, jams)
CHAIN_HOP_S = 15.0
CHAIN_HIT = 0.2
CHAIN_MIN_HITS = 3
CHAIN_ACCEPT = 0.06
REF_MERGE_GAP_S = 30.0
SNAP_S = 12.0              # reference cuts snap to a novelty peak this close
SNAP_MIN = 0.5
REF_EDGE_S = 15.0          # music-run edges move to a located reference edge this close
CHAIN_MIN_SEG_S = 60.0     # segments this long may match a reference by chunk chains
CHAIN_REF_ACCEPT = 0.08
CHAIN_MARGIN = 2.0
CHAIN_SIM_GAIN = 0.35 / 0.15
BEAT_MIN_FRAC = 0.2       # beat-snapped map needs this fraction of the take's beats paired
XNIGHT_ENV = "onset"        # cross-night alignment envelope: "onset" | "perc" | "both" (summed curves).
#   Measured on the real tour footage (song stage on result_fix2 groups): onset-aligned maps
#   read 10-15 ms for one song with BOTH an onset and a perc ruler; perc- or both-aligned
#   maps read 30-90 ms (perc ruler too) and lose sync on two other songs. Kept selectable.
BASE_LAG_CAP_MS = 600.0    # beat-base choice: a window that does not correlate costs this much
EXTEND_BARS = 8           # knot spacing (bars) of the smoothed DTW map outside the beat fit
BEAT_MIN_PAIRS = 24       # ... and at least this many beat pairs after outlier rejection
#       # a beat-snapped map needs this fraction of the take's beats
TAG_REF_MIN = 0.2          # DTW score at which a tagged take counts as matching its reference
SIM_RATE = 2.0             # CENS rate for similarity scoring (Hz)
SIM_WIN_S = 2.0
MAP_RATE = 5.0             # CENS rate for the song_map refinement
MAP_WIN_S = 1.0
MAP_KNOT_S = 2.0           # song_map knot spacing
MAP_TEMPO = (0.7, 1.4)     # allowed local slope of song_map (song s per owner s)
REFINE_MARGIN_S = 10.0


# =========================================================================== features helpers

def crop_features(feats: Features, start: float, end: float, start_offset: float = 0.0) -> Features:
    """Features of owner time [start, end] where ``feats`` frame 0 is owner time start_offset.

    The result's frame 0 is owner time ``start``; duration = end - start (clamped to the data).
    """
    a10 = max(0, int(round((start - start_offset) * CHROMA_RATE)))
    b10 = max(a10, int(round((end - start_offset) * CHROMA_RATE)))
    a100 = max(0, int(round((start - start_offset) * ENV_RATE)))
    b100 = max(a100, int(round((end - start_offset) * ENV_RATE)))
    ch = feats.chroma[:, a10:b10]
    return Features(clip_id=feats.clip_id,
                    duration=float(max(0.0, min(end, start_offset + feats.duration) - start)),
                    onset=feats.onset[a100:b100], energy_db=feats.energy_db[a100:b100],
                    chroma=ch, musicness=feats.musicness[a10:b10],
                    perc=None if getattr(feats, "perc", None) is None else feats.perc[a100:b100])


def group_features(group: EventGroup, feats: dict[str, Features]) -> Features:
    """Fuse the members' features onto event time (frame 0 = event time 0).

    Each member's 10 Hz chroma/musicness and 100 Hz onset/energy are resampled through the
    inverse of its clip->event TimeMap. Where several clips cover a moment they are averaged
    with weight 0.02 + musicness^2 (cleaner, more tonal sources dominate; crowd-noise phones
    fade out). The onset envelope is NOT averaged: per frame it comes from the covering member
    with the clearest beat (onset autocorrelation peak), resampled exactly at 100 Hz through its
    map, so beat tracking and onset alignment see real drum hits. ``perc`` (if the members have
    it) is carried the same way, from the same member. energy_db is the loudest member; NaN where no clip covers (coverage hole).
    """
    dur = float(group.duration)
    n10 = int(np.floor(dur * CHROMA_RATE)) + 1
    n100 = int(np.floor(dur * ENV_RATE)) + 1
    t10 = np.arange(n10) / CHROMA_RATE
    t100 = np.arange(n100) / ENV_RATE
    ch_acc = np.zeros((12, n10))
    w_acc = np.zeros(n10)
    mu_acc = np.zeros(n10)
    on_acc = np.zeros(n100)
    w100_acc = np.zeros(n100)
    on_best = np.zeros(n100)
    pe_best = np.zeros(n100)
    any_perc = False
    on_rank = np.full(n100, -1.0)          # beat clarity of the member that supplied on_best
    en = np.full(n100, -np.inf)
    for cid, m in group.maps.items():
        f = feats.get(cid)
        if f is None or f.chroma.shape[1] == 0:
            continue
        inv = m.inverse()
        lo, hi = group.spans.get(cid, (-np.inf, np.inf))
        # 10 Hz
        tc = np.asarray(inv(t10))
        nf = f.chroma.shape[1]
        fi = tc * CHROMA_RATE
        ok = (fi >= -0.5) & (fi <= nf - 0.5) & (t10 >= lo - 0.05) & (t10 <= hi + 0.05)
        if not ok.any():
            continue
        xs = np.arange(nf)
        ch = np.stack([np.interp(fi, xs, f.chroma[k]) for k in range(12)])
        mu = np.interp(fi, xs, f.musicness)
        w = np.where(ok, 0.02 + mu * mu, 0.0)
        ch_acc += ch * w
        mu_acc += mu * w
        w_acc += w
        # 100 Hz
        tc1 = np.asarray(inv(t100))
        ne = f.onset.shape[0]
        fe = tc1 * ENV_RATE
        ok1 = (fe >= -0.5) & (fe <= ne - 0.5) & (t100 >= lo - 0.05) & (t100 <= hi + 0.05)
        xe = np.arange(ne)
        on = np.interp(fe, xe, f.onset / (np.std(f.onset) + 1e-9))
        e = np.interp(fe, xe, f.energy_db)
        mu1 = np.interp(fe / ENV_RATE * CHROMA_RATE, xs, f.musicness)
        w1 = np.where(ok1, 0.02 + mu1 * mu1, 0.0)
        on_acc += on * w1
        w100_acc += w1
        # onset: the locally clearest-beat member wins frame by frame (no averaging across members:
        # different microphones / tape wobble smear drum hits when averaged)
        # local beat clarity (pulse curve, 10 s smoothing) of this member at each event frame
        from scipy.ndimage import uniform_filter1d
        pc = uniform_filter1d(_pulse_curve(f.onset, nf), size=int(10 * CHROMA_RATE), mode="nearest")
        clar = np.interp(fe / ENV_RATE * CHROMA_RATE, xs, pc)
        better = ok1 & (clar > on_rank)
        on_best = np.where(better, on, on_best)
        fp = getattr(f, "perc", None)
        if fp is not None and fp.size:
            any_perc = True
            pe = np.interp(fe, np.arange(fp.size), fp / (np.std(fp) + 1e-9))
        else:
            pe = on                       # member without perc: its onset stands in
        pe_best = np.where(better, pe, pe_best)
        on_rank = np.where(better, clar, on_rank)
        en = np.where(ok1, np.maximum(en, e), en)
    cov10 = w_acc > 0
    chroma = np.where(cov10[None, :], ch_acc / np.where(cov10, w_acc, 1.0)[None, :], 0.0)
    nrm = np.linalg.norm(chroma, axis=0)
    chroma = np.where(nrm[None, :] > 1e-9, chroma / np.where(nrm > 1e-9, nrm, 1.0)[None, :], 0.0)
    mus = np.where(cov10, mu_acc / np.where(cov10, w_acc, 1.0), 0.0)
    cov100 = w100_acc > 0
    onset = np.where(cov100, on_best, 0.0)
    energy = np.where(cov100, en, np.nan)
    return Features(clip_id=group.id, duration=dur, onset=onset.astype(np.float32),
                    energy_db=energy.astype(np.float32), chroma=chroma.astype(np.float32),
                    musicness=mus.astype(np.float32),
                    perc=np.where(cov100, pe_best, 0.0).astype(np.float32) if any_perc else None)


# =========================================================================== segmentation

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _key_profile(ch: np.ndarray) -> np.ndarray:
    """Mean chroma with its across-class mean removed, L2-normalised (the 'key' of a window).
    Centring matters: the log-compressed chroma has a large common floor."""
    v = ch.mean(axis=1).astype(np.float64)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def _beat_period(onset: np.ndarray) -> float:
    """Dominant beat period (s) from the onset autocorrelation, searched in 0.3..1.0 s."""
    x = np.asarray(onset, np.float64)
    x = x - x.mean()
    n = x.size
    lo, hi = int(0.3 * ENV_RATE), int(1.0 * ENV_RATE)
    if n < 2 * hi:
        return float("nan")
    F = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:hi]
    return (lo + int(np.argmax(ac[lo:hi]))) / ENV_RATE


def _tempo_change(pb: float, pa: float) -> float:
    """|log ratio| of two beat periods, folded so 2:1 and 3:2 (octave / meter errors) count as 0."""
    if not (np.isfinite(pa) and np.isfinite(pb)) or pa <= 0 or pb <= 0:
        return 0.0
    r = abs(np.log(pa / pb))
    return float(min(r, abs(r - np.log(2.0)), abs(r - np.log(1.5))))


def segue_profile(chroma: np.ndarray, onset: Optional[np.ndarray] = None) -> list[tuple[float, float, float]]:
    """Segue evidence along one music stretch (chroma at CHROMA_RATE, onset at ENV_RATE, both
    starting at the stretch start). Returns [(t, key_distance, tempo_change)] at 1 s steps for
    every t that leaves SEGUE_MIN_PIECE_S on both sides. Windows are SEGUE_WIN_S each side.
    """
    n = chroma.shape[1]
    dur = n / CHROMA_RATE
    out = []
    w = SEGUE_WIN_S
    for t in np.arange(SEGUE_MIN_PIECE_S, dur - SEGUE_MIN_PIECE_S + 1e-9, 1.0):
        a, b, c = int((t - w) * CHROMA_RATE), int(t * CHROMA_RATE), int((t + w) * CHROMA_RATE)
        kd = float(1.0 - _key_profile(chroma[:, max(0, a):b]) @ _key_profile(chroma[:, b:min(n, c)]))
        tc = 0.0
        if onset is not None and onset.size:
            i = int(t * ENV_RATE)
            span = int(SEGUE_TEMPO_WIN_S * ENV_RATE)
            tc = _tempo_change(_beat_period(onset[max(0, i - span):i]), _beat_period(onset[i:i + span]))
        out.append((float(t), kd, tc))
    return out


def _segue_splits(chroma: np.ndarray, onset: Optional[np.ndarray] = None) -> list[int]:
    """Frame indices (CHROMA_RATE, relative to the stretch) where one song segues into another.

    Conservative: split where the centred key profile of the SEGUE_WIN_S before vs after differs
    by >= SEGUE_KEY_STRONG. On 14 synthetic song pairs the largest within-song value was 0.70
    (intro -> verse, chorus with its own progression) and 11/14 segues exceeded 0.8 (the rest are
    left joined: under-splitting is preferred).
    Tempo is NOT used to split: the onset-autocorrelation beat period jumps between metrical
    levels (2:1, 3:2) inside songs, even with a beat-clarity gate (tried; it split 4 of 5
    synthetic songs). Same-key segues therefore stay joined (known limitation). ``onset`` is
    accepted for a future tempo tracker.
    Peaks are taken strongest first, at least SEGUE_MIN_PIECE_S apart and from the ends.
    """
    dur = chroma.shape[1] / CHROMA_RATE
    if dur < 2 * SEGUE_MIN_PIECE_S:
        return []
    prof = segue_profile(chroma)
    if not prof:
        return []
    t = np.array([p[0] for p in prof])
    kd = np.array([p[1] for p in prof])
    ok = kd >= SEGUE_KEY_STRONG
    chosen: list[float] = []
    for i in sorted(np.nonzero(ok)[0], key=lambda i: -kd[i]):
        if kd[i] < kd[max(0, i - 5):i + 6].max():
            continue                      # not the local peak of the key distance
        if any(abs(t[i] - c) < SEGUE_MIN_PIECE_S for c in chosen):
            continue
        chosen.append(float(t[i]))
    # refine each boundary with short windows (the long ones locate it only to a few seconds)
    n = chroma.shape[1]
    out = []
    for c in chosen:
        best, bt = -1.0, c
        for tt in np.arange(c - 6.0, c + 6.0 + 1e-9, 0.5):
            b = int(tt * CHROMA_RATE)
            w = int(12.0 * CHROMA_RATE)
            if b - w < 0 or b + w > n:
                continue
            d = float(1.0 - _key_profile(chroma[:, b - w:b]) @ _key_profile(chroma[:, b:b + w]))
            if d > best:
                best, bt = d, tt
        out.append(int(round(bt * CHROMA_RATE)))
    return sorted(out)


def _audible_extent(feats: Features, n: int) -> Optional[tuple[int, int]]:
    """(first, last+1) chroma frame whose level is within 40 dB of the clip's 95th percentile."""
    e = np.asarray(feats.energy_db, dtype=np.float64)
    if e.size == 0 or np.isnan(e).all():
        return None
    e = np.nan_to_num(e, nan=-200.0)
    on = np.nonzero(e > np.percentile(e, 95) - 40.0)[0]
    if on.size == 0:
        return None
    a = int(on[0] * CHROMA_RATE / ENV_RATE)
    b = min(n, int(np.ceil((on[-1] + 1) * CHROMA_RATE / ENV_RATE)))
    return (a, b) if b > a else None


def _merge_runs(runs: list[tuple[int, int]], feats: Features, covered: np.ndarray) -> list[tuple[int, int]]:
    """Re-join music runs that musicness split inside one song.

    Busy full-band sections can score low musicness for tens of seconds, which splits a song.
    Two neighbouring runs are merged when the gap is <= MERGE_GAP_S with no coverage hole, the
    key profiles of the SEGUE_WIN_S before and after agree (distance < MERGE_KEY), the gap's own
    chroma has the same key (< MERGE_KEY: applause / talk do not), and the tempo does not change.
    """
    if len(runs) < 2:
        return list(runs)
    C = feats.chroma
    w = int(SEGUE_WIN_S * CHROMA_RATE)
    span = int(SEGUE_TEMPO_WIN_S * ENV_RATE)
    r10 = ENV_RATE / CHROMA_RATE
    out = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = out[-1]
        gap = (a - pb) / CHROMA_RATE
        merge = False
        if gap <= MERGE_GAP_S and covered[pb:a].all():
            bef, aft, mid = C[:, max(pa, pb - w):pb], C[:, a:min(b, a + w)], C[:, pb:a]
            kd = 1.0 - _key_profile(bef) @ _key_profile(aft)
            kg = 1.0 - _key_profile(mid) @ _key_profile(np.concatenate([bef, aft], axis=1)) if a > pb else 0.0
            ib, ia = int(pb * r10), int(a * r10)
            tc = _tempo_change(_beat_period(feats.onset[max(0, ib - span):ib]),
                               _beat_period(feats.onset[ia:ia + span]))
            merge = kd < MERGE_KEY and kg < MERGE_KEY and tc < SEGUE_TEMPO
        if merge:
            out[-1] = (pa, b)
        else:
            out.append((a, b))
    return out


def _pulse_curve(onset: np.ndarray, n: int) -> np.ndarray:
    """Beat clarity per chroma frame: max normalised onset autocorrelation at lags 0.25-1.5 s
    in PULSE_WIN_S windows (1 s hop), interpolated to CHROMA_RATE. ~0.4-0.7 while a band
    plays with drums, ~0.1-0.2 for applause and talk (synthetic concerts, pro feed)."""
    x = np.asarray(onset, np.float64)
    W, H = int(PULSE_WIN_S * ENV_RATE), int(ENV_RATE)
    if x.size < W + H:
        return np.zeros(n)
    starts = np.arange(0, x.size - W, H)
    fr = np.lib.stride_tricks.sliding_window_view(x, W)[starts]
    fr = fr - fr.mean(axis=1, keepdims=True)
    F = np.fft.rfft(fr, 2 * W, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :int(1.5 * ENV_RATE) + 1]
    ac = ac / (ac[:, :1] + 1e-9)
    p = ac[:, int(0.25 * ENV_RATE):].max(axis=1)
    tc = (starts + W / 2) / ENV_RATE
    return np.interp(np.arange(n) / CHROMA_RATE, tc, p)


def _same_song(ca: np.ndarray, cb: np.ndarray) -> float:
    """Structural same-song evidence between two adjacent pieces of chroma (10 Hz): the best of
    'last STRUCT_Q_S of A found in B' and 'first STRUCT_Q_S of B found in A' (DTW score)."""
    q = int(STRUCT_Q_S * CHROMA_RATE)
    prep = lambda c: dtw.cens(dtw.remove_background(c, CHROMA_RATE, BG_WIN_S), in_rate=CHROMA_RATE,
                              out_rate=SIM_RATE, win_s=SIM_WIN_S, quantise=False)
    best = 0.0
    for qq, rr in ((ca[:, -q:], cb), (cb[:, :q], ca)):
        if qq.shape[1] < 5 * CHROMA_RATE or rr.shape[1] < 5 * CHROMA_RATE:
            continue
        Q, R = prep(qq), prep(rr)
        if Q.shape[1] > 2 * R.shape[1]:
            Q = Q[:, -2 * R.shape[1]:] if qq is ca[:, -q:] else Q[:, :2 * R.shape[1]]
        best = max(best, dtw.match(Q, R, return_path=False).score)
    return best


def _split_on_pulse_gaps(runs: list[tuple[int, int]], feats: Features, n: int) -> list[tuple[int, int]]:
    """Cut talk / applause out of music runs using beat clarity, checked structurally.

    Synthetic stage talk scores musicness 0.3-0.7 (sustained vowels), so musicness alone glues
    songs together across the talk. If the owner shows a clear beat while music plays (75th
    percentile of the pulse inside runs >= PULSE_INFORMATIVE; crowd-noise phones do not), every
    stretch of >= PULSE_GAP_S with pulse < PULSE_GAP inside a run is a candidate gap:
      * inside a run it is cut out when the music before and after are not the same song
        (_same_song < STRUCT_SAME) - a drum-less breakdown or bridge inside a song stays;
      * at a run edge it is trimmed when it does not recur in the rest of the run (talk before
        the first note), while a drum-less intro that recurs later stays.
    """
    if not runs or feats.onset.size == 0:
        return list(runs)
    pulse = _pulse_curve(feats.onset, n)
    inside = np.concatenate([pulse[a:b] for a, b in runs])
    if inside.size == 0 or np.percentile(inside, 75) < PULSE_INFORMATIVE:
        return list(runs)
    C = feats.chroma
    gmin = int(PULSE_GAP_S * CHROMA_RATE)
    out = []
    for a, b in runs:
        gaps = [(a + x, a + y) for x, y in _runs(pulse[a:b] < PULSE_GAP) if y - x >= gmin]
        pieces = []                     # music pieces between kept-out gaps
        cur = a
        for k, (ga, gb) in enumerate(gaps):
            if ga <= a + 1:             # leading gap
                if gb < b and _same_song(C[:, ga:gb], C[:, gb:b]) < STRUCT_EDGE:
                    cur = gb
                continue
            if gb >= b - 1:             # trailing gap
                if _same_song(C[:, cur:ga], C[:, ga:gb]) < STRUCT_EDGE:
                    pieces.append((cur, ga))
                    cur = None
                break
            nxt = gaps[k + 1][0] if k + 1 < len(gaps) else b
            if _same_song(C[:, cur:ga], C[:, gb:nxt]) < STRUCT_SAME:
                pieces.append((cur, ga))
                cur = gb
        if cur is not None:
            pieces.append((cur, b))
        out.extend(p for p in pieces if p[1] - p[0] >= int(MUSIC_MIN_S * CHROMA_RATE))
    return out


def _tempogram(onset: np.ndarray, hop_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Onset autocorrelation per NOV_TEMPO_WIN_S window (lags 0.25-2 s), hop 1 s.
    Returns (lags x frames, frame centre times). The tempogram shows each song as a block of
    constant line spacing; on the real tour footage it is the clearest song cue."""
    x = np.asarray(onset, np.float64)
    W, H = int(NOV_TEMPO_WIN_S * ENV_RATE), int(hop_s * ENV_RATE)
    lo, hi = int(0.25 * ENV_RATE), int(2.0 * ENV_RATE)
    if x.size < W + H:
        return np.zeros((hi - lo + 1, 0)), np.zeros(0)
    st = np.arange(0, x.size - W, H)
    fr = np.lib.stride_tricks.sliding_window_view(x, W)[st]
    fr = fr - fr.mean(axis=1, keepdims=True)
    F = np.fft.rfft(fr * np.hanning(W), 2 * W, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :hi + 1]
    ac = ac / (ac[:, :1] + 1e-9)
    return np.clip(ac[:, lo:], 0, None).T, (st + W / 2) / ENV_RATE


def _unit_cols(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    return X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)


def _checker(X: np.ndarray, L: int) -> np.ndarray:
    """Checkerboard novelty on the cosine SSM of unit columns X, half-width L frames, absolute
    scale (mean within-block similarity minus mean cross-block similarity, Gaussian taper)."""
    n = X.shape[1]
    out = np.zeros(n)
    g = np.arange(-L, L) + 0.5
    w = np.exp(-(g ** 2) / (2 * (L / 2) ** 2))
    W2 = w[:, None] * w[None, :]
    k = np.sign(g)[:, None] * np.sign(g)[None, :] * W2
    norm = W2[:L, :L].sum() * 2
    for i in range(L, n - L):
        B = X[:, i - L:i + L]
        out[i] = float(((B.T @ B) * k).sum() / norm)
    return out


def song_novelty(chroma: np.ndarray, onset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, novelty) at 1 Hz for a music stretch: tempo novelty + key novelty.

    Tempo: unit-normalised tempogram columns. Key: chroma with a 2-minute per-class background
    removed (kills mains hum / tape tones that are constant all show), smoothed over 10 s,
    unit-normalised. Checkerboard half-width NOV_L_S. Synthetic songs: within-song max 1.03,
    song changes 1.18-1.71. Real tour footage: song changes 0.75-1.45, but jams inside songs
    reach 1.39 - hence the song-length prior in _novelty_splits.
    """
    T, t = _tempogram(onset)
    if t.size < 2 * NOV_L_S + 2:
        return t, np.zeros(t.size)
    from scipy.ndimage import uniform_filter1d
    C = np.asarray(chroma, np.float64)
    C = np.maximum(C - uniform_filter1d(C, int(120 * CHROMA_RATE), axis=1, mode="nearest"), 0.0)
    idx = np.clip((t * CHROMA_RATE).astype(int), 0, C.shape[1] - 1)
    K = uniform_filter1d(C, int(10 * CHROMA_RATE), axis=1, mode="nearest")[:, idx]
    L = int(NOV_L_S)
    return t, _checker(_unit_cols(T), L) + _checker(_unit_cols(K), L)


def _novelty_splits(chroma: np.ndarray, onset: np.ndarray) -> list[int]:
    """Song boundaries inside one continuous music stretch (frames at CHROMA_RATE, relative).

    1. Any novelty peak >= NOV_STRONG splits (pieces >= NOV_MIN_PIECE_S).
    2. Song-length prior: a piece longer than LONG_PIECE_S is split at its highest novelty peak
       >= NOV_PRIOR (pieces >= NOV_MIN_PIECE_S), recursively. Live sets run songs together
       (no applause gap, crowd noise throughout), and a 10-minute "song" is far more often two
       songs than one jam; a jam that does get split still clusters (both halves match the
       other nights' version as subsequences).
    """
    dur = chroma.shape[1] / CHROMA_RATE
    if dur < 2 * NOV_MIN_PIECE_S:
        return []
    t, nov = song_novelty(chroma, onset)
    if t.size == 0:
        return []
    peaks = [i for i in range(1, len(nov) - 1) if nov[i] >= nov[max(0, i - 5):i + 6].max() and nov[i] > 0]
    cuts: list[float] = []

    def ok(tc, lo, hi):
        return tc - lo >= NOV_MIN_PIECE_S and hi - tc >= NOV_MIN_PIECE_S

    for i in sorted(peaks, key=lambda i: -nov[i]):
        if nov[i] < NOV_STRONG:
            break
        bounds = sorted([0.0, dur] + cuts)
        k = int(np.searchsorted(bounds, t[i]))
        if ok(t[i], bounds[k - 1], bounds[k]):
            cuts.append(float(t[i]))
    changed = True
    while changed:
        changed = False
        bounds = sorted([0.0, dur] + cuts)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi - lo <= LONG_PIECE_S:
                continue
            cand = [i for i in peaks if nov[i] >= NOV_PRIOR and ok(t[i], lo, hi)]
            if cand:
                cuts.append(float(t[max(cand, key=lambda i: nov[i])]))
                changed = True
                break
    return sorted(int(round(c * CHROMA_RATE)) for c in cuts)


def _chunk_chain(qc: np.ndarray, rc: np.ndarray) -> tuple[float, list]:
    """Where does query q (prepared chroma at SIM_RATE) occur in the longer r, tolerant of
    live versions that stretch or insert sections (jams)?

    q is cut into CHAIN_CHUNK_S chunks every CHAIN_HOP_S; each chunk is located in r by
    subsequence DTW. The best monotone chain of chunk hits (score >= CHAIN_HIT, local stretch
    0.5-2.5x, gaps <= 240 s) is returned with its score: sum of hit scores / number of chunks.
    On the real tour footage, studio reference vs live show gave 0.08-0.20 for songs that were
    played (with 3-16 hits) and no chain at all for songs that were not.
    Returns (score, [(q_t, r_start, r_end, s), ...]).
    """
    L, H = int(CHAIN_CHUNK_S * SIM_RATE), int(CHAIN_HOP_S * SIM_RATE)
    n = qc.shape[1]
    if n < 10 or rc.shape[1] < 10:
        return 0.0, []
    starts = list(range(0, max(1, n - L + 1), H))
    hits = []
    for a in starts:
        q = qc[:, a:a + L]
        if q.shape[1] < 10 or q.shape[1] > 2 * rc.shape[1]:
            continue
        m = dtw.match(q, rc, return_path=True)
        if m.score >= CHAIN_HIT:
            hits.append((a / SIM_RATE, m.ref_start / SIM_RATE, m.ref_end / SIM_RATE, m.score))
    if not hits:
        return 0.0, []
    hits.sort()
    k = len(hits)
    best = [h[3] for h in hits]
    prev = [-1] * k
    for j in range(k):
        for i in range(j):
            dq, dr = hits[j][0] - hits[i][0], hits[j][1] - hits[i][1]
            if dq > 0 and 0 < dr <= 240.0 and 0.5 <= dr / dq <= 2.5 and best[i] + hits[j][3] > best[j]:
                best[j], prev[j] = best[i] + hits[j][3], i
    j = int(np.argmax(best))
    chain = []
    while j >= 0:
        chain.append(hits[j])
        j = prev[j]
    return float(max(best)) / len(starts), chain[::-1]


def locate_references(feats: Features, references: dict[str, Features], start_offset: float = 0.0):
    """Where each reference song occurs in an owner (show recording).

    Returns [(name, start, end, score, n_hits, core_start, core_end)] in owner time, sorted by
    start, for chains with >= CHAIN_MIN_HITS hits and score >= CHAIN_ACCEPT. core = first to
    last matched chunk. start/end extrapolate the core to the reference's start/end at tempo 1
    but never into another located song's core; where two extensions overlap they meet half
    way. Approximate (live intros / jams that the studio version lacks are not "seen").
    """
    oc = _cens_sim(feats)
    regs = []
    for name, rf in references.items():
        sc, ch = _chunk_chain(_cens_sim(rf), oc)
        if len(ch) < CHAIN_MIN_HITS or sc < CHAIN_ACCEPT:
            continue
        c0, c1 = ch[0][1], ch[-1][2]
        a = max(0.0, c0 - ch[0][0])
        b = min(feats.duration, c1 + max(0.0, rf.duration - (ch[-1][0] + CHAIN_CHUNK_S)))
        regs.append([name, a, b, sc, len(ch), c0, c1])
    regs.sort(key=lambda r: r[5])
    # two chains whose cores overlap: keep the stronger
    keep = []
    for r in sorted(regs, key=lambda r: -r[3]):
        if all(r[6] <= k[5] or r[5] >= k[6] for k in keep):
            keep.append(r)
    regs = sorted(keep, key=lambda r: r[5])
    for k, r in enumerate(regs):
        if k:
            r[1] = max(r[1], regs[k - 1][6])
        if k + 1 < len(regs):
            r[2] = min(r[2], regs[k + 1][5])
    for k in range(1, len(regs)):
        if regs[k][1] < regs[k - 1][2]:
            m = 0.5 * (regs[k][1] + regs[k - 1][2])
            regs[k - 1][2] = regs[k][1] = m
    return [(r[0], start_offset + r[1], start_offset + r[2], r[3], r[4], start_offset + r[5], start_offset + r[6])
            for r in regs]


def _reference_cuts(regions, a_s: float, b_s: float) -> list[float]:
    """Boundaries (owner seconds) inside run [a_s, b_s] from located reference regions (their
    resolved start/end edges), at least 20 s from the run ends and from each other."""
    cuts = []
    for r in regions:
        for c in (r[1], r[2]):
            if a_s + 20.0 < c < b_s - 20.0 and all(abs(c - x) >= 20.0 for x in cuts):
                cuts.append(c)
    return sorted(cuts)


def _snap_cut(feats: Features, c: int, a: int, b: int) -> int:
    """Move a reference-derived cut (frame c inside run [a, b)) to the strongest song-novelty
    peak within +-SNAP_S: chain edges are extrapolated at tempo 1 and land seconds off."""
    w = int((SNAP_S + NOV_L_S + NOV_TEMPO_WIN_S) * CHROMA_RATE)
    lo, hi = max(a, c - w), min(b, c + w)
    if hi - lo < 4 * NOV_L_S * CHROMA_RATE:
        return c
    t, nov = song_novelty(feats.chroma[:, lo:hi],
                          feats.onset[int(lo * ENV_RATE / CHROMA_RATE):int(hi * ENV_RATE / CHROMA_RATE)])
    if t.size == 0:
        return c
    tt = t + lo / CHROMA_RATE
    m = np.abs(tt - c / CHROMA_RATE) <= SNAP_S
    if not m.any() or nov[m].max() < SNAP_MIN:
        return c
    return int(round(tt[m][int(np.argmax(nov[m]))] * CHROMA_RATE))


def _merge_runs_in_regions(runs, regions):
    """Join music runs whose gap (< REF_MERGE_GAP_S) lies inside one reference song's core."""
    out = []
    for a, b in runs:
        if out:
            pa, pb = out[-1]
            g0, g1 = pb / CHROMA_RATE, a / CHROMA_RATE
            if g1 - g0 < REF_MERGE_GAP_S and any(r[5] <= 0.5 * (g0 + g1) <= r[6] for r in regions):
                out[-1] = (pa, b)
                continue
        out.append((a, b))
    return out


def find_segments(owner: str, owner_kind: str, feats: Features, start_offset: float = 0.0,
                  references: Optional[dict[str, Features]] = None) -> list[SongSegment]:
    """Song segments of one owner (clip on clip time, or group on event time).

    1. musicness median-smoothed over 1.1 s, hysteresis MUSIC_ON / MUSIC_OFF.
    2. Non-music gaps shorter than SONG_GAP_S are bridged (never across a coverage hole).
    3. Owners shorter than SNIPPET_OWNER_S: all music is one segment (a snippet is one song).
       Otherwise runs that musicness split inside a song are re-joined (_merge_runs) and talk /
       applause that musicness mistakes for music is cut out by beat clarity plus a structural
       same-song test (_split_on_pulse_gaps).
    4. Stretches shorter than SONG_MIN_S are kept only when they touch an edge (cut-off song)
       or the owner is a snippet; anything under MUSIC_MIN_S is dropped.
    5. Long stretches are split at segues (see _segue_splits; conservative).
    6. A clip (<= FALLBACK_OWNER_S) where nothing was found gets its audible extent as one
       partial segment (crowd-dominated phones have musicness ~0 even during songs).
    is_partial: the segment touches the owner start/end or a coverage hole (EDGE_TOL_S), or is
    shorter than SONG_MIN_S. Ids are f"{owner}:{index:02d}" in time order.
    """
    mu = np.asarray(feats.musicness, dtype=np.float64)
    n = mu.size
    if n == 0:
        return []
    # coverage (group features carry NaN energy where no clip plays)
    e = np.asarray(feats.energy_db, dtype=np.float64)
    if e.size and np.isnan(e).any():
        idx = np.minimum((np.arange(n) * ENV_RATE / CHROMA_RATE).astype(int), e.size - 1)
        covered = ~np.isnan(e[idx])
    else:
        covered = np.ones(n, bool)
    m = median_filter(np.nan_to_num(mu), size=MUSIC_MEDIAN, mode="nearest")
    lab, nl = label(m > MUSIC_OFF)
    keep = np.zeros(nl + 1, bool)
    if nl:
        keep[np.unique(lab[m > MUSIC_ON])] = True
    keep[0] = False
    music = keep[lab] & covered
    # bridge short gaps that lie between music frames and contain no coverage hole
    gap = int(round(SONG_GAP_S * CHROMA_RATE))
    for a, b in _runs(~music):
        if a > 0 and b < n and (b - a) < gap and covered[a:b].all():
            music[a:b] = True
    owner_dur = n / CHROMA_RATE
    tol = int(round(EDGE_TOL_S * CHROMA_RATE))
    holes = ~covered
    runs = [(a, b) for a, b in _runs(music) if b - a >= CHROMA_RATE]
    if runs and owner_dur < SNIPPET_OWNER_S and covered[runs[0][0]:runs[-1][1]].all():
        runs = [(runs[0][0], runs[-1][1])]          # a snippet holds one song: never fragment it
    else:
        runs = _merge_runs(runs, feats, covered)
        runs = _split_on_pulse_gaps(runs, feats, n)
    regions = locate_references(feats, references) if references else []
    runs = _merge_runs_in_regions(runs, regions) if regions else runs
    segs: list[tuple[int, int, bool]] = []
    for a, b in runs:
        length = (b - a) / CHROMA_RATE
        if length < MUSIC_MIN_S:
            continue
        touch = a <= tol or b >= n - tol or holes[max(0, a - tol):a].any() or holes[b:b + tol].any()
        if length < SONG_MIN_S and not touch and owner_dur >= SNIPPET_OWNER_S:
            continue
        # reference regions first (they know where a song starts and ends, even through a jam)
        rc = [_snap_cut(feats, int(round(c * CHROMA_RATE)), a, b)
              for c in _reference_cuts(regions, a / CHROMA_RATE, b / CHROMA_RATE)]
        bounds = [a] + rc + [b]
        cuts = []
        for pa, pb in zip(bounds[:-1], bounds[1:]):
            mid = 0.5 * (pa + pb) / CHROMA_RATE
            if any(r[1] <= mid <= r[2] for r in regions):
                continue                     # inside a located reference song: keep whole
            if (pb - pa) / CHROMA_RATE < 2 * SEGUE_MIN_PIECE_S:
                continue
            ch, on = feats.chroma[:, pa:pb], feats.onset[int(pa * ENV_RATE / CHROMA_RATE):int(pb * ENV_RATE / CHROMA_RATE)]
            sc = _segue_splits(ch, on) + _novelty_splits(ch, on)
            for c in sorted(sc):
                if all(abs(pa + c - x) >= SEGUE_MIN_PIECE_S * CHROMA_RATE for x in cuts):
                    cuts.append(pa + c)
        bounds = sorted(set(bounds + cuts))
        if regions:                          # run edges: a located song edge nearby knows better
            def _edge(x, which):
                c = [r[which] * CHROMA_RATE for r in regions
                     if abs(r[which] * CHROMA_RATE - x) <= REF_EDGE_S * CHROMA_RATE]
                return int(round(min(c, key=lambda y: abs(y - x)))) if c else x
            bounds[0] = min(max(0, _edge(bounds[0], 1)), bounds[1] - 1)
            bounds[-1] = max(min(n, _edge(bounds[-1], 2)), bounds[-2] + 1)
        for k in range(len(bounds) - 1):
            pa, pb = bounds[k], bounds[k + 1]
            first, last = k == 0, k == len(bounds) - 2
            part_touch = (first and (a <= tol or holes[max(0, a - tol):a].any())) or \
                         (last and (b >= n - tol or holes[b:b + tol].any()))
            segs.append((pa, pb, part_touch or (pb - pa) / CHROMA_RATE < SONG_MIN_S))
    if not segs and owner_kind == "clip" and owner_dur <= FALLBACK_OWNER_S:
        # musicness found nothing (crowd-dominated phones score ~0 even during songs): a short
        # clip is most likely music, so offer its audible extent as one partial segment and let
        # the DTW scores decide whether it belongs to a song.
        fb = _audible_extent(feats, n)
        if fb is not None and (fb[1] - fb[0]) / CHROMA_RATE >= MUSIC_MIN_S:
            segs.append((fb[0], fb[1], True))
    out = []
    for i, (a, b, partial) in enumerate(segs):
        out.append(SongSegment(id=f"{owner}:{i:02d}", owner=owner, owner_kind=owner_kind,
                               start=float(start_offset + a / CHROMA_RATE),
                               end=float(start_offset + b / CHROMA_RATE), is_partial=bool(partial)))
    return out


# =========================================================================== song maps

def _fit_map(x: np.ndarray, y: np.ndarray, knot_s: float = MAP_KNOT_S,
             slope: tuple[float, float] = MAP_TEMPO, smooth: float = 0.5) -> TimeMap:
    """Monotonic piecewise-linear map y = f(x) through DTW path points (x, y).

    Knot targets: median of y - (x - knot) for points within +-0.75 knot spacing (robust to the
    DTW staircase). Then a bounded least-squares fit over the knot values parametrised as start
    value + slopes in [slope[0], slope[1]], with a small penalty on slope changes and two
    rounds of robust (Huber) reweighting.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return TimeMap.linear(1.0, float(np.median(y - x)), lo, lo + 1.0)
    nk = max(2, int(round((hi - lo) / knot_s)) + 1)
    kx = np.linspace(lo, hi, nk)
    half = 0.75 * (kx[1] - kx[0])
    ky = np.empty(nk)
    have = np.zeros(nk, bool)
    for k, c in enumerate(kx):
        sel = np.abs(x - c) <= half
        if sel.any():
            ky[k] = np.median(y[sel] - (x[sel] - c))
            have[k] = True
    if have.sum() < 2:
        return TimeMap.linear(1.0, float(np.median(y - x)), lo, hi)
    kx_h, ky_h = kx[have], ky[have]
    kx, ky = kx_h, ky_h
    nk = kx.size
    dx = np.diff(kx)
    # unknowns: v0, s_1..s_{nk-1}; value_k = v0 + sum_{i<k} s_i dx_i
    A = np.zeros((nk, nk))
    A[:, 0] = 1.0
    for k in range(1, nk):
        A[k, 1:k + 1] = dx[:k]
    R = np.zeros((max(0, nk - 2), nk))
    for k in range(nk - 2):
        R[k, k + 1], R[k, k + 2] = -smooth, smooth
    w = np.ones(nk)
    lb = np.concatenate([[-np.inf], np.full(nk - 1, slope[0])])
    ub = np.concatenate([[np.inf], np.full(nk - 1, slope[1])])
    sol = None
    for _ in range(3):
        M = np.vstack([A * w[:, None], R])
        rhs = np.concatenate([ky * w, np.zeros(R.shape[0])])
        sol = lsq_linear(M, rhs, bounds=(lb, ub), method="bvls").x
        res = A @ sol - ky
        scale = max(1.4826 * np.median(np.abs(res)), 0.15)
        w = 1.0 / np.maximum(1.0, np.abs(res) / (1.5 * scale))
        w = np.sqrt(w)
    vals = A @ sol
    return TimeMap(kx.astype(np.float64), vals.astype(np.float64))


def _prep(f: Features, rate: float, win_s: float) -> np.ndarray:
    """Background-removed, CENS-like smoothed chroma at ``rate`` Hz (see dtw.remove_background)."""
    return dtw.cens(dtw.remove_background(f.chroma, CHROMA_RATE, BG_WIN_S), in_rate=CHROMA_RATE,
                    out_rate=rate, win_s=win_s, quantise=False)


def _cens_sim(f: Features) -> np.ndarray:
    return _prep(f, SIM_RATE, SIM_WIN_S)


def _cens_map(f: Features) -> np.ndarray:
    return _prep(f, MAP_RATE, MAP_WIN_S)


def _align_map(q_feats: Features, r_feats: Features, q_t0: float, r_t0: float,
               allow_transpose: bool = False, coarse: Optional[dtw.DTWMatch] = None):
    """song_map from query owner time to reference time.

    q_t0 / r_t0: owner time of frame 0 of each feature set. The query must be the shorter one
    (subsequence DTW). Coarse 2 Hz match locates the region, then a 5 Hz DTW inside the region
    (+- REFINE_MARGIN_S) gives the path. Returns (TimeMap, coarse match).
    """
    qs, rs = _cens_sim(q_feats), _cens_sim(r_feats)
    if coarse is None:
        coarse = dtw.match(qs, rs, return_path=True, allow_transpose=allow_transpose)
    a = max(0.0, coarse.ref_start / SIM_RATE - REFINE_MARGIN_S)
    b = coarse.ref_end / SIM_RATE + REFINE_MARGIN_S
    qm = _cens_map(q_feats)
    rm = _cens_map(r_feats)
    ia, ib = int(a * MAP_RATE), min(rm.shape[1], int(np.ceil(b * MAP_RATE)) + 1)
    rr = rm[:, ia:ib]
    qk = np.roll(qm, coarse.transpose, axis=0) if coarse.transpose else qm
    if rr.shape[1] < 4 or qk.shape[1] < 2:
        p = coarse.path
        xq, yr = p[:, 0] / SIM_RATE, p[:, 1] / SIM_RATE
    else:
        C = dtw.cost_matrix(qk, rr)
        _, path = dtw.subseq_dtw(C, return_path=True, step_penalty=0.02)
        xq = path[:, 0] / MAP_RATE
        yr = (path[:, 1] + ia) / MAP_RATE
    tm = _fit_map(xq + q_t0, yr + r_t0)
    return tm, coarse


# =========================================================================== song assignment

def _similarity_matrix(feats_list: list[Features], allow_transpose: bool = False) -> np.ndarray:
    """Symmetric 0..1 similarity: subsequence DTW of the shorter (query) against the longer."""
    n = len(feats_list)
    S = np.eye(n)
    cs = [_cens_sim(f) for f in feats_list]
    lens = [c.shape[1] for c in cs]
    for i in range(n):
        others = [j for j in range(n) if j != i and (lens[j] > lens[i] or (lens[j] == lens[i] and j > i))]
        if not others or lens[i] < 2:
            continue
        res = dtw.match_many(cs[i], [cs[j] for j in others], allow_transpose=allow_transpose)
        for j, r in zip(others, res):
            S[i, j] = S[j, i] = r.score
    C = _chain_similarity(cs)
    return np.maximum(S, np.minimum(1.0, CHAIN_SIM_GAIN * C))


def _chain_similarity(cs: list[np.ndarray]) -> np.ndarray:
    """Chunk-chain similarity between long segments (both >= CHAIN_MIN_SEG_S): chunks of the
    shorter located in the longer (one batched DTW per chunk), best monotone chain, score =
    sum of hit scores / chunks. Catches the same song on two nights when the segments are cut
    differently or the live arrangement inserts / stretches sections, where the whole-segment
    DTW (tempo 0.5-2x, whole query must fit) drops below threshold. Real tour footage: same
    song on two nights 0.15-0.49, different songs 0; synthetic different songs p99 0.12 (songs
    sharing a chord progression). Scaled by CHAIN_SIM_GAIN so 0.15 lands on 0.35 (SONG_SIMILAR)."""
    n = len(cs)
    C = np.zeros((n, n))
    long_ = [i for i in range(n) if cs[i].shape[1] >= CHAIN_MIN_SEG_S * SIM_RATE]
    if len(long_) < 2:
        return C
    L, H = int(CHAIN_CHUNK_S * SIM_RATE), int(CHAIN_HOP_S * SIM_RATE)
    lens = {i: cs[i].shape[1] for i in long_}
    for i in long_:
        others = [j for j in long_ if j != i and (lens[j] > lens[i] or (lens[j] == lens[i] and j > i))]
        if not others:
            continue
        starts = list(range(0, max(1, lens[i] - L + 1), H))
        hits = {j: [] for j in others}
        for a in starts:
            q = cs[i][:, a:a + L]
            if q.shape[1] < 10:
                continue
            for j, m in zip(others, dtw.match_many(q, [cs[j] for j in others])):
                if m.score >= CHAIN_HIT:
                    hits[j].append((a / SIM_RATE, (m.ref_end - q.shape[1]) / SIM_RATE, m.ref_end / SIM_RATE, m.score))
        for j in others:
            sc = _best_chain(hits[j]) / len(starts)
            C[i, j] = C[j, i] = sc
    return C


def _best_chain(hits) -> float:
    """Sum of scores of the best monotone chain of chunk hits (q_t, r_start, r_end, score),
    local stretch 0.5-2.5x, gaps <= 240 s; 0 with fewer than CHAIN_MIN_HITS hits in it."""
    if len(hits) < CHAIN_MIN_HITS:
        return 0.0
    hits = sorted(hits)
    k = len(hits)
    best = [h[3] for h in hits]
    cnt = [1] * k
    for j in range(k):
        for i in range(j):
            dq, dr = hits[j][0] - hits[i][0], hits[j][1] - hits[i][1]
            if dq > 0 and 0 < dr <= 240.0 and 0.5 <= dr / dq <= 2.5 and best[i] + hits[j][3] > best[j]:
                best[j], cnt[j] = best[i] + hits[j][3], cnt[i] + 1
    j = int(np.argmax(best))
    return float(best[j]) if cnt[j] >= CHAIN_MIN_HITS else 0.0


def _agglomerate(clusters: list[list[int]], link, allowed, thresh: float) -> list[list[int]]:
    """Greedy agglomeration: repeatedly merge the allowed pair with the highest link >= thresh."""
    clusters = [list(c) for c in clusters]
    while len(clusters) > 1:
        best, bi, bj = -1.0, -1, -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if not allowed(clusters[a], clusters[b]):
                    continue
                s = link(clusters[a], clusters[b])
                if s > best:
                    best, bi, bj = s, a, b
        if bi < 0 or best < thresh:
            break
        clusters[bi] = clusters[bi] + clusters[bj]
        del clusters[bj]
    return clusters


def _average_linkage(S: np.ndarray, idx: list[int], thresh: float) -> list[list[int]]:
    """Agglomerative clustering (average linkage) of items idx using similarity S."""
    return _agglomerate([[i] for i in idx], lambda a, b: float(np.mean(S[np.ix_(a, b)])),
                        lambda a, b: True, thresh)


def _is_core(s: SongSegment) -> bool:
    d = s.end - s.start
    return d >= CORE_MIN_S or (not s.is_partial and d >= SONG_MIN_S)


def cluster_segments(segments: list[SongSegment], S: np.ndarray, thresh: float = None,
                     pre: Optional[list[list[int]]] = None, cannot: Optional[set] = None,
                     labels: Optional[dict[int, str]] = None) -> list[list[int]]:
    """Two-stage clustering of segment indices, optionally constrained by user hints.

    Stage A: core segments (_is_core: >= CORE_MIN_S, or not partial and >= SONG_MIN_S) -
    average linkage at thresh. is_partial alone is not used: a take whose music runs to the
    last sample of its file is flagged partial, and then no take was "complete" (synth v2).
    Stage B: the rest (snippets), longest first, join the cluster they fit best: mean
    similarity to the better half of the cluster's core members (a fragment of a take may not
    contain the snippet) or, for clusters made only of partials, the best member. A partial
    also joins when its best cluster reaches SNIP_RELAXED * thresh and beats the runner-up
    cluster by SNIP_MARGIN x (short noisy phone snippets score ~0.3 against their own song but
    ~0.05 against the others). Otherwise it seeds a new cluster that later partials may join.

    Constraints (user hints): ``pre`` = groups of indices that must stay together (tagged with
    one title / must-linked); they enter stage A as ready clusters, whatever their length.
    ``cannot`` = set of frozenset({i, j}) that may never share a cluster. ``labels`` = index ->
    song-title key; clusters with different titles never merge. A cluster holding hinted members
    links to another by the best of (average linkage, best single member's mean similarity),
    so a weak tagged crowd-phone clip does not stop real takes from joining its song.
    """
    thresh = SONG_SIMILAR if thresh is None else thresh
    pre = [list(g) for g in (pre or []) if g]
    cannot = cannot or set()
    labels = labels or {}
    hinted = {i for g in pre for i in g}

    def allowed(c1, c2):
        if len({labels[i] for i in c1 + c2 if i in labels}) > 1:
            return False
        return not any(frozenset((i, j)) in cannot for i in c1 for j in c2)

    def link(c1, c2):
        M = S[np.ix_(c1, c2)]
        v = float(M.mean())
        if hinted.intersection(c1) or hinted.intersection(c2):
            v = max(v, float(M.mean(axis=1).max()), float(M.mean(axis=0).max()))
        return v

    core = [i for i, s in enumerate(segments) if _is_core(s) and i not in hinted]
    clusters = _agglomerate(pre + [[i] for i in core], link, allowed, thresh)
    is_full = set(core) | hinted
    rest = sorted((i for i in range(len(segments)) if i not in is_full),
                  key=lambda i: -(segments[i].end - segments[i].start))
    for i in rest:
        sc = []
        for members in clusters:
            if not allowed([i], members):
                sc.append(-1.0)
                continue
            fm = [j for j in members if j in is_full]
            if fm:      # mean over the better half: a fragment of a take may not contain the snippet
                v = np.sort(S[i, fm])[::-1]
                s = float(np.mean(v[:max(1, (len(v) + 1) // 2)]))
            else:
                s = float(np.max(S[i, members]))
            if hinted.intersection(members):
                s = max(s, float(np.max(S[i, members])))
            sc.append(s)
        order = np.argsort(sc)[::-1] if sc else []
        best = sc[order[0]] if sc else -1.0
        second = sc[order[1]] if len(sc) > 1 else 0.0
        ok = best >= thresh or (best >= SNIP_RELAXED * thresh and best >= SNIP_MARGIN * max(second, 0.01))
        if sc and ok:
            clusters[int(order[0])].append(i)
        else:
            clusters.append([i])
    return clusters


def _ref_name(key: str) -> str:
    return Path(key).stem if ("/" in key or "\\" in key or "." in Path(key).name) else key


def _chain_reference(q: np.ndarray, rcs: list[np.ndarray]):
    """Best reference for a long segment by chunk chains in both directions (segment chunks
    found in the reference, reference chunks found in the segment). Accepts a chain with
    >= CHAIN_MIN_HITS hits and score >= CHAIN_REF_ACCEPT, and only when it beats the runner-up
    reference by CHAIN_MARGIN x. Returns (ref index, score, (k, 2) points seg_t -> ref_t) or None."""
    res = []
    for k, rc in enumerate(rcs):
        s1, c1 = _chunk_chain(q, rc)
        s2, c2 = _chunk_chain(rc, q)
        if s1 >= s2 and len(c1) >= CHAIN_MIN_HITS:
            pts = [(h[0], h[1]) for h in c1] + [(h[0] + CHAIN_CHUNK_S, h[2]) for h in c1]
            res.append((s1, k, pts))
        elif len(c2) >= CHAIN_MIN_HITS:
            pts = [(h[1], h[0]) for h in c2] + [(h[2], h[0] + CHAIN_CHUNK_S) for h in c2]
            res.append((s2, k, pts))
        else:
            res.append((0.0, k, []))
    res.sort(key=lambda r: -r[0])
    best = res[0]
    second = res[1][0] if len(res) > 1 else 0.0
    if best[0] < CHAIN_REF_ACCEPT or best[0] < CHAIN_MARGIN * max(second, 1e-3):
        return None
    pts = np.array(sorted(best[2]), float)
    return best[1], best[0], pts


def _title_key(title) -> str:
    return " ".join(str(title).split()).casefold()


def _align_to_master(s: SongSegment, master: SongSegment, seg_feats, allow_transpose) -> TimeMap:
    sf, mf = seg_feats[s.id], seg_feats[master.id]
    if sf.chroma.shape[1] <= mf.chroma.shape[1]:
        tm, _ = _align_map(sf, mf, s.start, master.start, allow_transpose)
        return TimeMap(tm.knots_src, tm.knots_dst - master.start)
    # partial longer than the master: align master inside it, invert
    tm, _ = _align_map(mf, sf, master.start, s.start, allow_transpose)
    inv = tm.inverse()
    return TimeMap(inv.knots_src, inv.knots_dst - master.start)


def _extend_with(snapped: TimeMap, coarse: TimeMap, t0: float, t1: float, bar: float) -> TimeMap:
    """Beat knots inside; outside them the coarse DTW map, re-fitted smoothly (knots every
    EXTEND_BARS bars, slope within beats.LOCAL_SLOPE of the edge slope) and joined
    continuously - the raw DTW knots are too jittery for picture warping."""
    kx, ky = snapped.knots_src, snapped.knots_dst
    parts_x, parts_y = [], []

    def side(lo, hi, anchor_x, anchor_y, edge_slope, before):
        if hi - lo < 0.5:
            return np.zeros(0), np.zeros(0)
        step = max(0.5, bar / 2)
        xs = np.arange(lo, hi - 0.25, step) if before else np.arange(lo + step, hi + 1e-9, step)
        if xs.size == 0:
            return np.zeros(0), np.zeros(0)
        ys = np.asarray(coarse(xs), np.float64)
        ys = ys - float(coarse(anchor_x)) + anchor_y           # join continuously
        a, b = beats_mod.LOCAL_SLOPE
        sl0, sl1 = max(MAP_TEMPO[0], a * edge_slope), min(MAP_TEMPO[1], b * edge_slope)
        x2 = np.concatenate([xs, [anchor_x]]) if before else np.concatenate([[anchor_x], xs])
        y2 = np.concatenate([ys, [anchor_y]]) if before else np.concatenate([[anchor_y], ys])
        fx, fy = beats_mod._smooth_fit(x2, y2, EXTEND_BARS * bar, sl0, sl1, beats_mod.SMOOTH)
        fy = fy - (fy[-1] if before else fy[0]) + anchor_y
        return (fx[:-1], fy[:-1]) if before else (fx[1:], fy[1:])

    e0 = (ky[1] - ky[0]) / (kx[1] - kx[0])
    e1 = (ky[-1] - ky[-2]) / (kx[-1] - kx[-2])
    px, py = side(t0, kx[0], kx[0], ky[0], e0, True)
    qx, qy = side(kx[-1], t1, kx[-1], ky[-1], e1, False)
    xs = np.concatenate([px, kx, qx])
    ys = np.concatenate([py, ky, qy])
    keep = np.concatenate([[True], (np.diff(xs) > 1e-6) & (np.diff(ys) > 1e-6)])
    return TimeMap(xs[keep], ys[keep])


def _map_ratio(m: TimeMap) -> float:
    sl = np.diff(m.knots_dst) / np.maximum(np.diff(m.knots_src), 1e-9)
    return float(np.clip(np.median(sl), MAP_TEMPO[0], MAP_TEMPO[1])) if sl.size else 1.0


def _pick_take_base(so, by_id, seg_feats, base_owner, track):
    """Beat-base take of a song: the longest member on ``base_owner`` (the plugin's master
    night) if any; else the most central member with reliable beats (highest median
    whole-segment DTW score against the other members - the take the others snap onto best)."""
    mem = [by_id[x] for x in so.segments if x in seg_feats and by_id[x].song_map is not None]
    if not mem:
        return None
    if base_owner is not None:
        on = [s for s in mem if s.owner == base_owner]
        if on:
            return max(on, key=lambda s: s.end - s.start)
    rel = [s for s in mem if track(s.id, seg_feats[s.id]).reliable and s.end - s.start >= CHAIN_MIN_SEG_S]
    if len(rel) < 2:
        return max(rel or mem, key=lambda s: s.end - s.start)
    cs = {s.id: _cens_sim(seg_feats[s.id]) for s in mem}

    def sim(a, b):
        q, r = (cs[a], cs[b]) if cs[a].shape[1] <= cs[b].shape[1] else (cs[b], cs[a])
        return dtw.match(q, r, return_path=False).score if q.shape[1] <= 2 * r.shape[1] else 0.0

    return max(rel, key=lambda c: (np.median([sim(c.id, o.id) for o in mem if o.id != c.id]),
                                   c.end - c.start))


def _xenvs(f: Features) -> list:
    """Envelopes used for cross-night alignment, per XNIGHT_ENV: onset, perc or both."""
    p = getattr(f, "perc", None)
    if XNIGHT_ENV == "onset" or p is None or p.size == 0:
        return [f.onset]
    if XNIGHT_ENV == "perc":
        return [p]
    return [f.onset, p]


def _xbeat_env(f: Features) -> np.ndarray:
    """Envelope for beat tracking of a take that is aligned across nights."""
    p = getattr(f, "perc", None)
    return p if (XNIGHT_ENV in ("perc", "both") and p is not None and p.size) else f.onset


def _sync_song(so, by_id, seg_feats, take, ref_f, ref_key, track):
    """Beat-snap one song's members onto ``take`` (a SongSegment) or, if None, the reference.
    Pure: returns (maps, infos, breaks) without touching the segments."""
    maps, infos, brk = {}, {}, {}
    if take is not None:
        base = track(take.id, seg_feats[take.id])
        base_times = base.times + take.start                      # base owner time
        Mb = take.song_map
        if ref_f is not None:
            # the base take itself is beat-snapped onto the reference where that works, so
            # song (reference) time read through it stays beat-accurate too
            rb = track("ref:" + ref_key, ref_f)
            if base.reliable and rb.reliable:
                ratio = _map_ratio(Mb)
                tb = track(take.id, seg_feats[take.id], rb.period / ratio)
                st = {}
                sn, frac = beats_mod.snap_map(Mb, tb.times + take.start, rb.times, rb.period,
                                              take.start, take.end, stats=st)
                if sn is not None and frac >= BEAT_MIN_FRAC and st.get("used", 0) >= BEAT_MIN_PAIRS:
                    Mb = _extend_with(sn, Mb, take.start, take.end, 4.0 * rb.period * ratio)
                    maps[take.id] = Mb
                    brk[take.id] = [float(x) for x in st.get("breaks", [])]
        Mb_inv = Mb.inverse()
    else:
        base = track("ref:" + ref_key, ref_f)
        base_times = base.times                                   # song (reference) time
    for sid in so.segments:
        s = by_id[sid]
        info = dict(beat_synced=False, snapped_frac=0.0, bpm=0.0, reliable=False,
                    base_reliable=bool(base.reliable), beat_base="take" if take is not None else "reference")
        infos[sid] = info
        brk.setdefault(sid, [])
        if take is not None and sid == take.id:
            info.update(beat_synced=bool(base.reliable), snapped_frac=1.0, bpm=60.0 / base.period,
                        reliable=bool(base.reliable))
            continue
        if s.song_map is None or sid not in seg_feats:
            continue
        coarse = s.song_map if take is None else Mb_inv.compose(s.song_map)
        ratio = _map_ratio(coarse)
        f = seg_feats[sid]
        # 1. onset-envelope alignment against the base: fixes whole-beat slips of the DTW map
        if take is not None:
            benv, bt0 = _xenvs(seg_feats[take.id]), take.start
        else:
            benv, bt0 = _xenvs(ref_f), 0.0
        tenv = _xenvs(f)
        if len(tenv) != len(benv):
            tenv, benv = [tenv[0]], [benv[0]]
        est = {}
        env_map = beats_mod.envelope_align(coarse, tenv, s.start, benv, bt0, s.start, s.end,
                                           base.period, stats=est)
        if env_map is not None:
            coarse = _extend_with(env_map, coarse, s.start, s.end, 4.0 * base.period * ratio)
            info.update(env_aligned=True, env_rms_ms=est.get("rms_ms", float("nan")))
            brk[sid] = [float(x) for x in est.get("breaks", [])]
        # 2. beat snapping on top (sharpens to the tracked beats where they are reliable)
        tb = track(sid, f, base.period / ratio)
        info.update(bpm=60.0 / tb.period, reliable=bool(tb.reliable))
        fine = None
        if tb.reliable and base.reliable:
            st = {}
            snapped, frac = beats_mod.snap_map(coarse, tb.times + s.start, base_times,
                                               base.period, s.start, s.end, stats=st)
            info["snapped_frac"] = frac
            info.update({k: v for k, v in st.items() if k != "breaks"})
            if snapped is not None and frac >= BEAT_MIN_FRAC and st.get("used", 0) >= BEAT_MIN_PAIRS:
                fine = _extend_with(snapped, coarse, s.start, s.end, 4.0 * base.period * ratio)
                brk[sid] = [float(x) for x in st.get("breaks", [])]
        if fine is None and env_map is not None:
            fine = coarse
        if fine is None:
            continue
        maps[sid] = fine if take is None else Mb.compose(fine)
        info["beat_synced"] = True
    return maps, infos, brk


def _cross_lag_cost(so, by_id, seg_feats, maps) -> float:
    """Mean drum-hit misalignment (ms) between every two members of different owners through
    their song maps (``maps`` overrides): onset cross-correlation in 4 s windows every 8 s
    (beats.onset_lag_ms); windows that do not correlate count as BASE_LAG_CAP_MS. This is the
    cross-night quality the beat base is chosen by."""
    mem = [by_id[x] for x in so.segments if x in seg_feats and (x in maps or by_id[x].song_map is not None)
           and by_id[x].end - by_id[x].start >= CHAIN_MIN_SEG_S]
    tot, cnt = 0.0, 0
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            a, b = mem[i], mem[j]
            if a.owner == b.owner:
                continue
            ma, mb = maps.get(a.id, a.song_map), maps.get(b.id, b.song_map)
            mbi = mb.inverse()
            fa, fb = seg_feats[a.id], seg_feats[b.id]
            # envelopes start at the segment starts: shift times into envelope time
            a2b = lambda t, ma=ma, mbi=mbi, a=a, b=b: np.asarray(mbi(ma(np.asarray(t) + a.start))) - b.start
            n_win = max(1, int((a.end - a.start - 4.5) // 8.0))
            lags = np.abs(beats_mod.onset_lag_ms(_xbeat_env(fa), _xbeat_env(fb), a2b, 0.0, a.end - a.start, max_lag_s=0.6))
            lags = np.minimum(lags, BASE_LAG_CAP_MS)
            tot += float(lags.sum()) + BASE_LAG_CAP_MS * max(0, n_win - lags.size)
            cnt += max(n_win, lags.size)
    return tot / cnt if cnt else float("inf")


def _sync_quality(infos) -> tuple:
    syn = [i for i in infos.values() if i.get("beat_synced") and "rms_ms" in i]
    return (sum(1 for i in infos.values() if i.get("beat_synced")),
            -float(np.mean([i["rms_ms"] for i in syn])) if syn else 0.0)


def _beat_sync(songs_list, segments, seg_feats, references, report, beat_base="auto", base_owner=None):
    """Make song maps beat-accurate and smooth (see beats.snap_map).

    Song time stays what assign_songs chose (reference file or master take). The BEAT base is
      * "reference": the reference file (or the master take for songs without one);
      * "take": a live take: base_owner's when given, else the reliable take (>= 60 s) with
        the lowest cross-night cost;
      * "auto": base_owner's take when given; else the reference / master and every reliable
        take are tried and the lowest cross-night cost wins (_cross_lag_cost: mean drum-hit
        misalignment between all member pairs of different owners, non-correlating windows
        count as BASE_LAG_CAP_MS) - live jams that differ from the studio version snap better
        onto a live take.
    With a take as beat base B (song map M_B), each other take X is snapped onto B's beats in
    B's own time (coarse guide M_B^-1 . M_X) and its song map becomes M_B . S_XB, so take->take
    alignment (M_B^-1 . M_X = S_XB) is beat-accurate whatever the reference looks like.
    Sets s.song_map, s.breaks (structure breaks, owner time) and report[seg id] = dict(
    beat_synced, snapped_frac, bpm, reliable, base_reliable, beat_base, pairs, used, pieces,
    knots, rms_ms)."""
    by_id = {s.id: s for s in segments}
    cache = {}

    def track(key, f, period=None):
        k = (key, None if period is None else round(period, 4))
        if k not in cache:
            cache[k] = beats_mod.beat_track(_xbeat_env(f), f.chroma, period=period)
        return cache[k]

    for so in songs_list:
        ref_key = so.reference[5:] if so.reference.startswith("file:") else None
        ref_f = references.get(ref_key) if (ref_key and references) else None
        if ref_key and ref_f is None:
            continue
        options = []
        master = None if ref_f is not None else by_id.get(so.reference.split(":", 1)[1])
        if ref_f is None and (master is None or master.id not in seg_feats):
            continue
        own = [by_id[x] for x in so.segments if base_owner is not None and by_id[x].owner == base_owner
               and x in seg_feats and by_id[x].song_map is not None]
        cands = [by_id[x] for x in so.segments if x in seg_feats and by_id[x].song_map is not None
                 and by_id[x].end - by_id[x].start >= CHAIN_MIN_SEG_S
                 and track(x, seg_feats[x]).reliable]
        if beat_base == "reference":
            options = [master]                         # None = the reference file
        elif own:
            options = [max(own, key=lambda x: x.end - x.start)]
        elif beat_base == "take":
            options = cands or [master if master is not None else None]
        else:                                          # auto
            options = [master] + [c for c in cands if master is None or c.id != master.id]
        best = None
        for take in options:
            res = _sync_song(so, by_id, seg_feats, take, ref_f, ref_key, track)
            cost = _cross_lag_cost(so, by_id, seg_feats, res[0]) if len(options) > 1 else 0.0
            if best is None or cost < best[0]:
                best = (cost, res)
        maps, infos, brk = best[1]
        for sid in so.segments:
            if sid in maps:
                by_id[sid].song_map = maps[sid]
            by_id[sid].breaks = brk.get(sid, [])
            report[sid] = infos[sid]


def _apply_anchors(anchors, segments, conflicts):
    """Make song maps pass through musical anchors (seg_a, t_a, seg_b, t_b) so that
    map_a(t_a) == map_b(t_b).

    For each anchor one side moves: never a song's base take, never a segment that already
    served as the fixed side of an earlier anchor; otherwise b. A moving segment's map gets a
    piecewise-linear offset that is exact at all of its anchors (constant beyond the outermost
    ones). Targets depend on other maps that may move too, so corrections are recomputed from
    the original maps until they settle (chains of anchors across several nights converge in
    as many rounds as the chain is long; cycles that cannot all hold are reported)."""
    by_id = {s.id: s for s in segments}
    moves: dict[str, list[tuple[float, str, float]]] = {}      # mover -> [(t, other seg, other t)]
    pinned = {s.id for s in segments if _is_base(s)}
    for a, ta, b, tb in anchors:
        sa, sb = by_id.get(a), by_id.get(b)
        if sa is None or sb is None:
            conflicts.append(f"anchor {a} / {b}: unknown segment")
            continue
        if sa.song_id is None or sa.song_id != sb.song_id or sa.song_map is None or sb.song_map is None:
            conflicts.append(f"anchor {a} / {b}: segments ended in different songs (ignored)")
            continue
        if b in pinned and a not in pinned:
            a, ta, b, tb = b, tb, a, ta
        elif b in pinned and a in pinned:
            conflicts.append(f"anchor {a} / {b}: both sides already fixed (ignored)")
            continue
        moves.setdefault(b, []).append((float(tb), a, float(ta)))
        pinned.add(a)
    orig = {sid: by_id[sid].song_map for sid in moves}
    for _ in range(len(moves) + 2):
        changed = False
        for sid, pts in moves.items():
            pts = sorted(pts)
            xs = np.array([p[0] for p in pts])
            ds = np.array([float(by_id[o].song_map(to)) - float(orig[sid](t)) for t, o, to in pts])
            m = orig[sid]
            kx = np.union1d(m.knots_src, xs)
            ky = np.asarray(m(kx), np.float64) + np.interp(kx, xs, ds)
            for i in range(1, ky.size):          # anchors win; keep the map strictly increasing
                ky[i] = max(ky[i], ky[i - 1] + 0.05 * (kx[i] - kx[i - 1]))
            new = TimeMap(kx, ky)
            cur = by_id[sid].song_map
            if cur.knots_src.size != kx.size or np.max(np.abs(cur(kx) - ky)) > 1e-9:
                changed = True
            by_id[sid].song_map = new
        if not changed:
            break
    for a, ta, b, tb in anchors:
        sa, sb = by_id.get(a), by_id.get(b)
        if sa is not None and sb is not None and sa.song_map is not None and sb.song_map is not None \
                and sa.song_id == sb.song_id and abs(float(sa.song_map(ta)) - float(sb.song_map(tb))) > 1e-3:
            conflicts.append(f"anchor {a} / {b}: could not be met together with the other anchors")


def _is_base(s: SongSegment) -> bool:
    return s.match_score >= 1.0 and s.song_map is not None and s.song_map.is_linear


def assign_songs(segments: list[SongSegment], seg_feats: dict[str, Features],
                 references: Optional[dict[str, Features]], allow_transpose: bool = False,
                 similar: Optional[float] = None, ref_accept: Optional[float] = None,
                 tags: Optional[dict[str, str]] = None,
                 must_link: Optional[list[tuple[str, str]]] = None,
                 cannot_link: Optional[list[tuple[str, str]]] = None,
                 conflicts: Optional[list] = None,
                 anchors: Optional[list[tuple[str, float, str, float]]] = None,
                 beat_sync: bool = True, beat_report: Optional[dict] = None,
                 beat_base: str = "auto", base_owner: Optional[str] = None) -> list[Song]:
    """Give every segment a song (fills song_id, song_map, match_score) and return the songs.

    references: name -> Features of a reference audio file. The key may be a path (then the
    name is its stem and Song.reference "file:<key>") or a plain name.
    With references, each segment's best reference by subsequence DTW (segment = query) is
    accepted at >= SONG_REF_ACCEPT; song time = reference file time. Segments no reference
    accepts are clustered among themselves (as without references). Without references:
    pairwise similarity -> cluster_segments at SONG_SIMILAR; master = longest member
    (partial ones weighted by MASTER_PARTIAL_WEIGHT), song time = seconds from the master's start.
    Songs are ordered by first appearance in ``segments``; unnamed ones are "Song 01", ...
    similar / ref_accept override config.SONG_SIMILAR / SONG_REF_ACCEPT (calibration, tests).
    Untagged segments without seg_feats (or under 0.2 s of chroma) are left without a song.

    User hints (docs/ARCHITECTURE.md "User hints"; all ids are segment ids):
      tags         segment id -> song title. Tagged segments always get a song with that title,
                   whatever the audio says (no reference auto-match for them). Titles compare
                   trimmed, whitespace-collapsed and case-insensitive; one title = one song,
                   named by the first spelling seen. Every tagged segment also seeds the
                   clustering: untagged segments similar to ANY tagged member join its song.
                   A title equal to a reference stem (case-insensitive) names that reference's
                   song; its time base is the reference when a tagged take matches it at all
                   (DTW >= TAG_REF_MIN or a chunk chain) - the other members are mapped through
                   that take - else the longest tagged live take, and segments that matched the
                   reference directly are re-aligned onto it. A tagged segment
                   without usable features gets song_map = time from its own start, score 0.
      must_link    pairs forced into one song (clustering only: a pair involving a segment
                   that went to a reference or has no features is reported and ignored).
      cannot_link  pairs that never share a song (also among reference matches: the weaker
                   untagged match is sent to clustering instead).
      anchors      musical anchors [(seg_a, owner_t_a, seg_b, owner_t_b)] (owner time of each
                   segment): the two moments are the same moment of the song. They imply
                   must_link(seg_a, seg_b), and after beat sync the maps are corrected to pass
                   exactly through every anchor (the master / reference side stays, the other
                   side gets a piecewise-linear offset, constant beyond its outermost anchors).
      beat_sync    snap song maps to tracked beats (beats.py): take beats go onto the base's
                   beats (reference file or master take), one knot per beat; members or bases
                   without a reliable beat keep the DTW map.
      beat_report  optional dict (out-param): segment id -> {beat_synced, snapped_frac, bpm,
                   reliable, base_reliable, beat_base, pairs, used, pieces, knots, rms_ms}.
      beat_base    "auto" | "reference" | "take": what the beats are snapped to (see _beat_sync).
                   Song time stays the reference / master take either way; with a take as beat
                   base, take->take alignment goes through that take's beats.
      base_owner   clip or group id of the preferred beat-base take (the master night).
      Also fills SongSegment.beat_synced and SongSegment.breaks (owner times of structure
      changes against the beat base: cut there, stretch elsewhere).
      conflicts    optional list (out-param); a human-readable string is appended for every hint
                   that was ignored: unknown ids, must/cannot contradictions (cannot_link wins),
                   links that contradict tags (tags win).
    """
    similar = SONG_SIMILAR if similar is None else similar
    ref_accept = SONG_REF_ACCEPT if ref_accept is None else ref_accept
    conflicts = conflicts if conflicts is not None else []
    anchors = list(anchors or [])
    anchor_pairs = {frozenset((a, b)) for a, _, b, _ in anchors}
    must_link = list(must_link or []) + [(a, b) for a, _, b, _ in anchors]
    order = {s.id: k for k, s in enumerate(segments)}
    by_id = {s.id: s for s in segments}

    def usable(s):
        return s.id in seg_feats and seg_feats[s.id].chroma.shape[1] >= 2

    def own_time(s):
        s.song_map, s.match_score = TimeMap.linear(1.0, -s.start, s.start, max(s.end, s.start + 1.0)), 0.0

    # ---- hints: normalise
    tag_key: dict[str, str] = {}
    titles: dict[str, str] = {}
    for sid, title in (tags or {}).items():
        if sid not in by_id:
            conflicts.append(f"tag {title!r}: unknown segment {sid}")
            continue
        key = _title_key(title)
        if not key:
            continue
        titles.setdefault(key, " ".join(str(title).split()))
        tag_key[sid] = key
    cannot_ids: set = set()
    for a, b in (cannot_link or []):
        if a not in by_id or b not in by_id:
            conflicts.append(f"cannot_link {a} / {b}: unknown segment")
        elif a == b:
            conflicts.append(f"cannot_link {a} / {b}: same segment")
        elif a in tag_key and tag_key.get(a) == tag_key.get(b):
            conflicts.append(f"cannot_link {a} / {b}: both tagged {titles[tag_key[a]]!r} (tag wins)")
        else:
            cannot_ids.add(frozenset((a, b)))
    ref_by_title = {}
    if references:
        for k in references:
            ref_by_title.setdefault(_title_key(_ref_name(k)), k)

    songs_tmp: list[tuple[int, dict]] = []      # (first appearance, song fields)
    todo = [s for s in segments if usable(s) and s.id not in tag_key]
    tagged_clu = [s for s in segments if s.id in tag_key and tag_key[s.id] not in ref_by_title]
    tagged_ref = [s for s in segments if s.id in tag_key and tag_key[s.id] in ref_by_title]

    # ---- references
    if references:
        rnames = list(references.keys())
        rindex = {k: i for i, k in enumerate(rnames)}
        rcs = [_cens_sim(references[k]) for k in rnames]
        matched: dict[int, list[SongSegment]] = {}
        unmatched = []
        for s in todo:
            q = _cens_sim(seg_feats[s.id])
            # segments far longer than a song cannot be (only) that song
            cand = [k for k, rc in enumerate(rcs) if q.shape[1] <= 2 * rc.shape[1]]
            best, bk = None, -1
            if cand and q.shape[1] >= 2:
                scores = dtw.match_many(q, [rcs[k] for k in cand], allow_transpose=allow_transpose)
                i = int(np.argmax([m.score for m in scores]))
                bk = cand[i]
                if scores[i].score >= ref_accept:
                    best = dtw.match(q, rcs[bk], return_path=True, allow_transpose=allow_transpose)
            if best is not None and best.score >= ref_accept:
                tm, _ = _align_map(seg_feats[s.id], references[rnames[bk]], s.start, 0.0,
                                   allow_transpose, coarse=best)
                s.song_map, s.match_score = tm, float(best.score)
                matched.setdefault(bk, []).append(s)
                continue
            # live versions with jams / extra sections: chunk chains (see _chunk_chain)
            ck = _chain_reference(q, rcs) if q.shape[1] >= CHAIN_MIN_SEG_S * SIM_RATE else None
            if ck is not None:
                k, sc, pts = ck
                s.song_map = _fit_map(pts[:, 0] + s.start, pts[:, 1])
                s.match_score = float(sc)
                matched.setdefault(k, []).append(s)
            else:
                unmatched.append(s)
        ref_tm: dict[str, tuple] = {}            # tagged segment -> (map owner->reference, score)
        for s in tagged_ref:                     # title names a reference
            k = rindex[ref_by_title[tag_key[s.id]]]
            if not usable(s):
                own_time(s)
                matched.setdefault(k, []).append(s)
                continue
            q = _cens_sim(seg_feats[s.id])
            best = dtw.match(q, rcs[k], return_path=True, allow_transpose=allow_transpose) \
                if q.shape[1] <= 2 * rcs[k].shape[1] else None
            if best is not None and best.score >= TAG_REF_MIN:
                tm, _ = _align_map(seg_feats[s.id], references[rnames[k]], s.start, 0.0,
                                   allow_transpose, coarse=best)
                ref_tm[s.id] = (tm, float(best.score))
            elif q.shape[1] >= CHAIN_MIN_SEG_S * SIM_RATE:
                ck = _chain_reference(q, [rcs[k]])
                if ck is not None:
                    ref_tm[s.id] = (_fit_map(ck[2][:, 0] + s.start, ck[2][:, 1]), float(ck[1]))
        # cannot_link inside one reference song: the weaker untagged match goes to clustering
        for k, segs in matched.items():
            while True:
                bad = next(((x, y) for x in segs for y in segs
                            if order[x.id] < order[y.id] and frozenset((x.id, y.id)) in cannot_ids
                            and (x.id not in tag_key or y.id not in tag_key)), None)
                if bad is None:
                    break
                drop = min((z for z in bad if z.id not in tag_key), key=lambda z: z.match_score)
                segs.remove(drop)
                drop.song_map, drop.match_score = None, 0.0
                unmatched.append(drop)
        todo = unmatched
    else:
        rnames, rindex, matched, ref_tm = [], {}, {}, {}

    # ---- clustering of the rest (with hint constraints). Every tagged segment with features
    # takes part - also those whose title names a reference - so it seeds its song: untagged
    # takes similar to ANY tagged member join it even when the studio reference does not match
    # the live arrangement.
    clu = sorted(todo + [s for s in tagged_clu + tagged_ref if usable(s)], key=lambda s: order[s.id])
    no_feat_tagged = [s for s in tagged_clu if not usable(s)]
    idx = {s.id: i for i, s in enumerate(clu)}
    parent = list(range(len(clu)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def members_of(root):
        return [j for j in range(len(clu)) if find(j) == root]

    first_of_key: dict[str, int] = {}
    for s in clu:                                 # one tag title = one group
        if s.id in tag_key:
            r = first_of_key.setdefault(tag_key[s.id], idx[s.id])
            parent[find(idx[s.id])] = find(r)
    for a, b in (must_link or []):
        if a not in by_id or b not in by_id:
            conflicts.append(f"must_link {a} / {b}: unknown segment")
            continue
        if a not in idx or b not in idx:
            if frozenset((a, b)) not in anchor_pairs:     # anchors are checked after assignment
                conflicts.append(f"must_link {a} / {b}: segment matched to a reference or without features (ignored)")
            continue
        ra, rb = find(idx[a]), find(idx[b])
        if ra == rb:
            continue
        ga, gb = members_of(ra), members_of(rb)
        keys = {tag_key[clu[j].id] for j in ga + gb if clu[j].id in tag_key}
        if len(keys) > 1:
            conflicts.append(f"must_link {a} / {b}: tagged with different titles (tag wins)")
        elif any(frozenset((clu[i].id, clu[j].id)) in cannot_ids for i in ga for j in gb):
            conflicts.append(f"must_link {a} / {b}: contradicts a cannot_link (cannot_link wins)")
        else:
            parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for j in range(len(clu)):
        groups.setdefault(find(j), []).append(j)
    pre = [g for g in groups.values() if len(g) > 1 or clu[g[0]].id in tag_key]
    labels = {idx[s.id]: tag_key[s.id] for s in clu if s.id in tag_key}
    cannot_idx = set()
    for p in cannot_ids:
        a, b = tuple(p)
        if a in idx and b in idx:
            cannot_idx.add(frozenset((idx[a], idx[b])))

    clusters = []
    S = None
    if clu:
        S = _similarity_matrix([seg_feats[s.id] for s in clu], allow_transpose)
        clusters = cluster_segments(clu, S, similar, pre=pre, cannot=cannot_idx, labels=labels)
    done_keys = set()
    ref_title: dict[int, str] = {}               # reference index -> tag key that named it
    for members in clusters:
        ms = [clu[i] for i in members]
        tk = next((labels[i] for i in members if i in labels), None)
        rk = rindex[ref_by_title[tk]] if tk and tk in ref_by_title else None
        ref_anchors = [i for i in members if clu[i].id in ref_tm] if rk is not None else []
        if ref_anchors:
            # time base = the reference, reached through the best-matching tagged take
            mi = max(ref_anchors, key=lambda i: ref_tm[clu[i].id][1])
        elif tk:
            # time base = a tagged live take (the longest)
            mi = max((i for i in members if i in labels), key=lambda i: clu[i].end - clu[i].start)
        else:
            # master: the longest take, non-partial ones preferred unless clearly shorter (a
            # full take whose music reaches the end of its file is flagged partial; a 70 s
            # fragment must not become the time base of 3-minute takes)
            mi = max(members, key=lambda i: (clu[i].end - clu[i].start) *
                     (1.0 if not clu[i].is_partial else MASTER_PARTIAL_WEIGHT))
        master = clu[mi]
        master.song_map = TimeMap.linear(1.0, -master.start, master.start, master.end)
        master.match_score = 1.0
        for i in members:
            if i != mi:
                clu[i].song_map = _align_to_master(clu[i], master, seg_feats, allow_transpose)
                clu[i].match_score = float(S[i, mi])
        if tk:
            done_keys.add(tk)
            extra = [s for s in no_feat_tagged if tag_key[s.id] == tk]
            for s in extra:
                own_time(s)
            ms = ms + extra
        if ref_anchors:
            base, sc0 = ref_tm[master.id]
            for s in ms:
                if s.song_map is None:
                    continue
                rel = s.song_map         # owner time -> seconds into the master take
                s.song_map = base.compose(TimeMap(rel.knots_src, rel.knots_dst + master.start))
            master.match_score = sc0
            matched.setdefault(rk, []).extend(ms)
            ref_title[rk] = tk
            continue
        if rk is not None:
            # the reference does not match any tagged take: the live take is the time base and
            # segments that matched the reference directly are re-aligned onto it
            for s in matched.pop(rk, []):
                if usable(s):
                    s.song_map = _align_to_master(s, master, seg_feats, allow_transpose)
                    s.match_score = 0.0
                ms.append(s)
        songs_tmp.append((min(order[s.id] for s in ms), dict(
            name=titles[tk] if tk else None, reference=f"segment:{master.id}",
            duration=float(master.end - master.start), segs=ms)))
    for k, segs in matched.items():
        if not segs:
            continue
        key = rnames[k]
        tk = ref_title.get(k) or next((tag_key[s.id] for s in segs if s.id in tag_key), None)
        songs_tmp.append((min(order[s.id] for s in segs), dict(
            name=titles[tk] if tk else _ref_name(key), reference=f"file:{key}",
            duration=float(references[key].duration), segs=segs)))
    for tk in dict.fromkeys(tag_key[s.id] for s in no_feat_tagged):   # titles with no audio at all
        if tk in done_keys:
            continue
        ms = [s for s in no_feat_tagged if tag_key[s.id] == tk]
        for s in ms:
            own_time(s)
        m = max(ms, key=lambda s: s.end - s.start)
        songs_tmp.append((min(order[s.id] for s in ms), dict(
            name=titles[tk], reference=f"segment:{m.id}", duration=float(m.end - m.start), segs=ms)))

    songs_tmp.sort(key=lambda x: x[0])
    songs: list[Song] = []
    n_unnamed = 0
    for k, (_, d) in enumerate(songs_tmp):
        sid = f"song{k + 1:02d}"
        name = d["name"]
        if name is None:
            n_unnamed += 1
            name = f"Song {n_unnamed:02d}"
        segs = sorted(d["segs"], key=lambda s: order[s.id])
        for s in segs:
            s.song_id = sid
        songs.append(Song(id=sid, name=name, reference=d["reference"], duration=d["duration"],
                          segments=[s.id for s in segs]))
    if beat_sync:
        rep = beat_report if beat_report is not None else {}
        _beat_sync(songs, segments, seg_feats, references, rep, beat_base, base_owner)
        for sg in segments:
            if sg.id in rep:
                sg.beat_synced = bool(rep[sg.id].get("beat_synced"))
    if anchors:
        _apply_anchors(anchors, segments, conflicts)
    return songs


# =========================================================================== point propagation

_PROP_TOL = 0.05


def propagate_points(times, from_owner: str, to_owner: str, segments: list[SongSegment],
                     groups: list[EventGroup]) -> list[Optional[float]]:
    """Convert time points from one owner's time to another's (owners: clip ids or group ids).

    * Same event group (or clip <-> its own group): exact, via the group's clip->event maps.
      Points the target clip did not record give None.
    * Otherwise via song time: the source point must lie in a segment with a song; the target
      gets the point in its first (earliest) segment of the same song whose span contains the
      song time (a song played twice resolves to the earlier take). None if nothing contains it.
    """
    gby = {g.id: g for g in groups}
    gof = {c: g for g in groups for c in g.maps}

    def base(owner):
        if owner in gby:
            return owner
        g = gof.get(owner)
        return g.id if g is not None else owner

    def to_base(owner, t):
        if owner in gby:
            return t
        g = gof.get(owner)
        return float(g.maps[owner](t)) if g is not None else t

    def from_base(owner, tb):
        if owner in gby:
            g = gby[owner]
            return tb if -_PROP_TOL <= tb <= g.duration + _PROP_TOL else None
        g = gof.get(owner)
        if g is None:
            return tb
        lo, hi = g.spans[owner]
        if not (lo - _PROP_TOL <= tb <= hi + _PROP_TOL):
            return None
        return float(g.maps[owner].inverse()(tb))

    bf, bt = base(from_owner), base(to_owner)
    out: list[Optional[float]] = []
    for t in times:
        x = to_base(from_owner, float(t))
        if bf == bt:
            out.append(from_base(to_owner, x))
            continue
        src = next((s for s in segments if s.owner == bf and s.song_id and s.song_map is not None
                    and s.start - _PROP_TOL <= x <= s.end + _PROP_TOL), None)
        if src is None:
            out.append(None)
            continue
        st = float(src.song_map(x))
        res = None
        for s in sorted((s for s in segments if s.owner == bt and s.song_id == src.song_id
                         and s.song_map is not None), key=lambda s: s.start):
            y = float(s.song_map.inverse()(st))
            if s.start - _PROP_TOL <= y <= s.end + _PROP_TOL:
                r = from_base(to_owner, y)
                if r is not None:
                    res = r
                    break
        out.append(res)
    return out


# ---- pure-python twin (no numpy) for the Resolve plugin ------------------------------------

def _pwl_py(src, dst, t):
    """Piecewise-linear map with end-slope extrapolation (same as models.TimeMap.__call__)."""
    n = len(src)
    if t <= src[0]:
        k = 0
    elif t >= src[-1]:
        k = n - 2
    else:
        k = max(0, min(n - 2, bisect.bisect_right(src, t) - 1))
    x0, x1, y0, y1 = src[k], src[k + 1], dst[k], dst[k + 1]
    return y0 + (t - x0) * (y1 - y0) / (x1 - x0)


def propagate_points_json(result: dict, times, from_owner: str, to_owner: str) -> list:
    """Same as propagate_points but on the AnalysisResult JSON dict (json.loads of to_json()).

    Standard library only; the Resolve plugin may copy this function together with _pwl_py.
    Uses result["groups"][*]["maps"/"spans"/"id"] and result["segments"][*]
    ["owner"/"start"/"end"/"song_id"/"song_map"]. Returns a list of float or None.
    """
    tol = 0.05
    groups = {g["id"]: g for g in result.get("groups", [])}
    gof = {}
    for g in result.get("groups", []):
        for c in g["maps"]:
            gof[c] = g

    def gdur(g):
        return max(v[1] for v in g["spans"].values()) if g["spans"] else 0.0

    def base(owner):
        if owner in groups:
            return owner
        g = gof.get(owner)
        return g["id"] if g is not None else owner

    def to_base(owner, t):
        if owner in groups:
            return t
        g = gof.get(owner)
        if g is None:
            return t
        m = g["maps"][owner]
        return _pwl_py(m["src"], m["dst"], t)

    def from_base(owner, tb):
        if owner in groups:
            return tb if -tol <= tb <= gdur(groups[owner]) + tol else None
        g = gof.get(owner)
        if g is None:
            return tb
        lo, hi = g["spans"][owner]
        if not (lo - tol <= tb <= hi + tol):
            return None
        m = g["maps"][owner]
        return _pwl_py(m["dst"], m["src"], tb)

    segs = [s for s in result.get("segments", []) if s.get("song_id") and s.get("song_map")]
    bf, bt = base(from_owner), base(to_owner)
    out = []
    for t in times:
        x = to_base(from_owner, float(t))
        if bf == bt:
            out.append(from_base(to_owner, x))
            continue
        src = None
        for s in segs:
            if s["owner"] == bf and s["start"] - tol <= x <= s["end"] + tol:
                src = s
                break
        if src is None:
            out.append(None)
            continue
        st = _pwl_py(src["song_map"]["src"], src["song_map"]["dst"], x)
        res = None
        for s in sorted((s for s in segs if s["owner"] == bt and s["song_id"] == src["song_id"]),
                        key=lambda s: s["start"]):
            y = _pwl_py(s["song_map"]["dst"], s["song_map"]["src"], st)
            if s["start"] - tol <= y <= s["end"] + tol:
                r = from_base(to_owner, y)
                if r is not None:
                    res = r
                    break
        out.append(res)
    return out
