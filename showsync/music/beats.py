"""Beat and downbeat tracking on the 100 Hz onset envelope, and beat-snapping of song maps.

numpy only. Used by music.songs to make song_map beat-accurate across performances.

* ``tempo_period``   dominant beat period from the onset autocorrelation with a log-normal
                     prior around 0.5 s (120 bpm) - live rock sits at 0.35-0.8 s.
* ``track_beats``    dynamic-programming beat tracker (Ellis 2007): maximise onset strength at
                     beats minus a log-period-deviation penalty between consecutive beats; beats
                     are refined to the onset peak within +-30 ms (parabolic, sub-frame).
* ``downbeat_phase`` 4/4 bar phase: the beat phase (0-3) with the largest chord change
                     (chroma between consecutive bars) plus onset accent.
* ``Beats``          result with a reliability flag: clear pulse (autocorrelation peak) and
                     beats that sit on onsets (mean onset at beats vs everywhere).
* ``snap_map``       refine a coarse take -> song-time map so take beats land on the base's
                     beats, smoothly: beat pairs -> outlier rejection -> structure-change split
                     -> bar-level monotone least-squares fit (see its docstring).
* ``onset_lag_ms``   measurement helper: drum-hit misalignment of two takes through their maps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from showsync.config import CHROMA_RATE, ENV_RATE
from showsync.models import TimeMap

PERIOD_RANGE = (0.3, 1.0)       # s per beat searched
PRIOR_CENTER = 0.5              # s, log-normal prior centre (120 bpm)
PRIOR_OCT = 0.9                 # prior width in octaves (1 sigma)
TIGHTNESS = 80.0                # DP penalty weight on (log(interval / period))^2
REFINE_S = 0.03                 # beat refinement window (+-)
MIN_CLARITY = 0.15              # autocorrelation peak needed for a reliable pulse
MIN_ONSET_RATIO = 1.4           # mean onset at beats / mean onset, needed for reliability
SNAP_FRAC = 0.35                # take beat may move at most this many beat periods
MAP_SLOPE = (0.7, 1.4)
OUTLIER_BARS = 4                # local trend window (+- bars) for outlier rejection
OUTLIER_S = 0.04                # pairs further than this from the local trend are dropped
MIN_PIECE_BEATS = 8             # a structure change must persist this many beats (2 bars)
LOCAL_SLOPE = (0.85, 1.18)      # bar-to-bar slope limits relative to the piece's tempo ratio
PIECE_MAX_RMS = 0.05            # a piece whose smooth fit is worse than this is dropped
ENV_WIN_S = 4.0                 # envelope_align: window
ENV_HOP_S = 2.0
ENV_MAXLAG_BEATS = 1.6          # lags searched (beats): resolves one-beat slips of the coarse map
ENV_NEIGH = 5                   # +- windows whose correlation curves are summed (~ +-10 s)
ENV_MIN_CORR = 0.15
ENV_OUTLIER_S = 0.06
ENV_KNOT_BARS = 2.0
SMOOTH = 4.0                    # penalty weight on slope changes between bars


@dataclass
class Beats:
    times: np.ndarray            # beat times (s, relative to the envelope start)
    period: float                # median beat period (s)
    downbeat: int                # index of the first downbeat in ``times`` (0-3)
    clarity: float               # autocorrelation peak (0..1)
    onset_ratio: float           # mean onset at beats / overall mean
    reliable: bool

    @property
    def downbeats(self) -> np.ndarray:
        return self.times[self.downbeat::4]


def _pos(onset: np.ndarray) -> np.ndarray:
    o = np.maximum(np.asarray(onset, np.float64), 0.0)
    sd = o.std()
    return o / sd if sd > 1e-9 else o


def tempo_period(onset: np.ndarray, rate: float = ENV_RATE) -> tuple[float, float]:
    """(period s, clarity) from the autocorrelation of the onset envelope."""
    o = _pos(onset)
    o = o - o.mean()
    n = o.size
    lo, hi = int(PERIOD_RANGE[0] * rate), int(PERIOD_RANGE[1] * rate)
    if n < 3 * hi:
        return PRIOR_CENTER, 0.0
    F = np.fft.rfft(o, 2 * n)
    ac = np.fft.irfft(np.abs(F) ** 2)[:2 * hi + 1]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(lo, hi + 1)
    # an octave-consistent score: the lag and its double both strong
    sc = ac[lags] + 0.5 * ac[np.minimum(2 * lags, 2 * hi)]
    prior = np.exp(-0.5 * (np.log2(lags / rate / PRIOR_CENTER) / PRIOR_OCT) ** 2)
    k = int(np.argmax(sc * prior))
    L = lags[k]
    # parabolic refinement of the lag
    if lo < L < hi:
        y0, y1, y2 = ac[L - 1], ac[L], ac[L + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        L = L + float(np.clip(d, -0.5, 0.5))
    return float(L / rate), float(max(0.0, ac[int(round(L))]))


def track_beats(onset: np.ndarray, rate: float = ENV_RATE, period: Optional[float] = None,
                tightness: float = TIGHTNESS) -> np.ndarray:
    """Beat times (s) by dynamic programming over the onset envelope."""
    o = _pos(onset)
    n = o.size
    if period is None:
        period, _ = tempo_period(onset, rate)
    P = period * rate
    if n < 2 * P:
        return np.zeros(0)
    d = np.arange(max(1, int(round(P / 2))), int(round(2 * P)) + 1)
    pen = tightness * np.log(d / P) ** 2
    score = o.copy()
    back = np.full(n, -1, np.int64)
    for t in range(int(d[0]), n):
        prev_idx = t - d
        ok = prev_idx >= 0
        cand = score[prev_idx[ok]] - pen[ok]
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] = o[t] + cand[k]
            back[t] = prev_idx[ok][k]
    # end at the best score in the last period
    tail = np.arange(max(0, n - int(round(P))), n)
    t = int(tail[np.argmax(score[tail])])
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    b = np.array(beats[::-1], np.float64)
    # refine to the onset peak (parabolic) within +-REFINE_S
    w = int(REFINE_S * rate)
    out = []
    for f in b.astype(int):
        a, c = max(1, f - w), min(n - 2, f + w)
        if c <= a:
            out.append(f)
            continue
        m = a + int(np.argmax(o[a:c + 1]))
        y0, y1, y2 = o[m - 1], o[m], o[m + 1]
        den = y0 - 2 * y1 + y2
        dd = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        out.append(m + float(np.clip(dd, -0.5, 0.5)))
    return np.asarray(out) / rate


def downbeat_phase(beats: np.ndarray, onset: np.ndarray, chroma: Optional[np.ndarray],
                   rate: float = ENV_RATE, crate: float = CHROMA_RATE) -> int:
    """Phase 0-3 of the downbeats in a 4/4 beat sequence: chord changes happen on the one."""
    nb = beats.size
    if nb < 9:
        return 0
    o = _pos(onset)
    acc = np.array([o[min(o.size - 1, int(round(b * rate)))] for b in beats])
    chg = np.zeros(nb)
    if chroma is not None and chroma.shape[1] > 0:
        C = np.asarray(chroma, np.float64)
        idx = np.clip((beats * crate).astype(int), 0, C.shape[1] - 1)
        # chroma averaged over each beat
        B = np.stack([C[:, idx[i]:max(idx[i] + 1, idx[min(i + 1, nb - 1)])].mean(axis=1) for i in range(nb)], axis=1)
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
        # change between the 2 beats before and the 2 beats from here
        for i in range(2, nb - 2):
            a = B[:, i - 2:i].mean(axis=1)
            c = B[:, i:i + 2].mean(axis=1)
            chg[i] = 1.0 - float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    s = z(chg) + 0.5 * z(acc)
    return int(np.argmax([s[p::4].mean() for p in range(4)]))


def beat_track(onset: np.ndarray, chroma: Optional[np.ndarray] = None,
               period: Optional[float] = None) -> Beats:
    """Beats of one segment. ``period`` forces the metrical level (used to track a take at the
    same level as its base: live takes were tracked at 88 / 118 / 176 bpm for one song
    otherwise, and beats of different levels snap onto off-beats)."""
    p0, clarity = tempo_period(onset)
    forced = period is not None
    if period is None:
        period = p0
    times = track_beats(onset, period=period)
    o = _pos(onset)
    if times.size >= 2:
        at = np.array([o[min(o.size - 1, int(round(t * ENV_RATE)))] for t in times])
        ratio = float(at.mean() / (o.mean() + 1e-9))
        per = float(np.median(np.diff(times)))
    else:
        ratio, per = 0.0, period
    ph = downbeat_phase(times, onset, chroma)
    # with a forced period (take tracked at its base's level) the own-tempo clarity is moot
    rel = (forced or clarity >= MIN_CLARITY) and ratio >= MIN_ONSET_RATIO and times.size >= 16
    return Beats(times, per, ph, clarity, ratio, bool(rel))


def _lis(y: np.ndarray) -> np.ndarray:
    """Indices of a longest strictly increasing subsequence of y."""
    import bisect
    tails, tail_idx, prev = [], [], np.full(y.size, -1)
    for i, v in enumerate(y):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v); tail_idx.append(i)
        else:
            tails[k] = v; tail_idx[k] = i
        prev[i] = tail_idx[k - 1] if k else -1
    out, i = [], tail_idx[-1] if tail_idx else -1
    while i >= 0:
        out.append(i); i = prev[i]
    return np.array(out[::-1], int)


def _local_residuals(x: np.ndarray, y: np.ndarray, half: float) -> np.ndarray:
    """Residual of each pair against a robust local line through its neighbours within
    +-half seconds (itself excluded): slope = median of neighbour-to-neighbour slopes,
    intercept = median offset. A least-squares line is dragged by beats snapped one beat off."""
    r = np.zeros(x.size)
    for i in range(x.size):
        m = np.abs(x - x[i]) <= half
        m[i] = False
        idx = np.nonzero(m)[0]
        if idx.size < 4:
            continue
        xs, ys = x[idx], y[idx]
        d = np.diff(xs)
        ok = d > 1e-6
        if not ok.any():
            continue
        k = float(np.median(np.diff(ys)[ok] / d[ok]))
        c = float(np.median(ys - k * (xs - x[i])))
        r[i] = y[i] - c
    return r


def _pieces(x: np.ndarray, y: np.ndarray, period: float, jump: float,
            min_n: int = MIN_PIECE_BEATS) -> list[np.ndarray]:
    """Split pairs where the offset changes by >= jump and stays changed for >= MIN_PIECE_BEATS
    pairs (a structure change: extra / missing bars). Returns index arrays."""
    n = x.size
    starts = [0]
    i = min_n
    while i + min_n <= n:
        a = slice(max(starts[-1], i - 2 * min_n), i)
        if x[a].size >= 3:
            A = np.stack([x[a], np.ones(x[a].size)], axis=1)
            c, *_ = np.linalg.lstsq(A, y[a], rcond=None)
            ahead = y[i:i + min_n] - (c[0] * x[i:i + min_n] + c[1])
            if (np.all(ahead > jump) or np.all(ahead < -jump)) and i - starts[-1] >= 2 * min_n and n - i >= 2 * min_n:
                starts.append(i)
                i += min_n
                continue
        i += 1
    bounds = starts + [n]
    return [np.arange(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _smooth_fit(x: np.ndarray, y: np.ndarray, knot_dx: float, lo: float, hi: float,
                smooth: float) -> tuple[np.ndarray, np.ndarray]:
    """Knots every knot_dx over [x0, xN]; values = least squares through (x, y) (linear
    interpolation between knots) with slopes in [lo, hi] and a penalty on slope changes."""
    from scipy.optimize import lsq_linear
    o = np.argsort(x, kind="stable")
    x, y = np.asarray(x, np.float64)[o], np.asarray(y, np.float64)[o]
    if x[-1] - x[0] < 1e-3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.array([x[0], x[0] + 1.0]), np.array([y[0], y[0] + lo])
    nk = max(2, int(round((x[-1] - x[0]) / knot_dx)) + 1)
    kx = np.linspace(x[0], x[-1], nk)
    # basis: value at knot k = v0 + sum_{j<k} s_j * dx_j
    dx = np.diff(kx)
    seg = np.clip(np.searchsorted(kx, x, side="right") - 1, 0, nk - 2)
    frac = (x - kx[seg]) / dx[seg]
    A = np.zeros((x.size, nk))
    A[:, 0] = 1.0
    cum = np.concatenate([[0.0], np.cumsum(dx)])
    for i in range(x.size):
        k = seg[i]
        A[i, 1:k + 1] = dx[:k]
        A[i, k + 1] = frac[i] * dx[k]
    R = np.zeros((max(0, nk - 2), nk))
    for k in range(nk - 2):
        R[k, k + 1], R[k, k + 2] = -smooth, smooth
    lb = np.concatenate([[-np.inf], np.full(nk - 1, lo)])
    ub = np.concatenate([[np.inf], np.full(nk - 1, hi)])
    sol = lsq_linear(np.vstack([A, R]), np.concatenate([y, np.zeros(R.shape[0])]),
                     bounds=(lb, ub), method="bvls").x
    ky = sol[0] + np.concatenate([[0.0], np.cumsum(sol[1:] * dx)])
    return kx, ky


def snap_map(coarse: TimeMap, take_beats: np.ndarray, base_beats: np.ndarray, base_period: float,
             t0: float, t1: float, slope: tuple[float, float] = MAP_SLOPE,
             stats: Optional[dict] = None) -> tuple[Optional[TimeMap], float]:
    """Smooth, beat-accurate version of ``coarse`` (take owner time -> song time) over [t0, t1].

    1. Pairs: each take beat goes to the base beat nearest to coarse(beat) within
       SNAP_FRAC * base_period; a longest increasing subsequence keeps them monotonic.
    2. Outliers: pairs whose residual against a local line over +-OUTLIER_BARS bars exceeds
       OUTLIER_S are dropped (twice) - beats snapped onto a neighbour, beat-position noise.
    3. Structure changes: the pairs are split where the offset jumps by >= half a beat and
       stays there for >= MIN_PIECE_BEATS beats (extra / skipped bars).
    4. Each piece: monotone piecewise-linear least-squares fit with one knot per bar
       (4 base beats), slopes within [LOCAL_SLOPE] x the piece's mean tempo ratio, penalty on
       slope changes. Between pieces the map changes slope once (the real discontinuity).
    Returns (map or None, fraction of take beats used). ``stats`` (optional dict) receives
    pairs, used, pieces, knots, rms_ms (residual of used pairs against the map), breaks
    (take times where a new piece starts: structure changes).
    """
    tb = take_beats[(take_beats >= t0) & (take_beats <= t1)]
    if tb.size < 8 or base_beats.size < 8:
        return None, 0.0
    pred = np.asarray(coarse(tb), np.float64)
    j = np.clip(np.searchsorted(base_beats, pred), 1, base_beats.size - 1)
    near = np.where(np.abs(base_beats[j - 1] - pred) <= np.abs(base_beats[j] - pred), j - 1, j)
    ok = np.abs(base_beats[near] - pred) <= SNAP_FRAC * base_period
    xs, ys = tb[ok], base_beats[near[ok]]
    if xs.size < 8:
        return None, float(ok.mean())
    st = {} if stats is not None else None
    tm, frac = _fit_pairs(xs, ys, base_period, slope, st, tb.size)
    if stats is not None:
        stats.update(st, pairs=int(ok.sum()))
    return tm, frac


def _env_curves(coarse, a, take_t0, b, base_t0, centers, off, lags, rate):
    """Normalised cross-correlation curves (windows x lags) of envelope a (take) against b
    (base) read through ``coarse``; NaN rows where a window falls outside either envelope."""
    curves = np.full((centers.size, lags.size), np.nan)
    for k, c in enumerate(centers):
        tt = c + off
        ia = (tt - take_t0) * rate
        if ia[0] < 0 or ia[-1] >= a.size - 1:
            continue
        xa = np.interp(ia, np.arange(a.size), a)
        xa = xa - xa.mean()
        na = np.linalg.norm(xa)
        if na < 1e-9:
            continue
        mb = np.asarray(coarse(tt), np.float64)
        pos = (mb[None, :] + lags[:, None] - base_t0) * rate
        if pos.min() < 0 or pos.max() >= b.size - 1:
            continue
        xb = np.interp(pos.ravel(), np.arange(b.size), b).reshape(pos.shape)
        xb = xb - xb.mean(axis=1, keepdims=True)
        curves[k] = (xb @ xa) / (na * np.linalg.norm(xb, axis=1) + 1e-9)
    return curves


def envelope_align(coarse: TimeMap, take_env: np.ndarray, take_t0: float, base_env: np.ndarray,
                   base_t0: float, t0: float, t1: float, base_period: float,
                   rate: float = ENV_RATE, slope: tuple[float, float] = MAP_SLOPE,
                   stats: Optional[dict] = None) -> Optional[TimeMap]:
    """Correct ``coarse`` (take owner time -> base time) by direct onset-envelope alignment.

    Beat snapping trusts the coarse map to within a third of a beat; when the DTW map is a beat
    off (common at fast tempi: 176 bpm = 340 ms), every take beat snaps onto the wrong base
    beat and the smooth fit follows it - a consistent one-beat slip. Here, in ENV_WIN_S windows
    every ENV_HOP_S, the take's onset envelope is compared with the base's envelope read
    through the coarse map at lags up to +-ENV_MAXLAG_BEATS beats; the correlation curves of
    +-ENV_NEIGH neighbouring windows are summed before picking the lag (one window alone is
    ambiguous between beats; the drum pattern over ~20 s is not), refined parabolically.
    Pairs (t, coarse(t) + lag) with a summed peak >= ENV_MIN_CORR are fitted like beat pairs
    (outliers, structure pieces, bar-level smooth monotone fit). Returns the map or None."""
    tes = take_env if isinstance(take_env, (list, tuple)) else [take_env]
    bes = base_env if isinstance(base_env, (list, tuple)) else [base_env]
    W = int(ENV_WIN_S * rate)
    L = int(round(ENV_MAXLAG_BEATS * base_period * rate))
    lags = np.arange(-L, L + 1) / rate
    centers = np.arange(t0 + ENV_WIN_S / 2 + 0.5, t1 - ENV_WIN_S / 2 - 0.5, ENV_HOP_S)
    if centers.size < 6:
        return None
    off = (np.arange(W) - W / 2) / rate
    curves = np.zeros((centers.size, lags.size))
    for a, b in zip(tes, bes):
        curves += _env_curves(coarse, _pos(a), take_t0, _pos(b), base_t0, centers, off, lags, rate)
    curves /= len(tes)
    xs, ys = [], []
    for k in range(centers.size):
        lo, hi = max(0, k - ENV_NEIGH), min(centers.size, k + ENV_NEIGH + 1)
        blk = curves[lo:hi]
        blk = blk[~np.isnan(blk).any(axis=1)]
        if blk.shape[0] < max(2, ENV_NEIGH) or np.isnan(curves[k]).any():
            continue
        s = blk.mean(axis=0)
        j = int(np.argmax(s))
        if s[j] < ENV_MIN_CORR:
            continue
        # the window's own lag near the consensus peak (+- a quarter beat)
        w = int(round(0.25 * base_period * rate))
        jl = max(0, j - w) + int(np.argmax(curves[k][max(0, j - w):j + w + 1]))
        d = 0.0
        if 0 < jl < lags.size - 1:
            y0, y1, y2 = curves[k][jl - 1], curves[k][jl], curves[k][jl + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        xs.append(centers[k])
        ys.append(float(coarse(centers[k])) + lags[jl] + d / rate)
    if len(xs) < 6:
        return None
    tm, used = _fit_pairs(np.asarray(xs), np.asarray(ys), base_period, slope, stats, len(centers),
                          min_pairs=6, piece_min=8, outlier_s=ENV_OUTLIER_S, knot_bars=ENV_KNOT_BARS)
    return tm


def _fit_pairs(xs, ys, base_period, slope, stats, n_total, min_pairs=8, piece_min=MIN_PIECE_BEATS,
               outlier_s=OUTLIER_S, knot_bars=1.0):
    """Shared fitting of (take time, base time) pairs: monotone (LIS), outliers against a
    robust local trend, structure pieces, bar-level smooth monotone least squares per piece.
    Returns (TimeMap or None, fraction of n_total used)."""
    if xs.size < min_pairs:
        return None, 0.0
    keep = _lis(ys)
    xs, ys = xs[keep], ys[keep]
    bar = 4.0 * base_period
    for _ in range(2):
        if xs.size < min_pairs:
            return None, 0.0
        r = _local_residuals(xs, ys, OUTLIER_BARS * bar)
        m = np.abs(r) <= outlier_s
        xs, ys = xs[m], ys[m]
    if xs.size < min_pairs:
        return None, 0.0
    kx_all, ky_all, kept_idx = [], [], []
    used_last = 0
    pieces = _pieces(xs, ys, base_period, max(0.5 * base_period, 0.12), piece_min)
    for idx in pieces:
        if idx.size < piece_min:
            continue
        x, y = xs[idx], ys[idx]
        r = float(np.polyfit(x, y, 1)[0]) if x.size >= 2 else 1.0
        lo = max(slope[0], LOCAL_SLOPE[0] * r)
        hi = min(slope[1], LOCAL_SLOPE[1] * r)
        if lo >= hi:
            continue
        kx, ky = _smooth_fit(x, y, knot_bars * bar / max(r, 1e-6), lo, hi, SMOOTH)
        if np.sqrt(np.mean((np.interp(x, kx, ky) - y) ** 2)) > PIECE_MAX_RMS:
            continue                      # an undetected structure change inside: leave it out
        if kx_all and ky[0] <= ky_all[-1][-1] + 1e-3:
            if idx.size <= used_last:
                continue
            kx_all.pop(); ky_all.pop(); kept_idx.pop()
        kx_all.append(kx); ky_all.append(ky); kept_idx.append(idx)
        used_last = idx.size
    if not kx_all:
        return None, 0.0
    kx = np.concatenate(kx_all)
    ky = np.concatenate(ky_all)
    keep = np.concatenate([[True], (np.diff(kx) > 1e-6) & (np.diff(ky) > 1e-6)])
    tm = TimeMap(kx[keep], ky[keep])
    ki = np.concatenate(kept_idx)
    if stats is not None:
        res = np.asarray(tm(xs[ki]), np.float64) - ys[ki]
        stats.update(breaks=[float(k[0]) for k in kx_all[1:]], used=int(ki.size), pieces=len(kx_all),
                     knots=int(keep.sum()), rms_ms=float(np.sqrt(np.mean(res ** 2)) * 1000.0))
    return tm, float(ki.size / max(1, n_total))


def onset_lag_ms(env_a: np.ndarray, env_b: np.ndarray, a_to_b, t_a0: float, t_a1: float,
                 win_s: float = 4.0, hop_s: float = 8.0, max_lag_s: float = 0.25,
                 min_corr: float = 0.3, rate: float = ENV_RATE) -> np.ndarray:
    """Residual drum-hit misalignment (ms) of take b against take a through a mapping
    a_to_b(t_a) -> t_b: in windows of win_s every hop_s, b's onset envelope is resampled at the
    mapped times and cross-correlated with a's over +-max_lag_s. Returns the lags (ms) of the
    windows whose normalised correlation peak >= min_corr (positive = b late)."""
    a = _pos(env_a)
    b = _pos(env_b)
    L = int(max_lag_s * rate)
    out = []
    for t in np.arange(t_a0 + max_lag_s, t_a1 - win_s - max_lag_s, hop_s):
        ta = t + np.arange(int(win_s * rate)) / rate
        tt = np.arange(-L, int(win_s * rate) + L) / rate + t
        tb = np.asarray(a_to_b(tt), np.float64)
        if tb.min() < 0 or tb.max() * rate >= b.size - 1:
            continue
        xa = np.interp(ta * rate, np.arange(a.size), a)
        xb = np.interp(tb * rate, np.arange(b.size), b)
        xa = xa - xa.mean()
        xb = xb - xb.mean()
        na = np.linalg.norm(xa)
        if na < 1e-9:
            continue
        cc = np.array([xa @ xb[l:l + xa.size] / (na * np.linalg.norm(xb[l:l + xa.size]) + 1e-9)
                       for l in range(2 * L + 1)])
        k = int(np.argmax(cc))
        if cc[k] < min_corr:
            continue
        d = 0.0
        if 0 < k < 2 * L:
            y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        out.append((k - L + d) / rate * 1000.0)
    return np.asarray(out)
