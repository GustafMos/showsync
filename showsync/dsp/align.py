"""Exact (event-time) sync of two clips:  t_a = map(t_b).

Pipeline (see docs/ARCHITECTURE.md, "dsp/align.py"):

1. Coarse. Internally the shorter clip S is slid over the longer clip L (the map is inverted at
   the end if S is clip a). S's onset envelope is cut into COARSE_WIN_S windows; each window is
   cross-correlated with all of L's envelope by FFT and normalised per lag by the energy of both
   signals under the overlapping part (true normalised xcorr, partial overlap at the edges
   allowed down to COARSE_MIN_OVERLAP_S). The envelope is Gaussian-smoothed (sigma 20 ms) so a
   30 s window stays coherent at 2000 ppm drift (60 ms smear over the window). Long clips use
   a capped number of windows: dense near both ends of S (an overlap of two clips always
   contains an end of the shorter one unless S lies wholly inside L) plus an even spread.
2. Line fit. RANSAC over (t_S, t_L) peak points with slope in [1-MAX_DRIFT, 1+MAX_DRIFT],
   score-weighted support with at most one point per window. The top few *distinct* lines are
   kept as candidates rather than only the best one.
3. Screen + fine. Up to 30 candidate lines get a cheap GCC-PHAT screen (3 probes of 2 s at
   the coarse slope, +-1000 ppm when the slope is unknown, a 500 ppm grid over +-MAX_DRIFT if
   nothing passes). The best 3 get a fine check: GCC-PHAT on raw audio (band FINE_BAND, window
   FINE_WIN_S, +-FINE_SEARCH_S), with the L segment resampled along the predicted line so drift
   does not smear the window; slope refined on a 250/50 ppm grid when the overlap is short.
   The winner is measured every FINE_HOP_S (at least 8 overlapping windows for snippets).
4. Fit. fit_map(): RANSAC + weighted least squares line, inliers within INLIER_MS. If the
   points follow a smooth curve (loose-line RMS > NONLINEAR_RESIDUAL_MS, or > INLIER_MS with
   clearly fewer strict inliers), the winner is re-measured every PIECEWISE_KNOT_S/8 along the
   curve and a robust LOESS curve is sampled at knots every PIECEWISE_KNOT_S (nonlinear=True).
5. Confidence. z = GCC-PHAT peak z-score per window (noise ~3.5..5.5). With s = clip((z - 5.5)
   / 4, 0, 1), hop/win the window overlap, n_in/f_in the inlier count/fraction:
       E     = sum(s over inliers) * sqrt(min(1, hop/win))
       n_eff = n_in * min(1, hop/win)                  independent consistent windows
       conf  = (1 - exp(-E/1.5)) * sqrt(clip((n_eff-1)/1.5) * clip((f_in-0.15)/0.35))
               * min(clip((1 - Q2/Q1) / 0.5), clip((0.75 - cs) / 0.4))
   Q1/Q2 = mean per-window strength (z - 5.5)+ of the winner and the runner-up candidate.
   cs = density ambiguity, judged on each candidate's OWN overlap (up to 6 candidates get a
   quick 10-window fine check): for every competing line (another placement >= 0.25 s away)
       cs = min(1, dens2/dens1) * clip(n_eff2 - 1) * min(1, frac2 / (0.6 * f_in))
   dens = mean (z - 5.5) of the windows agreeing with that line (strict INLIER_MS), n_eff2 =
   its independent agreeing windows, frac2 = its agreeing fraction; the winner's dens1/f_in
   come from its full pass. A studio track whose sections repeat digitally (music-video
   playback) gives competing placements that are internally as consistent as the true one,
   even when the true placement overlaps less -> confidence ~0 and PairMatch.note starts
   "ambiguous: alternative at <+-x> s" (shift of the alternative placement in clip a's time).
   Live repeats differ (crowd, fills, timing), so a wrong chorus has few, weak agreeing
   windows and costs nothing. Different performances / unrelated music give no line of
   consistent sample-level peaks, so n_eff or f_in collapse and conf ~0 (measured max 0.000 on 32 synth
   false pairs; true pairs >= 0.93 except two noted failures, see tests/test_dsp.py).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from ..config import (ANALYSIS_SR, ENV_RATE, COARSE_WIN_S, COARSE_HOP_S, COARSE_MIN_OVERLAP_S,
                      MAX_DRIFT, FINE_WIN_S, FINE_HOP_S, FINE_SEARCH_S, FINE_BAND, INLIER_MS,
                      NONLINEAR_RESIDUAL_MS, PIECEWISE_KNOT_S, MIN_INLIERS, CHROMA_RATE,
                      MATCH_ACCEPT)
from ..models import Features, PairMatch, TimeMap

# ---- internal tuning (not part of the contract)
_COARSE_SIGMA_S = 0.02        # envelope smoothing before coarse xcorr
_COARSE_MAX_WINDOWS = 48      # cap on coarse windows for long clips
_COARSE_EDGE_S = 150.0        # region at each end of S searched with every window
_COARSE_PEAKS = 3             # peaks kept per window
_COARSE_PEAKS_SINGLE = 20     # peaks kept when S is one window (snippet)
_COARSE_SUPPRESS_S = 0.5
_COARSE_RATIO_STEP = 4e-3     # coarse slope grid step (residual <= 2000 ppm)
_COARSE_EDGE_WINS = (10.0, 20.0)  # extra short windows at both ends of S
_RANSAC_TOL_S = 0.08
_N_CANDIDATES = 30           # coarse lines passed to the GCC screen
_SCREEN_WIN_S = 2.0
_SCREEN_PROBES = 3
_SCREEN_SLOPE_STEP = 5e-4     # 500 ppm grid: <= 0.5 ms smear in a 2 s probe
_SCREEN_PASS_Z = 7.0          # mean probe z that skips the slope-grid pass
_SCREEN_GRID_TOP = 5
_EXACT_DENS = 1.5             # mean (z - 5.5) of agreeing windows for an 'exact' placement
_EXACT_REL = 0.5              # ... and at least this fraction of the winner's density
_SCAN_Z = 8.0                 # full-scan peak z threshold
_SCAN_MAX_SAMPLES = 1 << 24   # full scan only for L up to ~35 min at 8 kHz (playback takes);
                              # longer clips use the coarse candidate lines instead
_MUSIC_BOTH = 0.5             # window mean musicness for 'both clips hold music'
import os as _os
_DEBUG = bool(_os.environ.get('SHOWSYNC_ALIGN_DEBUG'))
_CHROMA_AGREE = 0.6           # smoothed chroma cosine below this contradicts a placement
_CONTRA_MARGIN = 0.12         # a rival may be contradicted on at most this much more of the
                              # both-music frames than the winner
_HINT_SLOPE_SPAN_S = 60.0     # hint points this far apart give the slope
_ALT_MAX_TRIES = 16          # distinct placements refined when looking for exact alternatives
_AMB_LO, _AMB_HI = 0.35, 0.75    # competitor density ratio: no penalty .. confidence 0
_N_EVALUATE = 6               # screened candidates given a full fine check
_QUICK_WINDOWS = 10           # fine windows per candidate in the ranking pass
_MIN_FINE_POINTS = 8          # snippets: at least this many (overlapping) fine windows
_SLOPE_KNOWN_SPAN_S = 240.0   # coarse support span above which the coarse slope is trusted
_LOOSE_TOL_MS = 25.0          # line tolerance used to detect a smooth nonlinear curve
_Z0, _ZS, _E0 = 5.5, 4.0, 1.5
_MIN_RMS_DB = -65.0


# =========================================================================== GCC-PHAT

def _band_mask(n_fft: int, sr: int, band) -> np.ndarray:
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    return ((f >= band[0]) & (f <= band[1])).astype(np.float32)


def _gcc_batch(S: np.ndarray, Lw: np.ndarray, sr: int, M: int, band):
    """S (k, n) windows, Lw (k, n+2M) segments where perfect alignment puts S[i] at Lw[i+M].

    Returns (delta_samples, z, peak) for each row; delta > 0 means L content is later than
    predicted.
    """
    k, n = S.shape
    N = sfft.next_fast_len(n + 2 * M, real=True)
    X = sfft.rfft(S, N, axis=1, workers=-1)
    Y = sfft.rfft(Lw, N, axis=1, workers=-1)
    G = Y * np.conj(X)
    mag = np.abs(G)
    G /= mag + 1e-12 * (mag.max(axis=1, keepdims=True) + 1e-30)
    bm = _band_mask(N, sr, band)
    G *= bm
    cc = sfft.irfft(G, N, axis=1, workers=-1)[:, :2 * M + 1]
    cc *= N / (2.0 * max(bm.sum(), 1.0))           # 1.0 == perfectly coherent
    d = np.argmax(cc, axis=1)
    rows = np.arange(k)
    peak = cc[rows, d]
    # sub-sample: parabola through the peak and its neighbours
    dm = np.clip(d - 1, 0, 2 * M)
    dp = np.clip(d + 1, 0, 2 * M)
    y0, y1, y2 = cc[rows, dm], peak, cc[rows, dp]
    den = y0 - 2 * y1 + y2
    frac = np.where((np.abs(den) > 1e-12) & (d > 0) & (d < 2 * M), 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    frac = np.clip(frac, -0.5, 0.5)
    # noise level: mean / std of cc away from the peak (+-2 ms excluded)
    ex = max(4, int(0.002 * sr))
    tot1 = cc.sum(axis=1)
    tot2 = (cc * cc).sum(axis=1)
    cs1 = np.concatenate([np.zeros((k, 1)), np.cumsum(cc, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((k, 1)), np.cumsum(cc * cc, axis=1)], axis=1)
    a0 = np.clip(d - ex, 0, 2 * M + 1)
    a1 = np.clip(d + ex + 1, 0, 2 * M + 1)
    cnt = (2 * M + 1) - (a1 - a0)
    s1 = tot1 - (cs1[rows, a1] - cs1[rows, a0])
    s2 = tot2 - (cs2[rows, a1] - cs2[rows, a0])
    mu = s1 / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu * mu, 0.0))
    z = (peak - mu) / np.maximum(sd, 1e-9)
    return (d + frac - M), z, peak


def gcc_phat(x, y, sr: int = ANALYSIS_SR, max_lag_s: float = FINE_SEARCH_S, band=FINE_BAND):
    """GCC-PHAT delay of y relative to x:  y(t) ~ x(t - lag_s).  Returns (lag_s, peak_quality).

    x and y should cover roughly the same stretch of sound (equal length is simplest).
    peak_quality is the z-score of the PHAT peak against the rest of the searched lags:
    unrelated audio gives ~3.5..5, a reliable match is >= 8.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.size, y.size)
    M = max(1, int(round(max_lag_s * sr)))
    S = (x[:n] * _taper(n))[None, :]
    Lw = np.zeros((1, n + 2 * M))
    Lw[0, M:M + n] = y[:n]
    d, z, _ = _gcc_batch(S, Lw, sr, M, band)
    return float(d[0]) / sr, float(z[0])


_TAPER_CACHE: dict = {}


def _taper(n: int) -> np.ndarray:
    """Tukey(0.1) window, cached."""
    n = int(n)
    w = _TAPER_CACHE.get(n)
    if w is None:
        w = np.ones(n, dtype=np.float32)
        m = max(1, int(0.05 * n))
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = ramp
        w[n - m:] = ramp[::-1]
        if len(_TAPER_CACHE) > 256:
            _TAPER_CACHE.clear()
        _TAPER_CACHE[n] = w
    return w


# =========================================================================== fine measurement

def _fine_measure(aS: np.ndarray, aL: np.ndarray, sr: int, pred, centres, win_s: float,
                  search_s: float = FINE_SEARCH_S, band=FINE_BAND):
    """Measure the offset error of `pred` at window centres in S time.

    pred is a callable t_S -> t_L, or a tuple (r, c) of per-window arrays (line t_L = r t + c).

    Returns (t_S_actual, t_L_measured, z, valid) arrays (t_S_actual is the exact window centre
    after rounding to samples). Within each window the L segment is resampled along the local
    slope of pred.
    """
    centres = np.asarray(centres, dtype=np.float64)
    out_c = centres.copy()
    n = max(64, int(round(win_s * sr)))
    M = max(1, int(round(search_s * sr)))
    out_t = np.full(centres.size, np.nan)
    out_z = np.zeros(centres.size)
    valid = np.zeros(centres.size, dtype=bool)
    if centres.size == 0:
        return out_c, out_t, out_z, valid
    taper = _taper(n)
    j32 = np.arange(n + 2 * M, dtype=np.float32)
    i_n = np.arange(n)
    rms_floor = 10 ** (_MIN_RMS_DB / 20)
    B = 48
    for b0 in range(0, centres.size, B):
        cs = centres[b0:b0 + B]
        k0 = np.round((cs - win_s / 2) * sr).astype(np.int64)
        ok = (k0 >= 0) & (k0 + n <= aS.size)
        # local linear prediction over the window
        t0 = (k0 - M) / sr
        t1 = (k0 + n + M) / sr
        if callable(pred):
            p0 = np.asarray(pred(t0), dtype=np.float64) * sr
            p1 = np.asarray(pred(t1), dtype=np.float64) * sr
        else:                                   # per-window lines (r_i, c_i)
            rr, cc_ = np.asarray(pred[0])[b0:b0 + B], np.asarray(pred[1])[b0:b0 + B]
            p0 = (rr * t0 + cc_) * sr
            p1 = (rr * t1 + cc_) * sr
        slope = (p1 - p0) / (n + 2 * M)
        # the S window's image must lie inside L; the search margins may hang over (zeros)
        ok &= (p0 + slope * M >= 0) & (p0 + slope * (M + n) + 1 < aL.size)
        idx_rows = np.nonzero(ok)[0]
        if idx_rows.size == 0:
            continue
        Sm = aS[k0[idx_rows, None] + i_n[None, :]]
        base = np.floor(p0[idx_rows])
        pos = (p0[idx_rows] - base).astype(np.float32)[:, None] +             slope[idx_rows].astype(np.float32)[:, None] * j32[None, :]
        ipr = pos.astype(np.int32)                       # pos >= 0, floor == trunc
        fr = pos - ipr
        ip = base.astype(np.int64)[:, None] + ipr
        inside = (ip >= 0) & (ip < aL.size - 1)
        ip = np.clip(ip, 0, aL.size - 2)
        Lw = aL[ip]
        Lw += (aL[ip + 1] - Lw) * fr
        if not inside.all():
            Lw *= inside
        rs = np.sqrt(np.mean(Sm * Sm, axis=1))
        rl = np.sqrt(np.mean(Lw * Lw, axis=1))
        sig = (rs > rms_floor) & (rl > rms_floor)
        Sm = (Sm - Sm.mean(axis=1, keepdims=True)) * taper
        d, z, _ = _gcc_batch(Sm, Lw, sr, M, band)
        rows = b0 + idx_rows
        mid = M + (n - 1) / 2.0
        out_c[rows] = (k0[idx_rows] + (n - 1) / 2.0) / sr
        pc = (p0[idx_rows] + slope[idx_rows] * mid) / sr
        out_t[rows] = pc + d * slope[idx_rows] / sr
        out_z[rows] = np.where(sig, z, 0.0)
        valid[rows] = sig
    return out_c, out_t, out_z, valid


# =========================================================================== map fitting

def _wls_line(x, y, w):
    W = w.sum()
    if W <= 0 or x.size < 2:
        return 1.0, float(np.median(y - x)) if y.size else 0.0
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    vx = (w * (x - mx) ** 2).sum()
    if vx <= 1e-12:
        r = 1.0
    else:
        r = (w * (x - mx) * (y - my)).sum() / vx
    r = float(np.clip(r, 1 - MAX_DRIFT, 1 + MAX_DRIFT))
    return r, float(my - r * mx)


def _ransac_line(x, y, w, tol, rng, n_hyp=1500):
    """Best (r, c, inlier mask) with slope in the drift range; support = sum of inlier weights."""
    n = x.size
    if n == 0:
        return 1.0, 0.0, np.zeros(0, bool)
    hyps_r = [np.ones(n)]
    hyps_c = [y - x]                                  # every point with slope 1
    if n >= 2:
        if n * (n - 1) // 2 <= n_hyp:
            ii, jj = np.triu_indices(n, 1)
        else:
            ii = rng.integers(0, n, n_hyp)
            jj = rng.integers(0, n, n_hyp)
        dx = x[jj] - x[ii]
        good = np.abs(dx) > 1e-6
        ii, jj, dx = ii[good], jj[good], dx[good]
        r = (y[jj] - y[ii]) / dx
        keep = np.abs(r - 1) <= MAX_DRIFT * 1.05
        hyps_r.append(np.clip(r[keep], 1 - MAX_DRIFT, 1 + MAX_DRIFT))
        hyps_c.append(y[ii[keep]] - hyps_r[-1] * x[ii[keep]])
    R = np.concatenate(hyps_r)
    C = np.concatenate(hyps_c)
    best, best_s = 0, -1.0
    for h0 in range(0, R.size, 2048):
        res = np.abs(y[None, :] - (R[h0:h0 + 2048, None] * x[None, :] + C[h0:h0 + 2048, None]))
        s = ((res < tol) * w[None, :]).sum(axis=1)
        k = int(np.argmax(s))
        if s[k] > best_s:
            best_s, best = float(s[k]), h0 + k
    r, c = float(R[best]), float(C[best])
    for _ in range(3):
        inl = np.abs(y - (r * x + c)) < tol
        if inl.sum() < 2:
            break
        r, c = _wls_line(x[inl], y[inl], w[inl] + 1e-9)
    inl = np.abs(y - (r * x + c)) < tol
    return r, c, inl


def _loess(x, y, w, q, hw):
    """Weighted local quadratic regression of y(x) evaluated at q (tricube kernel, half width hw).
    Returns (value, ok) where ok marks queries with enough support."""
    q = np.asarray(q, dtype=np.float64)
    vals = np.full(q.size, np.nan)
    okm = np.zeros(q.size, dtype=bool)
    for q0 in range(0, q.size, 512):
        qq = q[q0:q0 + 512]
        dx = x[None, :] - qq[:, None]
        u = np.abs(dx) / hw
        k = np.where(u < 1, (1 - u ** 3) ** 3, 0.0) * w[None, :]
        s0, s1, s2 = k.sum(1), (k * dx).sum(1), (k * dx ** 2).sum(1)
        s3, s4 = (k * dx ** 3).sum(1), (k * dx ** 4).sum(1)
        t0, t1, t2 = (k * y).sum(1), (k * dx * y).sum(1), (k * dx ** 2 * y).sum(1)
        A = np.stack([np.stack([s0, s1, s2], -1), np.stack([s1, s2, s3], -1), np.stack([s2, s3, s4], -1)], -2)
        bvec = np.stack([t0, t1, t2], -1)
        npts = (k > 0).sum(1)
        good = npts >= 4
        # quadratic where well supported, else local linear, else weighted mean
        sol = np.full(qq.size, np.nan)
        if good.any():
            Ag = A[good] + np.eye(3)[None] * 1e-12 * (s0[good, None, None] + 1e-30)
            try:
                sol[good] = np.linalg.solve(Ag, bvec[good][..., None])[:, 0, 0]
            except np.linalg.LinAlgError:
                pass
        lin = ~np.isfinite(sol) & (npts >= 2)
        if lin.any():
            det = s0[lin] * s2[lin] - s1[lin] ** 2
            sol[lin] = np.where(np.abs(det) > 1e-18, (s2[lin] * t0[lin] - s1[lin] * t1[lin]) / np.where(det == 0, 1, det),
                                t0[lin] / np.maximum(s0[lin], 1e-30))
        mean = ~np.isfinite(sol) & (s0 > 0)
        sol[mean] = t0[mean] / s0[mean]
        vals[q0:q0 + 512] = sol
        okm[q0:q0 + 512] = npts >= 2
    return vals, okm


def _fit_curve(x, y, w, lo, hi, knot_s, line):
    """Robust smooth curve through points near `line` (r, c). Returns
    (out_map, dense_map, inlier_mask) or None."""
    r0, c0 = line
    res0 = y - (r0 * x + c0)
    hw = 0.6 * knot_s
    ww = w.copy()
    for it in range(4):
        f, ok = _loess(x, res0, ww, x, hw)
        rr = res0 - np.where(ok, f, 0.0)
        sel = ww > 0
        mad = 1.4826 * np.median(np.abs(rr[sel])) if sel.any() else 1e-3
        cst = max(6.0 * mad, 2.0 * INLIER_MS / 1000.0)
        u = np.clip(rr / cst, -1, 1)
        ww = w * (1 - u * u) ** 2 * ok
    tol = 1.5 * INLIER_MS / 1000.0
    inl = (np.abs(rr) < tol) & (w > 0) & ok
    nk = max(2, int(np.ceil((hi - lo) / knot_s - 1e-9)) + 1)
    knots = np.linspace(lo, hi, nk)
    kv, kok = _loess(x, res0, ww, knots, hw)
    if not np.all(np.isfinite(kv)):
        if np.isfinite(kv).sum() < 2:
            return None
        g = np.isfinite(kv)
        kv = np.interp(knots, knots[g], kv[g])
    dense_t = np.linspace(lo, hi, max(2, int(np.ceil(hi - lo)) + 1))
    dv, _ = _loess(x, res0, ww, dense_t, hw)
    g = np.isfinite(dv)
    if g.sum() < 2:
        return None
    dv = np.interp(dense_t, dense_t[g], dv[g])
    ky = r0 * knots + c0 + kv
    dy = r0 * dense_t + c0 + dv
    if not (np.all(np.diff(ky) > 0) and np.all(np.diff(dy) > 0)):
        return None
    return TimeMap(knots, ky), TimeMap(dense_t, dy), inl


def _fit(x, y, w, span=None, knot_s=PIECEWISE_KNOT_S, allow_nonlinear=True):
    """Internal fit_map that also returns the dense curve used to steer re-measurement."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y) & (w > 0)
    inl_full = np.zeros(x.size, dtype=bool)
    rng = np.random.default_rng(1234)
    if good.sum() == 0:
        return TimeMap.identity(), float("inf"), inl_full, False, None, 0.0, 0
    xg, yg, wg = x[good], y[good], w[good]
    lo, hi = (float(xg.min()), float(xg.max())) if span is None else (float(span[0]), float(span[1]))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    tol = INLIER_MS / 1000.0
    r, c, inl = _ransac_line(xg, yg, wg, tol, rng)
    best_map, best_inl, nonlinear, dense = TimeMap.linear(r, c, lo, hi), inl, False, None
    rms_loose = 0.0
    n_loose = 0
    if good.sum() >= 6:
        rl, cl, inl_l = _ransac_line(xg, yg, wg, _LOOSE_TOL_MS / 1000.0, rng)
        if inl_l.sum() >= max(6, 0.5 * xg.size):
            n_loose = int(inl_l.sum())
            rl, cl = _wls_line(xg[inl_l], yg[inl_l], wg[inl_l])
            rr = yg[inl_l] - (rl * xg[inl_l] + cl)
            rms_loose = float(np.sqrt(np.mean(rr ** 2))) * 1000
            pspan = xg[inl_l].max() - xg[inl_l].min()
            suspect = rms_loose > NONLINEAR_RESIDUAL_MS or (
                rms_loose > INLIER_MS and inl.sum() < 0.85 * inl_l.sum())
            if allow_nonlinear and suspect and pspan >= 2 * knot_s:
                fc = _fit_curve(xg, yg, wg * inl_l, lo, hi, knot_s, (rl, cl))
                if fc is not None:
                    om, dm, inl_c = fc
                    if inl_c.sum() >= max(MIN_INLIERS, 1.2 * inl.sum()) and inl_c.sum() >= 0.4 * xg.size:
                        best_map, best_inl, nonlinear, dense = om, inl_c, True, dm
    inl_full[np.nonzero(good)[0]] = best_inl
    if best_inl.sum() > 0:
        resid = y[inl_full] - best_map(x[inl_full])
        rms = float(np.sqrt(np.mean(resid ** 2))) * 1000
    else:
        rms = float("inf")
    return best_map, rms, inl_full, nonlinear, dense, rms_loose, n_loose


def fit_map(points_tb, points_ta, weights, span=None, knot_s: float = PIECEWISE_KNOT_S,
            allow_nonlinear: bool = True):
    """Robust map t_a = map(t_b) through measured points.

    Returns (TimeMap, residual_ms, inlier_mask, nonlinear).
    * Line: RANSAC (slope within MAX_DRIFT) + weighted least squares on inliers (|res| <
      INLIER_MS). residual_ms is the RMS of the inliers.
    * Nonlinear: if the points within _LOOSE_TOL_MS of the best loose line have RMS >
      NONLINEAR_RESIDUAL_MS and span >= 2 knot intervals, a robust local-quadratic smoother
      (LOESS, tricube half width 0.6*knot_s, Tukey reweighting) is fitted and sampled at knots
      every knot_s over `span` (or the point range). Knot values are local estimates of the
      curve (accurate AT the knots); between knots the map interpolates linearly. Used when it
      explains clearly more points within 1.5*INLIER_MS than the line; residual_ms is then the
      inlier RMS around the smooth curve.
    span=(lo, hi) sets the knot range in t_b.
    """
    m, rms, inl, nl, _, _, _ = _fit(points_tb, points_ta, weights, span, knot_s, allow_nonlinear)
    return m, rms, inl, nl


# =========================================================================== coarse stage

def _coarse_env(onset: np.ndarray) -> np.ndarray:
    x = np.asarray(onset, dtype=np.float32)
    x = np.nan_to_num(x)
    return gaussian_filter1d(x, _COARSE_SIGMA_S * ENV_RATE).astype(np.float32)


def _window_starts(dS: float) -> tuple[np.ndarray, float]:
    win = min(COARSE_WIN_S, dS)
    if dS <= COARSE_WIN_S + 1e-6:
        return np.array([0.0]), win
    starts = np.arange(0.0, dS - win + 1e-9, COARSE_HOP_S)
    if dS - win - starts[-1] > 1.0:
        starts = np.append(starts, dS - win)
    if starts.size > _COARSE_MAX_WINDOWS:
        edge = (starts <= _COARSE_EDGE_S) | (starts >= dS - win - _COARSE_EDGE_S)
        rest = np.nonzero(~edge)[0]
        n_rest = max(0, _COARSE_MAX_WINDOWS - int(edge.sum()))
        pick = rest[np.round(np.linspace(0, rest.size - 1, n_rest)).astype(int)] if n_rest and rest.size else []
        keep = edge.copy()
        keep[pick] = True
        starts = starts[keep]
    return starts, win


def _coarse_points(envS: np.ndarray, envL: np.ndarray, dS: float):
    """Normalised xcorr of S windows against L. Returns points (tS, tL, z, win_id).

    Besides the regular COARSE_WIN_S windows, clips longer than one window get short extra
    windows (10 s and 20 s) at both ends of S: a short edge overlap (S starting just before L
    ends) is then covered by a window that lies wholly inside it."""
    out = []
    for rho in _coarse_ratios():
        if rho == 1.0:
            e, d = envS, dS
        else:
            # S's envelope resampled onto a clock running rho times faster (t' = rho t)
            n2 = max(4, int(round(envS.size * rho)))
            e = np.interp(np.arange(n2) / rho, np.arange(envS.size), envS).astype(np.float32)
            d = dS * rho
        starts, win = _window_starts(d)
        pts = [_coarse_group(e, envL, starts, win, 0)]
        if d > win + 1.0:
            wid = int(starts.size)
            for w in _COARSE_EDGE_WINS:
                if w < win:
                    pts.append(_coarse_group(e, envL, np.array([0.0, d - w]), w, wid))
                    wid += 2
        pts = [p for p in pts if p.shape[0]]
        if pts:
            P = np.vstack(pts)
            P[:, 0] /= rho                   # back to S time; window ids shared across ratios
            out.append(np.column_stack([P, np.full(P.shape[0], rho)]))   # col 4: clock ratio
    return np.vstack(out) if out else np.zeros((0, 5))


def _coarse_ratios():
    """Clock ratios at which the coarse stage is run. The 20 ms envelope smoothing keeps a 30 s
    window coherent up to ~2000 ppm of residual slope (60 ms smear), so the drift range
    +-MAX_DRIFT is covered by ratios _COARSE_RATIO_STEP apart (3 ratios at 6000 ppm)."""
    k = int(np.ceil(max(0.0, MAX_DRIFT - _COARSE_RATIO_STEP / 2) / _COARSE_RATIO_STEP))
    return [1.0 + i * _COARSE_RATIO_STEP for i in range(-k, k + 1)]


def _coarse_group(envS, envL, starts, win, wid0):
    rate = float(ENV_RATE)
    nL = envL.size
    nw = max(4, int(round(win * rate)))
    min_ov = int(round(min(COARSE_MIN_OVERLAP_S, 0.8 * win) * rate))
    min_ov = max(4, min(min_ov, nw))
    N = sfft.next_fast_len(nL + nw, real=True)
    XL = sfft.rfft(envL, N, workers=-1)
    csx = np.concatenate([[0.0], np.cumsum(envL.astype(np.float64) ** 2)])
    lags = np.arange(-(nw - min_ov), nL - min_ov + 1)
    a0 = np.clip(lags, 0, nL)
    a1 = np.clip(lags + nw, 0, nL)
    Ex = csx[a1] - csx[a0]
    k_lo = np.maximum(0, -lags)
    k_hi = np.minimum(nw, nL - lags)
    n_ov = (k_hi - k_lo).astype(np.float64)
    neg = int(nw - min_ov)
    n_peaks = _COARSE_PEAKS_SINGLE if starts.size == 1 else _COARSE_PEAKS
    sup = int(_COARSE_SUPPRESS_S * rate)
    pts = []
    B = 8
    for b0 in range(0, starts.size, B):
        st = starts[b0:b0 + B]
        Wm = np.zeros((st.size, nw), dtype=np.float32)
        for i, s in enumerate(st):
            f0 = int(round(s * rate))
            seg = envS[f0:f0 + nw]
            Wm[i, :seg.size] = seg
        Wf = sfft.rfft(Wm, N, axis=1, workers=-1)
        cc = sfft.irfft(XL[None, :] * np.conj(Wf), N, axis=1, workers=-1)
        for i, s in enumerate(st):
            wv = Wm[i].astype(np.float64)
            csw = np.concatenate([[0.0], np.cumsum(wv ** 2)])
            Ew = csw[k_hi] - csw[k_lo]
            c = np.concatenate([cc[i, N - neg:], cc[i, :nL - min_ov + 1]]) if neg > 0 else cc[i, :nL - min_ov + 1]
            ok = (Ex > 0.05 * n_ov) & (Ew > 0.05 * n_ov)
            ncc = np.where(ok, c / np.sqrt(Ex * Ew + 1e-12), 0.0)
            score = ncc * np.sqrt(n_ov / nw)
            sd = float(np.std(score)) + 1e-9
            cand = np.argpartition(-score, min(400, score.size - 1))[:400]
            cand = cand[np.argsort(-score[cand])]
            taken = []
            for ci in cand:
                if all(abs(ci - t) > sup for t in taken):
                    taken.append(ci)
                    if len(taken) >= n_peaks:
                        break
            f0 = int(round(s * rate))
            for ci in taken:
                l = lags[ci]
                fr = 0.0
                if 0 < ci < score.size - 1:
                    y0, y1, y2 = score[ci - 1], score[ci], score[ci + 1]
                    den = y0 - 2 * y1 + y2
                    if abs(den) > 1e-12:
                        fr = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
                kmid = 0.5 * (k_lo[ci] + k_hi[ci])
                tS = (f0 + kmid) / rate
                tL = (kmid + l + fr) / rate
                pts.append((tS, tL, float(score[ci]) / sd, wid0 + b0 + i))
    if not pts:
        return np.zeros((0, 4))
    return np.array(pts, dtype=np.float64)


def _candidate_lines(pts: np.ndarray, rng) -> list[tuple[float, float, float, float]]:
    """Distinct lines (r, c, support, support_span) ranked by score-weighted support."""
    if pts.shape[0] == 0:
        return []
    x, y, z, wid = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3].astype(int)
    rho = pts[:, 4] if pts.shape[1] > 4 else np.ones(x.size)
    w = np.maximum(z - 2.0, 0.05)
    n = x.size
    order = np.argsort(wid, kind="stable")
    x, y, w, wid, z, rho = x[order], y[order], w[order], wid[order], z[order], rho[order]
    uw, first = np.unique(wid, return_index=True)
    hyps_r = [np.ones(n)]                             # single points: slope 1 (refined by the screen)
    hyps_c = [y - x]
    if uw.size > 1:
        n_h = 3000
        if n * (n - 1) // 2 <= n_h:
            ii, jj = np.triu_indices(n, 1)
        else:
            ii = rng.integers(0, n, n_h)
            jj = rng.integers(0, n, n_h)
        good = (wid[ii] != wid[jj]) & (np.abs(x[jj] - x[ii]) > 10.0)
        ii, jj = ii[good], jj[good]
        r = (y[jj] - y[ii]) / (x[jj] - x[ii])
        keep = np.abs(r - 1) <= MAX_DRIFT
        hyps_r.append(r[keep])
        hyps_c.append(y[ii[keep]] - r[keep] * x[ii[keep]])
    R = np.concatenate(hyps_r)
    C = np.concatenate(hyps_c)
    scores = np.zeros(R.size)
    for h0 in range(0, R.size, 1024):
        res = np.abs(y[None, :] - (R[h0:h0 + 1024, None] * x[None, :] + C[h0:h0 + 1024, None]))
        sup = np.where(res < _RANSAC_TOL_S, w[None, :], 0.0)
        # at most one point per window
        scores[h0:h0 + 1024] = np.maximum.reduceat(sup, first, axis=1).sum(axis=1)
    out = []
    xm = 0.5 * (x.min() + x.max())
    for h in np.argsort(-scores):
        if len(out) >= _N_CANDIDATES or scores[h] <= 0:
            break
        r, c = float(R[h]), float(C[h])
        inl = np.abs(y - (r * x + c)) < _RANSAC_TOL_S
        if inl.sum() >= 2 and np.ptp(x[inl]) > 10.0:
            r, c = _wls_line(x[inl], y[inl], w[inl])
            inl = np.abs(y - (r * x + c)) < _RANSAC_TOL_S
        if any(abs((r * xm + c) - (r2 * xm + c2)) < 0.25 for r2, c2, _, _ in out):
            continue
        span = float(np.ptp(x[inl])) if inl.any() else 0.0
        out.append((r, c, float(scores[h]), span))
    return out


# =========================================================================== match

def _overlap(r, c, dS, dL):
    """S-time interval whose image under t_L = r t_S + c lies in [0, dL]."""
    eps = 0.02                       # keep windows clear of the last samples (rounding)
    lo = max(0.0, (0.0 - c) / r) + eps
    hi = min(dS, (dL - c) / r) - eps
    return lo, hi


def _centres(lo, hi, win, hop, min_n=1):
    if hi - lo < win:
        return np.array([0.5 * (lo + hi)])
    n = int(np.floor((hi - lo - win) / hop + 1e-9)) + 1
    n = max(n, min_n)
    return np.linspace(lo + win / 2, hi - win / 2, n) if n > 1 else np.array([0.5 * (lo + hi)])


def _evidence(z, mask, hop, win):
    """Sum of per-window evidence over `mask`, discounted for overlapping windows."""
    s = np.clip((np.asarray(z) - _Z0) / _ZS, 0.0, 1.0)
    return float(s[mask].sum()) * min(1.0, (hop / max(win, 1e-9)) ** 0.5)


def _screen_one(aS, aL, sr, r, c, dS, dL, slopes):
    lo, hi = _overlap(r, c, dS, dL)
    ov = hi - lo
    if ov < 1.0:
        return None
    win = min(_SCREEN_WIN_S, 0.9 * ov)
    if ov > win * 1.05:
        probes = np.linspace(lo + win / 2, hi - win / 2, _SCREEN_PROBES)
    else:
        probes = np.array([0.5 * (lo + hi)])
    tm = 0.5 * (lo + hi)
    G = np.repeat(slopes, probes.size)
    Cg = c + (r - G) * tm
    _, _, z, v = _fine_measure(aS, aL, sr, (G, Cg), np.tile(probes, slopes.size), win)
    q = np.where(v, np.clip(z, 0, 20), 0.0).reshape(slopes.size, probes.size).sum(axis=1)
    k = int(np.argmax(q))
    rr = float(slopes[k])
    return float(q[k]) / probes.size, rr, c + (r - rr) * tm


def _screen(aS, aL, sr, cands, dS, dL):
    """Cheap GCC check of many candidate lines with a few short probe windows each.

    Pass 1 uses the coarse slope (tolerates ~250 ppm of slope error), plus +-1000 ppm when the
    coarse support was too short to measure the slope (24 vs 23.976 fps style mismatch).
    Candidates whose slope is unknown, if none passed clearly, get
    pass 2 over a 500 ppm slope grid spanning +-MAX_DRIFT (only the best _SCREEN_GRID_TOP of
    them, to bound the cost on non-matching pairs). Returns [(score, r, c, slope_known)] best
    first; score is the mean GCC z of the probes."""
    out = []
    for r, c, sup, span in cands:
        known = span >= _SLOPE_KNOWN_SPAN_S
        sl = np.array([r]) if known else np.unique(np.clip(np.concatenate(
            [r + np.array([-1e-3, 0.0, 1e-3]), _coarse_ratios()]), 1 - MAX_DRIFT, 1 + MAX_DRIFT))
        res = _screen_one(aS, aL, sr, r, c, dS, dL, sl)
        if res is not None:
            out.append([res[0], res[1], res[2], span >= _SLOPE_KNOWN_SPAN_S, r, c])
    out.sort(key=lambda e: -e[0])
    if out and out[0][0] < _SCREEN_PASS_Z:
        grid = 1.0 + np.arange(-MAX_DRIFT, MAX_DRIFT + 1e-12, _SCREEN_SLOPE_STEP)
        n = 0
        for e in out:
            if e[3]:
                continue
            res = _screen_one(aS, aL, sr, e[4], e[5], dS, dL, grid)
            if res is not None and res[0] > e[0]:
                e[0], e[1], e[2] = res
            n += 1
            if n >= _SCREEN_GRID_TOP:
                break
        out.sort(key=lambda e: -e[0])
    return [(e[0], e[1], e[2], e[3]) for e in out]


def _window_music(mus, t0, t1):
    """Mean musicness (CHROMA_RATE frames) over [t0, t1] for arrays of t0/t1."""
    if mus is None or np.size(mus) == 0:
        return np.ones(np.size(t0))
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(mus, dtype=np.float64))])
    i0 = np.clip(np.floor(np.asarray(t0) * CHROMA_RATE).astype(int), 0, mus.size)
    i1 = np.clip(np.ceil(np.asarray(t1) * CHROMA_RATE).astype(int), 0, mus.size)
    n = np.maximum(i1 - i0, 1)
    return np.where(i1 > i0, (cs[i1] - cs[i0]) / n, 0.0)


def _contradictions(feats, cen, win, r, c, agree=None):
    """Chroma contradiction of a placement t_L = r t_S + c over the span covered by the window
    centres: 10 Hz frames where BOTH clips clearly hold music (musicness > _MUSIC_BOTH) but the
    1 s-smoothed, mean-centred chroma of the two sides disagrees (cosine < _CHROMA_AGREE).
    Identical audio (true repeat or true placement) gives ~0-0.2 of the both-music frames; a
    placement that lays a unique section (intro, bridge, outro) or other chords over music gives
    0.3-1.0. GCC agreement is not used here: identical drums under different chords still give
    GCC peaks. Returns (n_contra_frames, n_both_frames)."""
    if feats is None or np.size(cen) == 0:
        return 0, 0
    fS, fL = feats
    cen = np.asarray(cen, dtype=np.float64)
    t0, t1 = float(cen.min() - win / 2), float(cen.max() + win / 2)
    iS = np.arange(max(0, int(np.ceil(t0 * CHROMA_RATE))), min(fS.chroma.shape[1], int(t1 * CHROMA_RATE) + 1))
    iL = np.round((r * iS / CHROMA_RATE + c) * CHROMA_RATE).astype(int)
    ok = (iL >= 0) & (iL < fL.chroma.shape[1])
    iS, iL = iS[ok], iL[ok]
    if iS.size < 10:
        return 0, 0
    a = fS.chroma[:, iS].astype(np.float64)
    b = fL.chroma[:, iL].astype(np.float64)
    a = uniform_filter1d(a - a.mean(axis=0), CHROMA_RATE, axis=1)
    b = uniform_filter1d(b - b.mean(axis=0), CHROMA_RATE, axis=1)
    cs = (a * b).sum(axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
    both = (np.asarray(fS.musicness)[iS] > _MUSIC_BOTH) & (np.asarray(fL.musicness)[iL] > _MUSIC_BOTH)
    if both.sum() < 10:
        return 0, 0
    return int((both & (cs < _CHROMA_AGREE)).sum()), int(both.sum())


def _evaluate(aS, aL, sr, r, c, dS, dL, slope_known, n_windows, mus=None):
    """Fine check of one candidate line on n_windows windows. Returns dict or None."""
    lo, hi = _overlap(r, c, dS, dL)
    ov = hi - lo
    if ov < 1.0:
        return None
    win = min(FINE_WIN_S, 0.9 * ov)
    if not slope_known:
        # refine the slope found by the screen (pivot at the overlap centre)
        tm = 0.5 * (lo + hi)
        probes = _centres(lo, hi, win, max(win, (ov - win) / 3), 1)[:4]
        best_r = r
        for span_, step in ((_SCREEN_SLOPE_STEP, 1e-4), (1e-4, 2.5e-5)):
            if step * win < 0.1e-3:
                break
            g = best_r + np.arange(-span_, span_ + 1e-12, step)
            g = g[np.abs(g - 1) <= MAX_DRIFT + 1e-12]
            G = np.repeat(g, probes.size)
            Cg = c + (r - G) * tm
            _, _, z, v = _fine_measure(aS, aL, sr, (G, Cg), np.tile(probes, g.size), win)
            q = np.where(v, z, 0.0).reshape(g.size, probes.size).sum(axis=1)
            best_r = float(g[int(np.argmax(q))])
        c = c + (r - best_r) * tm
        r = best_r
    hop = max(0.25, min(FINE_HOP_S, (ov - win) / max(n_windows - 1, 1)))
    cen = _centres(lo, hi, win, hop, min(n_windows, _MIN_FINE_POINTS))
    if cen.size > n_windows:
        cen = cen[np.round(np.linspace(0, cen.size - 1, n_windows)).astype(int)]
    hop_eff = (cen[-1] - cen[0]) / max(cen.size - 1, 1) if cen.size > 1 else win
    cen, tL, z, v = _fine_measure(aS, aL, sr, lambda t: r * t + c, cen, win)
    # loose support for ranking (nonlinear sources should still rank well)
    loose = np.zeros(int(v.sum()), bool)
    if v.sum() >= 1:
        _, _, loose = _ransac_line(cen[v], tL[v], np.clip(z[v] - _Z0, 0.0, None) + 1e-6,
                                   _LOOSE_TOL_MS / 1000.0, np.random.default_rng(7))
    E = _evidence(z[v], loose, hop_eff, win)
    # unsaturated strength for ranking / ambiguity (E saturates per window)
    Q = _strength(z[v], loose)
    # evidence density on this candidate's own overlap: windows that agree with one line
    # (strict INLIER_MS) and how strong they are; used for the ambiguity test
    dens, n_eff = 0.0, 0.0
    if v.sum() >= 2:
        zz = z[v]
        _, _, strict = _ransac_line(cen[v], tL[v], np.clip(zz - _Z0, 0.0, None) + 1e-6,
                                    INLIER_MS / 1000.0, np.random.default_rng(11))
        strict &= zz > _Z0
        if strict.any():
            dens = float(np.mean(zz[strict] - _Z0))
            n_eff = float(strict.sum()) * min(1.0, hop_eff / win)
    agree = np.zeros(cen.size, bool)
    if v.sum() >= 2:
        agree[np.nonzero(v)[0][strict]] = True
    n_con, n_both = _contradictions(mus, cen, win, r, c, agree)
    return dict(r=r, c=c, lo=lo, hi=hi, win=win, E=E, Q=Q, n_valid=int(v.sum()),
                dens=dens, n_eff=n_eff, frac=float(strict.mean()) if v.sum() >= 2 else 0.0,
                n_con=n_con, n_both=n_both)


def _strength(z, mask):
    """Unsaturated mean per-window GCC strength (z above noise) of windows in mask, counting
    the others as 0. Used to rank candidates and to measure ambiguity."""
    if np.size(z) == 0:
        return 0.0
    return float(np.where(mask, np.clip(np.asarray(z) - _Z0, 0.0, None), 0.0).mean())


def _weights(z, v):
    return np.where(v & (z > _Z0 - 1.0), z - _Z0 + 1.0, 0.0)


def _full_scan_offsets(aS, aL, sr, probes, win, r):
    """GCC-PHAT of a few S probe windows against ALL of L (only used once ambiguity between
    exact placements is suspected, e.g. a playback track with digitally identical repeats).
    Returns candidate line offsets c (t_L = r t_S + c) where any probe has a strong peak."""
    n = int(round(win * sr))
    N = sfft.next_fast_len(aL.size + n, real=True)
    if N > _SCAN_MAX_SAMPLES:
        return []
    Y = sfft.rfft(aL, N, workers=-1)
    bm = _band_mask(N, sr, FINE_BAND)
    out = []
    for t in probes:
        k0 = int(round((t - win / 2) * sr))
        if k0 < 0 or k0 + n > aS.size:
            continue
        x = (aS[k0:k0 + n] - aS[k0:k0 + n].mean()) * _taper(n)
        X = sfft.rfft(x, N, workers=-1)
        G = Y * np.conj(X)
        G /= np.abs(G) + 1e-12 * (np.abs(G).max() + 1e-30)
        cc = sfft.irfft(G * bm, N, workers=-1)[:aL.size - n + 1]
        if cc.size < 16:
            continue
        mu, sd = float(np.mean(cc)), float(np.std(cc)) + 1e-12
        zc = (cc - mu) / sd
        thr = max(_SCAN_Z, 0.35 * float(zc.max()))
        idx = np.nonzero(zc > thr)[0]
        # local maxima, at least 0.25 s apart
        idx = idx[np.argsort(-zc[idx])]
        taken = []
        for i in idx:
            if all(abs(i - j) > 0.25 * sr for j in taken):
                taken.append(i)
            if len(taken) >= 12:
                break
        for i in taken:
            tL = (i + (n - 1) / 2.0) / sr
            tS = (k0 + (n - 1) / 2.0) / sr
            out.append(tL - r * tS)
    return out


def _refine_linear(aS, aL, sr, r, c, dS, dL, mus=None):
    """Linear fine fit of one placement over its own overlap. Returns dict or None."""
    lo, hi = _overlap(r, c, dS, dL)
    ov = hi - lo
    if ov < 2.0:
        return None
    win = min(FINE_WIN_S, 0.9 * ov)
    hop = min(FINE_HOP_S, max(0.25, (ov - win) / max(_MIN_FINE_POINTS - 1, 1)))
    hop = max(hop, (ov - win) / 40.0)                   # at most ~40 windows per placement
    cen0 = _centres(lo, hi, win, hop, _MIN_FINE_POINTS if ov < 4 * FINE_HOP_S else 1)
    cen, tL, z, v = _fine_measure(aS, aL, sr, lambda t: r * t + c, cen0, win)
    m, rms, inl, _ = fit_map(cen, tL, _weights(z, v), span=(lo, hi), allow_nonlinear=False)
    if inl.sum() < MIN_INLIERS:
        return None
    hop_eff = (cen[-1] - cen[0]) / max(cen.size - 1, 1) if cen.size > 1 else win
    zin = z[inl & v]
    n_con, n_both = _contradictions(mus, cen, win, m.ratio, m.offset, inl & v)
    return dict(map=TimeMap.linear(m.ratio, m.offset, lo, hi), lo=lo, hi=hi, rms=rms,
                n_con=n_con, n_both=n_both,
                dens=float(np.mean(np.clip(zin - _Z0, 0.0, None))) if zin.size else 0.0,
                n_eff=float(inl.sum()) * min(1.0, hop_eff / win),
                frac=float(inl.sum()) / max(int(v.sum()), 1))


def _match_sl(fS: Features, fL: Features, aS, aL, sr, hint_line=None):
    dS, dL = aS.size / sr, aL.size / sr
    mus = (fS, fL)
    if hint_line is not None:
        # user sync point(s): no coarse search, no rivals; fine search +-FINE_SEARCH_S
        r, c, known = hint_line
        e = _evaluate(aS, aL, sr, r, c, dS, dL, known, _QUICK_WINDOWS, mus)
        evals, screened = ([e] if e is not None else []), []
    else:
        rng = np.random.default_rng(0)
        pts = _coarse_points(_coarse_env(fS.onset), _coarse_env(fL.onset), dS)
        cands = _candidate_lines(pts, rng)
        if not cands:
            return None
        screened = _screen(aS, aL, sr, cands, dS, dL)
        evals = []
        for q, r, c, known in screened[:_N_EVALUATE]:
            e = _evaluate(aS, aL, sr, r, c, dS, dL, known, _QUICK_WINDOWS, mus)
            if e is not None:
                e["screen"] = q
                evals.append(e)
    if not evals:
        return None
    evals.sort(key=lambda e: -e["Q"])
    best = evals[0]
    E2 = evals[1]["E"] if len(evals) > 1 else 0.0
    Q1 = best["Q"]
    Q2 = evals[1]["Q"] if len(evals) > 1 else 0.0
    r, c, lo, hi, win = best["r"], best["c"], best["lo"], best["hi"], best["win"]
    ov = hi - lo

    # ---- full fine pass along the winner
    hop = min(FINE_HOP_S, max(0.25, (ov - win) / max(_MIN_FINE_POINTS - 1, 1)))
    cen0 = _centres(lo, hi, win, hop, _MIN_FINE_POINTS if ov < 4 * FINE_HOP_S else 1)
    cen, tL, z, v = _fine_measure(aS, aL, sr, lambda t: r * t + c, cen0, win)
    m, rms, inl, nl, dense, rms_loose, n_loose = _fit(cen, tL, _weights(z, v), span=(lo, hi))
    if (not nl and rms_loose <= NONLINEAR_RESIDUAL_MS and inl.sum() >= 2
            and abs(m.ratio - r) * win > 0.0002):
        # the slope moved: re-measure along the corrected line (removes in-window smear)
        cen, tL, z, v = _fine_measure(aS, aL, sr, m, cen0, win)
        m, rms, inl, nl, dense, rms_loose, n_loose = _fit(cen, tL, _weights(z, v), span=(lo, hi))
    if nl or rms_loose > INLIER_MS and inl.sum() < 0.85 * n_loose:
        # nonlinear source: measure densely along the current curve and refit (curve improves)
        hop = max(0.5, min(hop, PIECEWISE_KNOT_S / 8))
        cen0 = _centres(lo, hi, win, hop, 1)
        steer = dense if dense is not None else m
        for _ in range(2):
            cen, tL, z, v = _fine_measure(aS, aL, sr, steer, cen0, win)
            m, rms, inl, nl, dense, rms_loose, n_loose = _fit(cen, tL, _weights(z, v), span=(lo, hi))
            if dense is None:
                break
            steer = dense
    if not nl:
        m = TimeMap.linear(m.ratio, m.offset, lo, hi)      # knots at the overlap ends
    n_valid = int(v.sum())
    n_in = int(inl.sum())
    if n_in < MIN_INLIERS:
        return None
    hop_eff = (cen[-1] - cen[0]) / max(cen.size - 1, 1) if cen.size > 1 else win
    E = _evidence(z, inl & v, hop_eff, win)
    f_in = n_in / max(n_valid, 1)
    n_eff = n_in * min(1.0, hop_eff / max(win, 1e-9))    # independent inlier windows
    c1 = float(np.clip((n_eff - 1.0) / 1.5, 0.0, 1.0))
    c2 = float(np.clip((f_in - 0.15) / 0.35, 0.0, 1.0))
    base = 1.0 - np.exp(-E / _E0)
    Q1 = max(Q1, _strength(z[v], inl[v]))           # the winner's full pass is the better estimate
    amb = float(np.clip((1.0 - Q2 / Q1) / 0.5, 0.0, 1.0)) if Q1 > 0 else 0.0
    # density ambiguity: a competing line (another placement) that is internally as
    # consistent as the winner, judged on each candidate's OWN overlap, over >= 2
    # independent windows -> genuinely ambiguous (identical repeats of a studio track).
    alt_shift, cs_best = None, 0.0
    tm = 0.5 * (lo + hi)
    zin = z[inl & v]
    dens1 = float(np.mean(np.clip(zin - _Z0, 0.0, None))) if zin.size else 0.0
    # contradicting windows of the winner (both sides music, GCC disagrees with the map)
    if nl:
        con1, both1 = 0, 0
    else:
        con1, both1 = _contradictions(mus, cen, win, m.ratio, m.offset, inl & v)

    frac1 = con1 / both1 if both1 else 0.0

    def _consistent(n_con, n_both):
        # not more contradicted (chroma, where both clips hold music) than the winner
        return (n_con / n_both if n_both else 0.0) <= frac1 + _CONTRA_MARGIN

    def _rival(d):
        return (min(1.0, d["dens"] / dens1)                          # as strong where it agrees
                * float(np.clip(d["n_eff"] - 1.0, 0.0, 1.0))          # >= 2 independent windows
                * min(1.0, d["frac"] / max(0.6 * f_in, 1e-9)))        # as consistent over its overlap

    # 1) raw rivalry (ignoring contradictions): is there similar audio at another placement?
    cs_raw, cs_best, Q2c = 0.0, 0.0, 0.0
    for e in evals[1:]:
        if abs((e["r"] * tm + e["c"]) - (r * tm + c)) < 0.25 or dens1 <= 0:
            continue
        cs = _rival(e)
        cs_raw = max(cs_raw, cs)
        if _consistent(e["n_con"], e["n_both"]):
            Q2c = max(Q2c, e["Q"])
            if cs > cs_best:
                cs_best, alt_shift = cs, (e["r"] * tm + e["c"]) - (r * tm + c)
    conf_free = float(base * np.sqrt(c1 * c2))          # confidence without ambiguity penalty
    alternatives = []
    searched = False
    n_eff1 = n_in * min(1.0, hop_eff / max(win, 1e-9))
    if hint_line is None and cs_raw > _AMB_LO and not nl and dens1 >= _EXACT_DENS and n_eff1 >= 2.0:
        # 2) Similar audio elsewhere and the winner is exact: look for every EXACT placement
        # (the audio really is identical there, and nothing contradicts it where both clips
        # hold music). Sources: evaluated candidates, a full GCC scan of a few winner windows
        # against all of L, the screened coarse lines. Each is refined linearly. The refined
        # results replace the quick ones for the penalty as well.
        offs = [e["c"] for e in evals[1:]]
        good_w = np.nonzero(inl & v)[0]
        if good_w.size:
            pick = good_w[np.round(np.linspace(0, good_w.size - 1, min(4, good_w.size))).astype(int)]
            offs += _full_scan_offsets(aS, aL, sr, cen[pick], win, m.ratio)
        offs += [q[2] for q in screened]
        c_win = m(tm) - m.ratio * tm
        seen = [c_win]
        n_tried = 0
        cs_best, alt_shift, searched = 0.0, None, True
        for c_alt in offs:
            if any(abs(c_alt - q) < 0.25 for q in seen) or n_tried >= _ALT_MAX_TRIES:
                continue
            seen.append(c_alt)
            n_tried += 1
            ref = _refine_linear(aS, aL, sr, m.ratio, c_alt, dS, dL, mus)
            if ref is None:
                continue
            consistent = _consistent(ref["n_con"], ref["n_both"])
            exact = (ref["dens"] >= max(_EXACT_DENS, _EXACT_REL * dens1) and ref["n_eff"] >= 2.0
                     and ref["frac"] >= 0.6 * f_in and ref["rms"] <= INLIER_MS and consistent)
            if _DEBUG:
                print("ALT", round(c_alt - c_win, 3), {k: (round(v, 3) if isinstance(v, float) else v)
                                                    for k, v in ref.items() if k != "map"},
                      "winner con/both", con1, both1, "dens1", round(dens1, 2), "f_in", round(f_in, 2), exact)
            if exact:
                alternatives.append(ref)
            elif consistent:
                cs = _rival(ref)
                if cs > cs_best:
                    cs_best, alt_shift = cs, c_alt - c_win
    swapped = False
    if alternatives:
        # the least contradicted exact placement becomes `map`; placements contradicted
        # clearly more than it (possibly the original winner) are dropped
        fr = lambda d: d["n_con"] / d["n_both"] if d["n_both"] else 0.0
        pool = alternatives + [dict(map=m, lo=lo, hi=hi, rms=rms, n_con=con1, n_both=both1)]
        top = min(pool, key=fr)
        alternatives = [d for d in pool if d is not top and fr(d) <= fr(top) + _CONTRA_MARGIN]
        if top["map"] is not m:
            m, rms, lo, hi = top["map"], top["rms"], top["lo"], top["hi"]
            swapped = True
    amb_d = float(np.clip((_AMB_HI - cs_best) / (_AMB_HI - _AMB_LO), 0.0, 1.0))
    # runner-up strength penalty only from rivals that are not contradicted
    amb = 1.0 if searched else (float(np.clip((1.0 - Q2c / Q1) / 0.5, 0.0, 1.0)) if Q1 > 0 else 0.0)
    if alternatives:
        conf = conf_free
    else:
        amb = min(amb, amb_d)
        conf = float(base * np.sqrt(c1 * c2) * amb)
    lo2 = max(0.0, float(m.inverse()(0.0)))
    hi2 = min(dS, float(m.inverse()(dL)))
    note = (f"drift {(m.ratio - 1) * 1e6:+.1f} ppm; evidence {E:.1f}; inliers {n_in}/{n_valid}; "
            f"screened {len(screened)}, evaluated {len(evals)}"
            + (f"; runner-up strength {Q2:.1f} vs {Q1:.1f}" if len(evals) > 1 else ""))
    if nl:
        note += f"; nonlinear (line rms {rms_loose:.1f} ms)"
    if swapped:
        note += "; least-contradicted placement chosen"
    if hint_line is not None:
        if conf < MATCH_ACCEPT:
            return None                  # no shared sound at the hinted offset
        note = "hinted; " + note
    if amb_d < 1.0:
        note += f"; competitor density ratio {cs_best:.2f}"
    elif amb < 0.6 and len(evals) > 1:
        alt_shift = (evals[1]["r"] * tm + evals[1]["c"]) - (r * tm + c)
    if alternatives:
        note += f"; {len(alternatives)} exact alternative placement(s)"
    return dict(map=m, conf=conf, overlap=max(0.0, hi2 - lo2), rms=rms, n_valid=n_valid,
                n_in=n_in, nonlinear=nl, note=note, E=E, E2=E2,
                alt_shift=alt_shift if (amb < 0.6 and not alternatives) else None,
                alternatives=[a["map"] for a in alternatives], tm=tm)


def _hint_line(hint, b_short):
    """(r, c, slope_known) in the internal S->L orientation from user sync point(s)."""
    pts = [hint] if isinstance(hint, tuple) and len(hint) == 2 and np.isscalar(hint[0]) else list(hint)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)          # rows (t_a, t_b)
    tS, tL = (pts[:, 1], pts[:, 0]) if b_short else (pts[:, 0], pts[:, 1])
    known = False
    r = 1.0
    if tS.size >= 2 and np.ptp(tS) >= _HINT_SLOPE_SPAN_S:
        r = float(np.clip(np.polyfit(tS, tL, 1)[0], 1 - MAX_DRIFT, 1 + MAX_DRIFT))
        known = True
    c = float(np.mean(tL - r * tS))
    return r, c, known


def match_pair(fa: Features, fb: Features, audio_a, audio_b, sr: int = ANALYSIS_SR,
               hint=None) -> PairMatch | None:
    """Exact sync of clip b onto clip a:  t_a = map(t_b).  None if nothing lines up.

    hint: optional user sync point (t_a, t_b) or list of them (markers on the same moment,
    frame accurate: +-60 ms assumed). With a hint the coarse search is skipped, the fine
    GCC-PHAT searches +-FINE_SEARCH_S around the hinted offset (slope from 2+ points at least
    _HINT_SLOPE_SPAN_S apart, else searched over +-MAX_DRIFT), the result is never ambiguous
    and its note starts with "hinted". None if the clips do not share sound there
    (confidence < MATCH_ACCEPT), so the caller can fall back to a manual edge.
    """
    aa = np.ascontiguousarray(np.asarray(audio_a, dtype=np.float32))
    ab = np.ascontiguousarray(np.asarray(audio_b, dtype=np.float32))
    if aa.size < sr or ab.size < sr:
        return None
    b_short = ab.size <= aa.size
    hl = _hint_line(hint, b_short) if hint is not None else None
    if b_short:
        res = _match_sl(fb, fa, ab, aa, sr, hl)      # t_a = m(t_b)
    else:
        res = _match_sl(fa, fb, aa, ab, sr, hl)      # t_b = m(t_a)
    if res is None:
        return None
    m = res["map"] if b_short else res["map"].inverse()
    note = res["note"]
    alts = res.get("alternatives") or []
    if alts:
        # shifts of the alternative placements in clip a's time, for the same b moment
        if b_short:
            tb0 = res["tm"]
            shifts = [float(am(tb0) - res["map"](tb0)) for am in alts]
            alts_out = alts
        else:
            alts_out = [am.inverse() for am in alts]
            tb0 = float(res["map"](res["tm"]))          # b time (b is the longer clip here)
            shifts = [float(am(tb0) - m(tb0)) for am in alts_out]
        note = "ambiguous: alternatives at " + ", ".join(f"{d:+.3f}" for d in shifts) + " s; " + note
    elif res.get("alt_shift") is not None:
        # shift of the alternative placement, in clip a's time (t_a of the same b moment)
        d = res["alt_shift"] if b_short else -res["alt_shift"]
        note = f"ambiguous: alternative at {d:+.3f} s; " + note
    return PairMatch(a=fa.clip_id, b=fb.clip_id, map=m, confidence=res["conf"],
                     overlap_s=res["overlap"], residual_ms=res["rms"], n_windows=res["n_valid"],
                     n_inliers=res["n_in"], nonlinear=res["nonlinear"], note=note,
                     ambiguous=bool(alts), alternatives=alts_out if alts else [])
