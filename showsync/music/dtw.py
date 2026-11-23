"""Chroma features and subsequence DTW for musical (song-level) matching. numpy only.

Everything here works on chroma matrices shaped (12, n) at some frame rate.

* ``cens``          CENS-like smoothing + downsampling (robust to timbre, dynamics, local tempo).
* ``cost_matrix``   1 - cosine between two chroma sequences.
* ``subseq_dtw``    subsequence DTW (query may start/end anywhere in the reference), steps
                    (1,1), (2,1), (1,2) so the local tempo ratio stays within 0.5x .. 2x.
                    One numpy vector operation per query row (no step depends on the current
                    row), so a 4-minute query against a 2-hour reference at 2 Hz takes well under
                    a second.
* ``match``         query vs one reference: normalised score 0..1, best path, matched region.
* ``match_many``    query vs many references in one DTW pass (references concatenated with
                    barrier columns), returns the score against each.

Score definition (see ``_score``): the DTW path cost per query frame, c_path, is compared with
the best path cost against the *time-reversed* reference, c_null (same chroma statistics and
smoothness, but no musical order): score = clip(1 - c_path / c_null, 0, 1). Identical material
gives ~1, unrelated material ~0-0.25 (calibrated in tests/test_songs.py). This null adapts to
query length (short snippets find cheap paths more easily), noise and key, which a plain
"mean cost" normaliser did not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_INF = 1e18
_STEP_DIAG, _STEP_SLOW, _STEP_FAST = 0, 1, 2   # (1,1) / (2,1) query moves faster / (1,2)

# CENS quantisation thresholds on the L1-normalised chroma (Mueller, FMP 7.2.1)
_CENS_THRESH = (0.05, 0.1, 0.2, 0.4)


def cens(chroma: np.ndarray, in_rate: float = 10.0, out_rate: float = 2.0, win_s: float = 2.0,
         quantise: bool = True) -> np.ndarray:
    """CENS-like features: L1 normalise, (optionally) quantise, Hann smooth over ``win_s``,
    downsample to ``out_rate``, L2 normalise. Silent frames become a uniform vector so that
    silence matches silence and has a neutral cost against music.

    Returns float32 (12, m) with m = ceil(n * out_rate / in_rate).
    """
    x = np.asarray(chroma, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 12:
        raise ValueError("chroma must be (12, n)")
    n = x.shape[1]
    if n == 0:
        return np.zeros((12, 0), np.float32)
    l1 = x.sum(axis=0)
    silent = l1 < 1e-6
    x = x / np.where(silent, 1.0, l1)
    if quantise:
        q = np.zeros_like(x)
        for th in _CENS_THRESH:
            q += (x > th)
        x = q
    win = max(1, int(round(win_s * in_rate)))
    if win > 1:
        h = np.hanning(win + 2)[1:-1]
        h /= h.sum()
        # pad by edge reflection so the borders are not attenuated
        pad = win // 2
        xp = np.pad(x, ((0, 0), (pad, win - 1 - pad)), mode="edge")
        # convolve each row (12 rows: a loop is fine)
        x = np.stack([np.convolve(xp[k], h, mode="valid") for k in range(12)])
    step = in_rate / out_rate
    idx = np.floor(np.arange(0.0, n, step)).astype(int)
    idx = idx[idx < n]
    x = x[:, idx]
    norm = np.linalg.norm(x, axis=0)
    uni = np.full(12, 1.0 / np.sqrt(12.0))
    x = np.where(norm[None, :] > 1e-9, x / np.where(norm > 1e-9, norm, 1.0)[None, :], uni[:, None])
    return x.astype(np.float32)


def remove_background(chroma: np.ndarray, rate: float = 10.0, win_s: float = 20.0) -> np.ndarray:
    """Subtract each pitch class's moving average over ``win_s`` and clip at 0.

    The analysis chroma is log-compressed per frame, so broadband noise (crowd, hiss) lifts all
    12 classes and flattens the vectors. Removing the slowly varying per-class floor keeps the
    chord changes. Measured on synthetic phone / Hi8 takes this raised correct-reference wins
    from 45/72 to 66/72 (crowd-dominated phones remain hard).
    """
    x = np.asarray(chroma, dtype=np.float64)
    if x.shape[1] == 0:
        return x
    from scipy.ndimage import uniform_filter1d
    w = max(1, int(round(win_s * rate)))
    return np.maximum(x - uniform_filter1d(x, w, axis=1, mode="nearest"), 0.0)


def cost_matrix(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """1 - cosine similarity; inputs are L2-normalised non-negative (12, n) / (12, m)."""
    c = 1.0 - q.T.astype(np.float32) @ r.astype(np.float32)
    np.clip(c, 0.0, 1.0, out=c)
    return c


def subseq_dtw(C: np.ndarray, return_path: bool = False, step_penalty: float = 0.0):
    """Subsequence DTW over cost matrix C (n query frames x m reference frames).

    Each query frame pays exactly one cell cost, so the accumulated cost divided by n is the
    mean cost per query frame. Steps: (1,1); (2,1) = query advances two frames for one
    reference frame (query is slower: reference tempo / query tempo = 0.5 locally);
    (1,2) = reference advances two (ratio 2). Off-diagonal steps pay ``step_penalty`` extra.

    Returns (acc_last_row / n, path) where path is an (k, 2) int array of (i, j) cells (or None).
    """
    C = np.asarray(C, dtype=np.float64)
    n, m = C.shape
    if n == 0 or m == 0:
        return np.full(m, np.inf), None
    prev2 = np.full(m, _INF)          # row i-2
    prev1 = np.zeros(m)               # virtual row -1: free start anywhere
    steps = np.empty((n, m), np.int8) if return_path else None
    for i in range(n):
        ci = C[i]
        diag = np.empty(m)
        fast = np.empty(m)
        slow = np.empty(m)
        start_fill = 0.0 if i == 0 else _INF
        diag[0] = start_fill
        diag[1:] = prev1[:-1]
        fast[:2] = start_fill
        fast[2:] = prev1[:-2]
        fast += step_penalty
        if i == 0:
            slow[:] = _INF
        else:
            slow[0] = _INF if i > 1 else 0.0
            slow[1:] = prev2[:-1]
            slow += C[i - 1] + step_penalty
        best = np.minimum(diag, np.minimum(fast, slow))
        if steps is not None:
            st = np.where(slow < np.minimum(diag, fast), _STEP_SLOW,
                          np.where(fast < diag, _STEP_FAST, _STEP_DIAG)).astype(np.int8)
            steps[i] = st
        cur = best + ci
        prev2, prev1 = prev1, cur
        if i == 0:
            prev2 = np.zeros(m)       # virtual row -1 for the (2,1) step out of row 1
    last = prev1 / n
    path = None
    if return_path:
        j = int(np.argmin(last))
        i = n - 1
        cells = []
        while i >= 0 and j >= 0:
            cells.append((i, j))
            s = steps[i, j]
            if s == _STEP_DIAG:
                i, j = i - 1, j - 1
            elif s == _STEP_FAST:
                i, j = i - 1, j - 2
            else:
                cells.append((i - 1, j))
                i, j = i - 2, j - 1
        cells = [(a, b) for a, b in cells if a >= 0 and b >= 0]
        path = np.array(cells[::-1], dtype=int)
    return last, path


@dataclass
class DTWMatch:
    score: float          # 0..1
    cost: float           # mean path cost per query frame
    null: float           # same against the time-reversed reference (normalisation)
    path: np.ndarray | None   # (k, 2) frames (query i, reference j) at the feature rate
    ref_start: int        # first / last matched reference frame
    ref_end: int
    transpose: int = 0    # semitones the query was rolled by


def _score(cost: float, null: float) -> float:
    if not (np.isfinite(cost) and np.isfinite(null)) or null <= 1e-6 or cost > 1.0 + 1e-9:
        return 0.0
    return float(np.clip(1.0 - cost / null, 0.0, 1.0))


def match(q: np.ndarray, r: np.ndarray, return_path: bool = True, allow_transpose: bool = False,
          step_penalty: float = 0.02) -> DTWMatch:
    """Best subsequence match of query chroma q inside reference chroma r (both from ``cens``).

    The score compares the path cost with the best path cost against the time-reversed
    reference (same chroma distribution, same smoothness, no musical order).
    """
    best = None
    rrev = r[:, ::-1]
    for k in (range(12) if allow_transpose else (0,)):
        qk = np.roll(q, k, axis=0) if k else q
        last, path = subseq_dtw(cost_matrix(qk, r), return_path=return_path, step_penalty=step_penalty)
        null_last, _ = subseq_dtw(cost_matrix(qk, rrev), step_penalty=step_penalty)
        j = int(np.argmin(last))
        cost, null = float(last[j]), float(null_last.min())
        sc = _score(cost, null)
        if best is None or sc > best.score:
            rs = int(path[0, 1]) if path is not None and len(path) else j
            best = DTWMatch(sc, cost, null, path, rs, j, k)
    return best


def match_many(q: np.ndarray, refs: list[np.ndarray], allow_transpose: bool = False,
               step_penalty: float = 0.02) -> list[DTWMatch]:
    """Score query q against every reference in one DTW pass (no paths); same score as match().

    References and their time reversals are concatenated with two barrier columns
    (prohibitive cost) between them so a path cannot cross from one into the next.
    """
    if not refs:
        return []
    bar = np.zeros((12, 2), np.float32)       # placeholder columns, cost overwritten below
    parts, spans, pos = [], [], 0
    for r in list(refs) + [r[:, ::-1] for r in refs]:
        parts.append(r)
        spans.append((pos, pos + r.shape[1]))
        pos += r.shape[1]
        parts.append(bar)
        pos += 2
    R = np.concatenate(parts, axis=1)
    barrier = np.ones(R.shape[1], bool)
    for a, b in spans:
        barrier[a:b] = False
    nr = len(refs)
    out = [None] * nr
    for k in (range(12) if allow_transpose else (0,)):
        qk = np.roll(q, k, axis=0) if k else q
        C = cost_matrix(qk, R)
        C[:, barrier] = 1e6
        last, _ = subseq_dtw(C, step_penalty=step_penalty)
        for idx in range(nr):
            a, b = spans[idx]
            seg = last[a:b]
            j = int(np.argmin(seg))
            ra, rb = spans[nr + idx]
            cost, null = float(seg[j]), float(last[ra:rb].min())
            sc = _score(cost, null)
            if out[idx] is None or sc > out[idx].score:
                out[idx] = DTWMatch(sc, cost, null, None, -1, j, k)
    return out
