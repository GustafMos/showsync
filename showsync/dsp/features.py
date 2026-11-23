"""Audio features for exact sync and song finding (numpy/scipy only).

All features are computed block-wise so 90 minutes of 8 kHz audio stays in bounded memory
(a block is ~80 s of STFT frames) and runs in a few seconds.

Frame convention: frame i of an ENV_RATE / CHROMA_RATE feature is centred on time
i / rate seconds (the signal is zero-padded by half a window at both ends), and a clip of
N samples has ceil(N / hop) frames.

onset_envelope
    STFT (n_fft 512, hop sr/ENV_RATE, Hann) -> 32 mel-spaced triangular bands 80..1500 Hz on
    the magnitude -> gain-free log compression log(1 + M / (0.1 * median_band)) where the
    median is taken per band over the processing block (so EQ, mic gain and AGC become a
    per-band additive constant that the temporal difference removes; this already equalises
    the bands, a boomy band cannot dominate) -> half-wave rectified first difference per band
    -> mean over bands -> subtract a 0.5 s moving average and divide by a 5 s moving standard
    deviation (adaptive whitening).
    Measured on tests/synth.py material: limiting the bands to <= 1500 Hz and NOT dividing
    each band by its own mean flux makes the envelope far more robust to crowd noise (claps,
    screams and babble live mostly above 1 kHz): for 15 s crowd_phone snippets against a board
    feed the true lag was the top coarse peak in 19/20 cases vs 3/20 with 80..3800 Hz and
    per-band normalisation. Kick, bass and snare bodies carry the timing below 1500 Hz.
energy_db
    RMS of the 512-sample analysis window, dBFS, at ENV_RATE.
chroma
    STFT (CHROMA_NFFT, hop sr/CHROMA_RATE) magnitude, bins CHROMA_FMIN..CHROMA_FMAX mapped to
    12 pitch classes (each bin split linearly between its two nearest semitones), per-frame
    scale-free log compression log(1 + 10 * c / max(c)), per-frame L2 normalisation, columns
    zeroed where the frame level is below max(-60 dBFS, 95th percentile - 50 dB).
musicness
    See the function docstring (tonal peak ratio x chroma stability x energy gate).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

from ..config import ANALYSIS_SR, ENV_RATE, CHROMA_RATE, CHROMA_NFFT, CHROMA_FMIN, CHROMA_FMAX
from ..models import Features

ONSET_NFFT = 512
ONSET_BANDS = 32
ONSET_FMIN = 80.0
ONSET_FMAX = 1500.0         # see module docstring: crowd noise lives above this
_BLOCK_FRAMES = 8192          # STFT frames per processing block
_EPS = 1e-10


# --------------------------------------------------------------------------- helpers

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


_FB_CACHE: dict = {}


def _mel_filterbank(sr: int, n_fft: int, n_bands: int, fmin: float, fmax: float) -> np.ndarray:
    key = (sr, n_fft, n_bands, fmin, fmax)
    if key in _FB_CACHE:
        return _FB_CACHE[key]
    fmax = min(fmax, sr / 2.0 - 1.0)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_bands + 2))
    fb = np.zeros((n_bands, freqs.size), dtype=np.float64)
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        w = np.maximum(0.0, np.minimum(up, down))
        if w.sum() <= 0:                       # band narrower than a bin: take nearest bin
            w[np.argmin(np.abs(freqs - c))] = 1.0
        fb[b] = w / w.sum()
    fb = fb.astype(np.float32)
    _FB_CACHE[key] = fb
    return fb


def _chroma_matrix(sr: int, n_fft: int, fmin: float, fmax: float) -> np.ndarray:
    key = ("chroma", sr, n_fft, fmin, fmax)
    if key in _FB_CACHE:
        return _FB_CACHE[key]
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    W = np.zeros((12, freqs.size), dtype=np.float64)
    sel = (freqs >= fmin) & (freqs <= min(fmax, sr / 2.0))
    midi = 12.0 * np.log2(freqs[sel] / 440.0) + 69.0
    lo = np.floor(midi)
    frac = midi - lo
    idx = np.nonzero(sel)[0]
    W[(lo.astype(int)) % 12, idx] += 1.0 - frac
    W[(lo.astype(int) + 1) % 12, idx] += frac
    W = W.astype(np.float32)
    _FB_CACHE[key] = W
    return W


def _frames(padded: np.ndarray, start_frame: int, n: int, hop: int, n_fft: int) -> np.ndarray:
    """n frames of length n_fft starting at frame start_frame (frame i begins at i*hop of padded)."""
    s = start_frame * hop
    need = s + (n - 1) * hop + n_fft
    seg = padded[s:need]
    if seg.size < need - s:
        seg = np.pad(seg, (0, need - s - seg.size))
    return np.lib.stride_tricks.as_strided(seg, shape=(n, n_fft),
                                           strides=(seg.strides[0] * hop, seg.strides[0]))


def _n_frames(n_samples: int, hop: int) -> int:
    return max(1, int(np.ceil(n_samples / hop)))


def _as_audio(audio) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=-1 if a.shape[-1] < a.shape[0] else 0).astype(np.float32)
    return np.ascontiguousarray(a)


# --------------------------------------------------------------------------- onset + energy

def _onset_and_energy(audio, sr: int):
    a = _as_audio(audio)
    hop = int(round(sr / ENV_RATE))
    n = _n_frames(a.size, hop)
    padded = np.pad(a, (ONSET_NFFT // 2, ONSET_NFFT // 2 + hop))
    win = np.hanning(ONSET_NFFT + 1)[:-1].astype(np.float32)
    fb = _mel_filterbank(sr, ONSET_NFFT, ONSET_BANDS, ONSET_FMIN, ONSET_FMAX)
    flux = np.zeros(n, dtype=np.float32)
    energy = np.zeros(n, dtype=np.float32)
    prev_M = None
    for f0 in range(0, n, _BLOCK_FRAMES):
        nb = min(_BLOCK_FRAMES, n - f0)
        fr = _frames(padded, f0, nb, hop, ONSET_NFFT)
        energy[f0:f0 + nb] = np.sqrt(np.mean(fr * fr, axis=1))
        spec = np.abs(sfft.rfft(fr * win, axis=1, workers=-1)).astype(np.float32)
        M = spec @ fb.T                                        # (nb, bands)
        Mx = M if prev_M is None else np.vstack([prev_M[None, :], M])
        med = np.median(M, axis=0)
        ref = 0.1 * np.maximum(med, 1e-7) + 1e-9
        Lg = np.log1p(Mx / ref)
        d = np.diff(Lg, axis=0) if prev_M is not None else np.vstack([np.zeros((1, Lg.shape[1]), Lg.dtype), np.diff(Lg, axis=0)])
        d = np.maximum(d, 0.0)
        flux[f0:f0 + nb] = d.mean(axis=1)
        prev_M = M[-1]
    energy_db = (20.0 * np.log10(energy + _EPS)).astype(np.float32)
    return flux, energy_db


def _whiten(flux: np.ndarray) -> np.ndarray:
    x = flux.astype(np.float64)
    ma = uniform_filter1d(x, size=max(1, int(0.5 * ENV_RATE)), mode="nearest")
    y = x - ma
    var = uniform_filter1d(y * y, size=max(1, int(5.0 * ENV_RATE)), mode="nearest")
    sd = np.sqrt(np.maximum(var, 0.0))
    floor = 0.05 * (np.median(sd) + 1e-9) + 1e-6
    return (y / np.maximum(sd, floor)).astype(np.float32)


# --------------------------------------------------------------------------- percussive envelope

PERC_KICK = (40.0, 150.0)
PERC_SNARE = (2000.0, 3900.0)
PERC_W_KICK, PERC_W_SNARE = 0.25, 0.75
PERC_HPSS = False       # measured: HPSS did not help on real tape captures (see perc_envelope)
_HPSS_T = 17            # harmonic median across time (frames, 0.17 s)
_SNARE_SUB = 16         # snare/hat band summed into this many sub-bands before the HPSS


def _median_axis(X: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Running median of odd `size` along `axis` (edges padded by replication)."""
    h = size // 2
    pad = [(0, 0)] * X.ndim
    pad[axis] = (h, h)
    Xp = np.pad(X, pad, mode="edge")
    W = np.lib.stride_tricks.sliding_window_view(Xp, size, axis=axis)
    return np.partition(W, h, axis=-1)[..., h]


def perc_envelope(audio, sr: int = ANALYSIS_SR, hpss: bool = PERC_HPSS) -> np.ndarray:
    """Percussive onset envelope at ENV_RATE (for cross-night beat alignment).

    Same STFT grid as onset_envelope (n_fft 512, hop sr/ENV_RATE). Per block (with margins
    so the medians and differences are seamless):
    1. Two bands: kick 40-150 Hz (the individual bins, ~16 Hz apart, plus 2 neighbours each
       side) and snare/hat 2-3.9 kHz summed into 16 sub-bands (~120 Hz each).
    2. Optional (hpss=True) harmonic/percussive separation on those magnitudes: H = median
       over time (17 frames), P = median over frequency (5 kick bins / 3 snare sub-bands);
       soft mask P^2 / (P^2 + H^2). Off by default: on the real SD-tape / 4K-camera footage it
       LOWERED peak sharpness slightly (same show z 3.8 vs 4.0; cross night 3.0/2.7 vs
       3.2/3.1) and costs ~4 s per 90 min.
    3. Per band: summed magnitude, log(1 + E / (0.1 * block median)), half-wave rectified
       first difference, adaptive whitening (as onset).
    4. perc = whiten(0.25 * kick + 0.75 * snare_hat). The 40-150 Hz kick band is only ~7 bins
       at n_fft 512 and noisy on the tape captures (same-show z 3.2-3.6 alone vs 4.8-5.3 for
       the snare/hat band), hence the lower weight.
    """
    a = _as_audio(audio)
    hop = int(round(sr / ENV_RATE))
    n = _n_frames(a.size, hop)
    padded = np.pad(a, (ONSET_NFFT // 2, ONSET_NFFT // 2 + hop))
    win = np.hanning(ONSET_NFFT + 1)[:-1].astype(np.float32)
    freqs = np.arange(ONSET_NFFT // 2 + 1) * sr / ONSET_NFFT
    kb = np.nonzero((freqs >= PERC_KICK[0]) & (freqs <= PERC_KICK[1]))[0]
    k_lo, k_hi = max(0, kb[0] - 2), min(freqs.size, kb[-1] + 3)
    sb = np.nonzero((freqs >= PERC_SNARE[0]) & (freqs <= min(PERC_SNARE[1], sr / 2 - 1)))[0]
    groups = np.array_split(sb, _SNARE_SUB)
    G = np.zeros((freqs.size, len(groups)), np.float32)
    for gi, g in enumerate(groups):
        G[g, gi] = 1.0
    k_in = (kb - k_lo)
    out = {"k": np.zeros(n, np.float32), "s": np.zeros(n, np.float32)}
    mg = _HPSS_T // 2 + 1
    for f0 in range(0, n, _BLOCK_FRAMES):
        nb = min(_BLOCK_FRAMES, n - f0)
        e0 = max(0, f0 - mg)
        e1 = min(n, f0 + nb + mg)
        fr = _frames(padded, e0, e1 - e0, hop, ONSET_NFFT)
        spec = np.abs(sfft.rfft(fr * win, axis=1, workers=-1)).astype(np.float32)
        for key, X, fsize, sel in (("k", spec[:, k_lo:k_hi], 5, k_in),
                                   ("s", spec @ G, 3, slice(None))):
            if hpss:
                H = _median_axis(X, _HPSS_T, 0)
                P = _median_axis(X, fsize, 1)
                X = X * (P * P / (P * P + H * H + 1e-12))
            E = X[:, sel].sum(axis=1)
            ref = 0.1 * max(float(np.median(E)), 1e-7) + 1e-9
            Lg = np.log1p(E / ref)
            d = np.maximum(np.diff(Lg, prepend=Lg[0]), 0.0)
            out[key][f0:f0 + nb] = d[f0 - e0:f0 - e0 + nb]
    comb = PERC_W_KICK * _whiten(out["k"]) + PERC_W_SNARE * _whiten(out["s"])
    return _whiten(comb)


def onset_envelope(audio, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Onset-strength envelope at ENV_RATE Hz (log-mel spectral flux, adaptively whitened)."""
    flux, _ = _onset_and_energy(audio, sr)
    return _whiten(flux)


def energy_db(audio, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Frame RMS level in dBFS at ENV_RATE Hz."""
    return _onset_and_energy(audio, sr)[1]


# --------------------------------------------------------------------------- chroma

def _chroma_and_level(audio, sr: int):
    a = _as_audio(audio)
    hop = int(round(sr / CHROMA_RATE))
    n = _n_frames(a.size, hop)
    padded = np.pad(a, (CHROMA_NFFT // 2, CHROMA_NFFT // 2 + hop))
    win = np.hanning(CHROMA_NFFT + 1)[:-1].astype(np.float32)
    W = _chroma_matrix(sr, CHROMA_NFFT, CHROMA_FMIN, CHROMA_FMAX)
    out = np.zeros((12, n), dtype=np.float32)
    level = np.zeros(n, dtype=np.float32)
    tonal = np.zeros(n, dtype=np.float32)
    freqs = np.arange(CHROMA_NFFT // 2 + 1) * sr / CHROMA_NFFT
    tb = np.nonzero((freqs >= CHROMA_FMIN) & (freqs <= min(CHROMA_FMAX, sr / 2)))[0]
    block = max(1, _BLOCK_FRAMES // 8)
    for f0 in range(0, n, block):
        nb = min(block, n - f0)
        fr = _frames(padded, f0, nb, hop, CHROMA_NFFT)
        level[f0:f0 + nb] = np.sqrt(np.mean(fr * fr, axis=1))
        spec = np.abs(sfft.rfft(fr * win, axis=1, workers=-1)).astype(np.float32)
        out[:, f0:f0 + nb] = W @ spec.T
        # tonal peak ratio: share of power in bins 10x above the local (31-bin) geometric mean
        P = spec[:, tb] ** 2 + 1e-12
        g = np.exp(uniform_filter1d(np.log(P), 31, axis=1, mode="nearest"))
        tonal[f0:f0 + nb] = (P * (P > 10.0 * g)).sum(axis=1) / P.sum(axis=1)
    level_db = 20.0 * np.log10(level + _EPS)
    c = out / (out.max(axis=0, keepdims=True) + _EPS)
    c = np.log1p(10.0 * c)
    c /= (np.linalg.norm(c, axis=0, keepdims=True) + _EPS)
    floor = max(-60.0, float(np.percentile(level_db, 95)) - 50.0)
    c[:, level_db < floor] = 0.0
    return c.astype(np.float32), level_db.astype(np.float32), tonal


def chroma(audio, sr: int = ANALYSIS_SR) -> np.ndarray:
    """(12, n) chroma at CHROMA_RATE Hz, columns L2-normalised, zero in silence."""
    return _chroma_and_level(audio, sr)[0]


# --------------------------------------------------------------------------- musicness

def musicness(chroma: np.ndarray, energy_db_at_chroma_rate: np.ndarray, tonal: np.ndarray | None = None) -> np.ndarray:
    """0..1 per chroma frame: high for sustained tonal music, low for applause/talk/silence.

    Three terms, multiplied, then smoothed with a 0.5 s Gaussian:
    1. Tonalness (the "spectral flatness" term). compute_features passes `tonal`: the share
       of STFT power (CHROMA_FMIN..CHROMA_FMAX) in bins >= 10x their local 31-bin geometric
       mean. Measured on synth material: songs ~0.8 (median), talk ~0.37, applause/crowd
       ~0.03, white noise 0.02.  term = clip((tonal - 0.25) / 0.4, 0, 1) after 1 s smoothing.
       Without `tonal` (contract signature), the centred-chroma norm is used instead:
       clip((norm - 0.12) / 0.15, 0, 1) (weaker: steady noise gets a little through).
    2. Chroma stability: cosine similarity between each mean-centred chroma column and the
       average centred column over +-1 s. Sustained harmony ~0.8, speech ~0.55.
       term = clip((stability - 0.5) / 0.3, 0, 1).
    3. Energy gate: logistic (3 dB) around max(-55 dBFS, 95th percentile - 35 dB).
    """
    C = np.asarray(chroma, dtype=np.float64)
    n = C.shape[1]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    e = np.asarray(energy_db_at_chroma_rate, dtype=np.float64)
    if e.size != n:
        e = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, max(e.size, 1)), e if e.size else np.zeros(1))
    Cc = C - C.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(Cc, axis=0)
    half = max(1, int(round(1.0 * CHROMA_RATE)))
    nb = uniform_filter1d(Cc, size=2 * half + 1, axis=1, mode="nearest")
    nbn = np.linalg.norm(nb, axis=0)
    stab = (Cc * nb).sum(axis=0) / (norm * nbn + 1e-9)
    stab = np.clip((stab - 0.5) / 0.3, 0.0, 1.0)
    if tonal is not None and np.size(tonal) == n:
        t = uniform_filter1d(np.asarray(tonal, dtype=np.float64), size=2 * half + 1, mode="nearest")
        ton = np.clip((t - 0.25) / 0.4, 0.0, 1.0)
    else:
        ton = np.clip((uniform_filter1d(norm, size=2 * half + 1, mode="nearest") - 0.12) / 0.15, 0.0, 1.0)
    floor = max(-55.0, float(np.percentile(e, 95)) - 35.0)
    gate = 1.0 / (1.0 + np.exp(-(e - floor) / 3.0))
    m = ton * stab * gate
    m = gaussian_filter1d(m, sigma=0.5 * CHROMA_RATE, mode="nearest")
    return np.clip(m, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- all features

def compute_features(clip_id: str, audio, sr: int = ANALYSIS_SR) -> Features:
    a = _as_audio(audio)
    flux, edb = _onset_and_energy(a, sr)
    onset = _whiten(flux)
    ch, level_db, tonal = _chroma_and_level(a, sr)
    mus = musicness(ch, level_db, tonal)
    return Features(clip_id=clip_id, duration=float(a.size) / sr, onset=onset,
                    energy_db=edb, chroma=ch, musicness=mus, perc=perc_envelope(a, sr))
