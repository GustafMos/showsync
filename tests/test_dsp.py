"""Tests for showsync.dsp (features + exact sync).

Part 1 uses small self-made signals (beat + chords, noise, filters, drift) so it runs without
tests/synth.py. Part 2 uses tests/synth.py when it is importable.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import signal

from showsync.config import ANALYSIS_SR, ENV_RATE, CHROMA_RATE, MATCH_ACCEPT
from showsync.dsp.features import compute_features, onset_envelope, chroma, musicness
from showsync.dsp.align import match_pair, gcc_phat, fit_map

SR = ANALYSIS_SR


# =========================================================================== simple signals

def make_music(seconds: float, seed: int, bpm: float = 120.0, sr: int = SR,
               repeat_bars: int | None = None) -> np.ndarray:
    """Beat + chord toy music with humanised timing. If repeat_bars is set the first
    repeat_bars bars are copied sample-identically later (an identical loop)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n + sr, dtype=np.float64)
    beat = 60.0 / bpm
    t = 0.0
    k = 0
    chords = [rng.integers(48, 60) + np.array([0, 4, 7]) + rng.integers(0, 2) * np.array([0, -1, 0])
              for _ in range(8)]
    while t < seconds:
        i = int((t + rng.normal(0, 0.008)) * sr)
        if i >= 0:
            if k % 2 == 0:                      # kick
                L = int(0.15 * sr)
                tt = np.arange(L) / sr
                out[i:i + L] += 0.8 * np.sin(2 * np.pi * (60 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)
            else:                               # snare
                L = int(0.12 * sr)
                out[i:i + L] += 0.4 * rng.standard_normal(L) * np.exp(-np.arange(L) / sr * 25)
            L = int(0.03 * sr)                  # hat on every half beat
            out[i:i + L] += 0.12 * rng.standard_normal(L) * np.exp(-np.arange(L) / sr * 120)
            j = int((t + beat / 2 + rng.normal(0, 0.006)) * sr)
            if 0 <= j < n:
                out[j:j + L] += 0.1 * rng.standard_normal(min(L, n + sr - j))[:out[j:j + L].size]
        if k % 4 == 0:                          # chord per bar, random voicing / velocity
            ch = chords[(k // 4) % len(chords)] if rng.random() > 0.2 else chords[rng.integers(0, 8)]
            L = int(4 * beat * sr)
            s = max(0, i)
            tt = np.arange(L) / sr
            env = np.exp(-tt * rng.uniform(0.3, 1.0))
            tone = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * tt + rng.uniform(0, 6.28)) *
                       (0.5 + 0.5 * np.sin(2 * np.pi * 440 * 2 ** ((m - 57) / 12) * tt) ** 2) for m in ch)
            seg = out[s:s + L]
            seg += 0.15 * rng.uniform(0.6, 1.0) * (tone * env)[:seg.size]
        # occasional fill
        if rng.random() < 0.05:
            for q in range(4):
                j = int((t + q * beat / 4) * sr)
                Lf = int(0.05 * sr)
                if 0 <= j < n:
                    out[j:j + Lf] += 0.3 * rng.standard_normal(Lf)[:out[j:j + Lf].size] * np.exp(-np.arange(Lf)[:out[j:j + Lf].size] / sr * 60)
        t += beat
        k += 1
    out = out[:n]
    if repeat_bars:
        L = int(repeat_bars * 4 * beat * sr)
        out[n - L:] = out[:L]
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def device(x: np.ndarray, seed: int, kind: str = "phone", sr: int = SR, crowd: float = 0.0,
           drift_ppm: float = 0.0) -> np.ndarray:
    """Crude device simulation: EQ, reverb, noise, clipping, drift (resampling)."""
    rng = np.random.default_rng(seed)
    y = x.astype(np.float64)
    if kind == "phone":
        b, a = signal.butter(2, [300, 3400], btype="band", fs=sr)
        y = signal.lfilter(b, a, y)
        y = np.clip(y * 3.0, -0.6, 0.6)
    elif kind == "boomy":
        b, a = signal.butter(2, 600, btype="low", fs=sr)
        y = signal.lfilter(b, a, y) * 2 + 0.2 * y
    elif kind == "room":
        L = int(0.6 * sr)
        ir = rng.standard_normal(L) * np.exp(-np.arange(L) / sr * 8)
        ir[0] = 6.0
        y = signal.fftconvolve(y, ir)[:y.size] / 8
    y = y + crowd * rng.standard_normal(y.size) * np.std(y)
    if drift_ppm:
        n_new = int(round(y.size * (1 + drift_ppm * 1e-6)))
        y = signal.resample(y, n_new)
    return (y / (np.abs(y).max() + 1e-9) * 0.7).astype(np.float32)


def _check(m, true_a_of_b, t_b):
    err = (m.map(t_b) - true_a_of_b(t_b)) * 1000
    return float(np.sqrt(np.mean(err ** 2)))


def _toy_pair(src, a0, la, b0, lb, kind_b="phone", drift_ppm=0.0, seed=0):
    """a = room-mic recording [a0, a0+la) of src; b = device recording starting at src time b0.
    Returns A, B, true(t_b) -> t_a."""
    A = device(src, 11 + seed, "room", crowd=0.2)[int(a0 * SR):int((a0 + la) * SR)]
    Bf = device(src, 12 + seed, kind_b, crowd=0.3, drift_ppm=drift_ppm)
    k = 1 + drift_ppm * 1e-6
    B = Bf[int(round(b0 * k * SR)):int(round((b0 * k + lb) * SR))]
    true = lambda tb: (b0 * k + np.asarray(tb)) / k - a0
    return A, B, true


def _match(A, B):
    return match_pair(compute_features("a", A), compute_features("b", B), A, B)


def _err_ms(m, true, lb, la):
    tb = np.linspace(0, lb, 500)
    ta = true(tb)
    ok = (ta >= 0.2) & (ta <= la - 0.2)
    return float(np.sqrt(np.mean((m.map(tb[ok]) - ta[ok]) ** 2))) * 1000


@pytest.fixture(scope="module")
def toy_src():
    return make_music(400, 1)


# =========================================================================== features

def test_feature_shapes_rates_and_silence():
    x = make_music(20, 3)
    x = np.concatenate([x, np.zeros(5 * SR, np.float32)])
    f = compute_features("c", x)
    assert f.duration == pytest.approx(25.0)
    assert f.onset.shape == (int(np.ceil(x.size / (SR // ENV_RATE))),)
    assert f.energy_db.shape == f.onset.shape
    assert f.chroma.shape == (12, int(np.ceil(x.size / (SR // CHROMA_RATE))))
    assert f.musicness.shape == (f.chroma.shape[1],)
    for arr in (f.onset, f.energy_db, f.chroma, f.musicness):
        assert arr.dtype == np.float32 and np.all(np.isfinite(arr))
    norms = np.linalg.norm(f.chroma, axis=0)
    assert np.allclose(norms[10:180], 1.0, atol=1e-3)          # music: unit columns
    assert np.all(norms[-30:] == 0)                              # digital silence: zero columns
    assert f.energy_db[-100:].max() < -150
    assert f.musicness[20:180].mean() > 0.3 and f.musicness[-25:].max() < 0.05   # toy: drum-heavy


def test_onset_envelope_gain_and_eq_invariant():
    x = make_music(60, 4)
    b, a = signal.butter(2, 500, btype="high", fs=SR)
    y = (0.05 * signal.lfilter(b, a, x)).astype(np.float32)      # -26 dB and thin EQ
    ex, ey = onset_envelope(x), onset_envelope(y)
    c = np.corrcoef(ex[100:-100], ey[100:-100])[0, 1]
    assert c > 0.6, c


def test_musicness_tone_noise_talkish():
    t = np.arange(20 * SR) / SR
    chord = sum(np.sin(2 * np.pi * f0 * t) for f0 in (220, 277.2, 329.6)).astype(np.float32) * 0.2
    noise = (np.random.default_rng(0).standard_normal(t.size) * 0.2).astype(np.float32)
    glide = (0.3 * np.sin(2 * np.pi * np.cumsum(150 + 80 * np.sin(2 * np.pi * 3 * t)) / SR)).astype(np.float32)
    m = {k: compute_features(k, v).musicness[20:-20].mean() for k, v in
         (("chord", chord), ("noise", noise), ("glide", glide))}
    assert m["chord"] > 0.8 and m["noise"] < 0.1 and m["glide"] < 0.5, m


def test_features_speed_90_minutes():
    x = (np.random.default_rng(1).standard_normal(90 * 60 * SR) * 0.1).astype(np.float32)
    t = time.time()
    compute_features("long", x)
    assert time.time() - t < 30.0


def test_perc_envelope_follows_drums_not_sustained_tones():
    """perc: same length/dtype as onset; on drums + loud sustained chords + noise its peaks sit
    on the drum hits (not on chord changes)."""
    rng = np.random.default_rng(4)
    n = 30 * SR
    t = np.arange(n) / SR
    x = np.zeros(n)
    hits = np.arange(0.5, 29.5, 0.5)
    for k, h in enumerate(hits):
        i = int(h * SR)
        L = int(0.08 * SR)
        tt = np.arange(L) / SR
        if k % 2 == 0:                                   # kick
            x[i:i + L] += 0.6 * np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 30)
        else:                                            # snare / hat
            b, a = signal.butter(2, 2000, "high", fs=SR)
            x[i:i + L] += 0.5 * signal.lfilter(b, a, rng.standard_normal(L)) * np.exp(-tt * 40)
    # loud sustained chords changing every 1.3 s (off the beat grid) + noise
    for c0 in np.arange(0.0, 30.0, 1.3):
        seg = (t >= c0) & (t < c0 + 1.3)
        f0 = rng.choice([196.0, 220.0, 247.0, 262.0])
        x[seg] += 0.4 * sum(np.sin(2 * np.pi * f0 * m * t[seg]) / m for m in (1, 1.25, 1.5, 2, 3))
    x += 0.05 * rng.standard_normal(n)
    f = compute_features("p", x.astype(np.float32))
    assert f.perc is not None and f.perc.shape == f.onset.shape and f.perc.dtype == np.float32
    peaks, _ = signal.find_peaks(f.perc, height=np.percentile(f.perc, 97), distance=20)
    on_hit = np.min(np.abs(peaks[:, None] / ENV_RATE - hits[None, :]), axis=1) < 0.03
    assert on_hit.mean() > 0.8, on_hit.mean()


# =========================================================================== helpers

def test_gcc_phat_known_delay():
    rng = np.random.default_rng(2)
    x = make_music(5, 5)
    d = 0.0123
    y = np.concatenate([np.zeros(int(d * SR)), x])[:x.size] + 0.05 * rng.standard_normal(x.size)
    lag, q = gcc_phat(x, y, SR, 0.05)
    assert abs(lag - d) < 0.1e-3 and q > 8
    lag2, q2 = gcc_phat(x, rng.standard_normal(x.size), SR, 0.05)
    assert q2 < 6.5


def test_fit_map_linear_with_outliers():
    rng = np.random.default_rng(3)
    tb = np.arange(5, 600, 10.0)
    ta = 1.00004 * tb + 12.5 + rng.normal(0, 0.0002, tb.size)
    bad = rng.choice(tb.size, 12, replace=False)
    ta[bad] += rng.uniform(-0.1, 0.1, 12)
    m, rms, inl, nl = fit_map(tb, ta, np.ones(tb.size))
    assert not nl and rms < 0.5 and inl.sum() >= tb.size - 12
    assert abs(m.ratio - 1.00004) < 2e-6 and abs(m(300.0) - (1.00004 * 300 + 12.5)) < 0.3e-3


def test_fit_map_nonlinear_curve():
    tb = np.arange(2, 400, 2.5)
    wow = 0.012 * np.sin(2 * np.pi * tb / 90.0)
    ta = tb + 3.0 + wow
    m, rms, inl, nl = fit_map(tb, ta, np.ones(tb.size), span=(0, 400))
    assert nl and m.knots_src.size >= 20
    k = m.knots_src
    err = np.abs(m(k) - (k + 3.0 + 0.012 * np.sin(2 * np.pi * k / 90.0)))
    inside = (k >= tb[0]) & (k <= tb[-1])           # end knots are extrapolated beyond the data
    assert err[inside].max() < 1.5e-3 and err.max() < 3e-3, err


# =========================================================================== toy matching

@pytest.mark.parametrize("case", ["full", "snippet10", "a_short", "edge40", "drift1000", "snippet_drift2000"])
def test_toy_matches(toy_src, case):
    p = dict(full=(0, 400, 0, 400, 0), snippet10=(0, 400, 200, 10, 0), a_short=(150, 20, 0, 400, 0),
             edge40=(0, 200, 160, 200, 0), drift1000=(0, 400, 5, 390, 1000),
             snippet_drift2000=(0, 400, 300, 20, -2000))[case]
    a0, la, b0, lb, d = p
    A, B, true = _toy_pair(toy_src, a0, la, b0, lb, drift_ppm=d)
    m = _match(A, B)
    assert m is not None
    assert m.confidence >= MATCH_ACCEPT, m.note
    assert _err_ms(m, true, len(B) / SR, la) < 1.0, m.note


def test_toy_unrelated_rejected(toy_src):
    other = make_music(400, 2)
    for b0, lb in ((0, 400), (100, 15)):
        A, _, _ = _toy_pair(toy_src, 0, 400, 0, 1)
        B = device(other, 5, "phone", crowd=0.3)[int(b0 * SR):int((b0 + lb) * SR)]
        m = _match(A, B)
        assert m is None or m.confidence < MATCH_ACCEPT


def test_speed_snippet_and_long_pair():
    """30 s snippet vs 90 min, and 90 min vs 90 min (matching only, features excluded)."""
    base = make_music(300, 7)
    rng = np.random.default_rng(0)
    long_a = np.concatenate([base * rng.uniform(0.5, 1.0) + 0.01 * rng.standard_normal(base.size).astype(np.float32)
                             for _ in range(18)]).astype(np.float32)
    long_b = device(long_a, 3, "phone", crowd=0.3)
    snip = long_b[int(1000 * SR):int(1030 * SR)]
    fa, fb, fs = compute_features("a", long_a), compute_features("b", long_b), compute_features("s", snip)
    t = time.time()
    match_pair(fa, fs, long_a, snip)
    t_snip = time.time() - t
    t = time.time()
    m = match_pair(fa, fb, long_a, long_b)
    t_long = time.time() - t
    print(f"\nspeed: 30 s snippet vs 90 min {t_snip:.2f} s; 90 min vs 90 min {t_long:.2f} s")
    assert t_snip < 1.5 and t_long < 10.0
    assert m is not None and abs(m.map(2000.0) - 2000.0) < 1e-3


# =========================================================================== synth-based tests

try:
    from tests import synth as _synth
    _synth.render(_synth.random_song(0, target_s=10.0), sr=48000)
    HAVE_SYNTH = True
except Exception:                                   # pragma: no cover - synth not ready
    _synth = None
    HAVE_SYNTH = False

needs_synth = pytest.mark.skipif(not HAVE_SYNTH, reason="tests/synth.py not importable")
_SYN_SR = 48000


def _rec(src, kind, seed, start, length, drift=0.0, **kw):
    x, tr = _synth.record(src, _SYN_SR, kind, seed, start=start, length=length, drift_ppm=drift, **kw)
    return signal.resample_poly(x, 1, _SYN_SR // SR).astype(np.float32), tr


def _truth_err(m, A, ta, B, tb):
    """RMS map error (ms) over the true overlap; t_a(t_b) from the synth truth maps."""
    tbs = np.linspace(0, len(B) / SR, 2000)
    true = _synth.map_inverse_eval(ta["map"], _synth.map_eval(tb["map"], tbs))
    ok = (true >= 0.5) & (true <= len(A) / SR - 0.5)
    e = (m.map(tbs[ok]) - true[ok]) * 1000
    return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


_CACHE: dict = {}


def _concert():
    if "concert" not in _CACHE:
        specs = [_synth.random_song(100 + i) for i in range(3)]
        src, truth = _synth.make_concert(specs, 1)
        other_perf, _ = _synth.make_concert(specs, 77)              # same songs, other night
        unrelated, _ = _synth.make_concert([_synth.random_song(200 + i) for i in range(2)], 2)
        A, ta = _rec(src, "pro", 1, 0.0, None)
        _CACHE["concert"] = dict(src=src, truth=truth, other=other_perf, unrel=unrelated, A=A, ta=ta,
                                 fa=compute_features("a", A))
    return _CACHE["concert"]


def _run(c, B):
    return match_pair(c["fa"], compute_features("b", B), c["A"], B)


@needs_synth
@pytest.mark.parametrize("kind", ["pro", "recorder", "phone", "crowd_phone", "camera_mic"])
@pytest.mark.parametrize("length", [12.0, 300.0])
def test_synth_same_event_all_kinds(kind, length):
    c = _concert()
    B, tb = _rec(c["src"], kind, 21, 140.0, length, drift=40.0 if kind != "pro" else 0.0)
    m = _run(c, B)
    assert m is not None and m.confidence >= MATCH_ACCEPT, (m.note if m else None)
    rms, mx = _truth_err(m, c["A"], c["ta"], B, tb)
    assert rms < 1.0 and not m.nonlinear, (rms, m.note)


@needs_synth
@pytest.mark.parametrize("drift", [40.0, -1000.0, 2150.0, -5000.0])
def test_synth_drift(drift):
    c = _concert()
    for start, length in ((20.0, 400.0), (300.0, 20.0)):
        B, tb = _rec(c["src"], "camera_mic", 22, start, length, drift=drift)
        m = _run(c, B)
        assert m is not None and m.confidence >= MATCH_ACCEPT, (m.note if m else None)
        rms, _ = _truth_err(m, c["A"], c["ta"], B, tb)
        assert rms < 1.0, (rms, m.note)
        if length > 100:
            assert abs((m.map.ratio - 1) * 1e6 - (-drift / (1 + drift * 1e-6))) < 3.0


@needs_synth
def test_synth_edge_overlap_both_ends():
    c = _concert()
    A2, ta2 = _rec(c["src"], "pro", 1, 0.0, 250.0)
    fa2 = compute_features("a2", A2)
    for start in (232.0, -100.0):                       # 18 s / 20 s of overlap
        B, tb = _rec(c["src"], "phone", 23, start, 120.0)
        m = match_pair(fa2, compute_features("b", B), A2, B)
        assert m is not None and m.confidence >= MATCH_ACCEPT, (m.note if m else None)
        rms, _ = _truth_err(m, A2, ta2, B, tb)
        assert rms < 1.0 and 15 < m.overlap_s < 25, (rms, m.overlap_s)


@needs_synth
def test_synth_snippets_land_on_the_right_chorus():
    c = _concert()
    n = 0
    for song in c["truth"]:
        ch = [s for s in song["sections"] if s[0] == "chorus"]
        for k in range(1, len(ch)):
            s0 = ch[k][1]
            B, tb = _rec(c["src"], "phone", 24 + k, s0 + 1.0, 10.0)
            m = _run(c, B)
            assert m is not None and m.confidence >= MATCH_ACCEPT, (m.note if m else None)
            rms, _ = _truth_err(m, c["A"], c["ta"], B, tb)
            assert rms < 1.0, (song["song"], k, m.note)
            n += 1
    assert n >= 3


@needs_synth
def test_synth_identical_loop_is_not_confidently_wrong():
    """A song rendered without humanisation repeats its choruses sample-identically (a
    loop played twice). Matching a snippet of chorus 2 against the board feed is genuinely
    ambiguous: the result must be correct or come back below MATCH_ACCEPT."""
    spec = _synth.random_song(321)
    src, secs = _synth.render(spec, _SYN_SR)
    A, ta = _rec(src, "pro", 3, 0.0, None)
    ch = [s for s in secs if s[0] == "chorus"]
    B, tb = _rec(src, "pro", 4, ch[1][1] + 1.0, 8.0)
    m = match_pair(compute_features("a", A), compute_features("b", B), A, B)
    if m is not None and m.confidence >= MATCH_ACCEPT:
        rms, _ = _truth_err(m, A, ta, B, tb)
        assert rms < 1.0, m.note


@needs_synth
def test_synth_hi8_nonlinear():
    """Hi8 with slow tape-speed variation (+-0.15 %, periods 23-91 s): piecewise map,
    < 3 ms at the knots."""
    c = _concert()
    B, tb = _rec(c["src"], "hi8", 25, 20.0, 400.0, wow_ms=0.5, flutter_ms=0.3, speed_var_ppm=1500)
    m = _run(c, B)
    assert m is not None and m.confidence >= MATCH_ACCEPT, (m.note if m else None)
    assert m.nonlinear
    k = m.map.knots_src
    true_k = _synth.map_inverse_eval(c["ta"]["map"], _synth.map_eval(tb["map"], k))
    assert np.sqrt(np.mean((m.map(k) - true_k) ** 2)) * 1000 < 3.0
    rms, _ = _truth_err(m, c["A"], c["ta"], B, tb)
    assert rms < 6.0


@needs_synth
@pytest.mark.parametrize("which", ["other", "unrel"])
@pytest.mark.parametrize("kind", ["pro", "phone", "crowd_phone", "camera_mic"])
def test_synth_must_reject(which, kind):
    """Another performance of the same songs, or unrelated songs, never reaches MATCH_ACCEPT."""
    c = _concert()
    s = c[which]
    D = len(s) / _SYN_SR
    for length, start in ((12.0, 0.3 * D), (300.0, 5.0)):
        B, _ = _rec(s, kind, 26, start, min(length, D - start - 1))
        m = _run(c, B)
        assert m is None or m.confidence < MATCH_ACCEPT, m.note


def _mv_fixture(fixtures_dir):
    import json
    from showsync.media.decode import decode_audio
    d = fixtures_dir / "mv_playback"
    if not (d / "truth.json").exists():
        pytest.skip("mv_playback fixture missing")
    truth = json.loads((d / "truth.json").read_text())
    names = sorted(truth["files"])
    aud = {n: decode_audio(str(d / n)) for n in names}
    ref_rel = truth["songs"][next(iter(truth["songs"]))]["ref"]
    aud["ref"] = decode_audio(str(d / ref_rel))
    ft = {n: compute_features(n, aud[n]) for n in aud}
    maps = {n: truth["files"][n]["map"] for n in names}
    maps["ref"] = {"kind": "linear", "ratio": 1.0, "offset": 0.0}   # song time == event time
    return truth, names, aud, ft, maps


def _mv_err_ms(mp, a, b, aud, maps):
    tb = np.linspace(0, len(aud[b]) / SR, 400)
    true = _synth.map_inverse_eval(maps[a], _synth.map_eval(maps[b], tb))
    ok = (true >= 0) & (true <= len(aud[a]) / SR)
    e = np.abs(mp(tb) - true)
    return float(np.median(e[ok] if ok.any() else e)) * 1000


@needs_synth
def test_mv_playback_identical_repeats_return_alternatives(fixtures_dir):
    """Music-video playback (tests/_out/fixtures/mv_playback): every take hears the same studio
    track whose sections repeat digitally, so several placements can be equally exact.
    * Accepted pairs that share sound: the true placement is `map` or one of `alternatives`
      (within 1 ms), unless they share < 5 s (then a repeat placement may win; reported).
    * Pairs that share no sound may only be accepted as ambiguous=True.
    * Ambiguous results carry alternatives and the "ambiguous:" note."""
    truth, names, aud, ft, maps = _mv_fixture(fixtures_dir)
    span = {n: truth["files"][n]["exact_span"] for n in names}
    problems, found, noov_amb, marginal = [], 0, 0, []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = match_pair(ft[a], ft[b], aud[a], aud[b])
            if m is None or m.confidence < MATCH_ACCEPT:
                continue
            if m.ambiguous:
                assert m.alternatives and m.note.startswith("ambiguous: alternatives at"), m.note
            shared = min(span[a][1], span[b][1]) - max(span[a][0], span[b][0])
            hit = min(_mv_err_ms(x, a, b, aud, maps) for x in [m.map] + list(m.alternatives)) < 1.0
            if shared >= 5.0:
                found += hit
                if not hit:
                    problems.append(("true placement missing", a, b, m.note))
            elif shared > 0:
                if not hit:
                    marginal.append((a, b, round(shared, 1), m.ambiguous))
            elif not m.ambiguous:
                problems.append(("accepted without shared sound", a, b, m.note))
            else:
                noov_amb += 1
    print(chr(10) + f"mv_playback: true placements found {found}; no-shared-sound pairs accepted as "
          f"ambiguous {noov_amb}; <5 s shared, repeat placement won: {marginal}")
    assert not problems, problems
    assert found >= 6


@needs_synth
def test_mv_playback_takes_vs_reference(fixtures_dir):
    """Each take vs the reference track: the truth is in map/alternatives; a take that holds
    >= 5 s of unique material (intro/outro: sections that occur once) is not ambiguous
    (crowd_phone excepted: its chroma is too degraded to contradict reliably)."""
    truth, names, aud, ft, maps = _mv_fixture(fixtures_dir)
    song = next(iter(truth["songs"]))
    spec = _synth.random_song(int(song.split("_")[-1]), 70)
    _, secs = _synth.render(spec, _SYN_SR, seed=0)
    if abs(secs[-1][2] - truth["songs"][song]["duration"]) > 0.5:
        pytest.skip("cannot reproduce the fixture song layout")
    names_count = {}
    for nm, _, _ in secs:
        names_count[nm] = names_count.get(nm, 0) + 1
    unique = [(s0, s1) for nm, s0, s1 in secs if names_count[nm] == 1]
    rows, bad = [], []
    for n in names:
        sg = truth["files"][n]["songs"][0]
        u = sum(max(0.0, min(s1, sg["song_end"]) - max(s0, sg["song_start"])) for s0, s1 in unique)
        m = match_pair(ft["ref"], ft[n], aud["ref"], aud[n])
        if m is None:
            rows.append((n, round(u, 1), None, None, None))
            continue
        hit = min(_mv_err_ms(x, "ref", n, aud, maps) for x in [m.map] + list(m.alternatives)) < 1.0
        rows.append((n, round(u, 1), round(m.confidence, 3), m.ambiguous, hit))
        if m.confidence >= MATCH_ACCEPT and not hit:
            bad.append(("truth missing", n, m.note))
        if (m.confidence >= MATCH_ACCEPT and m.ambiguous and u >= 5.0
                and truth["files"][n]["kind"] != "crowd_phone"):
            bad.append(("ambiguous despite unique material", n, m.note))
    print(chr(10) + "ref vs take (take, unique_s, conf, ambiguous, truth_in_map_or_alts):")
    for r_ in rows:
        print("  ", r_)
    assert not bad, bad


@needs_synth
def test_hint_fixes_ambiguous_pair_and_rejects_unrelated(fixtures_dir):
    truth, names, aud, ft, maps = _mv_fixture(fixtures_dir)
    a, b = "take_2.wav", "take_3.mp4"
    tb = np.array([8.0, 15.0])
    ta = _synth.map_inverse_eval(maps[a], _synth.map_eval(maps[b], tb))
    free = match_pair(ft[a], ft[b], aud[a], aud[b])
    free_ok = free is not None and np.max(np.abs(free.map(tb) - ta)) < 1e-3 and not free.ambiguous
    assert not free_ok                                  # free match is ambiguous or wrong
    m = match_pair(ft[a], ft[b], aud[a], aud[b], hint=(float(ta[1]) - 0.04, float(tb[1])))
    assert m is not None and not m.ambiguous and m.note.startswith("hinted")
    assert np.max(np.abs(m.map(tb) - ta)) < 1e-3
    # two hint points (slope known only if far apart; here close -> slope searched)
    m2 = match_pair(ft[a], ft[b], aud[a], aud[b], hint=[(float(ta[0]) + 0.03, 8.0), (float(ta[1]) - 0.05, 15.0)])
    assert m2 is not None and np.max(np.abs(m2.map(tb) - ta)) < 1e-3
    noise = (np.random.default_rng(0).standard_normal(len(aud[b])) * 0.1).astype(np.float32)
    assert match_pair(ft[a], compute_features("n", noise), aud[a], noise, hint=(10.0, 10.0)) is None
    other = make_music(60, 9)
    assert match_pair(ft[a], compute_features("o", other), aud[a], other, hint=(12.0, 20.0)) is None


# =========================================================================== calibration table

def calibration_table(seed: int = 5):
    """Confidence / error for true and false pairs on synth material (python -m tests.test_dsp).
    Returns a list of row dicts and prints a table plus the confidence distributions."""
    c = _concert()
    A, ta, D = c["A"], c["ta"], len(c["src"]) / _SYN_SR
    rng = np.random.default_rng(seed)
    rows = []

    def run(name, B, tb, truth, a=None):
        AA, tta, ffa = (A, ta, c["fa"]) if a is None else a
        t0 = time.time()
        m = match_pair(ffa, compute_features("b", B), AA, B)
        dt = time.time() - t0
        conf = m.confidence if m else 0.0
        err = _truth_err(m, AA, tta, B, tb)[0] if (m is not None and truth) else float("nan")
        ok = (conf >= MATCH_ACCEPT and err < (6.0 if "hi8" in name else 1.0)) if truth else conf < MATCH_ACCEPT
        rows.append(dict(case=name, true=truth, conf=conf, err_ms=err, ok=ok, t=dt))
        print(f"{name:40s} {'T' if truth else 'F'} conf={conf:.3f} err_ms={err:8.3f} "
              f"{'PASS' if ok else 'FAIL'} {dt:.2f}s", flush=True)

    for kind in ["pro", "recorder", "phone", "crowd_phone", "camera_mic"]:
        for L in (10.0, 20.0, 45.0, 300.0):
            for _ in range(2 if L < 300 else 1):
                st = float(rng.uniform(5, D - L - 5))
                d = float(rng.choice([0.0, 40.0, -60.0, 1000.0])) if kind != "pro" else 0.0
                B, tb = _rec(c["src"], kind, int(rng.integers(1 << 30)), st, L, drift=d)
                run(f"{kind} L={L:.0f} drift={d:.0f}", B, tb, True)
    for svp in (800, 1500):
        B, tb = _rec(c["src"], "hi8", 7, 20.0, 400.0, wow_ms=0.5, flutter_ms=0.3, speed_var_ppm=svp)
        run(f"hi8 speed_var {svp} ppm", B, tb, True)
    B, tb = _rec(c["src"], "hi8", 8, 20.0, 300.0)
    run("hi8 synth default (0.5 Hz wow)", B, tb, True)
    A2, ta2 = _rec(c["src"], "pro", 1, 0.0, 250.0)
    a2 = (A2, ta2, compute_features("a2", A2))
    for kind in ("phone", "crowd_phone", "camera_mic"):
        B, tb = _rec(c["src"], kind, 9, 230.0, 200.0)
        run(f"edge end 20 s {kind}", B, tb, True, a2)
        B, tb = _rec(c["src"], kind, 9, -100.0, 120.0)
        run(f"edge start 20 s {kind}", B, tb, True, a2)
    for song in c["truth"]:
        ch = [s for s in song["sections"] if s[0] == "chorus"]
        if len(ch) >= 2:
            for kind in ("phone", "crowd_phone"):
                B, tb = _rec(c["src"], kind, 11, ch[1][1] + 1.0, 10.0)
                run(f"chorus-2 snippet {song['song']} {kind}", B, tb, True)
    for lab in ("other", "unrel"):
        s = c[lab]
        D2 = len(s) / _SYN_SR
        for kind in ["pro", "phone", "crowd_phone", "camera_mic"]:
            for L in (10.0, 20.0, 45.0, 300.0):
                st = float(rng.uniform(5, D2 - L - 5))
                B, tb = _rec(s, kind, int(rng.integers(1 << 30)), st, L)
                run(f"{'other perf' if lab == 'other' else 'unrelated'} {kind} L={L:.0f}", B, tb, False)
    T = np.array([r["conf"] for r in rows if r["true"]])
    F = np.array([r["conf"] for r in rows if not r["true"]])
    print(f"\ntrue pairs  n={T.size} conf min {T.min():.3f} p5 {np.percentile(T, 5):.3f} "
          f"p25 {np.percentile(T, 25):.3f} median {np.median(T):.3f}")
    print(f"false pairs n={F.size} conf max {F.max():.3f} p95 {np.percentile(F, 95):.3f} median {np.median(F):.3f}")
    print(f"failed: {[r['case'] for r in rows if not r['ok']]}")
    return rows


if __name__ == "__main__":
    calibration_table()
