"""Sanity tests of the synthetic generator itself."""
import json
import subprocess

import numpy as np
import pytest
from scipy import signal

from tests import synth as S

SR = 48000


@pytest.fixture(scope="module")
def song_audio():
    sp = S.random_song(5, 40)
    a, secs = S.render(sp, SR, seed=0)
    return sp, a, secs


def test_determinism():
    assert S.random_song(7) == S.random_song(7)
    assert S.random_song(7) != S.random_song(8)
    sp = S.random_song(7, 25)
    a1, _ = S.render(sp, seed=3, humanize_ms=5, variation=0.3)
    a2, _ = S.render(sp, seed=3, humanize_ms=5, variation=0.3)
    assert np.array_equal(a1, a2)
    b1, _ = S.render(sp, seed=1)
    b2, _ = S.render(sp, seed=2)
    assert np.array_equal(b1, b2)                       # canonical render ignores seed
    c, _ = S.render(sp, seed=4, humanize_ms=5, variation=0.3)
    assert not np.array_equal(a1, c)
    x1, t1 = S.record(a1, SR, "phone", 3, start=1, length=10)
    x2, t2 = S.record(a1, SR, "phone", 3, start=1, length=10)
    assert np.array_equal(x1, x2) and t1["map"] == t2["map"]


def test_lengths_and_sections(song_audio):
    sp, a, secs = song_audio
    assert secs[0][1] == 0
    for (_, _, e), (_, s, _) in zip(secs, secs[1:]):
        assert e == pytest.approx(s)
    assert len(a) / SR == pytest.approx(secs[-1][2] + S.TAIL_S, abs=0.01)
    assert secs[-1][2] == pytest.approx(S.song_duration(sp))
    assert 30 < secs[-1][2] < 55
    assert np.abs(a).max() <= 1.0 and a.dtype == np.float32
    a2, s2 = S.render(sp, SR, tempo_scale=1.08)
    assert s2[-1][2] == pytest.approx(secs[-1][2] / 1.08)


def _corr(x, y):
    n = min(len(x), len(y))
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def test_choruses_repeat_within_song_not_across():
    sp = S.random_song(11, 60)
    sp.structure = "VCVC"
    a, secs = S.render(sp, SR)
    ch = [(s, e) for n, s, e in secs if n == "chorus"]
    c1 = a[int(ch[0][0] * SR) + SR // 2:int(ch[0][1] * SR) - SR // 2]
    c2 = a[int(ch[1][0] * SR) + SR // 2:int(ch[1][1] * SR) - SR // 2]
    assert _corr(c1, c2) > 0.8
    sp2 = S.random_song(12, 60)
    sp2.structure = "VCVC"
    b, secs2 = S.render(sp2, SR)
    cb = [(s, e) for n, s, e in secs2 if n == "chorus"][0]
    c3 = b[int(cb[0] * SR) + SR // 2:int(cb[1] * SR) - SR // 2]
    assert abs(_corr(c1, c3)) < 0.3
    v = [(s, e) for n, s, e in secs if n == "verse"][0]
    cv = a[int(v[0] * SR):int(v[1] * SR)]
    assert abs(_corr(c1, cv)) < 0.5                     # verse differs from chorus


def test_live_performances_differ_but_same_song():
    sp = S.random_song(13, 30)
    a, _ = S.render(sp, SR, seed=1, tempo_scale=0.95, humanize_ms=10, variation=0.5)
    b, _ = S.render(sp, SR, seed=2, tempo_scale=1.05, humanize_ms=10, variation=0.5)
    assert len(a) != len(b)
    assert abs(_corr(a, b)) < 0.2


def test_concert_truth():
    specs = [S.random_song(20 + i, 30) for i in range(3)]
    a, tr = S.make_concert(specs, 1, gap_scale=0.4)
    assert [t["song"] for t in tr] == [s.name for s in specs]
    assert tr[0]["end"] < tr[1]["start"]                # gap after song 1
    assert tr[1]["end"] == pytest.approx(tr[2]["start"])  # segue
    assert len(a) / SR > tr[-1]["end"]


def _lag(ref, dev, sr, band=(200, 3000)):
    """Positive result: ref is delayed relative to dev."""
    n = len(ref)
    sos = signal.butter(2, band, "band", fs=sr, output="sos")
    r, d = signal.sosfilt(sos, ref), signal.sosfilt(sos, dev)
    X = np.fft.rfft(r, 2 * n) * np.conj(np.fft.rfft(d, 2 * n))
    cc = np.fft.irfft(X / (np.abs(X) + 1e-9), 2 * n)
    cc = np.concatenate([cc[-n:], cc[:n]])
    k = int(np.argmax(cc))
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (k - n) / sr


@pytest.fixture(scope="module")
def long_src():
    sp = S.random_song(5, 100)
    a, _ = S.render(sp, SR, seed=0, variation=0.2, humanize_ms=3)
    return a


def test_truth_map_matches_audio_with_drift(long_src):
    dev, tr = S.record(long_src, SR, "pro", 1, start=3.0, length=80, drift_ppm=100)
    m = tr["map"]
    assert m["ratio"] == pytest.approx(1 / (1 + 1e-4))
    for t_dev in (2.0, 40.0, 76.0):
        s0 = float(S.map_eval(m, t_dev))
        w = int(1.0 * SR)
        d = dev[int(t_dev * SR):int(t_dev * SR) + w]
        r = long_src[int(round(s0 * SR)):int(round(s0 * SR)) + w]
        assert abs(_lag(r, d, SR)) < 0.0003
        # a naive offset-only map is off by t*100ppm: the drift really changed sample positions
        naive = long_src[int(round((3.0 + t_dev) * SR)):int(round((3.0 + t_dev) * SR)) + w]
        assert abs(_lag(naive, d, SR)) == pytest.approx(t_dev * 1e-4, abs=0.0004)


def test_map_inverse():
    for m in ({"kind": "linear", "ratio": 0.9999, "offset": 3.2},
              {"kind": "pwl", "dev": [0, 1, 2, 3], "src": [5, 6.01, 6.9, 8.0]}):
        t = np.array([0.3, 1.7, 2.9])
        assert np.allclose(S.map_inverse_eval(m, S.map_eval(m, t)), t, atol=1e-9)
    m = S.shift_map({"kind": "linear", "ratio": 1.0, "offset": 1.0}, 2.0)
    assert m["offset"] == 3.0


def test_hi8_wow_and_truth(long_src):
    dev, tr = S.record(long_src, SR, "hi8", 2, start=2.0, length=60, speed_var_ppm=300)
    m = tr["map"]
    assert m["kind"] == "pwl"
    d, s = np.array(m["dev"]), np.array(m["src"])
    resid = s - np.polyval(np.polyfit(d, s, 1), d)
    peak = np.abs(resid).max() * 1000
    assert 0.5 < peak < 25.0   # realistic Hi8 (v2): ~1 ms wow + slow speed wander
    assert np.all(np.diff(s) > 0)
    assert tr["dropouts"]
    for t_dev in (5.0, 30.0, 55.0):
        s0 = float(S.map_eval(m, t_dev))
        w = int(0.5 * SR)
        a = long_src[int(round(s0 * SR)):int(round(s0 * SR)) + w]
        b = dev[int(t_dev * SR):int(t_dev * SR) + w]
        assert abs(_lag(a, b, SR, (200, 3000))) < 0.002   # wow moves up to ~1 ms inside a 0.5 s window


def _stats(x, sr=SR):
    f, p = signal.welch(x, sr, nperseg=4096)
    tot = p.sum()
    return {"rms": float(np.sqrt(np.mean(x ** 2))), "centroid": float((f * p).sum() / tot),
            "lo": float(p[f < 150].sum() / tot), "hi": float(p[f > 11000].sum() / tot)}


def test_kinds_differ(long_src):
    st = {}
    for k in S.KINDS:
        d, _ = S.record(long_src, SR, k, 4, start=5, length=40)
        st[k] = _stats(d)
    assert st["phone"]["lo"] < 0.2 * st["pro"]["lo"]
    assert st["hi8"]["hi"] < 0.2 * st["pro"]["hi"] + 1e-6
    assert st["camera_mic"]["centroid"] < st["pro"]["centroid"]
    assert st["phone"]["centroid"] > st["pro"]["centroid"]
    assert abs(st["recorder"]["centroid"] - st["pro"]["centroid"]) > 20
    assert st["crowd_phone"]["rms"] > 0.05 and st["phone"]["rms"] > 0.05

    def env_corr(kind):
        d, _ = S.record(long_src, SR, kind, 4, start=5, length=40)
        ref = long_src[5 * SR:45 * SR]
        sos = signal.butter(2, [200, 2000], "band", fs=SR, output="sos")

        def e(x):
            return np.sqrt(np.mean(signal.sosfilt(sos, x)[:len(x) // 4800 * 4800].reshape(-1, 4800) ** 2, axis=1))
        return float(np.corrcoef(e(d), e(ref))[0, 1])
    assert env_corr("crowd_phone") < env_corr("pro") - 0.2


def test_record_out_sr_and_negative_start(long_src):
    d, tr = S.record(long_src, SR, "recorder", 1, start=-2.0, length=10, out_sr=44100)
    assert len(d) == 441000
    assert float(S.map_eval(tr["map"], 2.0)) == pytest.approx(0.0, abs=1e-9)
    assert np.abs(d[:44100]).max() < 0.05               # before the source: only noise floor


def test_fixtures_manifest_and_media(fixtures_dir):
    man = json.loads((fixtures_dir / "manifest.json").read_text())
    assert man["synth_version"] == S.SYNTH_VERSION
    assert set(man["scenarios"]) == {"concert_full", "snippets_only", "mv_playback", "live_takes", "two_nights", "negative"}
    assert (fixtures_dir / "mv_playback" / "refs").is_dir()
    n = 0
    for name, sc in man["scenarios"].items():
        assert (fixtures_dir / name / "truth.json").exists()
        for rel, f in sc["files"].items():
            p = fixtures_dir / name / rel
            out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)],
                                 capture_output=True, text=True, check=True).stdout
            info = json.loads(out)
            assert any(s["codec_type"] == "audio" for s in info["streams"]), rel
            assert abs(float(info["format"]["duration"]) - f["dev_duration"]) < 0.15, rel
            assert f["exact_span"][1] >= f["exact_span"][0]
            n += 1
    assert n >= 30
    cf = man["scenarios"]["concert_full"]
    assert cf["files"]["cam_a.mov"]["start_tc"]
    tags = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                           str(fixtures_dir / "concert_full" / "cam_a.mov")], capture_output=True, text=True).stdout
    assert cf["files"]["cam_a.mov"]["start_tc"] in tags
    assert len(cf["expected_exact_groups"]) == 1 and len(cf["expected_exact_groups"][0]) == len(cf["files"])


def test_vfr_video_is_irregular(fixtures_dir):
    p = fixtures_dir / "concert_full" / "phone_vfr.mp4"
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True).stdout
    pts = np.sort(np.array([float(x) for x in out.split()]))
    d = np.diff(pts)
    assert d.std() > 0.003 and d.max() > 1.5 * np.median(d)


def test_scenario_truth_logic(fixtures_dir):
    man = json.loads((fixtures_dir / "manifest.json").read_text())["scenarios"]
    tn = man["two_nights"]
    assert len(tn["expected_exact_groups"]) == 2
    assert len(tn["expected_songs"]) == 3 and all(len(v) >= 2 for v in tn["expected_songs"].values())
    assert man["negative"]["expected_exact_groups"] == []
    mv = man["mv_playback"]
    assert len(mv["expected_exact_groups"]) == 1 and len(mv["expected_exact_groups"][0]) >= 4
    lt = man["live_takes"]
    assert lt["expected_exact_groups"] == [] and len(list(lt["expected_songs"].values())[0]) == 4
    sn = man["snippets_only"]
    grouped = {f for g in sn["expected_exact_groups"] for f in g}
    assert 0 < len(grouped) < len(sn["files"])
