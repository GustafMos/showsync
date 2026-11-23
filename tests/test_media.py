import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from showsync.models import TimeMap  # noqa: E402
from showsync.media import cache, conform, decode, probe as probe_mod  # noqa: E402


def ff(*args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, args)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWSYNC_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def cfr_video(tmp_path):
    p = tmp_path / "cfr.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=4",
       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p)
    return str(p)


@pytest.fixture
def vfr_video(tmp_path):
    p = tmp_path / "vfr.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=6",
       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
       "-vf", "setpts='(N/30+0.009*sin(N*1.7))/TB'", "-fps_mode", "passthrough",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p)
    return str(p)


def test_clip_id():
    a = probe_mod.clip_id("foo.mp4")
    assert len(a) == 12 and a == probe_mod.clip_id(os.path.abspath("foo.mp4"))
    assert a != probe_mod.clip_id("bar.mp4")


def test_probe_cfr(cfr_video):
    t0 = time.time()
    i = probe_mod.probe(cfr_video)
    assert time.time() - t0 < 2.0
    assert i.has_audio and i.has_video
    assert i.audio_sr == 44100 and i.audio_channels == 1
    assert i.fps == pytest.approx(30.0) and (i.fps_num, i.fps_den) == (30, 1)
    assert (i.width, i.height) == (160, 120)
    assert i.video_codec == "h264" and i.audio_codec == "aac"
    assert i.duration == pytest.approx(4.0, abs=0.1)
    assert not i.is_vfr, i.vfr_detail
    assert i.start_tc is None
    assert i.clip_id == probe_mod.clip_id(cfr_video)


def test_probe_vfr(vfr_video):
    i = probe_mod.probe(vfr_video)
    assert i.is_vfr
    assert i.vfr_detail


def test_probe_timecode_and_audio_only(tmp_path):
    p = tmp_path / "tc.mov"
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=2", "-timecode", "10:00:00:00",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", p)
    i = probe_mod.probe(str(p))
    assert i.start_tc == "10:00:00:00"
    assert not i.has_audio and i.fps == 25.0
    w = tmp_path / "a.wav"
    ff("-f", "lavfi", "-i", "sine=duration=1", w)
    j = probe_mod.probe(str(w))
    assert j.has_audio and not j.has_video and j.fps == 0.0 and j.device == ""


def test_probe_device_tags(tmp_path):
    p = tmp_path / "dev.mkv"
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=1", "-c:v", "libx264",
       "-metadata", "make=Acme", "-metadata", "model=Cam9", "-metadata",
       "creation_time=2024-05-01T20:00:00Z", p)
    i = probe_mod.probe(str(p))
    assert "Acme" in i.device and "Cam9" in i.device
    assert i.creation_time and i.creation_time.startswith("2024-05-01")


def test_decode_mono_sr_and_cache(cfr_video):
    a = decode.decode_audio(cfr_video, sr=8000)
    assert a.dtype == np.float32 and a.ndim == 1
    assert abs(len(a) - 4 * 8000) < 8000 * 0.1
    assert cache.load_npy(cache.cache_key(cfr_video, "audio-v2", 8000, 0)) is not None
    called = []
    orig = decode.run_f32
    decode.run_f32 = lambda *a_, **k: called.append(1) or orig(*a_, **k)
    try:
        b = decode.decode_audio(cfr_video, sr=8000)
    finally:
        decode.run_f32 = orig
    assert not called
    assert np.array_equal(a, b)
    # different sr is a different entry
    c = decode.decode_audio(cfr_video, sr=16000)
    assert abs(len(c) - 2 * len(a)) < 40


def test_decode_mmap(cfr_video):
    a = decode.decode_audio(cfr_video, sr=8000, mmap=True)  # miss path
    assert isinstance(a, np.memmap) and a.dtype == np.float32 and a.ndim == 1
    b = decode.decode_audio(cfr_video, sr=8000, mmap=True)  # hit path
    assert isinstance(b, np.memmap)
    assert np.array_equal(a, decode.decode_audio(cfr_video, sr=8000))
    assert not a.flags.writeable


def test_decode_no_audio(tmp_path):
    p = tmp_path / "v.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=1", "-c:v", "libx264", p)
    with pytest.raises(ValueError, match="no audio"):
        decode.decode_audio(str(p))


def test_cache_roundtrip_and_key(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    k1 = cache.cache_key(str(f), "a", 1)
    assert k1 == cache.cache_key(str(f), "a", 1) != cache.cache_key(str(f), "a", 2)
    cache.save_npz(k1, x=np.arange(5))
    assert np.array_equal(cache.load_npz(k1)["x"], np.arange(5))
    assert cache.load_npz("nope") is None
    f.write_bytes(b"abcd")
    assert cache.cache_key(str(f), "a", 1) != k1
    assert not list(cache.cache_dir().glob("*.tmp"))


def test_conform_cfr(vfr_video, tmp_path):
    out = str(tmp_path / "cfr_out.mov")
    assert conform.conform_cfr(vfr_video, out, fps=30.0) == out
    i = probe_mod.probe(out)
    assert not i.is_vfr, i.vfr_detail
    assert i.fps == pytest.approx(30.0)
    assert i.has_audio and i.audio_codec.startswith("pcm")
    assert i.duration == pytest.approx(6.0, abs=0.15)
    assert conform.nearest_standard_rate(29.5) == 29.97


def _clicks(sr, seconds, period, first=1.0):
    x = np.zeros(sr * seconds, np.float32)
    ts = np.arange(first, seconds - 1, period)
    x[np.round(ts * sr).astype(int)] = 1.0
    return x, ts


def _peaks(y, expected, sr):
    got = []
    for e in expected:
        c = int(round(e * sr))
        w = y[c - 200:c + 200]
        got.append(c - 200 + int(np.argmax(np.abs(w))))
    return np.array(got)


def test_export_corrected_audio_linear(tmp_path):
    sr = 48000
    x, ts = _clicks(sr, 60, 5.0)
    src = tmp_path / "clicks.wav"
    wavfile.write(src, sr, np.stack([x, 0.5 * x], axis=1))  # stereo kept
    r, o = 1.0001, 0.5
    out = str(tmp_path / "fixed.wav")
    conform.export_corrected_audio(str(src), out, TimeMap.linear(r, o, 0.0, 60.0), 60.0, sr)
    rate, y = wavfile.read(out)
    assert rate == sr and y.shape == (60 * sr, 2) and y.dtype == np.float32
    expect = (r * ts + o) * sr
    got = _peaks(y[:, 0], (r * ts + o), sr)
    assert np.max(np.abs(got - expect)) <= 1.0, (got - expect)
    assert np.max(np.abs(y[:, 1] - 0.5 * y[:, 0])) < 1e-4
    assert abs(y[:, 0][got[0]]) > 0.5


def test_export_corrected_audio_piecewise_and_negative_offset(tmp_path):
    sr = 48000
    x, ts = _clicks(sr, 40, 4.0, first=2.0)
    src = tmp_path / "c2.wav"
    wavfile.write(src, sr, x)
    m = TimeMap(np.array([0.0, 20.0, 40.0]), np.array([-1.0, 19.0 + 0.002, 39.0 - 0.004]))
    out = str(tmp_path / "fixed2.wav")
    conform.export_corrected_audio(str(src), out, m, 38.0, sr)
    _, y = wavfile.read(out)
    assert y.shape == (38 * sr,)
    tgt = m(ts)
    keep = (tgt > 0.1) & (tgt < 37.9)
    got = _peaks(y, tgt[keep], sr)
    assert np.max(np.abs(got - tgt[keep] * sr)) <= 1.0
