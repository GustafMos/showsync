"""User hints end-to-end on the synthetic music-video fixture (identical repeats)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from showsync.models import Hints, SongTag, SyncPoint
from showsync.pipeline import analyze
from tests.synth import map_inverse_eval


def _scenario(fixtures_dir, name):
    d = os.path.join(fixtures_dir, name)
    t = json.load(open(os.path.join(d, "truth.json"), encoding="utf-8"))
    return d, t


def _run(d, files, tmp_path, hints=None, refs=True):
    hp = None
    if hints is not None:
        hp = str(tmp_path / "hints.json")
        open(hp, "w", encoding="utf-8").write(hints.to_json())
    rd = os.path.join(d, "refs")
    return analyze(files, refs=[rd] if refs and os.path.isdir(rd) else None, hints_path=hp)


def _truth_point(t, fa, fb, ev):
    """Same event moment ev expressed in device time of fa and fb."""
    return map_inverse_eval(t["files"][fa]["map"], ev), map_inverse_eval(t["files"][fb]["map"], ev)


def _pair_err_ms(res, d, t, fa, fb):
    cid = {os.path.basename(m.path): c for c, m in res.clips.items()}
    g = next(g for g in res.groups if cid[fa] in g.maps and cid[fb] in g.maps)
    lo = max(t["files"][fa]["exact_span"][0], t["files"][fb]["exact_span"][0])
    hi = min(t["files"][fa]["exact_span"][1], t["files"][fb]["exact_span"][1])
    ev = np.linspace(lo + 0.5, hi - 0.5, 20)
    da = np.array([map_inverse_eval(t["files"][fa]["map"], e) for e in ev])
    db = np.array([map_inverse_eval(t["files"][fb]["map"], e) for e in ev])
    pred = g.maps[cid[fb]].inverse()(g.maps[cid[fa]](da))
    return float(np.abs(pred - db).max() * 1000), g


def test_sync_point_resolves_repeat_ambiguity(fixtures_dir, tmp_path):
    d, t = _scenario(fixtures_dir, "mv_playback")
    files = [os.path.join(d, f) for f in sorted(t["files"])]
    base = _run(d, files, tmp_path)
    amb = [os.path.basename(base.clips[c].path) for g in base.groups for c in g.alt_shifts
           if base.clips[c].role == "clip"]
    assert amb, "fixture should contain repeat-ambiguous takes"
    # give one point per ambiguous take, against take_1 at a moment both cover
    pts = []
    for f in amb:
        if f == "take_1.mp4":
            continue
        lo = max(t["files"][f]["exact_span"][0], t["files"]["take_1.mp4"]["exact_span"][0])
        hi = min(t["files"][f]["exact_span"][1], t["files"]["take_1.mp4"]["exact_span"][1])
        if hi - lo < 2:
            continue
        ta, tb = _truth_point(t, "take_1.mp4", f, 0.5 * (lo + hi))
        pts.append(SyncPoint(os.path.join(d, "take_1.mp4"), ta + 0.03, os.path.join(d, f), tb, "#hit"))
    res = _run(d, files, tmp_path, Hints(sync_points=pts))
    for sp in pts:
        f = os.path.basename(sp.b)
        err, g = _pair_err_ms(res, d, t, "take_1.mp4", f)
        # snapped to the audio placement (the point itself was 30 ms off); the crowd-drowned
        # phone take is less precise but still a tenth of a frame
        assert err < (5.0 if f == "take_5.mp4" else 1.0), f"{f}: {err} ms"
        cid = next(c for c, m in res.clips.items() if os.path.basename(m.path) == f)
        assert cid not in g.alt_shifts


def test_clip_without_audio_joins_by_marker(fixtures_dir, tmp_path):
    d, t = _scenario(fixtures_dir, "mv_playback")
    silent = str(tmp_path / "silent_take_3.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(d, "take_3.mp4"), "-an",
                    "-c:v", "copy", silent], check=True)
    ev = 40.0
    ta, tb = _truth_point(t, "take_1.mp4", "take_3.mp4", ev)
    files = [os.path.join(d, "take_1.mp4"), os.path.join(d, "take_4.wav"), silent]
    res = _run(d, files, tmp_path, Hints(sync_points=[SyncPoint(files[0], ta, silent, tb, "#kick")]))
    cid = {os.path.basename(m.path): c for c, m in res.clips.items()}
    g = next(g for g in res.groups if cid["take_1.mp4"] in g.maps)
    assert cid["silent_take_3.mp4"] in g.maps
    e = g.maps[cid["silent_take_3.mp4"]](tb) - g.maps[cid["take_1.mp4"]](ta)
    assert abs(e) < 1e-6
    assert any(m.kind == "manual" for m in g.edges)


def test_different_event_and_ignore(fixtures_dir, tmp_path):
    d, t = _scenario(fixtures_dir, "concert_full")
    files = [os.path.join(d, f) for f in ("cam_a.mov", "cam_b.mp4", "phone_1.mp4")]
    h = Hints(different_event=[[files[0], files[1]]], ignore=[files[2]])
    res = _run(d, files, tmp_path, h, refs=False)
    assert len(res.clips) == 2                                   # phone ignored
    assert not any(len(g.maps) > 1 for g in res.groups)          # cams kept apart


def test_song_tag_names_song(fixtures_dir, tmp_path):
    d, t = _scenario(fixtures_dir, "live_takes")
    files = [os.path.join(d, f) for f in sorted(t["files"])]
    res = _run(d, files, tmp_path, Hints(song_tags=[SongTag(files[0], "Midnight Bus")]), refs=False)
    names = {s.name for s in res.songs}
    if not any("tags" in str(w) or "cannot apply" in w for w in res.warnings):
        assert "Midnight Bus" in names
    else:
        pytest.skip("songs module does not support tags yet")


def test_manual_points_make_exact_edge_even_without_audio_match(monkeypatch):
    """A drum-side camera vs a room tape: both have audio, the matcher finds nothing, but the user's
    verified ("manual") points say same event -> exact manual edge, not a cross-night anchor."""
    import numpy as np
    from showsync import hints as H
    from showsync.dsp import align
    from showsync.models import Hints, SyncPoint
    monkeypatch.setattr(align, "match_pair", lambda *a, **k: None)
    feats = {"A": object(), "B": object()}
    audio = {"A": np.zeros(10), "B": np.zeros(10)}
    for source, expect_edge in (("manual", True), ("marker", False)):
        h = Hints(sync_points=[SyncPoint(a="A", ta=121.58, b="B", tb=240.0, source=source),
                               SyncPoint(a="A", ta=121.58 + 250 * 0.99781, b="B", tb=490.0, source=source)])
        musical = []
        out = H.apply_sync_points([], h, lambda p: p, feats, audio, [], lambda c: c, musical)
        if expect_edge:
            assert len(out) == 1 and out[0].kind == "manual" and not musical
            assert abs(float(out[0].map(240.0)) - 121.58) < 1e-6 and abs(out[0].map.ratio - 0.99781) < 1e-6
        else:
            assert out == [] and len(musical) == 2
