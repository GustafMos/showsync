"""Tests for the Resolve plugin logic (resolve_plugin/showsync_core.py, resolve_ops.py). No Resolve needed."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolve_plugin"))

import showsync_core as core   # noqa: E402
import resolve_ops as ops      # noqa: E402

TL_FPS = 24.0


def _clip(cid, name, dur, fps=25.0, video=True, start_tc=None, creation=None):
    return {"clip_id": cid, "path": "C:/media/%s" % name, "duration": dur, "has_audio": True, "has_video": video,
            "fps": fps if video else 0.0, "is_vfr": False, "vfr_detail": "", "start_tc": start_tc,
            "creation_time": creation}


def make_result():
    hi8_src = [20.0 * i for i in range(0, 21)]                       # 0..400 s
    hi8_dst = [t + 50 + 0.06 * math.sin(i * 1.3) for i, t in enumerate(hi8_src)]   # wow of +-60 ms
    clips = {
        "A": _clip("A", "a_pro.mov", 600.0, 25.0, start_tc="18:30:15:00", creation="2026-05-01T20:00:00"),
        "B": _clip("B", "b_phone.mp4", 300.0, 30.0),
        "C": _clip("C", "c_hi8.avi", 400.0, 25.0),
        "D": _clip("D", "d_cam.mov", 200.0, 29.97, start_tc="00:00:00;00"),
        "E": _clip("E", "e_rec.wav", 100.0, video=False),
        "L": _clip("L", "l_snip.mp4", 40.0, 30.0),
    }
    groups = [
        {"id": "g1", "anchor": "A", "anchor_tc": "18:30:15:00", "conflicts": ["B overlaps its own device"],
         "maps": {"A": {"src": [0, 600], "dst": [0, 600]},
                  "B": {"src": [0, 300], "dst": [100.0, 100.0 + 1.0005 * 300]},
                  "C": {"src": hi8_src, "dst": hi8_dst}},
         "spans": {"A": [0, 600], "B": [100.0, 400.15], "C": [50.0, 450.0]}, "edges": []},
        {"id": "g2", "anchor": "D", "anchor_tc": None, "conflicts": [],
         "maps": {"D": {"src": [0, 200], "dst": [0, 200]}, "E": {"src": [0, 100], "dst": [50, 150]}},
         "spans": {"D": [0, 200], "E": [50, 150]}, "edges": []},
    ]
    seg = lambda i, owner, kind, a, b, song, m, partial=False: {
        "id": i, "owner": owner, "owner_kind": kind, "start": a, "end": b, "song_id": song,
        "song_map": m, "match_score": 0.8, "is_partial": partial}
    segments = [
        seg("s1", "g1", "group", 10, 300, "song1", {"src": [0, 600], "dst": [-10, 590]}),
        seg("s2", "g1", "group", 320, 580, "song2", {"src": [0, 600], "dst": [-320, 280]}),
        seg("s3", "L", "clip", 0, 40, "song1", {"src": [0, 40], "dst": [150, 190]}, partial=True),
        seg("s4", "g2", "group", 0, 200, "song2", {"src": [0, 200], "dst": [5, 205]}),
    ]
    songs = [{"id": "song1", "name": "Alpha", "reference": "file:alpha.wav", "duration": 290.0, "segments": ["s1", "s3"]},
             {"id": "song2", "name": "Beta", "reference": "segment:s2", "duration": 260.0, "segments": ["s2", "s4"]}]
    matches = [{"a": "A", "b": "B", "confidence": 0.9}, {"a": "A", "b": "C", "confidence": 0.7},
               {"a": "D", "b": "E", "confidence": 0.8}]
    return core.Result({"version": 1, "clips": clips, "groups": groups, "loners": ["L"], "segments": segments,
                        "songs": songs, "matches": matches, "warnings": ["demo warning"]}, "")


@pytest.fixture
def res():
    return make_result()


# ---------------------------------------------------------------- timecode

def test_tc_nondrop_roundtrip():
    for fps in (24, 25, 30, 23.976, 50):
        for n in (0, 1, 23, 24, 86400, 123456):
            assert core.tc_to_frames(core.frames_to_tc(n, fps), fps) == n
    assert core.frames_to_tc(86400, 24) == "01:00:00:00"
    assert core.tc_to_frames("01:00:00:00", 25) == 90000


def test_tc_dropframe():
    assert core.frames_to_tc(1799, 29.97, True) == "00:00:59;29"
    assert core.frames_to_tc(1800, 29.97, True) == "00:01:00;02"
    assert core.frames_to_tc(17982, 29.97, True) == "00:10:00;00"
    assert core.frames_to_tc(107892, 29.97, True) == "01:00:00;00"
    assert core.tc_to_frames("00:01:00;02", 29.97) == 1800
    assert core.tc_to_frames("01:00:00;00", 29.97) == 107892
    assert core.frames_to_tc(3600, 59.94, True) == "00:01:00;04"
    for n in list(range(0, 40000, 331)) + [107892, 2589407]:
        assert core.tc_to_frames(core.frames_to_tc(n, 29.97, True), 29.97, True) == n
        assert core.tc_to_frames(core.frames_to_tc(n, 59.94, True), 59.94, True) == n
    # 25 fps has no drop frame: flag ignored
    assert core.frames_to_tc(1500, 25, True) == "00:01:00:00"


# ---------------------------------------------------------------- time map

def test_timemap_extrapolates():
    m = core.TimeMap([0, 10, 20], [5, 15, 40])
    assert m(5) == 10 and m(-2) == 3 and m(30) == 65
    assert m.inverse()(65) == 30 and not m.is_linear
    assert core.TimeMap([0, 1], [3, 5]).ratio == 2


# ---------------------------------------------------------------- drift splitting

def _dense_err(m, cfps, piece, tfps):
    a = piece.sf / cfps
    worst = 0.0
    n = 400
    for i in range(n + 1):
        t = a + (piece.ef - piece.sf) / cfps * i / n
        worst = max(worst, abs(m(t) - m(a) - (t - a)) * tfps)
    return worst


def test_no_drift_single_piece():
    m = core.TimeMap([0, 600], [12.0, 612.0])
    ps = core.split_for_drift(m, 25.0, 15000, TL_FPS)
    assert len(ps) == 1 and (ps[0].sf, ps[0].ef) == (0, 15000)
    assert ps[0].rec == 288 and ps[0].drift_err_frames < 1e-6


def test_linear_drift_split_error_below_half_frame():
    m = core.TimeMap([0, 300], [100.0, 100.0 + 1.0005 * 300])
    ps = core.split_for_drift(m, 30.0, 9000, TL_FPS)
    assert 7 <= len(ps) <= 9
    assert ps[0].sf == 0 and ps[-1].ef == 9000
    for a, b in zip(ps, ps[1:]):
        assert a.ef == b.sf                                    # contiguous in the source
    for p in ps:
        assert p.drift_err_frames < core.DRIFT_RESYNC_FRAC
        assert _dense_err(m, 30.0, p, TL_FPS) < core.DRIFT_RESYNC_FRAC
        assert p.quant_err_frames <= 0.5 + 1e-9
        assert p.rec == round(m(p.sf / 30.0) * TL_FPS)         # record frame from the map at piece start


def test_hi8_nonlinear_split(res):
    m = res.group_map("g1", "C")
    assert not m.is_linear
    ps = core.split_for_drift(m, 25.0, 10000, TL_FPS)
    assert len(ps) > 3
    for p in ps:
        assert _dense_err(m, 25.0, p, TL_FPS) < core.DRIFT_RESYNC_FRAC
    assert ps[-1].ef == 10000


def test_negative_drift_and_tight_frac():
    m = core.TimeMap([0, 100], [0.0, 99.99])
    for p in core.split_for_drift(m, 24.0, 2400, 24.0, frac=0.25):
        assert _dense_err(m, 24.0, p, 24.0) < 0.25


# ---------------------------------------------------------------- plans

def test_event_timeline_plan(res):
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(song_timelines=False, song_markers=False))
    names = [t.name for t in plan.timelines]
    assert names == ["ShowSync G01 Alpha", "ShowSync G02 Beta"]
    tp = plan.timelines[0]
    assert tp.n_video == tp.n_audio == 3 and tp.start_tc == "01:00:00:00"
    # A: one piece per media type at record frame 0 (earliest member), full length
    a = [p for p in tp.placements if p.clip_id == "A"]
    assert {p.media_type for p in a} == {1, 2} and all(p.record_frame == 0 and p.end_frame == 15000 for p in a)
    # B placed at 100 s -> frame 2400, track order by event start: A(1) B(2)? C starts at 50 s -> C is 2
    tracks = {p.clip_id: p.track for p in tp.placements}
    assert tracks == {"A": 1, "C": 2, "B": 3}
    b = sorted((p for p in tp.placements if p.clip_id == "B" and p.media_type == 1), key=lambda p: p.start_frame)
    assert b[0].record_frame == 2400 and len(b) == 1                 # 500 ppm drift: one piece at a speed
    assert abs(b[0].speed_pct - 100 / 1.0005) < 1e-6 and (b[0].start_frame, b[0].end_frame) == (0, 9000)
    assert any("B" in w or "b_phone" in w for w in tp.warnings if "speed 99.95" in w)
    assert all(p.drift_err_frames < 0.5 for p in tp.placements)
    assert tp.max_drift_err_frames < 0.5
    # audio-only recorder in group 2: audio placement only
    e = [p for p in plan.timelines[1].placements if p.clip_id == "E"]
    assert {p.media_type for p in e} == {2} and e[0].record_frame == 1200
    # song markers on the event timeline
    assert [(m.name, m.frame) for m in tp.markers] == [("Alpha", 240), ("Beta", 7680)]


def test_song_timeline_plan(res):
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(event_timelines=False, song_markers=False, song_align="offset"))
    names = [t.name for t in plan.timelines]
    assert names == ["ShowSync Song - Alpha", "ShowSync Song - Beta"]
    alpha = plan.timelines[0]
    # every member covering the segment, anchor first then by coverage (A 290 s, C 250 s, B 200 s), then the loner
    order = []
    for p in sorted(alpha.placements, key=lambda q: q.track):
        if p.media_type == 1 and p.clip_id not in order:
            order.append(p.clip_id)
    assert order == ["A", "C", "B", "L"] and alpha.n_video == 4
    tr = {p.clip_id: p.track for p in alpha.placements if p.clip_id}
    assert tr == {"A": 1, "C": 2, "B": 3, "L": 4}
    assert [p.track for p in alpha.placements if p.optional] == [5]                 # reference song on top
    a = next(p for p in alpha.placements if p.clip_id == "A" and p.media_type == 1)
    assert (a.start_frame, a.end_frame, a.record_frame, a.speed_pct) == (250, 7500, 0, 100.0)  # 10..300 s, song 0
    # B keeps its exact offset to A: its clip starts at event 100 s = song 90 s, played at the drift speed
    b = next(p for p in alpha.placements if p.clip_id == "B" and p.media_type == 1)
    assert b.record_frame == 90 * 24 and abs(b.speed_pct - 100 / 1.0005) < 1e-6 and b.start_frame == 0
    l = next(p for p in alpha.placements if p.clip_id == "L")
    assert l.record_frame == 3600 and l.speed_pct == 100.0                         # loner at song 150 s, plain
    beta = plan.timelines[1]
    tracks_clip = {}
    for p in beta.placements:
        tracks_clip.setdefault(p.track, p.clip_id)
    assert [tracks_clip[t] for t in sorted(tracks_clip)] == ["A", "C", "B", "D", "E"]   # group 1, then group 2
    assert beta.n_video == 5 and all(p.record_frame >= 0 for p in beta.placements)
    d = next(p for p in beta.placements if p.clip_id == "D" and p.media_type == 1)
    assert d.record_frame == 5 * 24
    # song start marker at song time 0
    assert [(m.frame, m.name) for m in alpha.markers] == [(0, "Alpha")] and alpha.markers[0].timeline == alpha.name


def test_clip_markers(res):
    mk = core.build_song_clip_markers(res, TL_FPS)
    by = {(m.clip_id, m.name): m for m in mk}
    assert by[("L", "Alpha")].frame == 0 and by[("L", "Alpha")].color == "Blue"
    # B in group 1 is covered by s1 from event 100: clip time 0 -> frame 0; song 2 from 320: clip t=(320-100)/1.0005
    assert by[("A", "Alpha")].frame == 250
    assert by[("A", "Beta")].color == "Green"
    assert ("E", "Beta") in by and by[("E", "Beta")].frame == 0
    assert all(m.custom.startswith("showsync") for m in mk)


def test_tc_plan_event(res):
    ops_, untouched = core.build_tc_plan(res, "event", TL_FPS)
    tc = {o.clip_id: o.new_tc for o in ops_}
    assert tc["A"] == "18:30:15:00"
    assert tc["B"] == "18:31:55:00"                       # anchor TC + 100 s
    assert tc["D"] == "02:00:00;00"                       # group 2 -> hour 2, drop frame kept from old TC
    assert tc["E"] == "02:00:50:00"
    assert {u["clip_id"] for u in untouched} == {"L"}
    old = {o.clip_id: o.old_tc for o in ops_}
    assert old["A"] == "18:30:15:00" and old["D"] == "00:00:00;00"


def test_tc_plan_song(res):
    ops_, untouched = core.build_tc_plan(res, "song", TL_FPS)
    tc = {o.clip_id: o.new_tc for o in ops_}
    assert tc == {"L": "01:02:30:00", "D": "02:00:05;00", "E": "02:00:55:00"} or \
        (tc["L"] == "01:02:30:00" and tc["D"].startswith("02:00:05") and tc["E"].startswith("02:00:55"))
    assert {u["clip_id"] for u in untouched} == {"A", "B", "C"}     # not inside a single song segment


# ---------------------------------------------------------------- propagation

def test_propagate_exact_and_song(res):
    mk = [{"frame": 5000, "name": "solo", "color": "Red", "note": "n", "duration": 25}]   # A t=200 s
    out = core.propagate_markers(res, "A", mk, 25.0, TL_FPS)
    by = {m.clip_id: m for m in out}
    assert by["B"].via == "event"
    assert by["B"].frame == round((200 - 100) / 1.0005 * 30)
    assert abs(by["C"].frame / 25 - res.group_map("g1", "C").inverse()(200)) < 0.03
    assert "A" not in by and "D" not in by
    assert by["B"].duration == 30 and by["B"].color == "Red" and by["B"].custom == "showsync:prop:A"
    # A t=170 s is song time 160 s in Alpha; L (snippet at song 150..190) gets clip time 10 s -> frame 300 @30
    out2 = core.propagate_markers(res, "A", [{"frame": 4250, "name": "x", "color": "Blue"}], 25.0, TL_FPS)
    l = [m for m in out2 if m.clip_id == "L"]
    assert len(l) == 1 and l[0].via == "song" and l[0].frame == 300
    # song time out of the snippet: no marker on L
    out3 = core.propagate_markers(res, "A", [{"frame": 5250, "name": "x"}], 25.0, TL_FPS)
    assert not [m for m in out3 if m.clip_id == "L"]


def test_propagate_from_loner_via_song(res):
    out = core.propagate_markers(res, "L", [{"frame": 300, "name": "m"}], 30.0, TL_FPS)   # L t=10 s, song 160 s
    by = {m.clip_id: m for m in out}
    assert by["A"].frame == 4250 and by["A"].via == "song"       # event 170 s at 25 fps
    assert "D" not in by                                         # other song


def test_summary_rows(res):
    rows = {r["clip_id"]: r for r in core.summarize_result(res, TL_FPS)}
    assert rows["B"]["group"] == "G01" and rows["B"]["drift_ppm"] == "500.0"
    assert rows["B"]["offset_tc"] == "00:01:40:00" and "Alpha" in rows["B"]["songs"]
    assert rows["C"]["warnings"].startswith("nonlinear")
    assert rows["L"]["group"] == "" and "no exact match" in rows["L"]["warnings"]
    assert rows["A"]["confidence"] == "0.90"


# ---------------------------------------------------------------- backup / undo with a fake layer

class FakeLayer(ops.LayerBase):
    def __init__(self, paths, res=None):
        info = {}
        if res is not None:
            for c in res.clips.values():
                info[core.norm_path(c["path"])] = (c["fps"] or TL_FPS, int(c["duration"] * (c["fps"] or TL_FPS)))
        self.clips = {core.norm_path(p): {"tc": "00:00:00:00", "meta": {"Comments": "keep me"} if p.endswith("a_pro.mov") else {},
                                           "markers": {}, "path": p,
                                           "fps": info.get(core.norm_path(p), (25.0, 10 ** 9))[0],
                                           "frames": info.get(core.norm_path(p), (25.0, 10 ** 9))[1]} for p in paths}
        self.speeds = []
        self.ceil_dur = False
        self.refuse_disable = False
        self.current = None
        self.short_by_one = False
        self.collisions = 0
        self.deleted = 0
        self.timelines = {}
        self.fail_set_tc = set()

    def timeline_start_frame(self, tl): return 86400
    def clip_for_path(self, path): return self.clips.get(core.norm_path(path))
    def timeline_names(self): return list(self.timelines)

    def create_timeline(self, name, start_tc, fps):
        if name in self.timelines:
            return None
        self.timelines[name] = {"items": [], "markers": {}, "tracks": (0, 0), "names": {}, "tc": start_tc, "fps": fps}
        return self.timelines[name]

    def set_track_count(self, tl, v, a, subtypes=None): tl["tracks"] = (v, a)
    def set_track_name(self, tl, t, i, n): tl["names"][(t, i)] = n
    def append_item(self, tl, info):
        clip = info["mediaPoolItem"]
        n = min(info["endFrame"], clip["frames"]) - info["startFrame"]          # endFrame exclusive, clamped to media
        if n < 1:
            return None
        raw = n / clip["fps"] * tl["fps"]
        dur = max(1, math.ceil(raw - 1e-9) if self.ceil_dur else round(raw))
        if self.short_by_one:
            dur = max(1, dur - 1)
        # appending onto an occupied frame fails (verified live)
        for o in tl["items"]:
            if (o["mediaType"], o["trackIndex"]) == (info["mediaType"], info["trackIndex"]) and                     o["recordFrame"] < info["recordFrame"] + dur and info["recordFrame"] < o["recordFrame"] + o["_dur"]:
                self.collisions += 1
                return None
        item = dict(info, _dur=dur, _speed=100.0, _clip=clip, _tl=tl)
        tl.get("disabled", set()).discard(("video" if info["mediaType"] == 1 else "audio", info["trackIndex"]))
        tl["items"].append(item)                      # (adding an item re-enables its track: models the live bug)
        return item

    def current_timeline(self):
        return self.current

    def set_current_timeline(self, tl):
        self.current = tl
        return True

    def set_track_enable(self, tl, ttype, index, enabled):
        tl.setdefault("disabled", set())
        if self.refuse_disable or self.current is not tl:
            return True                                   # verified live: only the CURRENT timeline takes the change
        (tl["disabled"].discard if enabled else tl["disabled"].add)((ttype, index))
        return True

    def is_track_enabled(self, tl, ttype, index):
        return (ttype, index) not in tl.get("disabled", set())

    def delete_item(self, tl, item):
        tl["items"].remove(item)
        self.deleted += 1
        return True

    def set_speed(self, item, pct):
        item["_speed"] = pct                    # verified semantics: the timeline duration stays fixed
        self.speeds.append(pct)
        return True

    def item_extent(self, item, media_type):
        clip = item["_clip"]
        consumed = min(round(item["_dur"] / item["_tl"]["fps"] * item["_speed"] / 100.0 * clip["fps"]),
                       clip["frames"] - item["startFrame"])
        if media_type != 1:
            return item["_dur"], None, None
        return item["_dur"], item["startFrame"], item["startFrame"] + consumed
    def add_timeline_marker(self, tl, frame, color, name, note, duration, custom):
        tl["markers"][frame] = custom
        return True
    def delete_timeline(self, name): return self.timelines.pop(name, None) is not None
    def get_start_tc(self, clip): return clip["tc"]
    def set_start_tc(self, clip, tc):
        if clip["path"] in self.fail_set_tc:
            return False
        clip["tc"] = tc
        return True
    def get_metadata(self, clip, key): return clip["meta"].get(key, "")
    def set_metadata(self, clip, key, value):
        if value:
            clip["meta"][key] = value
        else:
            clip["meta"].pop(key, None)
        return True
    def add_clip_marker(self, clip, frame, color, name, note, duration, custom):
        clip["markers"][frame] = custom
        return True
    def delete_clip_marker(self, clip, frame, custom):
        if clip["markers"].get(frame, "").startswith("showsync"):
            del clip["markers"][frame]
            return True
        return False


def _snapshot(layer):
    return {k: (v["tc"], dict(v["meta"]), dict(v["markers"])) for k, v in layer.clips.items()}, set(layer.timelines)


def test_apply_and_undo_roundtrip(res, tmp_path):
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    layer.clips[core.norm_path("C:/media/b_phone.mp4")]["markers"][77] = "user-marker"   # user's own, must survive
    layer.timelines["ShowSync G01 Alpha"] = {"items": [], "markers": {}, "fps": 24.0}                   # name clash -> suffix
    before = _snapshot(layer)
    opts = core.ApplyOptions(tc_mode="event", propagate_source="A", propagate_fps=25.0,
                             propagate_markers=[{"frame": 5000, "name": "solo", "color": "Red"}])
    plan = core.build_apply_plan(res, TL_FPS, opts)
    rpath = str(tmp_path / "result.json")
    open(rpath, "w").write("{}")
    bk = core.Backup.new(rpath)
    errs = ops.apply_plan(layer, plan, bk, log=lambda m: None)
    assert errs == []
    assert "ShowSync G01 Alpha (2)" in layer.timelines and "ShowSync Song - Alpha" in layer.timelines
    tl = layer.timelines["ShowSync G01 Alpha (2)"]
    assert min(i["recordFrame"] for i in tl["items"]) == 86400            # offset by the timeline start frame
    assert all(i["endFrame"] >= i["startFrame"] for i in tl["items"])
    assert tl["tracks"] == (3, 3) and tl["tc"] == "01:00:00:00"
    a = layer.clips[core.norm_path("C:/media/a_pro.mov")]
    assert a["tc"] == "18:30:15:00" and a["meta"]["Comments"] == "keep me\nShowSync orig TC=00:00:00:00"
    assert layer.clips[core.norm_path("C:/media/d_cam.mov")]["tc"] == "02:00:00;00"
    assert os.path.exists(bk.file) and core.find_latest_backup(rpath) == bk.file
    # re-applying TC must not overwrite the first recorded original
    bk2 = core.Backup.new(rpath)
    bk2.run_id = "second"
    ops.apply_plan(layer, core.build_apply_plan(res, TL_FPS, core.ApplyOptions(
        event_timelines=False, song_timelines=False, song_markers=False, tc_mode="event")), bk2, log=lambda m: None)
    assert a["meta"]["Comments"].count("ShowSync orig TC=") == 1
    errs = ops.undo_backup(layer, bk2, log=lambda m: None)
    assert errs == []
    # undo first backup from disk
    loaded = core.Backup.load(bk.file)
    assert not loaded.undone and loaded.timelines and loaded.tc_changes and loaded.clip_markers
    assert ops.undo_backup(layer, loaded, log=lambda m: None) == []
    assert _snapshot(layer)[0] == before[0]
    assert set(layer.timelines) == before[1]
    assert layer.clips[core.norm_path("C:/media/b_phone.mp4")]["markers"] == {77: "user-marker"}
    assert core.find_latest_backup(rpath) != bk.file                       # marked undone


def test_apply_reports_missing_clip_and_failed_tc(res, tmp_path):
    paths = [c["path"] for c in res.clips.values() if not c["path"].endswith("e_rec.wav")]
    layer = FakeLayer(paths, res)
    layer.fail_set_tc.add("C:/media/b_phone.mp4")
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(tc_mode="event", song_markers=False))
    bk = core.Backup.new(str(tmp_path / "r.json"))
    errs = ops.apply_plan(layer, plan, bk, log=lambda m: None)
    assert any("e_rec.wav" in e for e in errs) and any("Start TC failed" in e for e in errs)
    # partial apply still undoes cleanly
    assert ops.undo_backup(layer, bk, log=lambda m: None) == []
    assert layer.timelines == {}
    assert layer.clips[core.norm_path("C:/media/a_pro.mov")]["tc"] == "00:00:00:00"


def test_progress_and_settings(tmp_path, monkeypatch):
    assert core.parse_progress('{"stage": "match", "done": 3, "total": 9, "msg": "x"}')["done"] == 3
    assert core.parse_progress("warning: something") is None
    monkeypatch.setenv("SHOWSYNC_APPDATA", str(tmp_path))
    s = core.load_settings()
    s["engine_python"] = "X"
    core.save_settings(s)
    assert core.load_settings()["engine_python"] == "X"


# ---------------------------------------------------------------- reference role and alt_shifts

def make_ref_result():
    r = make_result()
    d = r.d
    d["clips"]["R"] = {**_clip("R", "beta.wav", 260.0, video=False), "path": "C:/ref/beta.wav", "role": "reference"}
    g2 = d["groups"][1]
    g2["maps"]["R"] = {"src": [0, 260], "dst": [-5.0, 255.0]}
    g2["spans"]["R"] = [-5.0, 255.0]
    d["songs"][1]["reference"] = "file:C:/ref/beta.wav"
    d["groups"][0]["alt_shifts"] = {"B": [60.0, -150.0, 120.0]}
    return core.Result(d, "")


def test_reference_in_group_is_optional_top_audio_track():
    res = make_ref_result()
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(tc_mode="event", song_align="offset"))
    g2 = next(t for t in plan.timelines if t.name.startswith("ShowSync G02"))
    assert g2.n_video == 2 and g2.n_audio == 3                      # D, E + reference on audio 3
    ref = [p for p in g2.placements if p.clip_id == "R"]
    assert ref and all(p.track == 3 and p.media_type == 2 and p.optional for p in ref)
    assert not any(p.optional for p in g2.placements if p.clip_id != "R")
    song = next(t for t in plan.timelines if t.name == "ShowSync Song - Beta")
    top = [p for p in song.placements if p.track == song.n_audio]
    assert top and top[0].path == "C:/ref/beta.wav" and top[0].optional and top[0].record_frame == 0
    # references never get timecode changes or markers
    assert "R" not in {o.clip_id for o in plan.tc_ops} and "R" not in {u["clip_id"] for u in plan.tc_untouched}
    assert "R" not in {m.clip_id for m in plan.clip_markers}


def test_missing_reference_is_skipped_quietly(tmp_path):
    res = make_ref_result()
    layer = FakeLayer([c["path"] for c in res.clips.values() if c.get("role") != "reference"], res)
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(song_markers=False))
    bk = core.Backup.new(str(tmp_path / "r.json"))
    assert ops.apply_plan(layer, plan, bk, log=lambda m: None) == []
    # imported reference IS placed
    layer2 = FakeLayer([c["path"] for c in res.clips.values()], res)
    plan2 = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(song_markers=False))
    assert ops.apply_plan(layer2, plan2, core.Backup.new(str(tmp_path / "r2.json")), log=lambda m: None) == []
    items = layer2.timelines["ShowSync G02 Beta"]["items"]
    assert any(i["trackIndex"] == 3 and i["mediaType"] == 2 for i in items)


def test_alt_shifts_markers_and_summary():
    res = make_ref_result()
    tp = core.build_event_timeline(res, 0, TL_FPS)
    alts = [m for m in tp.markers if m.name == "Also fits"]
    # B span starts at 100 s: alternatives at 160 s and 220 s; -150 s falls before the group start
    assert sorted(m.frame for m in alts) == [160 * 24, 220 * 24]
    assert all(m.color == "Cyan" and "this take also fits at 01:02:40:00" in m.note or "01:03:40:00" in m.note
               for m in alts)
    assert any("before the group start" in w for w in tp.warnings)
    rows = {r["clip_id"]: r for r in core.summarize_result(res, TL_FPS)}
    assert rows["B"]["repeat_ambiguous"] == "yes (3)" and "repeat-ambiguous" in rows["B"]["warnings"]
    assert rows["A"]["repeat_ambiguous"] == "" and rows["R"]["clip"].endswith("[reference]")


# ---------------------------------------------------------------- hints: markers, tags, merge, picks

def _mk(frame, name, dur=1):
    return {"frame": frame, "name": name, "color": "Blue", "note": "", "duration": dur}


def test_marker_sync_points_fps_and_no_audio_clip():
    clips = [
        {"path": "C:/m/a.mov", "fps": 25.0, "markers": [_mk(250, "#kick"), _mk(10, "other")]},
        {"path": "C:/m/b_noaudio.mp4", "fps": 29.97, "markers": [_mk(300, "#Kick"), _mk(5, "#solo")]},
        {"path": "C:/m/c.wav", "fps": 24.0, "markers": [_mk(48, " #KICK ")]},
        {"path": "C:/m/d.mov", "fps": 25.0, "markers": [_mk(1, "#kick"), _mk(9, "#kick")]},   # duplicate: first wins
    ]
    pts = core.marker_sync_points(clips)
    assert len(pts) == 6                                   # 4 clips share #kick -> 6 pairs; #solo is alone
    ab = next(p for p in pts if (p["a"], p["b"]) == ("C:/m/a.mov", "C:/m/b_noaudio.mp4"))
    assert ab["ta"] == 10.0 and abs(ab["tb"] - 300 / (30000 / 1001)) < 1e-9
    assert ab["source"] == "marker" and ab["label"] == "#kick"
    ac = next(p for p in pts if p["b"] == "C:/m/c.wav")
    assert ac["tb"] == 2.0
    assert not [p for p in pts if p["label"] == "#solo"]
    assert core.marker_sync_points([clips[0]]) == []


def test_song_marker_ranges():
    mk = [_mk(500, "song: Alpha", 250), _mk(100, "SONG:Beta"), _mk(5, "Song title"), _mk(9, "#x")]
    tags = core.song_tags_from_markers("C:/m/a.mov", 25.0, mk)
    assert tags == [{"clip": "C:/m/a.mov", "song": "Alpha", "start": 20.0, "end": 30.0},
                    {"clip": "C:/m/a.mov", "song": "Beta", "start": 4.0, "end": None}]


def test_hints_merge_keep_and_shape(tmp_path):
    h = core.Hints()
    assert h.relate("same_song", ["C:/m/a.mov", "C:/m/b.mp4", "C:/m/c.wav"]) == 3
    assert h.relate("same_song", ["C:/m/B.mp4", "c:/m/a.mov"]) == 0                 # dedupe, any order/case
    assert h.relate("different_event", ["C:/m/a.mov", "C:/m/c.wav"]) == 1
    assert h.relate("ignore", ["C:/m/z.mov", "C:/m/z.mov"]) == 1
    h.tag_song(["C:/m/a.mov"], "Mine")
    h.add_pick({"a": "C:/m/a.mov", "ta": 1.0, "b": "C:/m/b.mp4", "tb": 2.0, "label": "p", "source": "pick"})
    h.add_pick({"a": "C:/m/b.mp4", "ta": 3.0, "b": "C:/m/a.mov", "tb": 4.0, "label": "p2", "source": "pick"})
    assert len([s for s in h.sync_points if s["source"] == "pick"]) == 1             # newer pick replaces
    pts1 = [{"a": "C:/m/a.mov", "ta": 1.0, "b": "C:/m/c.wav", "tb": 2.0, "label": "#k", "source": "marker"}]
    tags1 = [{"clip": "C:/m/a.mov", "song": "FromMarker", "start": 1.0, "end": None}]
    h.merge_marker_derived(pts1, tags1)
    n1 = h.count()
    h.merge_marker_derived(pts1, tags1)                                               # idempotent
    assert h.count() == n1
    # markers removed by the user: derived entries vanish, user-made ones stay
    h.merge_marker_derived([], [])
    assert [s["source"] for s in h.sync_points] == ["pick"]
    assert [t["song"] for t in h.song_tags] == ["Mine"]
    path = str(tmp_path / "hints.json")
    h.save(path)
    h2 = core.Hints.load(path)
    assert h2.to_dict() == h.to_dict() and h2.count() == h.count()
    assert core.Hints.load(str(tmp_path / "missing.json")).count() == 0
    h2.clear()
    assert h2.count() == 0
    assert core.hints_path_for(["C:/a", "C:/b"], "d") == core.hints_path_for(["C:/B", "c:/a"], "d")
    # the engine's own parser accepts the file (skipped when the engine package is not importable)
    models = pytest.importorskip("showsync.models")
    parsed = models.Hints.from_json(open(path, encoding="utf-8").read())
    assert parsed.same_song and parsed.sync_points[0].source == "pick" and parsed.song_tags[0].song == "Mine"


def test_pick_sync_point_reproduces_alternative():
    res = make_ref_result()
    g = res.groups[0]
    opts = core.alt_options(res, "B", TL_FPS)
    assert [o["shift"] for o in opts] == [0.0, 60.0, 120.0]            # -150 s would start before the group
    assert opts[0]["label"].startswith("current") and opts[1]["tc"] == "01:02:40:00"
    for shift in (0.0, 60.0, 120.0):
        sp = core.pick_sync_point(res, "B", shift)
        assert sp["a"] == "C:/media/a_pro.mov" and sp["b"] == "C:/media/b_phone.mp4" and sp["source"] == "pick"
        assert sp["tb"] == 150.0                                                      # middle of B (300 s)
        # B at tb, moved by the picked shift, lands on the anchor at ta
        event_alt = res.group_map("g1", "B")(sp["tb"]) + shift
        assert abs(res.group_map("g1", "A")(sp["ta"]) - event_alt) < 1e-9
    with pytest.raises(ValueError):
        core.pick_sync_point(res, "A", 0.0)
    assert core.group_timeline_tc(res, "B", 60.0, TL_FPS) == "01:02:40:00"


# ---------------------------------------------------------------- song handles

def _handles_result():
    r = make_result()
    r.d["segments"][0]["song_map"] = {"src": [0, 600], "dst": [90, 690]}     # A: event 10 s -> song 100 s
    return core.Result(r.d, "")


def test_song_handles_clamp_and_record_frames():
    res = _handles_result()
    song = res.songs[0]
    p0 = core.build_song_timeline(res, song, TL_FPS)
    a0 = next(p for p in p0.placements if p.clip_id == "A")
    assert (a0.start_frame, a0.end_frame, a0.record_frame) == (250, 7500, 2400)      # cut to the segment
    p1 = core.build_song_timeline(res, song, TL_FPS, handles=20.0)
    a1 = next(p for p in p1.placements if p.clip_id == "A" and p.media_type == 1)
    # 10 s - 20 s clamps at the clip start (0); tail 300 + 20; song time of the cut start = 100 - 10 s = 90 s
    assert (a1.start_frame, a1.end_frame, a1.record_frame) == (0, 8000, 90 * 24)
    p2 = core.build_song_timeline(res, song, TL_FPS, handles=5.0)
    a2 = next(p for p in p2.placements if p.clip_id == "A" and p.media_type == 1)
    assert (a2.start_frame, a2.end_frame, a2.record_frame) == (125, 7625, 95 * 24)   # 5..305 s, song 95 s
    pw = core.build_song_timeline(res, song, TL_FPS, whole_clips=True)
    aw = next(p for p in pw.placements if p.clip_id == "A" and p.media_type == 1)
    assert (aw.start_frame, aw.end_frame, aw.record_frame) == (0, 15000, 90 * 24)
    # end clamp: L is a 40 s clip fully inside its segment: handles cannot extend it
    l = next(p for p in p1.placements if p.clip_id == "L")
    assert (l.start_frame, l.end_frame) == (0, 1200)


def test_song_handles_trimmed_at_song_start():
    res = make_result()                 # Beta: A segment 320..580 starts exactly at song time 0
    tp = core.build_song_timeline(res, res.songs[1], TL_FPS, handles=20.0)
    a = next(p for p in tp.placements if p.clip_id == "A" and p.media_type == 1)
    assert a.record_frame == 0 and a.start_frame == 320 * 25                         # head handle cannot go before 0
    assert a.end_frame == 600 * 25                                                    # tail handle clamped to media end
    assert any("head handle trimmed" in w for w in tp.warnings)
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(event_timelines=False, song_markers=False, song_align="offset",
                                                                handles_s=20.0))
    assert plan.timelines[1].placements[0].end_frame == 600 * 25


# ---------------------------------------------------------------- speed-based drift handling

def _pieces_err(m, ps, cfps, tfps=TL_FPS):
    worst = 0.0
    for pc in ps:
        a, b = pc.sf / cfps, pc.ef / cfps
        for i in range(201):
            t = a + (b - a) * i / 200
            worst = max(worst, abs(pc.rec / tfps + (t - a) * 100.0 / pc.speed_pct - m(t)) * tfps)
    return worst


def test_tape_2190ppm_one_piece_at_speed():
    m = core.TimeMap([0, 3600], [10.0, 10.0 + 1.00219 * 3600])
    ps = core.fit_pieces(m, 50.0, 180000, TL_FPS)
    assert len(ps) == 1 and (ps[0].sf, ps[0].ef) == (0, 180000)
    assert abs(ps[0].speed_pct - 100 / 1.00219) < 1e-9 and abs(ps[0].speed_pct - 99.7815) < 1e-3
    assert ps[0].rec == 240 and ps[0].drift_err_frames < 1e-6
    assert _pieces_err(m, ps, 50.0) <= 0.5 + 1e-6                      # quantisation of rec only


def test_tiny_drift_short_clip_stays_plain():
    m = core.TimeMap([0, 30], [5.0, 5.0 + 1.00005 * 30])               # 1.5 ms over the clip: far below a frame
    ps = core.fit_pieces(m, 25.0, 750, TL_FPS)
    assert len(ps) == 1 and ps[0].speed_pct == 100.0


def test_hi8_wobble_few_pieces():
    src = [20.0 * i for i in range(21)]
    dst = [t * 1.001 + 50 + 0.03 * math.sin(2 * math.pi * t / 600.0) for t in src]    # smooth wow + drift
    m = core.TimeMap(src, dst)
    ps = core.fit_pieces(m, 25.0, 10000, TL_FPS)
    assert 2 <= len(ps) <= 5, len(ps)
    assert all(pc.drift_err_frames < 0.5 for pc in ps)
    assert all(abs(pc.speed_pct - 100 / 1.001) < 0.2 for pc in ps)
    assert _pieces_err(m, ps, 25.0) <= 0.5 + 0.5 + 1e-6
    # the old 1:1 splitter needs far more pieces for the same map
    assert len(core.split_for_drift(m, 25.0, 10000, TL_FPS)) > 3 * len(ps)


def test_event_plan_reports_speed_and_residual(res):
    tp = core.build_event_timeline(res, 0, TL_FPS)
    line = [w for w in tp.warnings if w.startswith("b_phone.mp4")][0]
    assert "speed 99.9500 %" in line and "1 piece(s)" in line and "max residual" in line
    c_pieces = [p for p in tp.placements if p.clip_id == "C" and p.media_type == 1]
    assert 1 <= len(c_pieces) < 15 and all(p.speed_pct != 0 for p in c_pieces)


def _speed_placement(speed, sf, ef, rec=0, mt=1, cfps=50.0, nclip=180000):
    return core.Placement("T", "C:/media/t.mov", 1, mt, sf, ef, rec, speed_pct=speed, clip_fps=cfps, clip_frames=nclip)


@pytest.mark.parametrize("mt", [1, 2])
@pytest.mark.parametrize("slope", [1.00219, 0.9995])
def test_append_piece_covers_source_with_fake_setspeed(slope, mt):
    layer = FakeLayer(["C:/media/t.mov"])
    clip = layer.clips[core.norm_path("C:/media/t.mov")]
    clip["fps"], clip["frames"] = 50.0, 180000
    tl = layer.create_timeline("T", "01:00:00:00", 24.0)
    p = _speed_placement(100.0 / slope, 0, 180000, rec=240, mt=mt)
    logs = []
    assert ops.append_piece(layer, tl, clip, p, 24.0, 86400, logs.append) == ""
    items = tl["items"]
    assert items[0]["recordFrame"] == 86400 + 240
    assert 1 <= len(items) <= 4
    # items are contiguous on the timeline and every one carries the speed
    for a, b in zip(items, items[1:]):
        assert b["recordFrame"] == a["recordFrame"] + a["_dur"]
        assert abs(b["_speed"] - 100 / slope) < 1e-12
    # the source is covered from frame 0 to the end without gaps (within one frame at the end)
    covered = 0
    for it in items:
        assert it["startFrame"] <= covered + 1
        covered = it["startFrame"] + round(it["_dur"] / 24.0 * it["_speed"] / 100.0 * 50.0)
    assert 180000 - covered < 1.5
    total_tl = sum(i["_dur"] for i in items)
    assert abs(total_tl - 3600 * slope * 24) < 2.0                       # whole clip lasts 3600 s * slope


def test_append_piece_plain_is_single_call_and_failures_are_reported():
    layer = FakeLayer(["C:/media/t.mov"])
    clip = layer.clips[core.norm_path("C:/media/t.mov")]
    clip["fps"] = 50.0
    tl = layer.create_timeline("T", "01:00:00:00", 24.0)
    assert ops.append_piece(layer, tl, clip, _speed_placement(100.0, 100, 600, rec=5), 24.0, 86400, print) == ""
    assert len(tl["items"]) == 1 and tl["items"][0]["endFrame"] == 600 and not layer.speeds
    layer.set_speed = lambda item, pct: False
    assert "SetSpeed" in ops.append_piece(layer, tl, clip, _speed_placement(99.78, 0, 1000, rec=5000), 24.0, 86400, print)


def test_apply_applies_speed_to_video_and_audio_items(tmp_path):
    res = make_result()
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(song_timelines=False, song_markers=False))
    errs = ops.apply_plan(layer, plan, core.Backup.new(str(tmp_path / "r.json")), log=lambda m: None)
    assert errs == []
    tl = layer.timelines["ShowSync G01 Alpha"]
    b_items = [i for i in tl["items"] if i["trackIndex"] == 3]
    assert {i["mediaType"] for i in b_items} == {1, 2}
    assert all(abs(i["_speed"] - 100 / 1.0005) < 1e-9 for i in b_items)
    assert all(i["_speed"] == 100.0 for i in tl["items"] if i["trackIndex"] == 1)   # anchor A stays plain


# ---------------------------------------------------------------- contiguous slots, no collisions

def _wobble_result():
    r = make_result()
    d = r.d
    src = [20.0 * i for i in range(21)]
    d["groups"][0]["maps"]["C"] = {"src": src, "dst": [t * 1.00219 + 50 + 0.03 * math.sin(2 * math.pi * t / 600.0)
                                                         for t in src]}
    return core.Result(d, "")


def test_plan_slots_are_contiguous():
    res = _wobble_result()
    tp = core.build_event_timeline(res, 0, TL_FPS)
    for mt in (1, 2):
        cs = sorted((p for p in tp.placements if p.clip_id == "C" and p.media_type == mt), key=lambda p: p.record_frame)
        assert len(cs) >= 2
        for a, b in zip(cs, cs[1:]):
            assert a.record_end == b.record_frame and a.end_frame == b.start_frame      # no gap, no overlap
        assert all(p.record_end > p.record_frame for p in cs)


@pytest.mark.parametrize("ceil_dur", [False, True])
def test_apply_fills_slots_without_gaps_or_collisions(ceil_dur, tmp_path):
    res = _wobble_result()
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    layer.ceil_dur = ceil_dur                    # rounding up forces overshoot -> delete + shorten + retry
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(song_timelines=False, song_markers=False))
    bk = core.Backup.new(str(tmp_path / "r.json"))
    errs = ops.apply_plan(layer, plan, bk, log=lambda m: None)
    assert errs == [], errs
    assert layer.collisions == 0
    assert bk.summary and all("max gap 0 frame(s)" in l for l in bk.summary)
    tl = layer.timelines["ShowSync G01 Alpha"]
    for track in (1, 2, 3):
        for mt in (1, 2):
            it = sorted((i for i in tl["items"] if i["trackIndex"] == track and i["mediaType"] == mt),
                        key=lambda i: i["recordFrame"])
            assert it
            for a, b in zip(it, it[1:]):
                assert b["recordFrame"] == a["recordFrame"] + a["_dur"], (track, mt)        # contiguous
            assert all(i["_dur"] >= 2 for i in it[:-1])                                     # no 1-frame stubs inside
    # video and audio of a clip occupy identical slots
    v = [(i["recordFrame"], i["_dur"]) for i in tl["items"] if i["trackIndex"] == 2 and i["mediaType"] == 1]
    a_ = [(i["recordFrame"], i["_dur"]) for i in tl["items"] if i["trackIndex"] == 2 and i["mediaType"] == 2]
    assert sorted(v)[0][0] == sorted(a_)[0][0] and sum(d for _, d in v) == sum(d for _, d in a_)
    if ceil_dur:
        assert layer.deleted > 0


def test_old_independent_rounding_would_collide():
    """The fake rejects an append on occupied frames: place two items overlapping by one frame."""
    layer = FakeLayer(["C:/media/t.mov"])
    clip = layer.clips[core.norm_path("C:/media/t.mov")]
    clip["fps"] = 24.0
    tl = layer.create_timeline("T", "01:00:00:00", 24.0)
    mk = lambda rec: {"mediaPoolItem": clip, "startFrame": 0, "endFrame": 100, "mediaType": 1, "trackIndex": 1,
                      "recordFrame": rec}
    assert layer.append_item(tl, mk(1000)) is not None
    assert layer.append_item(tl, mk(1099)) is None and layer.collisions == 1
    assert layer.append_item(tl, mk(1100)) is not None


def test_append_piece_fills_slot_exactly_with_remainder_item():
    layer = FakeLayer(["C:/media/t.mov"])
    clip = layer.clips[core.norm_path("C:/media/t.mov")]
    clip["fps"], clip["frames"] = 50.0, 180000
    tl = layer.create_timeline("T", "01:00:00:00", 24.0)
    slope = 1.00219
    slot_len = int(round(3600 * slope * 24))
    p = core.Placement("T", "C:/media/t.mov", 1, 1, 0, 180000, 240, speed_pct=100 / slope, clip_fps=50.0,
                       clip_frames=180000, record_end=240 + slot_len)
    occ = []
    assert ops.append_piece(layer, tl, clip, p, 24.0, 86400, lambda m: None, occ) == ""
    occ.sort()
    assert occ[0][0] == 86400 + 240 and len(occ) == 2
    assert occ[0][1] == occ[1][0] and occ[-1][1] <= 86400 + 240 + slot_len          # never past the slot end
    assert slot_len - (occ[-1][1] - occ[0][0]) <= 1                                  # at most 1 frame short
    sm = ops.occupancy_summary([(86400 + 240, 86400 + 240 + slot_len)], occ)
    assert sm["max_gap"] == 0 and sm["overlaps"] == 0 and sm["items"] == 2


# ---------------------------------------------------------------- no 1-frame stubs, no empty tracks

def _stub_setup(short=True):
    layer = FakeLayer(["C:/media/t.mov"])
    clip = layer.clips[core.norm_path("C:/media/t.mov")]
    clip["fps"], clip["frames"] = 25.0, 100000
    layer.short_by_one = short
    return layer, clip, layer.create_timeline("T", "01:00:00:00", 24.0)


def test_one_frame_remainder_not_created_at_tail():
    layer, clip, tl = _stub_setup()
    p = core.Placement("T", "C:/media/t.mov", 1, 1, 4690, 11659, 0, clip_fps=25.0, clip_frames=100000, record_end=6691)
    occ = []
    assert ops.append_piece(layer, tl, clip, p, 24.0, 86400, lambda m: None, occ, followed=False) == ""
    assert len(occ) == 1                                                       # main item only, no 1-frame stub
    sm = ops.occupancy_summary([(86400, 86400 + 6691)], occ)
    assert sm["max_gap"] == 0 and sm["tail_short"] == 1 and sm["items"] == 1


def test_one_frame_remainder_kept_when_another_slot_follows():
    layer, clip, tl = _stub_setup()
    p = core.Placement("T", "C:/media/t.mov", 1, 1, 4690, 11659, 0, clip_fps=25.0, clip_frames=100000, record_end=6691)
    occ = []
    assert ops.append_piece(layer, tl, clip, p, 24.0, 86400, lambda m: None, occ, followed=True) == ""
    assert len(occ) == 2 and occ[1][1] - occ[1][0] == 1 and occ[1][1] == 86400 + 6691


def test_larger_remainder_still_filled():
    layer, clip, tl = _stub_setup(short=False)
    clip["frames"] = 100000
    p = core.Placement("T", "C:/media/t.mov", 1, 1, 0, 3000, 0, speed_pct=99.7, clip_fps=25.0,
                       clip_frames=100000, record_end=2900)
    occ = []
    assert ops.append_piece(layer, tl, clip, p, 24.0, 86400, lambda m: None, occ, followed=False) == ""
    assert max(e for _, e in occ) <= 86400 + 2900


def test_missing_clips_leave_no_empty_tracks(tmp_path):
    res = make_result()
    paths = [c["path"] for c in res.clips.values() if not c["path"].endswith(("b_phone.mp4", "c_hi8.avi"))]
    layer = FakeLayer(paths, res)
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(event_timelines=False, song_markers=False, song_align="offset"))
    bk = core.Backup.new(str(tmp_path / "r.json"))
    errs = ops.apply_plan(layer, plan, bk, log=lambda m: None)
    assert all("clip not in media pool" in e for e in errs) and errs
    tl = layer.timelines["ShowSync Song - Alpha"]
    assert sorted({i["trackIndex"] for i in tl["items"]}) == [1, 2]            # A and L only, renumbered
    assert tl["tracks"] == (2, 2)
    assert tl["names"][("video", 2)].endswith("l_snip.mp4")


# ---------------------------------------------------------------- master-night song timeline

def make_master_result(x_tempo=1.03, z_speeds=None):
    """Master night = group gm (M1, 25 fps). Song time = master event time - 20 s. Other takes (loners):
    X is a take played x_tempo times slower, Y has an extra 8 s (4 bars) jam at clip 100-108 s, Z has a
    wobbly tempo map (knot every 2 s)."""
    clips = {c: _clip(c, "%s.mov" % c.lower(), 300.0, 25.0) for c in ("M1", "X", "Y", "Z")}
    seg = lambda i, owner, kind, a, b, m: {"id": i, "owner": owner, "owner_kind": kind, "start": a, "end": b,
                                          "song_id": "s1", "song_map": m, "match_score": 0.8, "is_partial": False}
    zs = list(range(0, 302, 2))
    sig, acc = [], 0.0
    for t in zs:
        sig.append(acc)
        acc += 2.0 * (z_speeds(t) if z_speeds else 1.0)
    segments = [
        seg("sM", "gm", "group", 0, 240, {"src": [0, 300], "dst": [-20, 280]}),
        seg("sX", "X", "clip", 10, 250, {"src": [0, 300], "dst": [-10 / x_tempo, 290 / x_tempo]}),
        seg("sY", "Y", "clip", 0, 300, {"src": [0, 100, 108, 300], "dst": [0, 100, 100.001, 292.001]}),
        seg("sZ", "Z", "clip", 0, 240, {"src": [float(t) for t in zs], "dst": sig}),
    ]
    groups = [{"id": "gm", "anchor": "M1", "anchor_tc": None, "conflicts": [],
               "maps": {"M1": {"src": [0, 300], "dst": [0, 300]}}, "spans": {"M1": [0, 300]}, "edges": []}]
    songs = [{"id": "s1", "name": "Tune", "reference": "segment:sM", "duration": 240.0,
              "segments": ["sM", "sX", "sY", "sZ"]}]
    return core.Result({"version": 1, "clips": clips, "groups": groups, "loners": ["X", "Y", "Z"],
                        "segments": segments, "songs": songs, "matches": [], "warnings": []}, "")


def _master_plan(res, **kw):
    kw.setdefault("master_id", "gm")      # the default master is the longest take (here Y), so name the night
    return core.build_master_timeline(res, res.songs[0], TL_FPS, **kw)


def test_master_slower_take_is_one_piece_near_103_percent():
    res = make_master_result()
    tp = _master_plan(res)
    x = sorted((p for p in tp.placements if p.clip_id == "X"), key=lambda p: p.record_frame)
    assert len(x) == 1 and x[0].media_type == 1                          # picture only, one piece
    assert abs(x[0].speed_pct - 103.0) < 1e-6
    assert x[0].drift_err_frames < 0.5
    assert x[0].record_frame == round(20 * 24) and x[0].start_frame == 250      # clip 10 s sits at master 20 s
    line = [w for w in tp.warnings if "x.mov" in w][0]
    assert "1 piece(s)" in line and "0 structure cut(s)" in line and "103.00" in line
    # master night: M1 with audio and video on the top slot, timeline time = master time
    m = [p for p in tp.placements if p.clip_id == "M1"]
    assert {p.media_type for p in m} == {1, 2} and len({p.track for p in m}) == 1
    assert m[0].track == max(p.track for p in tp.placements)
    assert (m[0].start_frame, m[0].end_frame, m[0].record_frame, m[0].speed_pct) == (0, 6000, 0, 100.0)


def test_master_extra_jam_is_cut_and_replaced():
    res = make_master_result()
    tp = _master_plan(res)
    y = sorted((p for p in tp.placements if p.clip_id == "Y"), key=lambda p: p.start_frame)
    assert len(y) == 2 and all(abs(p.speed_pct - 100.0) < 1e-6 for p in y)
    assert y[0].end_frame == 2500 and y[1].start_frame == 2700               # the 8 s jam (200 frames) is dropped
    assert y[0].record_end == y[1].record_frame == 2880                      # and the picture re-joins on the beat
    assert all(p.drift_err_frames < 0.5 for p in y)
    line = [w for w in tp.warnings if "y.mov" in w][0]
    assert "1 structure cut(s)" in line and "2 piece(s)" in line


def test_master_wobbly_tempo_few_pieces_within_band():
    wob = lambda t: 1.0 + 0.04 * math.sin(2 * math.pi * t / 4.0) + 0.02 * (t / 240.0 - 0.5)   # bar wobble + accelerando
    res = make_master_result(z_speeds=wob)
    tp = _master_plan(res, max_warp_pct=5.0)
    z = sorted((p for p in tp.placements if p.clip_id == "Z"), key=lambda p: p.start_frame)
    assert 1 <= len(z) <= 5, len(z)
    assert all(95.0 <= p.speed_pct <= 105.0 for p in z)
    assert all(p.drift_err_frames <= 2.0 for p in z)                          # vs the raw wobbly map (fit tolerance 1.5)
    zmap = core.TimeMap.from_dict(res.segments[3]["song_map"])
    for p in z:                                                               # independent check against the map
        a, b = p.start_frame / 25.0, p.end_frame / 25.0
        for i in range(41):
            t = a + (b - a) * i / 40
            placed = p.record_frame / 24.0 + (t - a) * 100.0 / p.speed_pct
            assert abs(placed - (zmap(t) + 20.0)) * 24 <= 2.0 + 0.5 + 1e-6
    assert all(p.record_end >= p.record_frame + 1 for p in z)


def test_master_take_beyond_warp_limit_is_not_placed():
    res = make_master_result(x_tempo=1.08)
    tp = _master_plan(res, max_warp_pct=5.0)
    assert not [p for p in tp.placements if p.clip_id == "X"]
    assert any("not placed" in w and "5.0" in w for w in tp.warnings)
    tp2 = _master_plan(res, max_warp_pct=10.0)
    assert [p for p in tp2.placements if p.clip_id == "X"]


def test_warp_runs_forgive_jitter_but_cut_structure():
    # per-beat jitter of a 0.5 s beat: net deviation 0.1 s < 0.25 s -> still usable
    reg, br = core.warp_runs(core.TimeMap([0.0, 0.5, 1.0, 1.5, 2.0, 10.0], [0.0, 0.6, 1.0, 1.6, 2.0, 10.0]), 5.0)
    assert reg == [(0.0, 10.0)] and br == []
    # an 8 s jam that stands still in master time
    reg, br = core.warp_runs(core.TimeMap([0, 100, 108, 200], [0, 100, 100.001, 192.001]), 5.0)
    assert reg == [(0.0, 100.0), (108.0, 200.0)] and br == [(100.0, 108.0)]


def test_master_audio_other_modes_multicam_and_apply(tmp_path):
    res = make_master_result()
    off = _master_plan(res)
    assert {p.track for p in off.placements if p.media_type == 2} == {4}        # master audio only, on the top slot
    assert off.n_video == 4 and off.n_audio == 4 and not off.muted_audio_slots
    muted = _master_plan(res, other_audio="muted")
    assert sorted(muted.muted_audio_slots) == [1, 2, 3]
    assert {p.clip_id for p in muted.placements if p.media_type == 2 and p.track <= 3} == {"X", "Y", "Z"}
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(
        event_timelines=False, song_timelines=False, song_markers=False, master_mode=True, master_group="gm",
        other_audio="muted", multicam=True))
    assert any("multicam clip skipped" in w for w in plan.warnings)
    logs = []
    bk = core.Backup.new(str(tmp_path / "r.json"))
    errs = ops.apply_plan(layer, plan, bk, log=logs.append)
    assert errs == [] and layer.collisions == 0, errs
    tl = layer.timelines["ShowSync Master - Tune"]
    assert tl["disabled"] == {("audio", 1), ("audio", 2), ("audio", 3)}
    assert tl["tracks"] == (4, 4)
    assert any("multicam clip skipped" in l for l in logs) and any("structure cut" in l for l in logs)
    assert any("max gap 0" in l for l in bk.summary)
    assert all(i["_speed"] == 100.0 for i in tl["items"] if i["trackIndex"] == 4)    # master angle stays plain
    assert any(abs(i["_speed"] - 103.0) < 1e-6 for i in tl["items"] if i["mediaType"] == 1)


def test_master_group_parameter_selects_other_master():
    res = make_master_result()
    tp = _master_plan(res, master_id="Z")                                        # a loner take as the master
    assert any(p.clip_id == "Z" and p.media_type == 2 for p in tp.placements)    # Z carries the audio now
    assert not any(p.clip_id == "M1" and p.media_type == 2 for p in tp.placements)
    tp2 = _master_plan(res, master_id="nope")
    assert any("has no take of this song" in w for w in tp2.warnings)
    default = core.build_master_timeline(res, res.songs[0], TL_FPS)                # no master named: longest take (Y)
    assert any(p.clip_id == "Y" and p.media_type == 2 for p in default.placements)


# ---------------------------------------------------------------- jittery per-beat maps

def _with_jittery_take(seed=7, tempo=1.02, jit=0.03, outliers=(40, 130, 300, 411), outlier_s=0.2, jam=None):
    """make_master_result plus take J: 120 bpm beat knots (every 0.5 s) with +-jit random noise per knot,
    one-beat outlier knots, on a `tempo` tempo difference. jam = (clip_t0, extra_s) inserts an extra stretch."""
    import random
    rnd = random.Random(seed)
    res = make_master_result()
    d = res.d
    d["clips"]["J"] = _clip("J", "j.mov", 300.0, 25.0)
    src = [0.5 * k for k in range(0, 481)]                          # 0..240 s
    dst = []
    for k, t in enumerate(src):
        sig = t / tempo
        if jam and t >= jam[0]:
            sig = (jam[0] if t < jam[0] + jam[1] else t - jam[1]) / tempo
        v = sig + rnd.uniform(-jit, jit)
        if k in outliers:
            v += outlier_s * rnd.choice((-1, 1))
        dst.append(v)
    for k in range(1, len(dst)):                                    # a real map is monotone
        dst[k] = max(dst[k], dst[k - 1] + 1e-4)
    d["segments"].append({"id": "sJ", "owner": "J", "owner_kind": "clip", "start": 0.0, "end": 240.0, "song_id": "s1",
                          "song_map": {"src": src, "dst": dst}, "match_score": 0.8, "is_partial": False})
    d["loners"].append("J")
    d["songs"][0]["segments"].append("sJ")
    return core.Result(d, ""), src, dst


def test_jittery_beat_map_gives_few_pieces_and_no_cuts():
    res, src, dst = _with_jittery_take()
    tp = _master_plan(res)
    j = sorted((p for p in tp.placements if p.clip_id == "J"), key=lambda p: p.start_frame)
    assert 1 <= len(j) <= 3, len(j)
    line = [w for w in tp.warnings if "j.mov" in w][0]
    assert "0 structure cut(s)" in line, line                                # beat jitter / outliers never cut
    assert all(abs(p.speed_pct - 100.0 * 1.02) < 0.3 for p in j)             # the 2 % tempo difference
    assert all(95.0 <= p.speed_pct <= 105.0 for p in j)
    # contiguous coverage: pieces follow each other in the source and on the timeline
    for a, b in zip(j, j[1:]):
        assert a.end_frame == b.start_frame and a.record_end == b.record_frame
    # picture follows the true (un-jittered) tempo map within the tolerance, outlier beats excluded
    odd = {40, 130, 300, 411}
    for p in j:
        for k, t in enumerate(src):
            if k in odd or not (p.start_frame / 25.0 <= t <= p.end_frame / 25.0):
                continue
            placed = p.record_frame / 24.0 + (t - p.start_frame / 25.0) * 100.0 / p.speed_pct
            truth = t / 1.02 + 20.0
            assert abs(placed - truth) * 24 <= 1.5 + 1.0, (k, abs(placed - truth) * 24)
    assert all(p.end_frame - p.start_frame >= 8 * 25 for p in j[:-1])        # pieces of at least 8 s


def test_jitter_with_a_real_jam_still_cuts_once():
    res, src, dst = _with_jittery_take(seed=3, jam=(100.0, 8.0))
    tp = _master_plan(res)
    j = sorted((p for p in tp.placements if p.clip_id == "J"), key=lambda p: p.start_frame)
    line = [w for w in tp.warnings if "j.mov" in w][0]
    assert "1 structure cut(s)" in line, line
    gaps = [(a.end_frame, b.start_frame) for a, b in zip(j, j[1:]) if b.start_frame - a.end_frame > 25]
    assert len(gaps) == 1 and abs((gaps[0][1] - gaps[0][0]) / 25.0 - 8.0) < 1.0     # the 8 s jam is skipped
    assert len(j) <= 5


def test_warp_regions_ignore_outlier_knots_and_jitter():
    import random
    rnd = random.Random(1)
    ks = [0.5 * k for k in range(0, 241)]
    ys = [t * 0.98 + rnd.uniform(-0.03, 0.03) + (0.4 if k in (50, 51) else 0.0) for k, t in enumerate(ks)]
    reg, br = core.warp_runs(core.TimeMap(ks, ys), 5.0)
    assert reg == [(0.0, 120.0)] and br == []
    # a skipped section: +8 s jump at t = 60 persists -> exactly one break
    ys2 = [y + (8.0 if t >= 60 else 0.0) for t, y in zip(ks, ys)]
    reg, br = core.warp_runs(core.TimeMap(ks, ys2), 5.0)
    assert len(reg) == 2 and len(br) == 1 and 55 <= br[0][0] <= 60.5 and br[0][1] <= 70


# ---------------------------------------------------------------- muted tracks verified, monotonic source joins

def test_muted_audio_tracks_disabled_after_placement_and_verified(tmp_path):
    res = make_master_result()
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(
        event_timelines=False, song_timelines=False, song_markers=False, master_mode=True, master_group="gm",
        other_audio="muted"))
    errs = ops.apply_plan(layer, plan, core.Backup.new(str(tmp_path / "r.json")), log=lambda m: None)
    assert errs == []
    tl = layer.timelines["ShowSync Master - Tune"]
    for idx in (1, 2, 3):                                   # every non-master audio track, read back as disabled
        assert not layer.is_track_enabled(tl, "audio", idx)
    assert layer.is_track_enabled(tl, "audio", 4)           # the master's audio stays on
    assert all(layer.is_track_enabled(tl, "video", i) for i in (1, 2, 3, 4))
    # a layer that does not honour SetTrackEnable is reported, not silently accepted
    bad = FakeLayer([c["path"] for c in res.clips.values()], res)
    bad.refuse_disable = True
    errs = ops.apply_plan(bad, plan, core.Backup.new(str(tmp_path / "r2.json")), log=lambda m: None)
    assert sum("still enabled" in e for e in errs) == 3


def test_consecutive_pieces_have_monotonic_source_steps():
    layer = FakeLayer(["C:/media/t.mov"])
    clip = layer.clips[core.norm_path("C:/media/t.mov")]
    clip["fps"], clip["frames"] = 50.0, 100000
    tl = layer.create_timeline("T", "01:00:00:00", 24.0)
    sp = 99.94
    mk = lambda sf, ef, rec, end: core.Placement("T", "C:/media/t.mov", 1, 1, sf, ef, rec, speed_pct=sp,
                                                clip_fps=50.0, clip_frames=100000, record_end=end)
    first = mk(0, 1666, 0, 800)
    st = {}
    occ = []
    assert ops.append_piece(layer, tl, clip, first, 24.0, 86400, lambda m: None, occ, True, st) == ""
    consumed_end = st["src_end"]
    assert consumed_end == 1666
    # the map re-derives the next piece's start 1 frame behind the consumed end (and another case: 1 ahead)
    for planned, expected in ((consumed_end - 1, consumed_end), (consumed_end + 2, consumed_end + 2)):
        layer2 = FakeLayer(["C:/media/t.mov"])
        c2 = layer2.clips[core.norm_path("C:/media/t.mov")]
        c2["fps"], c2["frames"] = 50.0, 100000
        tl2 = layer2.create_timeline("T", "01:00:00:00", 24.0)
        st2 = {}
        ops.append_piece(layer2, tl2, c2, first, 24.0, 86400, lambda m: None, [], True, st2)
        ops.append_piece(layer2, tl2, c2, mk(planned, planned + 1666, 800, 1600), 24.0, 86400, lambda m: None, [], False, st2)
        a, b = sorted(tl2["items"], key=lambda i: i["recordFrame"])[:2]
        # backwards step corrected to the consumed end; a forward gap keeps the planned start (sync never drifts)
        assert b["startFrame"] == expected
    # a real cut (gap larger than the tolerance) keeps the planned start
    layer3 = FakeLayer(["C:/media/t.mov"])
    c3 = layer3.clips[core.norm_path("C:/media/t.mov")]
    c3["fps"], c3["frames"] = 50.0, 100000
    tl3 = layer3.create_timeline("T", "01:00:00:00", 24.0)
    st3 = {}
    ops.append_piece(layer3, tl3, c3, first, 24.0, 86400, lambda m: None, [], True, st3)
    ops.append_piece(layer3, tl3, c3, mk(2400, 4066, 800, 1600), 24.0, 86400, lambda m: None, [], False, st3)
    assert sorted(tl3["items"], key=lambda i: i["recordFrame"])[-1]["startFrame"] == 2400


def test_apply_source_steps_are_monotonic_everywhere():
    res = _wobble_result()
    for kw in ({}, {"ceil_dur": True}, {"short_by_one": True}):
        layer = FakeLayer([c["path"] for c in res.clips.values()], res)
        for k, v in kw.items():
            setattr(layer, k, v)
        plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(song_timelines=False, song_markers=False))
        errs = ops.apply_plan(layer, plan, core.Backup.new("x.json"), log=lambda m: None)
        assert errs == [] or all("gap" not in e and "overlap" not in e for e in errs), errs
        tl = layer.timelines["ShowSync G01 Alpha"]
        for track in (1, 2, 3):
            for mt in (1, 2):
                it = sorted((i for i in tl["items"] if i["trackIndex"] == track and i["mediaType"] == mt),
                            key=lambda i: i["recordFrame"])
                for a, b in zip(it, it[1:]):
                    cons = round(a["_dur"] / 24.0 * a["_speed"] / 100.0 * a["_clip"]["fps"])
                    a_end = a["startFrame"] + cons
                    assert b["startFrame"] >= a_end - 0, (track, mt, a["startFrame"], a_end, b["startFrame"])
                    assert b["startFrame"] - a_end <= ops.SOURCE_JOIN_TOL or b["startFrame"] - a_end > 25   # no re-reads


# ---------------------------------------------------------------- one warp per non-master group

def _two_angle_night(tape_ppm=2190, wobble_s=0.02, tempo=1.02):
    """make_master_result plus a second night 'gn' (song time = (event-10)/tempo + 40): camera C (25 fps,
    event = clip + 5) and tape T (50 fps, +2190 ppm and slow wobble against the same event time line)."""
    res = make_master_result()
    d = res.d
    d["clips"]["C"] = _clip("C", "c.mov", 300.0, 25.0)
    d["clips"]["T"] = _clip("T", "t.mov", 300.0, 50.0)
    tsrc = [20.0 * i for i in range(16)]
    tdst = [t * (1 + tape_ppm * 1e-6) + 3.0 + wobble_s * math.sin(2 * math.pi * t / 300.0) for t in tsrc]
    d["groups"].append({"id": "gn", "anchor": "C", "anchor_tc": None, "conflicts": [],
                        "maps": {"C": {"src": [0, 300], "dst": [5, 305]}, "T": {"src": tsrc, "dst": tdst}},
                        "spans": {"C": [5.0, 305.0], "T": [tdst[0], tdst[-1]]}, "edges": []})
    d["segments"].append({"id": "sN", "owner": "gn", "owner_kind": "group", "start": 10.0, "end": 250.0,
                          "song_id": "s1", "song_map": {"src": [0, 320], "dst": [40 - 10 / tempo, 40 + 310 / tempo]},
                          "match_score": 0.8, "is_partial": False})
    d["songs"][0]["segments"].append("sN")
    return core.Result(d, ""), tsrc, tdst


def _event_at_timeline_frame(tp, res, cid, frame, tfps=24.0):
    """Event time of clip `cid` shown at timeline frame `frame`, from the planned video placements."""
    cfps = exact = None
    for p in tp.placements:
        if p.clip_id == cid and p.media_type == 1 and p.record_frame <= frame < p.record_end:
            a = p.start_frame / core.exact_fps(p.clip_fps)
            t = a + (frame - p.record_frame) / tfps * p.speed_pct / 100.0
            return res.group_map("gn", cid)(t)
    return None


def test_same_night_angles_share_one_warp():
    res, tsrc, tdst = _two_angle_night()
    tp = _master_plan(res)
    # one group warp line, per-angle lines mention extra pieces only
    gw = [w for w in tp.warnings if "G02 warp" in w]
    assert len(gw) == 1 and "shared by all angles" in gw[0]
    c = [p for p in tp.placements if p.clip_id == "C"]
    t = [p for p in tp.placements if p.clip_id == "T"]
    assert c and t and all(p.media_type == 1 for p in c + t)
    assert all(abs(p.speed_pct - 102.0) < 0.5 for p in c)                       # camera follows the group warp
    assert all(100.0 < p.speed_pct < 103.0 for p in t)                         # tape: warp and its own +2190 ppm
    # at many timeline frames both angles show the same event time (within one frame: two rounded placements)
    frames = range(int(70 * 24), int(180 * 24), 11)
    worst, n = 0.0, 0
    for f in frames:
        ec, et = _event_at_timeline_frame(tp, res, "C", f), _event_at_timeline_frame(tp, res, "T", f)
        if ec is None or et is None:
            continue
        worst = max(worst, abs(ec - et) * 24)
        n += 1
    assert n > 200
    assert worst <= 1.0, worst
    # and each angle follows the group warp, not its own musical fit (timeline time of event e)
    sm_n = core.TimeMap.from_dict(res.segments[-1]["song_map"])
    for f in frames:
        ec = _event_at_timeline_frame(tp, res, "C", f)
        if ec is not None:
            tau_expected = sm_n(ec) + 20.0                                      # master e = song + 20 (cut0 = 0)
            assert abs(tau_expected - f / 24.0) * 24 <= 1.5 + 1.0
    # the tape's own wobble needs extra pieces, the camera needs none beyond the group warp
    line_c = [w for w in tp.warnings if "c.mov" in w][0]
    line_t = [w for w in tp.warnings if "t.mov" in w][0]
    assert "0 extra piece(s)" in line_c
    assert "extra piece(s) beyond the group warp" in line_t


def test_group_warp_is_not_refitted_per_angle_when_clocks_differ_a_lot():
    res, _tsrc, _tdst = _two_angle_night(tape_ppm=2190, wobble_s=0.0)
    tp = _master_plan(res)
    t = [p for p in tp.placements if p.clip_id == "T"]
    c = [p for p in tp.placements if p.clip_id == "C"]
    assert len(t) <= len(c) + 2                                                # no wobble: tape adds ~no pieces
    for f in range(int(70 * 24), int(180 * 24), 13):
        ec, et = _event_at_timeline_frame(tp, res, "C", f), _event_at_timeline_frame(tp, res, "T", f)
        assert ec is not None and et is not None and abs(ec - et) * 24 <= 1.0


# ---------------------------------------------------------------- master audio rule, current timeline, titles

def _master_night_result(n_angles=3):
    """make_master_result with a master night of 1 anchor (not in the pool) + n_angles audio-bearing angles."""
    res = make_master_result()
    d = res.d
    for k in range(1, n_angles + 1):
        d["clips"]["MA%d" % k] = _clip("MA%d" % k, "ma%d.mov" % k, 300.0, 25.0)
        d["groups"][0]["maps"]["MA%d" % k] = {"src": [0, 300], "dst": [0.1 * k, 300 + 0.1 * k]}
        d["groups"][0]["spans"]["MA%d" % k] = [0.1 * k, 300 + 0.1 * k]
    return core.Result(d, "")


def _master_opts(**kw):
    return core.ApplyOptions(event_timelines=False, song_timelines=False, song_markers=False, master_mode=True,
                             master_group="gm", other_audio="muted", **kw)


def test_master_angles_keep_audio_when_the_anchor_is_missing_from_the_pool(tmp_path):
    res = _master_night_result(3)
    paths = [c["path"] for c in res.clips.values() if not c["path"].endswith("m1.mov")]   # anchor M1 not imported
    layer = FakeLayer(paths, res)
    plan = core.build_apply_plan(res, TL_FPS, _master_opts())
    tp = plan.timelines[0]
    assert len(tp.master_audio_slots) == 4 and sorted(tp.muted_audio_slots) == [1, 2, 3]
    errs = ops.apply_plan(layer, plan, core.Backup.new(str(tmp_path / "r.json")), log=lambda m: None)
    assert all("clip not in media pool" in e for e in errs), errs
    tl = layer.timelines["ShowSync Master - Tune"]
    # other nights: A1..A3 disabled; the three placed master angles (A4..A6) all enabled
    assert tl["tracks"][1] == 6
    for idx in (1, 2, 3):
        assert not layer.is_track_enabled(tl, "audio", idx)
    for idx in (4, 5, 6):
        assert layer.is_track_enabled(tl, "audio", idx)
    assert tl["disabled"] == {("audio", 1), ("audio", 2), ("audio", 3)}


def test_never_disable_every_audio_track(tmp_path):
    res = _master_night_result(3)
    paths = [c["path"] for c in res.clips.values() if not c["path"].startswith(("C:/media/m", "C:/media/ma"))]
    layer = FakeLayer(paths, res)               # no master clip at all in the pool
    plan = core.build_apply_plan(res, TL_FPS, _master_opts())
    errs = ops.apply_plan(layer, plan, core.Backup.new(str(tmp_path / "r.json")), log=lambda m: None)
    assert any("no master audio track would remain enabled" in e for e in errs)
    tl = layer.timelines["ShowSync Master - Tune"]
    assert not tl.get("disabled")               # nothing was disabled


def test_track_enable_needs_the_current_timeline_and_the_view_is_restored(tmp_path):
    res = make_master_result()
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    user_tl = {"items": [], "markers": {}, "fps": 24.0}
    layer.timelines["User timeline"] = user_tl
    layer.current = user_tl                      # the user is looking at another timeline
    plan = core.build_apply_plan(res, TL_FPS, _master_opts())
    errs = ops.apply_plan(layer, plan, core.Backup.new(str(tmp_path / "r.json")), log=lambda m: None)
    assert errs == []
    tl = layer.timelines["ShowSync Master - Tune"]
    assert tl["disabled"] == {("audio", 1), ("audio", 2), ("audio", 3)}      # only possible with tl made current
    assert layer.current is user_tl                                          # and the user's timeline is back
    # a layer that ignores enable changes on non-current timelines is what the fake models: direct call has no effect
    other = layer.timelines["ShowSync Master - Tune"]
    layer.current = user_tl
    layer.set_track_enable(other, "audio", 4, False)
    assert layer.is_track_enabled(other, "audio", 4)


def test_event_timeline_names_use_date_not_unnamed_songs():
    res = make_result()
    res.songs[0]["name"] = "Song 07"                       # unnamed (clustering) songs -> never a label
    res.songs[1]["name"] = "Song 08"
    assert core.group_title(res, 0) == "ShowSync G01 2026-05-01"                      # date of A (creation_time)
    res.songs[0]["name"] = "01 - Auto"                      # a named song is used
    assert core.group_title(res, 0) == "ShowSync G01 01 - Auto"
    res2 = make_result()
    res2.songs[1]["name"] = "Song 03"
    assert core.group_title(res2, 1) == "ShowSync G02"      # no named song and no creation_time
    res2.d["clips"]["D"]["creation_time"] = "2026-11-17T20:15:00"
    res3 = core.Result(res2.d, "")
    assert core.group_title(res3, 1) == "ShowSync G02 2026-11-17"
    # first named song wins over an earlier unnamed one
    res4 = make_result()
    res4.songs[0]["name"] = "Song 01"
    res4.d["segments"].append({"id": "s5", "owner": "g1", "owner_kind": "group", "start": 300.0, "end": 310.0,
                               "song_id": "song2", "song_map": {"src": [0, 600], "dst": [0, 600]},
                               "match_score": 0.8, "is_partial": False})
    assert core.group_title(core.Result(res4.d, ""), 0) == "ShowSync G01 Beta"


def test_master_id_accepts_group_label_and_skips_second_take_in_master_night():
    res = _master_night_result(1)
    res.d["segments"].append({"id": "sM2", "owner": "gm", "owner_kind": "group", "start": 250.0, "end": 290.0,
                              "song_id": "s1", "song_map": {"src": [0, 300], "dst": [-20, 280]},
                              "match_score": 0.8, "is_partial": False})
    res.d["songs"][0]["segments"].append("sM2")
    res = core.Result(res.d, "")
    tp = core.build_master_timeline(res, res.songs[0], TL_FPS, master_id="G01")          # label instead of id
    assert any(p.clip_id == "M1" and p.media_type == 2 for p in tp.placements)
    assert any("second take of this song inside the master night skipped" in w for w in tp.warnings)


# ---------------------------------------------------------------- real data acceptance (skipped when absent)

REAL_RESULT = os.environ.get("SHOWSYNC_REAL_RESULT") or ""     # path to a real analysis result JSON


def _angle_spreads(res, tp, gid, step=7):
    """Apply tp to a fake layer, then per timeline frame the event times shown by the placed angles of group gid."""
    layer = FakeLayer([c["path"] for c in res.clips.values()], res)
    plan = core.ApplyPlan(fps=TL_FPS)
    plan.timelines.append(tp)
    errs = ops.apply_plan(layer, plan, core.Backup.new(os.path.join(os.environ.get("TEMP", "."), "acc.json")),
                          log=lambda m: None)
    assert errs == [], errs
    tl = layer.timelines[tp.name]
    base = 86400
    path_cid = {res.clip_path(c): c for c in res.clips}
    items = [i for i in tl["items"] if i["mediaType"] == 1 and i["mediaPoolItem"]["path"] in path_cid and
             (res.group_of(path_cid[i["mediaPoolItem"]["path"]]) or {}).get("id") == gid]
    lo = min(i["recordFrame"] for i in items) - base
    hi = max(i["recordFrame"] + i["_dur"] for i in items) - base
    spreads, off = [], {}
    for tf in range(lo, hi, step):
        ev = {}
        for it in items:
            r = it["recordFrame"] - base
            if r <= tf < r + it["_dur"]:
                cid = path_cid[it["mediaPoolItem"]["path"]]
                cf = core.exact_fps(it["_clip"]["fps"])
                t = (it["startFrame"] + (tf - r) / core.exact_fps(TL_FPS) * it["_speed"] / 100.0 * cf) / cf
                ev[cid] = res.group_map(gid, cid)(t)
        if len(ev) >= 2:
            vals = sorted(ev.values())
            ref = vals[len(vals) // 2]
            for c, e in ev.items():
                off.setdefault(c, []).append((e - ref) * 1000.0)
            spreads.append((max(ev.values()) - min(ev.values())) * TL_FPS)
    return sorted(spreads), off


@pytest.mark.skipif(not os.path.exists(REAL_RESULT), reason="real analysis result not available")
@pytest.mark.parametrize("song_id", ["song08", "song05"])          # two songs of the real result
def test_real_data_same_night_angles_agree_in_master_timelines(song_id):
    res = core.Result.load(REAL_RESULT)
    song = next(s for s in res.songs if s["id"] == song_id)
    tp = core.build_master_timeline(res, song, TL_FPS, master_id="G02", tol_frames=0.5)
    for gid in ("G03", "G02"):                                    # another night and the master night itself
        spreads, off = _angle_spreads(res, tp, gid)
        assert len(spreads) > 100
        p50, p95 = spreads[len(spreads) // 2], spreads[int(len(spreads) * 0.95)]
        assert p50 <= 0.35 and p95 <= 0.9 and spreads[-1] <= 1.1, (gid, p50, p95, spreads[-1])
        for c, v in off.items():                                  # no systematic offset between angles (the live bug)
            v.sort()
            assert abs(v[len(v) // 2]) <= 20.0, (res.clip_name(c), v[len(v) // 2])      # live bug: 40-100 ms



def test_song_timeline_with_takes_from_several_owners_is_speed_matched():
    """Real footage (2026-11-23): song timelines that placed each night with one offset drifted
    0.6-1.5 s between nights. With takes from several owners the song timeline is now built like a
    master: the longest take at its own time, the others following the song map, their audio muted."""
    res = _master_night_result()
    plan = core.build_apply_plan(res, TL_FPS, core.ApplyOptions(event_timelines=False, song_markers=False))
    multi = [s for s in res.songs if len({x["owner"] for x in res.song_segments(s["id"])}) > 1]
    assert multi
    for s in multi:
        tp = next(t for t in plan.timelines if t.name == core._safe_name("ShowSync Song - %s" % s["name"]))
        assert tp.kind == "song"
        assert tp.master_audio_slots                       # the base take's audio plays
        assert tp.muted_audio_slots                        # other takes' audio is muted (they are speed-matched)
