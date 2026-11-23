from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from showsync.models import (AnalysisResult, EventGroup, MediaInfo, PairMatch, Song, SongSegment,
                             TimeMap)
from showsync.report.html import fmt_time, write_report

OUT = Path(__file__).parent / "_out"


def _info(cid, name, dur, device="", fps=25.0, video=True, **kw):
    return MediaInfo(clip_id=cid, path=f"D:/shoot/{name}", duration=dur, has_audio=True,
                     has_video=video, fps=fps, device=device, **kw)


def sample_result() -> AnalysisResult:
    clips = {
        "a": _info("a", "CAM_A_main_wide.mov", 3600, "Sony FX6", start_tc="19:58:10:00"),
        "b": _info("b", "CAM_B_stage_left.mov", 3500, "Canon C70", fps=23.976),
        "c": _info("c", "zoom_board_mix.wav", 3650, "Zoom F8", video=False, fps=0.0),
        "d": _info("d", "iphone_crowd_clip.mp4", 42, "iPhone 14"),
        "e": _info("e", "hi8_handycam_capture.avi", 1800, "", fps=29.97),
        "f": _info("f", "unrelated_soundcheck.mp4", 300, "iPhone 14"),
        "g": _info("g", "silent_gopro.mp4", 120, "GoPro"),
    }
    clips["g"].has_audio = False
    lin = TimeMap.linear
    maps = {"a": lin(1, 20.0), "b": lin(1.00018, 75.0), "c": lin(1.0, 0.0),
            "d": lin(1.0, 1500.0), "e": lin(0.9997, 900.0)}
    knots = np.linspace(0, 1800, 10)
    maps["e"] = TimeMap(knots, 900 + knots * 0.9997 + 0.004 * np.sin(knots / 90))
    spans = {k: (float(maps[k](0)), float(maps[k](clips[k].duration))) for k in maps}
    edges = [PairMatch("c", "a", lin(1, 20.0), 0.97, 3600, 0.4, 360, 340),
             PairMatch("a", "b", lin(1 / 1.00018, 55 / 1.00018), 0.88, 3500, 0.9, 350, 300),
             PairMatch("c", "d", lin(1, 1500.0), 0.61, 42, 1.2, 4, 3),
             PairMatch("c", "e", lin(1, 900.0), 0.72, 1800, 3.1, 180, 150, nonlinear=True)]
    clips["h"] = _info("h", "ref_opening_number.wav", 300, "", video=False, fps=0.0, role="reference")
    maps["h"] = lin(1.0, 120.0)
    spans["h"] = (120.0, 420.0)
    edges[0].ambiguous = True
    edges[0].alternatives = [lin(1, 20.0 + 28.8)]
    edges[0].note = "identical choruses"
    g = EventGroup("G01", "a", maps, spans, edges,
                   conflicts=["CAM_B_stage_left.mov and iphone_crowd_clip.mp4: sync disagrees by 31 ms"],
                   anchor_tc="19:57:50:00", alt_shifts={"d": [-28.8, 28.8]})
    segs = [SongSegment("s1", "G01", "group", 120, 420, "song1", None, 0.9),
            SongSegment("s2", "G01", "group", 500, 800, "song2", None, 0.8),
            SongSegment("s3", "G01", "group", 1500, 1542, "song1", None, 0.66, True),
            SongSegment("s4", "f", "clip", 10, 200, "song2", None, 0.58),
            SongSegment("s5", "f", "clip", 210, 290, None, None, 0.0)]
    songs = [Song("song1", "Opening Number", "segment:s1", 300, ["s1", "s3"]),
             Song("song2", "Slow Burn", "file:D:/refs/Slow Burn.wav", 300, ["s2", "s4"])]
    return AnalysisResult(1, clips, [g], ["f", "g", "h"], segs, songs, edges,
                          warnings=["iphone_crowd_clip.mp4 has a variable frame rate"])


class _Check(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []
        self.void = {"meta", "link", "br", "hr", "img", "input", "line", "rect", "path", "circle"}

    def handle_starttag(self, tag, attrs):
        if tag not in self.void:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append((tag, list(self.stack[-3:])))
        else:
            self.stack.pop()


def test_report_written_and_sane(tmp_path):
    r = sample_result()
    p = write_report(r, str(tmp_path / "r.html"))
    text = Path(p).read_text(encoding="utf-8")
    assert text.lower().startswith("<!doctype html>")
    for i in r.clips.values():
        assert Path(i.path).name in text
    for needle in ("G01", "Opening Number", "Slow Burn", "prefers-color-scheme", "nonlinear",
                   "Loners", "Conflicts", "ref_opening_number.wav", "also fits at -28.8 s, +28.8 s", "Ambiguous matches", "ghost", "19:57:50:00", "ppm"):
        assert needle in text
    assert "http://" not in text.replace("http://www.w3.org", "") and "https://" not in text
    chk = _Check()
    chk.feed(text)
    assert not chk.errors and not chk.stack, (chk.errors, chk.stack)


def test_report_empty_result(tmp_path):
    r = AnalysisResult(1, {}, [], [], [], [], [])
    assert Path(write_report(r, str(tmp_path / "e.html"))).exists()


def test_render_sample():
    OUT.mkdir(exist_ok=True)
    p = write_report(sample_result(), str(OUT / "report_sample.html"))
    assert Path(p).stat().st_size > 5000


def test_fmt_time():
    assert fmt_time(75.5) == "1:15.500"
    assert fmt_time(3725.0, ms=False) == "1:02:05"
