import numpy as np
import pytest

from showsync.graph import groups as G
from showsync.graph.groups import build_groups, seconds_to_tc, tc_to_seconds
from showsync.models import MediaInfo, PairMatch, TimeMap


def info(cid, dur, fps=25.0, **kw):
    return MediaInfo(clip_id=cid, path=f"C:/x/{cid}.mp4", duration=dur, has_audio=True,
                     has_video=True, fps=fps, **kw)


def match(truth, a, b, conf=0.9, err=0.0):
    """truth[c] = (ratio, offset): event = ratio*t + offset. Returns PairMatch t_a = map(t_b)."""
    ra, oa = truth[a]
    rb, ob = truth[b]
    m = TimeMap.linear(rb / ra, (ob - oa) / ra + err)
    return PairMatch(a=a, b=b, map=m, confidence=conf, overlap_s=10, residual_ms=0.5,
                     n_windows=5, n_inliers=5)


def check_truth(g, truth, infos, tol=2e-3):
    """Recovered event maps equal truth up to a common shift."""
    shifts = []
    for c, (r, o) in truth.items():
        if c not in g.maps:
            continue
        t = np.linspace(0, infos[c].duration, 9)
        shifts.append(r * t + o - g.maps[c](t))
    base = shifts[0][0]
    for s in shifts:
        assert np.allclose(s, base, atol=tol)


def test_chain_through_snippet():
    truth = {"A": (1.0, 0.0), "B": (1.0, 100.0), "C": (1.0, 150.0)}
    infos = {"A": info("A", 300), "B": info("B", 200), "C": info("C", 20)}
    ms = [match(truth, "B", "A", 0.9), match(truth, "B", "C", 0.7)]   # C only matches B
    groups, loners = build_groups(ms, infos)
    assert loners == [] and len(groups) == 1
    g = groups[0]
    assert g.id == "G01" and g.anchor == "A"
    check_truth(g, truth, infos)
    assert g.spans["A"][0] == pytest.approx(0.0, abs=1e-6)
    assert g.spans["C"] == pytest.approx((150.0, 170.0), abs=1e-3)
    assert g.duration == pytest.approx(300.0, abs=1e-3)
    assert g.conflicts == []


def test_earliest_covered_moment_is_zero():
    truth = {"A": (1.0, 0.0), "B": (1.0, -30.0)}
    infos = {"A": info("A", 300), "B": info("B", 100)}
    g = build_groups([match(truth, "A", "B")], infos)[0][0]
    assert g.spans["B"][0] == pytest.approx(0.0, abs=1e-6)
    assert g.spans["A"][0] == pytest.approx(30.0, abs=1e-3)


def test_triangle_with_one_bad_edge():
    truth = {"A": (1.0, 0.0), "B": (1.0, 40.0), "C": (1.0, 90.0)}
    infos = {k: info(k, 200) for k in truth}
    ms = [match(truth, "A", "B", 0.9), match(truth, "B", "C", 0.85),
          match(truth, "A", "C", 0.55, err=12.0)]                      # wrong place
    g = build_groups(ms, infos)[0][0]
    check_truth(g, truth, infos)
    assert len(g.edges) == 2
    assert any("dropped" in c for c in g.conflicts)


def test_triangle_small_disagreement_refined_and_reported():
    truth = {"A": (1.0, 0.0), "B": (1.0, 40.0), "C": (1.0, 90.0)}
    infos = {k: info(k, 200) for k in truth}
    ms = [match(truth, "A", "B", 0.9), match(truth, "B", "C", 0.9),
          match(truth, "A", "C", 0.9, err=0.3)]
    g = build_groups(ms, infos)[0][0]
    assert len(g.edges) == 3
    assert any("disagrees by" in c for c in g.conflicts)
    ms[2] = match(truth, "A", "C", 0.9, err=0.004)
    g = build_groups(ms, infos)[0][0]
    assert g.conflicts == []


def test_drift_composes_through_chain():
    truth = {"A": (1.0, 0.0), "B": (1.0005, 20.0), "C": (0.9996, 55.0)}   # +500 ppm, -400 ppm
    infos = {k: info(k, 1800) for k in truth}
    infos["A"].duration = 1900                      # A is the anchor (ratio 1)
    ms = [match(truth, "A", "B"), match(truth, "B", "C")]
    g = build_groups(ms, infos)[0][0]
    assert g.anchor == "A"
    check_truth(g, truth, infos, tol=1e-3)
    assert g.maps["B"].ratio == pytest.approx(1.0005, abs=1e-8)
    assert g.maps["C"].ratio == pytest.approx(0.9996, abs=1e-8)


def test_loners_and_two_groups_ordering():
    truth = {"A": (1.0, 0.0), "B": (1.0, 5.0), "C": (1.0, 0.0), "D": (1.0, 7.0)}
    infos = {k: info(k, 100) for k in "ABCDEF"}
    infos["C"].duration = 400
    infos["D"].duration = 400
    infos["F"].has_audio = False
    ms = [match(truth, "A", "B"), match(truth, "C", "D")]
    groups, loners = build_groups(ms, infos)
    assert loners == ["E", "F"]
    assert [g.id for g in groups] == ["G01", "G02"]
    assert set(groups[0].maps) == {"C", "D"}            # no clocks: bigger first


def test_group_order_by_creation_time():
    truth = {"A": (1.0, 0.0), "B": (1.0, 5.0), "C": (1.0, 0.0), "D": (1.0, 7.0)}
    infos = {k: info(k, 100) for k in "ABCD"}
    infos["A"].creation_time = "2024-05-02T20:00:00Z"
    infos["B"].creation_time = "2024-05-02T20:00:05Z"
    infos["C"].creation_time = "2024-05-01T20:00:00Z"
    infos["D"].creation_time = "2024-05-01T20:00:07Z"
    groups, _ = build_groups([match(truth, "A", "B"), match(truth, "C", "D")], infos)
    assert set(groups[0].maps) == {"C", "D"}


def test_same_device_overlap_conflict():
    truth = {"A": (1.0, 0.0), "B": (1.0, 10.0), "C": (1.0, 0.0)}
    infos = {"A": info("A", 300, device="CamX"), "B": info("B", 100, device="CamX"),
             "C": info("C", 300, device="Zoom")}
    ms = [match(truth, "C", "A"), match(truth, "C", "B")]
    g = build_groups(ms, infos)[0][0]
    assert any("same device" in c for c in g.conflicts)
    truth["B"] = (1.0, 310.0)
    infos["C"].duration = 500
    g = build_groups([match(truth, "C", "A"), match(truth, "C", "B")], infos)[0][0]
    assert g.conflicts == []


def test_anchor_is_longest_steady_clip_and_borrows_tc():
    truth = {"A": (1.0, 0.0), "B": (1.0, 60.0)}
    infos = {"A": info("A", 500), "B": info("B", 100, start_tc="10:00:00:00")}
    infos["A"].start_tc = "00:00:00:00"            # reset-to-zero TC is not real
    g = build_groups([match(truth, "A", "B")], infos)[0][0]
    assert g.anchor == "A"                         # longest clip defines the clock
    assert g.anchor_tc == "09:59:00:00"            # time of day borrowed from B (starts 60 s in)
    infos["B"].start_tc = None
    assert build_groups([match(truth, "A", "B")], infos)[0][0].anchor_tc is None


def test_pair_pruning():
    a = info("a", 100, device="Cam", creation_time="2024-05-01T20:00:00Z")
    b = info("b", 100, device="Cam", creation_time="2024-05-01T20:10:00Z")
    c = info("c", 100, device="Cam", creation_time="2024-05-01T20:01:00Z")
    p = info("p", 100, device="Phone", creation_time="2030-01-01T00:00:00Z")
    assert G.can_skip_pair(a, b)
    assert not G.can_skip_pair(a, c)            # 60 s apart, 100 s long: overlaps
    assert not G.can_skip_pair(a, p)            # different device: never trust clocks
    assert not G.can_skip_pair(a, info("q", 100, device="Cam"))


def test_match_all_uses_match_pair_and_threshold(monkeypatch):
    import sys
    import types
    import showsync.dsp
    from showsync.models import Features
    align = types.ModuleType("showsync.dsp.align")     # stands in until/besides the real module
    monkeypatch.setitem(sys.modules, "showsync.dsp.align", align)
    monkeypatch.setattr(showsync.dsp, "align", align, raising=False)
    z = np.zeros(10, np.float32)
    feats = {c: Features(c, 10.0, z, z, np.zeros((12, 2), np.float32), z) for c in "abcd"}
    audio = {c: np.zeros(8000, np.float32) for c in "abcd"}
    infos = {c: info(c, 10.0) for c in "abcd"}
    infos["d"].has_audio = False

    def fake(fa, fb, aa, ab, sr=8000):
        conf = {("a", "b"): 0.9, ("a", "c"): 0.3}.get((fa.clip_id, fb.clip_id))
        if conf is None:
            return None
        return PairMatch(fa.clip_id, fb.clip_id, TimeMap.identity(), conf, 5, 1, 3, 3)

    monkeypatch.setattr(align, "match_pair", fake, raising=False)
    seen = []
    out = G.match_all(feats, audio, infos, progress=lambda d, t, m: seen.append((d, t)))
    assert [(m.a, m.b) for m in out] == [("a", "b")]
    assert seen[-1] == (3, 3)


@pytest.mark.parametrize("fps,drop,tc", [
    (25, False, "01:02:03:04"), (24, False, "23:59:59:23"), (30, False, "00:10:00:00"),
    (29.97, True, "00:01:00;02"), (29.97, True, "00:10:00;00"), (29.97, True, "01:09:59;29"),
    (29.97, False, "01:00:00:00"), (59.94, True, "00:01:00;04"), (59.94, True, "00:09:59;59"),
])
def test_tc_roundtrip(fps, drop, tc):
    assert seconds_to_tc(tc_to_seconds(tc, fps), fps, drop) == tc


def test_dropframe_known_values():
    assert tc_to_seconds("01:00:00;00", 29.97) * 30000 / 1001 == pytest.approx(107892)
    assert seconds_to_tc(107892 * 1001 / 30000, 29.97, True) == "01:00:00;00"
    assert seconds_to_tc(1800 * 1001 / 30000, 29.97, True) == "00:01:00;02"
    assert seconds_to_tc(1799 * 1001 / 30000, 29.97, True) == "00:00:59;29"
    assert tc_to_seconds("01:00:00:00", 29.97) == pytest.approx(3603.6, abs=1e-3)
    assert tc_to_seconds("00:00:01:12", 25) == pytest.approx(1.48)


def test_tc_wraps_at_24h():
    assert seconds_to_tc(-1.0, 25) == "23:59:59:00"
    assert seconds_to_tc(86400 + 2.0, 25) == "00:00:02:00"


# ---- identical repeats (playback music videos): ambiguous edges with exact alternatives

def amb(truth, a, b, wrong_first=True, period=28.8, conf=0.95):
    m = match(truth, a, b, conf)
    alt = TimeMap.linear(m.map.ratio, m.map.offset + period)
    if wrong_first:
        m.map, alts = alt, [m.map]
    else:
        alts = [alt]
    m.ambiguous, m.alternatives = True, alts
    return m


def test_ambiguous_edge_resolved_by_unambiguous_neighbours():
    truth = {"R": (1.0, 0.0), "T1": (1.0, 10.0), "T2": (1.0, 60.0)}
    infos = {"R": info("R", 200, role="reference"), "T1": info("T1", 90), "T2": info("T2", 80)}
    ms = [amb(truth, "R", "T1"),                  # wrong placement listed first
          match(truth, "R", "T2", 0.9),
          match(truth, "T2", "T1", 0.6)]          # unambiguous: settles T1
    g = build_groups(ms, infos)[0][0]
    assert g.anchor == "R"
    check_truth(g, truth, infos)
    assert "T1" not in g.alt_shifts


def test_purely_ambiguous_pair_keeps_alternatives():
    truth = {"R": (1.0, 0.0), "T1": (1.0, 10.0)}
    infos = {"R": info("R", 200, role="reference"), "T1": info("T1", 30)}
    g = build_groups([amb(truth, "R", "T1", wrong_first=False)], infos)[0][0]
    check_truth(g, truth, infos)
    assert g.alt_shifts["T1"] == [pytest.approx(28.8, abs=1e-6)]


def test_two_ambiguous_edges_vote_for_the_common_placement():
    truth = {"R": (1.0, 0.0), "T1": (1.0, 10.0), "T2": (1.0, 60.0)}
    infos = {"R": info("R", 200, role="reference"), "T1": info("T1", 90), "T2": info("T2", 80)}
    ms = [amb(truth, "R", "T1", period=28.8),     # T1 at truth or +28.8
          match(truth, "R", "T2", 0.9),
          amb(truth, "T2", "T1", period=-14.4)]   # T1 at truth or another place
    g = build_groups(ms, infos)[0][0]
    check_truth(g, truth, infos)
    assert "T1" not in g.alt_shifts


def test_corroboration_accepts_agreeing_weak_matches():
    from showsync.graph.groups import corroborate
    truth = {"A": (1.0, 0.0), "B": (1.0, 50.0), "X": (1.0, 120.0), "Y": (1.0, 300.0)}
    infos = {c: info(c, d) for c, d in (("A", 600), ("B", 400), ("X", 60), ("Y", 60))}
    strong = [match(truth, "A", "B", 0.95)]
    weak = [match(truth, "A", "X", 0.3), match(truth, "B", "X", 0.4),   # agree -> accept X
            match(truth, "A", "Y", 0.3), match(truth, "B", "Y", 0.3, err=0.5)]  # disagree -> not Y
    acc, notes = corroborate(strong, weak, infos)
    groups, loners = build_groups(acc, infos)
    assert set(groups[0].maps) == {"A", "B", "X"} and loners == ["Y"]
    check_truth(groups[0], {k: truth[k] for k in "ABX"}, infos)
    assert notes and "X" in notes[0]


def _flip(m):
    from dataclasses import replace
    return replace(m, a=m.b, b=m.a, map=m.map.inverse(),
                   alternatives=[x.inverse() for x in (m.alternatives or [])])


def test_group_maps_do_not_depend_on_pair_orientation():
    """Regression (real concert footage): a short phone clip matched against a long
    tape was checked at 1-2 points when the tape was `b` and 41 when it was `a`, so the offset
    refinement moved clips by 20-25 ms depending on clip ids (which change when files move)."""
    truth = {"T": (1.002, 0.0), "A": (1.0, 200.0), "C": (1.0, 1500.0), "D": (1.0, 2600.0)}
    infos = {"T": info("T", 4200), "A": info("A", 3000), "C": info("C", 60), "D": info("D", 90)}

    def skew(m, ppm, off=0.0):          # slightly wrong slope/offset, as real short-overlap fits are
        return PairMatch(a=m.a, b=m.b, map=TimeMap.linear(m.map.ratio * (1 + ppm * 1e-6), m.map.offset + off),
                         confidence=m.confidence, overlap_s=m.overlap_s, residual_ms=m.residual_ms,
                         n_windows=5, n_inliers=5)

    ms = [match(truth, "A", "T", 0.95), skew(match(truth, "T", "C", 0.8), 300, 0.004),
          skew(match(truth, "A", "C", 0.9), -200), skew(match(truth, "T", "D", 0.8), -250, -0.003),
          match(truth, "A", "D", 0.9)]
    ref = build_groups(ms, infos)[0][0]
    for k in range(len(ms)):
        alt = list(ms)
        alt[k] = _flip(alt[k])
        g = build_groups(alt, infos)[0][0]
        for c in truth:
            t = np.linspace(0, infos[c].duration, 7)
            assert np.allclose(g.maps[c](t), ref.maps[c](t), atol=1e-4), (k, c)


def test_single_weak_match_accepted_when_it_reproduces_tape_drift():
    """Real case: a drum-side phone matched only the tape, weakly (0.27),
    but its slope restored the tape's +2190 ppm error to a steady phone clock exactly."""
    from showsync.graph.groups import corroborate
    truth = {"CAM": (1.0, 0.0), "T": (1.0022, 0.0), "P1": (1.0, 300.0), "X": (1.0, 2500.0)}
    infos = {"CAM": info("CAM", 2400), "T": info("T", 3500), "P1": info("P1", 600), "X": info("X", 1500)}
    strong = [match(truth, "CAM", "T", 0.95), match(truth, "CAM", "P1", 0.9), match(truth, "T", "P1", 0.9)]

    def weak(ratio_err_ppm=0.0, conf=0.27):
        m = match(truth, "T", "X", conf)
        return PairMatch(a="T", b="X", map=TimeMap.linear(m.map.ratio * (1 + ratio_err_ppm * 1e-6), m.map.offset),
                         confidence=conf, overlap_s=900, residual_ms=5.3, n_windows=20, n_inliers=12)

    acc, notes = corroborate(strong, [weak()], infos)
    groups, loners = build_groups(acc, infos)
    assert "X" not in loners and any("speed error" in n for n in notes)
    check_truth(groups[0], truth, infos, tol=5e-3)
    # a wrong match: its slope does not undo the tape's drift (X would run at +2190 ppm)
    acc, notes = corroborate(strong, [weak(ratio_err_ppm=-2190)], infos)
    assert "X" in build_groups(acc, infos)[1]
    # too weak, even with the right slope
    acc, notes = corroborate(strong, [weak(conf=0.12)], infos)
    assert "X" in build_groups(acc, infos)[1]


def test_steady_clocks_become_straight_lines_tapes_keep_their_wow():
    """Real footage (2026-11-23): Canon/phone maps carried 25-73 ms of piecewise fit noise inherited
    through a tape; straight lines fit the audio better and timelines stopped chasing the noise."""
    rng = np.random.default_rng(3)
    T = 3000.0
    ts = np.linspace(0, T, 300)
    wow = 0.03 * np.sin(ts / 97.0)                    # real tape wow, +-30 ms
    tape_event = 1.0022 * ts + wow                    # tape clip time -> event (camera A clock)
    infos = {"A": info("A", 2800), "T": info("T", T), "C": info("C", 1200), "D": info("D", 900)}
    ms = [PairMatch(a="A", b="T", map=TimeMap(ts, tape_event), confidence=0.95, overlap_s=2800,
                    residual_ms=2.0, n_windows=50, n_inliers=48, nonlinear=True)]
    for cid, off, n in (("C", 600.0, 120), ("D", 1700.0, 90)):
        tc = np.linspace(0, infos[cid].duration, n)
        ev = tc + off                                  # truth: steady clock, ratio 1
        # measured C/D -> tape clip time, through the (wobbly) tape, plus +-15 ms fit noise
        t_tape = np.interp(ev, tape_event, ts) + rng.normal(0, 0.015, n)
        ms.append(PairMatch(a="T", b=cid, map=TimeMap(tc, t_tape), confidence=0.8, overlap_s=infos[cid].duration,
                            residual_ms=4.0, n_windows=40, n_inliers=38, nonlinear=True))
    g = build_groups(ms, infos)[0][0]
    assert g.anchor == "A"
    for cid, off in (("C", 600.0), ("D", 1700.0)):
        assert g.maps[cid].is_linear, cid
        t = np.linspace(0, infos[cid].duration, 50)
        err = (g.maps[cid](t) - g.maps["A"](t * 0 + 0) ) - (t + off)
        assert np.ptp(err) < 0.012, (cid, np.ptp(err))          # straight, right slope
    assert not g.maps["T"].is_linear                            # the tape keeps its measured wow
