"""Tests for showsync.music (song segmentation, song matching, song maps, point propagation).

Part 1: small self-made signals (DTW mechanics, speed, helpers) - no synth needed.
Part 2: tests/synth.py material (songs, live takes, concerts, device simulations). Material is
generated once per module (~1-2 min). Run with -s to see the tables.

Thresholds: the scores here come from the reversed-reference normalised DTW (see dtw.py).
REC_SIMILAR / REC_REF_ACCEPT are the values this module's calibration recommends for
config.SONG_SIMILAR / SONG_REF_ACCEPT; tests that involve degraded devices (phone, Hi8,
camera mic) pass them explicitly until config.py is updated. Clean material is also tested with
the config defaults.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from scipy.signal import resample_poly

from showsync.config import ANALYSIS_SR, CHROMA_RATE, SONG_REF_ACCEPT, SONG_SIMILAR
from showsync.models import AnalysisResult, EventGroup, Features, SongSegment, TimeMap, _to_jsonable
from showsync.music import dtw, songs

REC_SIMILAR = 0.35
REC_REF_ACCEPT = 0.35


# =========================================================================== part 1: mechanics

def _rand_chroma(n, seed, rate=10.0, chord_s=1.5, loop=None):
    """Piecewise-constant random 'chords' (12, n) at `rate` Hz, L2-normalised. With `loop` the
    chords cycle through that many (one key, like a song); otherwise every chord is new."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / (chord_s * rate))) + 1
    chords = np.abs(rng.normal(size=(12, loop or k))) ** 3
    idx = (np.arange(n) / (chord_s * rate)).astype(int) % (loop or k)
    c = chords[:, idx] + 0.05 * np.abs(rng.normal(size=(12, n)))
    return (c / np.linalg.norm(c, axis=0)).astype(np.float32)


def _stretch(c, factor):
    """Resample a chroma sequence in time: factor > 1 makes it longer (slower)."""
    n = c.shape[1]
    m = int(round(n * factor))
    src = np.clip(np.arange(m) / factor, 0, n - 1)
    return np.stack([np.interp(src, np.arange(n), c[k]) for k in range(12)]).astype(np.float32)


def test_subseq_dtw_finds_tempo_changed_query():
    ref = _rand_chroma(6000, 1)                      # 10 min at 10 Hz
    for factor in (0.6, 0.8, 1.0, 1.25, 1.7):
        q = _stretch(ref[:, 2000:3200], factor)      # 2 min of the reference at another tempo
        R, Q = dtw.cens(ref), dtw.cens(q)
        m = dtw.match(Q, R)
        assert m.score > 0.6, (factor, m.score)
        assert abs(m.ref_start / 2.0 - 200.0) < 3.0, (factor, m.ref_start)
        assert abs(m.ref_end / 2.0 - 320.0) < 3.0, (factor, m.ref_end)
    other = dtw.cens(_rand_chroma(1200, 2))
    assert dtw.match(other, dtw.cens(ref)).score < 0.25


def test_match_many_equals_match():
    refs = [dtw.cens(_rand_chroma(n, s)) for n, s in ((900, 3), (2000, 4), (1500, 5))]
    q = dtw.cens(_stretch(_rand_chroma(2000, 4)[:, 500:900], 1.1))
    many = dtw.match_many(q, refs)
    for r, mm in zip(refs, many):
        assert abs(dtw.match(q, r, return_path=False).score - mm.score) < 1e-6
    assert int(np.argmax([m.score for m in many])) == 1


def test_dtw_speed_song_vs_two_hour_concert():
    ref = _rand_chroma(72000, 7)                     # 2 h at 10 Hz
    q = _stretch(ref[:, 30000:32400], 1.05)          # 4 min, 5 % slower
    t0 = time.perf_counter()
    R, Q = dtw.cens(ref), dtw.cens(q)
    m = dtw.match(Q, R)
    dt = time.perf_counter() - t0
    print(f"\n4-min query vs 2-h reference: {dt:.2f} s (score {m.score:.2f})")
    assert abs(m.ref_start / 2.0 - 3000.0) < 3.0
    assert dt < 5.0


def test_fit_map_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    x = np.arange(0, 120, 0.2)
    y = 10 + 0.95 * x + rng.normal(0, 0.3, x.size)
    y[200:210] += 30                                 # path glitch
    tm = songs._fit_map(x, y)
    slopes = np.diff(tm.knots_dst) / np.diff(tm.knots_src)
    assert slopes.min() >= songs.MAP_TEMPO[0] - 1e-6 and slopes.max() <= songs.MAP_TEMPO[1] + 1e-6
    err = np.abs(tm(x) - (10 + 0.95 * x))
    assert np.median(err) < 0.15 and err.max() < 2.0


def test_crop_features():
    n10 = 600
    f = Features("c", 60.0, np.arange(6000, dtype=np.float32), np.zeros(6000, np.float32),
                 _rand_chroma(n10, 1), np.linspace(0, 1, n10).astype(np.float32))
    g = songs.crop_features(f, 110.0, 130.0, start_offset=100.0)
    assert g.chroma.shape == (12, 200) and g.onset.shape == (2000,)
    assert g.onset[0] == 1000 and abs(g.duration - 20.0) < 1e-9


def _seg(owner, i, a, b, song, tm, kind="clip"):
    return SongSegment(id=f"{owner}:{i:02d}", owner=owner, owner_kind=kind, start=a, end=b,
                       song_id=song, song_map=tm)


def test_propagate_points_exact_and_song_and_json_twin():
    # group G: clip A is the anchor, clip B starts 50 s later and runs 0.1 % fast
    g = EventGroup(id="G", anchor="A",
                   maps={"A": TimeMap.identity(), "B": TimeMap.linear(0.999, 50.0, 0, 100)},
                   spans={"A": (0.0, 300.0), "B": (50.0, 249.8)}, edges=[])
    segs = [_seg("G", 0, 10.0, 200.0, "song01", TimeMap.linear(1.0, -10.0, 10, 200), "group"),
            _seg("L", 0, 5.0, 100.0, "song01", TimeMap.linear(1.1, -5.5, 5, 100)),     # other night, faster
            _seg("L", 1, 120.0, 150.0, "song02", TimeMap.linear(1.0, -120.0, 120, 150))]
    # exact: A -> B
    r = songs.propagate_points([60.0, 10.0], "A", "B", segs, [g])
    assert abs(r[0] - 10.0 / 0.999) < 1e-6 and r[1] is None
    # song: B time 20 -> event 69.98 -> song 59.98 -> L time 5 + 59.98/1.1
    r = songs.propagate_points([20.0, 1000.0], "B", "L", segs, [g])
    assert abs(r[0] - (5.0 + (0.999 * 20 + 50 - 10) / 1.1)) < 1e-6 and r[1] is None
    # back: L -> A (A is in G, so via G's segment)
    r = songs.propagate_points([50.0, 130.0], "L", "A", segs, [g])
    assert abs(r[0] - (10.0 + (50.0 * 1.1 - 5.5))) < 1e-6 and r[1] is None
    # JSON twin gives the same numbers
    res = {"groups": [_to_jsonable(g)], "segments": [_to_jsonable(s) for s in segs]}
    res = json.loads(json.dumps(res))
    for args in (([60.0, 10.0], "A", "B"), ([20.0, 1000.0], "B", "L"), ([50.0, 130.0], "L", "A"),
                 ([30.0], "G", "B")):
        a = songs.propagate_points(args[0], args[1], args[2], segs, [g])
        b = songs.propagate_points_json(res, *args)
        assert all((x is None and y is None) or abs(x - y) < 1e-9 for x, y in zip(a, b)), (args, a, b)


def test_group_features_coverage_hole_marks_partial():
    """Two clips in one group with a hole between them: music running into the hole is partial."""
    rate = CHROMA_RATE
    def clip(cid, dur, seed):
        n = int(dur * rate)
        return Features(cid, dur, np.zeros(int(dur * 100), np.float32), np.full(int(dur * 100), -20, np.float32),
                        _rand_chroma(n, seed, loop=4), np.full(n, 0.8, np.float32))
    fa, fb = clip("a", 100.0, 1), clip("b", 100.0, 1)
    g = EventGroup(id="G", anchor="a", maps={"a": TimeMap.identity(), "b": TimeMap.linear(1.0, 150.0, 0, 100)},
                   spans={"a": (0.0, 100.0), "b": (150.0, 250.0)}, edges=[])
    gf = songs.group_features(g, {"a": fa, "b": fb})
    assert gf.chroma.shape[1] == int(250 * rate) + 1
    assert np.isnan(gf.energy_db[12000:14000]).all()
    segs = songs.find_segments("G", "group", gf)
    assert len(segs) == 2
    assert all(s.is_partial for s in segs)
    assert abs(segs[0].end - 100.0) < 1.0 and abs(segs[1].start - 150.0) < 1.0


# =========================================================================== part 2: synth material

synth = pytest.importorskip("tests.synth")
SR48 = 48000


def _rec(audio48, kind, seed, start=0.0, length=None, drift_ppm=0.0):
    """synth.record at 16 kHz (it cannot output 8 kHz: its filters need fs/2 > 5 kHz), then 8 kHz."""
    x, tr = synth.record(audio48, SR48, kind, seed, start=start, length=length, drift_ppm=drift_ppm,
                         out_sr=16000)
    return resample_poly(x, 1, 2).astype(np.float32), tr


def _to8k(a48):
    return resample_poly(a48, 1, 6).astype(np.float32)


def _feats(cid, a8):
    from showsync.dsp.features import compute_features
    return compute_features(cid, a8, ANALYSIS_SR)


def _tm_from_truth(m):
    if m["kind"] == "linear":
        return TimeMap.linear(m["ratio"], m["offset"], 0.0, 100.0)
    return TimeMap(np.asarray(m["dev"], float), np.asarray(m["src"], float))


SPECS_SEEDS = list(range(600, 606))
UNREL_SEEDS = [700, 701, 702]
TAKE_TEMPI = [0.92, 0.97, 1.03, 1.08]
TAKE_KINDS = ["pro", "hi8", "camera_mic", "phone"]


class Clip:
    """A clip with features and a truth function owner_time -> (song, canonical song time)."""
    def __init__(self, cid, feats, truth_fn, kind, role):
        self.cid, self.feats, self.truth, self.kind, self.role = cid, feats, truth_fn, kind, role


def _concert_truth(truth, dev_map):
    def fn(t):
        src = float(synth.map_eval(dev_map, t))
        for s in truth:
            if s["start"] <= src <= s["end"]:
                return s["song"], (src - s["start"]) * s["tempo_scale"]
        return None
    return fn


def _take_truth(song, ts, dev_map, dur):
    def fn(t):
        src = float(synth.map_eval(dev_map, t))
        return (song, src * ts) if 0.0 <= src <= dur else None
    return fn


@pytest.fixture(scope="module")
def mat():
    """All synthetic material for part 2 (generated once)."""
    t0 = time.perf_counter()
    specs = [synth.random_song(s) for s in SPECS_SEEDS]
    unrel = [synth.random_song(s) for s in UNREL_SEEDS]
    M = {"specs": specs, "refs": {}, "takes": [], "snips": [], "unrel": []}
    _SPEC_BY_NAME.update({sp.name: sp for sp in specs + unrel})
    rng = np.random.default_rng(42)
    # canonical reference renders (also the music-video playback track)
    for sp in specs:
        a, _ = synth.render(sp, SR48)
        M["refs"][sp.name] = (a, _feats("ref_" + sp.name, _to8k(a)))
    # live takes of the first three songs, each through another device, plus snippets
    for si, sp in enumerate(specs[:3]):
        for k, (ts, kind) in enumerate(zip(TAKE_TEMPI, TAKE_KINDS)):
            a, _ = synth.render(sp, SR48, seed=100 * si + k, tempo_scale=ts, humanize_ms=7, variation=0.3)
            dur = len(a) / SR48
            x, tr = _rec(a, kind, 10 * si + k)
            cid = f"take_{sp.name}_{k}"
            M["takes"].append(Clip(cid, _feats(cid, x), _take_truth(sp.name, ts, tr["map"], dur), kind, "take"))
            if k in (1, 2):
                ln = float(rng.uniform(15, 40))
                st = float(rng.uniform(5, dur - ln - 5))
                kind_s = "phone" if k == 1 else "camera_mic"
                x, tr = _rec(a, kind_s, 10 * si + k + 500, start=st, length=ln)
                cid = f"snip_{sp.name}_{k}"
                M["snips"].append(Clip(cid, _feats(cid, x), _take_truth(sp.name, ts, tr["map"], dur), kind_s, "snip"))
    for j, sp in enumerate(unrel):
        a, _ = synth.render(sp, SR48, seed=j, tempo_scale=1.0, humanize_ms=7, variation=0.3)
        x, tr = _rec(a, "pro", 900 + j)
        cid = f"unrel_{sp.name}"
        M["unrel"].append(Clip(cid, _feats(cid, x), _take_truth(sp.name, 1.0, tr["map"], len(a) / SR48), "pro", "take"))
    # night 1: five songs; recorded by a pro feed (anchor), a drifting phone (part) and a Hi8
    a1, t1 = synth.make_concert(specs[:5], seed=1, sr=SR48)
    M["night1_truth"] = t1
    n1 = {}
    for kind, seed, start, length, ppm in (("pro", 1, 0.0, None, 0.0), ("phone", 2, 300.0, 330.0, 40.0),
                                          ("hi8", 3, 0.0, None, 0.0)):
        x, tr = _rec(a1, kind, seed, start=start, length=length, drift_ppm=ppm)
        n1[kind] = Clip(f"n1_{kind}", _feats(f"n1_{kind}", x), _concert_truth(t1, tr["map"]), kind, "concert")
        n1[kind].map = tr["map"]
        n1[kind].dur = tr["dev_duration"]
    M["night1"] = n1
    # night 2: other order, one song night 1 did not have; one pro recording
    a2, t2 = synth.make_concert([specs[3], specs[0], specs[5], specs[1]], seed=2, sr=SR48)
    x, tr = _rec(a2, "pro", 4)
    M["night2_truth"] = t2
    M["night2"] = Clip("n2_pro", _feats("n2_pro", x), _concert_truth(t2, tr["map"]), "pro", "concert")
    M["gen_s"] = time.perf_counter() - t0
    print(f"\n[synth material generated in {M['gen_s']:.0f} s]")
    return M


def _segments_of(clips):
    segs, sf = [], {}
    for c in clips:
        for s in songs.find_segments(c.cid, "clip", c.feats):
            segs.append(s)
            sf[s.id] = songs.crop_features(c.feats, s.start, s.end)
    return segs, sf


def _truth_song_of_segment(seg, clip):
    """Majority truth song over the segment (None if it holds no song)."""
    votes = {}
    for t in np.arange(seg.start + 0.5, seg.end, 1.0):
        tr = clip.truth(t)
        if tr:
            votes[tr[0]] = votes.get(tr[0], 0) + 1
    if not votes:
        return None
    name, n = max(votes.items(), key=lambda kv: kv[1])
    return name if n >= 0.5 * (seg.end - seg.start) else None


def _night1_group(M):
    n1 = M["night1"]
    maps = {c.cid: _tm_from_truth(c.map) for c in n1.values()}
    spans = {c.cid: (float(maps[c.cid](0.0)), float(maps[c.cid](c.dur))) for c in n1.values()}
    g = EventGroup(id="grp_n1", anchor="n1_pro", maps=maps, spans=spans, edges=[])
    return g, {c.cid: c.feats for c in n1.values()}


def _boundary_table(segs, truth):
    rows = []
    for s in truth:
        best = max(segs, key=lambda g: min(g.end, s["end"]) - max(g.start, s["start"]), default=None)
        if best is None or min(best.end, s["end"]) - max(best.start, s["start"]) <= 0:
            rows.append((s["song"], s["start"], s["end"], None, None))
        else:
            rows.append((s["song"], s["start"], s["end"], best.start - s["start"], best.end - s["end"]))
    return rows


def _print_boundaries(title, rows, segs):
    print(f"\n{title}: {len(segs)} segments")
    print("  song        true start   true end   start err   end err")
    for name, a, b, ea, eb in rows:
        e = "   (missed)" if ea is None else f"   {ea:+7.1f}    {eb:+7.1f}"
        print(f"  {name:10s} {a:9.1f} {b:10.1f} {e}")


def test_concert_segmentation(mat):
    c = mat["night1"]["pro"]
    t0 = time.perf_counter()
    segs = songs.find_segments(c.cid, "clip", c.feats)
    dt = time.perf_counter() - t0
    rows = _boundary_table(segs, mat["night1_truth"])
    _print_boundaries(f"night 1 pro feed ({c.feats.duration:.0f} s, {dt:.2f} s)", rows, segs)
    assert len(segs) == len(mat["night1_truth"])          # includes the segue split
    errs = np.abs([e for r in rows for e in r[3:] if e is not None])
    assert all(r[3] is not None for r in rows)
    assert np.median(errs) < 2.0 and errs.max() < 15.0
    # the segue (4th -> 5th song, no gap) must be split within a few seconds
    seg_t = mat["night1_truth"][4]["start"]
    assert min(abs(s.start - seg_t) for s in segs) < 10.0


def test_group_features_segmentation(mat):
    g, feats = _night1_group(mat)
    gf = songs.group_features(g, feats)
    segs = songs.find_segments(g.id, "group", gf)
    rows = _boundary_table(segs, mat["night1_truth"])
    _print_boundaries("night 1 fused group (pro + drifting phone part + Hi8)", rows, segs)
    assert len(segs) == len(mat["night1_truth"])
    errs = np.abs([e for r in rows for e in r[3:] if e is not None])
    assert np.median(errs) < 2.0 and errs.max() < 15.0
    assert segs[0].id == "grp_n1:00" and segs[0].owner_kind == "group"


def _cluster_report(songs_list, segs, clip_of, title):
    """Purity / completeness of song clusters against truth."""
    by_id = {s.id: s for s in segs}
    truth_of = {s.id: _truth_song_of_segment(s, clip_of[s.owner]) for s in segs}
    print(f"\n{title}")
    for so in songs_list:
        names = [truth_of[x] for x in so.segments]
        print(f"  {so.name:9s} ref={so.reference:28s} truth={names}")
    return truth_of


def _check_clusters(songs_list, truth_of):
    """Every truth song is in exactly one cluster and every cluster holds one truth song."""
    cluster_of = {}
    for so in songs_list:
        names = {truth_of[x] for x in so.segments if truth_of[x] is not None}
        assert len(names) <= 1, f"{so.name} mixes songs {names}"
        for nm in names:
            cluster_of.setdefault(nm, set()).add(so.id)
    split = {k: v for k, v in cluster_of.items() if len(v) > 1}
    assert not split, f"songs split over clusters: {split}"


_SPEC_BY_NAME = {}


def _section_pos(song, t):
    """(section letter, bars in section, seconds into the section) at canonical song time t."""
    sp = _SPEC_BY_NAME.get(song)
    if sp is None:
        return None
    bar, secs, _ = synth._layout(sp, 1.0)
    for ch, b0, nb in secs:
        if b0 * bar <= t < (b0 + nb) * bar:
            return ch, nb, t - b0 * bar
    return None


def _repeat_err(song, t_true, t_map):
    """Error that treats harmonically identical places as equivalent: same section letter
    (chorus 1 vs chorus 2) and the same position in the section's 4-bar chord cycle (the synth
    tiles a 4-chord progression, one chord per bar, so chroma repeats every 4 bars)."""
    a, b = _section_pos(song, t_true), _section_pos(song, t_map)
    raw = abs(t_map - t_true)
    if a and b and a[0] == b[0]:
        cyc = 4 * 240.0 / _SPEC_BY_NAME[song].tempo_bpm
        d = (a[2] - b[2]) % cyc
        return min(raw, d, cyc - d)
    return raw


def _map_errors(songs_list, segs, clip_of, modulo_repeats=False):
    """|song_map error| in canonical song seconds, sampled every 0.5 s inside each segment."""
    by_id = {s.id: s for s in segs}
    errs = {}
    for so in songs_list:
        if not so.reference.startswith("segment:"):
            continue
        m = by_id[so.reference.split(":", 1)[1]]
        mc = clip_of[m.owner]
        for sid in so.segments:
            s = by_id[sid]
            c = clip_of[s.owner]
            e = []
            for t in np.arange(s.start + 0.5, s.end - 0.5, 0.5):
                tr = c.truth(t)
                if tr is None:
                    continue
                trm = mc.truth(m.start + float(s.song_map(t)))
                if trm is None or trm[0] != tr[0]:
                    continue
                e.append(_repeat_err(tr[0], tr[1], trm[1]) if modulo_repeats else abs(trm[1] - tr[1]))
            if e:
                errs[sid] = np.array(e)
    return errs


def test_live_takes_cluster_and_song_map(mat):
    clips = mat["takes"] + mat["unrel"]
    clip_of = {c.cid: c for c in clips}
    segs, sf = _segments_of(clips)
    t0 = time.perf_counter()
    sl = songs.assign_songs(segs, sf, None, similar=REC_SIMILAR)
    dt = time.perf_counter() - t0
    truth_of = _cluster_report(sl, segs, clip_of, f"live takes ({len(segs)} segments, assign {dt:.2f} s)")
    _check_clusters(sl, truth_of)
    # each multi-take song has all its takes in one cluster -> 3 + 3 unrelated songs at least
    named = {truth_of[x] for so in sl for x in so.segments if truth_of[x]}
    assert len(named) == 6
    errs = _map_errors(sl, segs, clip_of)
    allv = np.concatenate([v for k, v in errs.items()])
    by_kind = {}
    for sid, v in errs.items():
        by_kind.setdefault(clip_of[sid.split(":")[0]].kind, []).append(v)
    print("  song_map error (canonical s): median {:.2f}  p95 {:.2f}  ({} points)".format(
        np.median(allv), np.percentile(allv, 95), allv.size))
    for k, vs in by_kind.items():
        v = np.concatenate(vs)
        print(f"    {k:11s} median {np.median(v):.2f}  p95 {np.percentile(v, 95):.2f}")
    assert np.median(allv) < 0.35
    assert np.percentile(allv, 95) < 1.5


def test_snippets_join_their_songs(mat):
    clips = mat["takes"] + mat["snips"] + mat["unrel"]
    clip_of = {c.cid: c for c in clips}
    segs, sf = _segments_of(clips)
    sl = songs.assign_songs(segs, sf, None, similar=REC_SIMILAR)
    truth_of = _cluster_report(sl, segs, clip_of, "takes + snippets")
    _check_clusters(sl, truth_of)
    snip_ids = [s.id for s in segs if s.owner.startswith("snip_")]
    assert snip_ids and all(next(s for s in segs if s.id == i).is_partial for i in snip_ids)
    errs = _map_errors(sl, segs, clip_of)
    errs_r = _map_errors(sl, segs, clip_of, modulo_repeats=True)
    for i in snip_ids:
        if i in errs:
            print(f"  {i:28s} median map error {np.median(errs[i]):6.2f} s   modulo repeats {np.median(errs_r[i]):.2f} s")
    # placement inside the song may land on an identical repeat (chorus 2 for chorus 1); modulo
    # repeats every snippet must be accurate
    assert all(np.median(errs_r[i]) < 0.5 for i in snip_ids if i in errs_r)
    # every snippet sits with the full takes of its song
    for i in snip_ids:
        so = next(so for so in sl if i in so.segments)
        others = [truth_of[x] for x in so.segments if x != i]
        assert truth_of[i] in others, (i, truth_of[i], others)


def test_two_nights_setlists_match(mat):
    g, feats = _night1_group(mat)
    gf = songs.group_features(g, feats)
    s1 = songs.find_segments(g.id, "group", gf)
    n2 = mat["night2"]
    s2 = songs.find_segments(n2.cid, "clip", n2.feats)
    segs = s1 + s2
    sf = {s.id: songs.crop_features(gf, s.start, s.end) for s in s1}
    sf.update({s.id: songs.crop_features(n2.feats, s.start, s.end) for s in s2})
    # truth for the group owner: event time == night-1 source time
    gclip = Clip(g.id, gf, _concert_truth(mat["night1_truth"], {"kind": "linear", "ratio": 1.0, "offset": 0.0}), "group", "concert")
    clip_of = {g.id: gclip, n2.cid: n2}
    t0 = time.perf_counter()
    sl = songs.assign_songs(segs, sf, None, similar=REC_SIMILAR)
    dt = time.perf_counter() - t0
    truth_of = _cluster_report(sl, segs, clip_of, f"two nights ({len(segs)} segments, assign {dt:.2f} s)")
    _check_clusters(sl, truth_of)
    shared = {mat["specs"][i].name for i in (0, 1, 3)}
    for nm in shared:
        so = [so for so in sl if any(truth_of[x] == nm for x in so.segments)]
        assert len(so) == 1 and {x.split(":")[0] for x in so[0].segments} == {g.id, n2.cid}, nm
    assert [so.name for so in sl][:2] == ["Song 01", "Song 02"]
    errs = _map_errors(sl, segs, clip_of)
    allv = np.concatenate(list(errs.values()))
    print(f"  song_map error across nights: median {np.median(allv):.2f}  p95 {np.percentile(allv, 95):.2f}")
    assert np.median(allv) < 0.5

    # propagate a marker from night 1 (phone clip, exact member of the group) to night 2
    phone = mat["night1"]["phone"]
    groups = [g]
    hits = 0
    for t in np.arange(20.0, 300.0, 23.0):
        tr = phone.truth(t)
        r = songs.propagate_points([t], phone.cid, n2.cid, segs, groups)[0]
        if tr is None or tr[0] not in shared:
            continue
        assert r is not None, t
        tr2 = n2.truth(r)
        assert tr2 is not None and tr2[0] == tr[0]
        assert abs(tr2[1] - tr[1]) < 1.0, (t, tr, tr2)
        hits += 1
    assert hits >= 3
    # exact within the group: phone -> hi8
    hi8 = mat["night1"]["hi8"]
    for t in (10.0, 100.0, 250.0):
        r = songs.propagate_points([t], phone.cid, hi8.cid, segs, groups)[0]
        src = synth.map_eval(phone.map, t)
        assert abs(synth.map_eval(hi8.map, r) - src) < 0.01
    # the JSON twin agrees on the full AnalysisResult
    res = AnalysisResult(version=1, clips={}, groups=groups, loners=[n2.cid], segments=segs, songs=sl, matches=[])
    d = json.loads(res.to_json())
    ts = list(np.arange(5.0, 320.0, 17.0))
    a = songs.propagate_points(ts, phone.cid, n2.cid, segs, groups)
    b = songs.propagate_points_json(d, ts, phone.cid, n2.cid)
    assert all((x is None and y is None) or abs(x - y) < 1e-6 for x, y in zip(a, b))


def test_unrelated_songs_not_merged_config_defaults(mat):
    """Clean material: 6 reference renders + 3 unrelated live takes are 9 different songs,
    with the thresholds currently in config.py."""
    clips = [Clip("ref_" + k, f, None, "pro", "ref") for k, (a, f) in mat["refs"].items()] + mat["unrel"]
    segs, sf = _segments_of(clips)
    sl = songs.assign_songs(segs, sf, None)
    owners = [{x.split(":")[0] for x in so.segments} for so in sl]
    print(f"\nunrelated: {len(segs)} segments -> {len(sl)} songs")
    assert all(len(o) == 1 for o in owners), owners


def test_reference_matching_mv_playback_and_live(mat):
    refs = {f"refs/{k}.wav": f for k, (a, f) in mat["refs"].items()}
    sp = mat["specs"][2]
    ref_audio = mat["refs"][sp.name][0]
    # music-video set: playback of the track starting mid-song, filmed by a camera mic
    x, tr = _rec(ref_audio, "camera_mic", 77, start=47.3, length=60.0)
    mv = Clip("mv_take", _feats("mv_take", x), _take_truth(sp.name, 1.0, tr["map"], len(ref_audio) / SR48), "camera_mic", "mv")
    live = mat["takes"][5]           # song 1, live take through a Hi8, tempo 0.97
    unrel = mat["unrel"][0]
    clips = [mv, live, unrel]
    segs, sf = _segments_of(clips)
    t0 = time.perf_counter()
    sl = songs.assign_songs(segs, sf, refs, ref_accept=REC_REF_ACCEPT, similar=REC_SIMILAR)
    dt = time.perf_counter() - t0
    print(f"\nreference matching ({len(segs)} segments x {len(refs)} refs, {dt:.2f} s)")
    for so in sl:
        print(f"  {so.name:10s} {so.reference:24s} {so.segments}")
    by_owner = {}
    for s in segs:
        by_owner.setdefault(s.owner, []).append(s)
    for c, want in ((mv, sp.name), (live, mat["specs"][1].name)):
        for s in by_owner[c.cid]:
            so = next(so for so in sl if s.id in so.segments)
            assert so.name == want and so.reference == f"file:refs/{want}.wav"
            e = [abs(float(s.song_map(t)) - c.truth(t)[1]) for t in np.arange(s.start + 0.5, s.end - 0.5, 0.5)
                 if c.truth(t)]
            print(f"  {c.cid:24s} score {s.match_score:.2f}  map error median {np.median(e):.3f} p95 {np.percentile(e, 95):.3f} s")
            assert np.median(e) < (0.15 if c is mv else 0.35)
    for s in by_owner.get(unrel.cid, []):
        so = next(so for so in sl if s.id in so.segments)
        assert so.reference.startswith("segment:") and so.name.startswith("Song ")


def test_calibration_table(mat):
    """True vs false pair similarity by device class; prints the table the thresholds come from."""
    clips = mat["takes"] + mat["snips"] + mat["unrel"] + \
        [Clip("ref_" + k, f, None, "ref", "ref") for k, (a, f) in mat["refs"].items()]
    rows = []
    for c in clips:
        ss = songs.find_segments(c.cid, "clip", c.feats)
        if not ss:
            continue
        s = max(ss, key=lambda s: s.end - s.start)
        song = c.cid[4:] if c.role == "ref" else _truth_song_of_segment(s, c)
        rows.append((c, songs.crop_features(c.feats, s.start, s.end), song))
    S = songs._similarity_matrix([r[1] for r in rows])
    cls = lambda c: "ref" if c.role == "ref" else ("snip" if c.role == "snip" else
                                                   ("clean" if c.kind == "pro" else "degraded"))
    cat = {}
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][2] is None or rows[j][2] is None:
                continue
            key = "+".join(sorted([cls(rows[i][0]), cls(rows[j][0])]))
            cat.setdefault((key, rows[i][2] == rows[j][2]), []).append(S[i, j])
    print("\ncalibration: pair similarity (reversed-reference normalised subsequence DTW)")
    print("  pair class          same   n     min    p5   median   p95    max")
    for (key, same), v in sorted(cat.items()):
        v = np.array(v)
        print(f"  {key:18s} {'yes' if same else 'no ':4s} {v.size:4d}  {v.min():.2f}  {np.percentile(v, 5):.2f}  "
              f"{np.median(v):.2f}    {np.percentile(v, 95):.2f}  {v.max():.2f}")
    false = np.concatenate([np.array(v) for (k, same), v in cat.items() if not same])
    true_clean = np.concatenate([np.array(v) for (k, same), v in cat.items()
                                 if same and "degraded" not in k and "snip" not in k])
    print(f"  false pairs: p99 {np.percentile(false, 99):.2f}, max {false.max():.2f};"
          f"  clean true pairs: min {true_clean.min():.2f}")
    print(f"  config SONG_SIMILAR={SONG_SIMILAR} SONG_REF_ACCEPT={SONG_REF_ACCEPT};"
          f" recommended {REC_SIMILAR} / {REC_REF_ACCEPT}")
    assert np.percentile(false, 99) < REC_SIMILAR
    assert true_clean.min() > SONG_SIMILAR


# =========================================================================== user hints

def _song_of(sl, sid):
    return next((so for so in sl if sid in so.segments), None)


@pytest.fixture(scope="module")
def hint_mat(mat):
    """Takes of song_600 / song_601 plus a crowd-dominated phone take of song_600."""
    sp = mat["specs"][0]
    a, _ = synth.render(sp, SR48, seed=4242, tempo_scale=1.02, humanize_ms=7, variation=0.3)
    x, tr = _rec(a, "crowd_phone", 4243)
    crowd = Clip("crowd_600", _feats("crowd_600", x), _take_truth(sp.name, 1.02, tr["map"], len(a) / SR48),
                 "crowd_phone", "take")
    clips = [c for c in mat["takes"] if "600" in c.cid or "601" in c.cid] + [crowd]
    segs, sf = _segments_of(clips)
    first = {}
    for s in segs:                       # longest segment per clip
        if s.owner not in first or s.end - s.start > first[s.owner].end - first[s.owner].start:
            first[s.owner] = s
    return clips, segs, sf, first


def _fresh(segs):
    return [SongSegment(id=s.id, owner=s.owner, owner_kind=s.owner_kind, start=s.start, end=s.end,
                        is_partial=s.is_partial) for s in segs]


def test_hint_tags_force_weak_clip_and_attract_untagged(hint_mat):
    clips, segs, sf, first = hint_mat
    crowd, pro600 = first["crowd_600"].id, first["take_song_600_0"].id
    segs = _fresh(segs)
    conf = []
    sl = songs.assign_songs(segs, sf, None, tags={crowd: "  My  Song ", pro600: "my song"}, conflicts=conf)
    so = _song_of(sl, crowd)
    print(f"\ntag test: {[(x.name, x.segments) for x in sl]}  conflicts={conf}")
    assert so.name == "My Song" and pro600 in so.segments
    # untagged takes of the same song join the tagged song; song_601 stays apart
    for c in ("take_song_600_1", "take_song_600_2", "take_song_600_3"):
        assert _song_of(sl, first[c].id) is so, c
    assert _song_of(sl, first["take_song_601_0"].id) is not so
    assert conf == []
    assert all(s.song_map is not None for s in segs if s.song_id == so.id)


def test_hint_title_selects_reference_and_no_feature_segment(hint_mat, mat):
    clips, segs, sf, first = hint_mat
    refs = {f"refs/{k}.wav": f for k, (a, f) in mat["refs"].items()}
    crowd = first["crowd_600"].id
    segs = _fresh(segs) + [SongSegment(id="silent:00", owner="silent", owner_kind="clip", start=3.0, end=40.0)]
    sl = songs.assign_songs(segs, sf, refs, tags={crowd: "SONG_600", "silent:00": "Unheard"})
    so = _song_of(sl, crowd)
    assert so.reference == "file:refs/song_600.wav" and so.name == "SONG_600"
    assert _song_of(sl, first["take_song_600_0"].id) is so          # untagged, same reference
    sil = next(s for s in segs if s.id == "silent:00")
    so2 = _song_of(sl, "silent:00")
    assert so2.name == "Unheard" and abs(float(sil.song_map(13.0)) - 10.0) < 1e-9


def test_hint_must_and_cannot_link(hint_mat):
    clips, segs, sf, first = hint_mat
    a0, a1 = first["take_song_600_0"].id, first["take_song_600_1"].id
    b0 = first["take_song_601_0"].id
    segs = _fresh(segs)
    sl = songs.assign_songs(segs, sf, None, must_link=[(a0, b0)], cannot_link=[(a0, a1)])
    assert _song_of(sl, a0) is _song_of(sl, b0)
    assert _song_of(sl, a0) is not _song_of(sl, a1)
    # takes of song_601 other than b0 still find each other
    s601 = {_song_of(sl, first[f"take_song_601_{k}"].id).id for k in (1, 2, 3)}
    assert len(s601) == 1


def test_hint_conflicts_reported_tags_win(hint_mat):
    clips, segs, sf, first = hint_mat
    a0, a1 = first["take_song_600_0"].id, first["take_song_600_1"].id
    b0, b1 = first["take_song_601_0"].id, first["take_song_601_1"].id
    segs = _fresh(segs)
    conf = []
    sl = songs.assign_songs(segs, sf, None, tags={a0: "A", b0: "B", a1: "a"},
                            must_link=[(a0, b0), (b1, "nope:00"), (b1, first["take_song_601_2"].id)],
                            cannot_link=[(a0, a1), (b1, first["take_song_601_2"].id)], conflicts=conf)
    print("\nconflicts:", *conf, sep="\n  ")
    assert _song_of(sl, a0).name == "A" and _song_of(sl, b0).name == "B"
    assert _song_of(sl, a0) is _song_of(sl, a1)                      # tag beats cannot_link
    assert _song_of(sl, a0) is not _song_of(sl, b0)                  # tags beat must_link
    assert _song_of(sl, b1) is not _song_of(sl, first["take_song_601_2"].id)   # cannot beats must
    assert len(conf) == 4
    assert any("tag wins" in c and "cannot_link" in c for c in conf)
    assert any("different titles" in c for c in conf)
    assert any("unknown" in c for c in conf)
    assert any("cannot_link wins" in c for c in conf)


def test_reference_guided_segmentation_splits_same_key_segue(mat):
    """Night 2 ends with a same-key segue (song_605 -> song_601) that audio-only segmentation
    leaves joined; with the reference songs given, find_segments cuts it at the located edge."""
    n2 = mat["night2"]
    refs = {f"refs/{k}.wav": f for k, (a, f) in mat["refs"].items()}
    plain = songs.find_segments(n2.cid, "clip", n2.feats)
    t0 = time.perf_counter()
    guided = songs.find_segments(n2.cid, "clip", n2.feats, references=refs)
    dt = time.perf_counter() - t0
    regions = songs.locate_references(n2.feats, refs)
    rows = _boundary_table(guided, mat["night2_truth"])
    _print_boundaries(f"night 2 reference-guided ({dt:.1f} s; audio-only gave {len(plain)})", rows, guided)
    for r in regions:
        print(f"  located {r[0]:22s} {r[1]:7.1f}-{r[2]:7.1f}  chain {r[3]:.2f} ({r[4]} hits)")
    assert len(guided) == len(mat["night2_truth"]) > len(plain)
    errs = np.abs([e for r in rows for e in r[3:] if e is not None])
    assert np.median(errs) < 3.0 and errs.max() < 20.0
    located = {r[0] for r in regions}
    assert {f"refs/{t['song']}.wav" for t in mat["night2_truth"]} <= located


def test_hint_tag_spreads_to_other_nights_despite_reference(hint_mat, mat):
    """A tag on ONE live take seeds its song: the other takes of that song join it, also when
    the title names a reference whose (studio) audio does not match the live arrangement."""
    clips, segs0, sf, first = hint_mat
    clip_of = {c.cid: c for c in clips}
    tagged = first["take_song_600_0"].id
    others = [first[f"take_song_600_{k}"].id for k in (1, 2, 3)]
    # 1) reference named like the tag but with unrelated audio (studio version "doesn't match")
    sf2 = dict(sf)
    refs = {"refs/song_b.wav": mat["unrel"][0].feats}
    segs = [s for s in _fresh(segs0) if not s.owner.startswith("crowd")]
    sl = songs.assign_songs(segs, sf2, refs, tags={tagged: "song_b"})
    so = _song_of(sl, tagged)
    print(f"\nnon-matching reference: {so.name} {so.reference} {so.segments}")
    assert so.name == "song_b" and so.reference == f"segment:{tagged}"
    assert all(_song_of(sl, x) is so for x in others)
    assert _song_of(sl, first["take_song_601_0"].id) is not so
    # 2) matching reference (the canonical render): time base is the reference file, and the
    #    untagged takes are mapped onto it through the tagged take
    refs = {f"refs/{k}.wav": f for k, (a, f) in mat["refs"].items()}
    segs = [s for s in _fresh(segs0) if not s.owner.startswith("crowd")]
    sl = songs.assign_songs(segs, sf2, refs, tags={tagged: "SONG_600"})
    so = _song_of(sl, tagged)
    assert so.reference == "file:refs/song_600.wav" and so.name == "SONG_600"
    by_id = {s.id: s for s in segs}
    for x in [tagged] + others:
        s = by_id[x]
        assert _song_of(sl, x) is so
        c = clip_of[s.owner]
        e = [abs(float(s.song_map(t)) - c.truth(t)[1]) for t in np.arange(s.start + 1, s.end - 1, 1.0) if c.truth(t)]
        print(f"  {x:24s} map error vs reference time: median {np.median(e):.2f} s")
        assert np.median(e) < 0.5


# =========================================================================== beat-accurate maps

@pytest.fixture(scope="module")
def beat_mat():
    """One song: canonical render + 3 live takes (tempo 0.93 / 1.0 / 1.07, 5 ms humanize, no
    tempo wobble so that beat truth is exact) through pro / Hi8 / phone, plus a take of another song."""
    sp = synth.random_song(612)
    ref, _ = synth.render(sp, SR48)
    takes = []
    for k, (ts, kind) in enumerate(((0.93, "pro"), (1.0, "hi8"), (1.07, "phone"))):
        a, _ = synth.render(sp, SR48, seed=70 + k, tempo_scale=ts, humanize_ms=5, variation=0.0)
        x, tr = _rec(a, kind, 170 + k)
        cid = f"beat_take_{k}"
        c = Clip(cid, _feats(cid, x), _take_truth(sp.name, ts, tr["map"], len(a) / SR48), kind, "take")
        c.ts, c.map = ts, tr["map"]
        takes.append(c)
    other = synth.random_song(613)
    a, _ = synth.render(other, SR48, seed=5, humanize_ms=5)
    x, tr = _rec(a, "pro", 180)
    oc = Clip("beat_other", _feats("beat_other", x), _take_truth(other.name, 1.0, tr["map"], len(a) / SR48), "pro", "take")
    return sp, _feats("beat_ref", _to8k(ref)), takes, oc


def _beat_errors(sp, s, c, base_t0=0.0):
    """|song_map - truth| (ms) at the take's true beat times (canonical seconds)."""
    beat = 60.0 / sp.tempo_bpm
    out = []
    for k in range(int(synth.song_duration(sp) / beat)):
        canon = k * beat
        src = canon / c.ts                                  # render time of beat k
        dev = float(synth.map_inverse_eval(c.map, src))     # device (owner) time
        if s.start + 1 <= dev <= s.end - 1:
            out.append(abs(float(s.song_map(dev)) - base_t0 - canon) * 1000.0)
    return np.asarray(out)


def test_beat_sync_maps_land_on_beats(beat_mat):
    sp, ref_f, takes, oc = beat_mat
    segs, sf = _segments_of(takes + [oc])
    refs = {f"refs/{sp.name}.wav": ref_f}
    res = {}
    for mode in (False, True):
        ss = _fresh(segs)
        rep = {}
        sl = songs.assign_songs(ss, sf, refs, beat_sync=mode, beat_report=rep)
        so = next(so for so in sl if so.reference == f"file:refs/{sp.name}.wav")
        errs = []
        for c in takes:
            s = next(x for x in ss if x.owner == c.cid and x.song_id == so.id)
            e = _beat_errors(sp, s, c)
            errs.append(e)
            print(f"  beat_sync={mode!s:5s} {c.cid} ({c.kind:5s}, tempo {c.ts}): beat error median "
                  f"{np.median(e):5.1f} ms  p90 {np.percentile(e, 90):5.1f} ms  synced={rep.get(s.id, {}).get('beat_synced')}")
        res[mode] = np.concatenate(errs)
        assert next(x for x in ss if x.owner == "beat_other").song_id != so.id
    print(f"  all takes: DTW median {np.median(res[False]):.1f} / p90 {np.percentile(res[False], 90):.1f} ms;"
          f"  beat-synced median {np.median(res[True]):.1f} / p90 {np.percentile(res[True], 90):.1f} ms")
    assert np.median(res[True]) < 20.0 and np.percentile(res[True], 90) < 40.0
    assert np.median(res[True]) <= np.median(res[False])


def test_anchors_force_maps_and_link(beat_mat):
    sp, ref_f, takes, oc = beat_mat
    segs, sf = _segments_of(takes + [oc])
    a = next(s for s in segs if s.owner == takes[0].cid)
    b = next(s for s in segs if s.owner == takes[2].cid)
    o = next(s for s in segs if s.owner == "beat_other")
    # deliberately one beat off the truth on b: the anchor must win
    beat = 60.0 / sp.tempo_bpm
    ta = a.start + 40.0
    canon = float(takes[0].truth(ta)[1]) + beat
    tb = float(synth.map_inverse_eval(takes[2].map, canon / takes[2].ts))
    conf = []
    ss = _fresh(segs)
    sl = songs.assign_songs(ss, sf, None, anchors=[(a.id, ta, b.id, tb), (a.id, a.start + 20.0, o.id, o.start + 20.0)],
                            conflicts=conf)
    sa, sb, so_ = (next(x for x in ss if x.id == i) for i in (a.id, b.id, o.id))
    assert sa.song_id == sb.song_id
    assert abs(float(sa.song_map(ta)) - float(sb.song_map(tb))) < 1e-6
    # the anchor to an unrelated song acts as must_link: same song, and the maps meet there too
    assert so_.song_id == sa.song_id
    assert abs(float(sa.song_map(a.start + 20.0)) - float(so_.song_map(o.start + 20.0))) < 1e-6
    assert conf == []
    assert all(np.all(np.diff(x.song_map.knots_dst) > 0) for x in ss if x.song_map is not None)


def test_beat_base_take_aligns_take_to_take(beat_mat):
    """beat_base='take': song time stays the reference, but take->take alignment goes through
    the chosen base take's beats; base_owner picks it. Take->take beat error vs truth."""
    sp, ref_f, takes, oc = beat_mat
    segs, sf = _segments_of(takes)
    refs = {f"refs/{sp.name}.wav": ref_f}
    ss = _fresh(segs)
    rep = {}
    sl = songs.assign_songs(ss, sf, refs, beat_base="take", base_owner=takes[1].cid, beat_report=rep)
    so = _song_of(sl, ss[0].id)
    assert so.reference == f"file:refs/{sp.name}.wav"
    base = next(s for s in ss if s.owner == takes[1].cid)
    assert rep[base.id]["beat_base"] == "take" and rep[base.id]["snapped_frac"] == 1.0
    beat = 60.0 / sp.tempo_bpm
    for c in (takes[0], takes[2]):
        s = next(x for x in ss if x.owner == c.cid)
        e = []
        for k in range(int(synth.song_duration(sp) / beat)):
            canon = k * beat
            t = float(synth.map_inverse_eval(c.map, canon / c.ts))
            if not (s.start + 1 <= t <= s.end - 1):
                continue
            tb_pred = float(base.song_map.inverse()(s.song_map(t)))           # take -> base take
            tb_true = float(synth.map_inverse_eval(takes[1].map, canon / takes[1].ts))
            e.append(abs(tb_pred - tb_true) * 1000.0)
        print(f"  {c.cid} -> {takes[1].cid}: take-to-take beat error median {np.median(e):.1f} ms p90 {np.percentile(e, 90):.1f} ms"
              f"  synced={s.beat_synced} breaks={s.breaks}")
        assert s.beat_synced and np.median(e) < 20.0 and np.percentile(e, 90) < 40.0
        assert s.breaks == []                    # same arrangement: no structure breaks
