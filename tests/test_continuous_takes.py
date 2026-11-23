"""One recording plays a song front to back: its pieces must continue each other in song time."""
from showsync.models import Song, SongSegment, TimeMap
from showsync.pipeline import _continuous_takes


def seg(sid, owner, start, end, song_start):
    return SongSegment(id=sid, owner=owner, owner_kind="group", start=start, end=end, song_id="s",
                       song_map=TimeMap.linear(1.0, song_start - start, start, end))


def test_later_piece_mapped_onto_an_earlier_repeat_is_put_in_sequence():
    """Real case (2026-11-23): a tape's second piece of a song, 32 s after the first, was mapped back
    onto an earlier chorus, i.e. the same camera at two places of the song at once."""
    a = seg("G:1", "G", 1724.5, 1853.1, 245.6)
    b = seg("G:2", "G", 1885.0, 1942.0, 325.1)       # wrongly inside a's song time
    other = seg("H:1", "H", 0.0, 200.0, 250.0)
    notes = _continuous_takes([a, b, other], [Song(id="s", name="Auto", reference="", duration=500,
                                                   segments=["G:1", "G:2", "H:1"])])
    end_a = 245.6 + (1853.1 - 1724.5)
    assert len(notes) == 1 and "in sequence" in notes[0]
    assert abs(float(b.song_map(1885.0)) - (end_a + 31.9)) < 1e-6     # continues at real-time pace
    assert a.end == 1885.0 and abs(float(a.song_map(1885.0)) - (end_a + 31.9)) < 1e-6   # gap filled
    assert abs(float(other.song_map(0.0)) - 250.0) < 1e-9            # other owners untouched


def test_consistent_pieces_and_far_apart_pieces_are_left_alone():
    a = seg("G:1", "G", 100.0, 200.0, 0.0)
    b = seg("G:2", "G", 210.0, 300.0, 110.5)          # continues within tolerance
    far = seg("G:3", "G", 900.0, 1000.0, 10.0)        # 10 min later: a different performance
    notes = _continuous_takes([a, b, far], [Song(id="s", name="X", reference="", duration=300,
                                                 segments=["G:1", "G:2", "G:3"])])
    assert notes == []
    assert abs(float(b.song_map(210.0)) - 110.5) < 1e-9 and abs(float(far.song_map(900.0)) - 10.0) < 1e-9
