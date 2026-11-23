"""Synthetic ground truth for ShowSync: songs, concerts, device simulations, fixture folders.

Everything is deterministic by seed. numpy + scipy only (ffmpeg is used by write_video).

CLI:  python -m tests.synth <out_dir> [--quick]

TRUTH SCHEMA
============
Time-map dict ("map"), always DEVICE time -> SOURCE (event) time, seconds:
    {"kind": "linear", "ratio": r, "offset": o}        src = r*dev + o
    {"kind": "pwl", "dev": [...], "src": [...]}         sampled every 20 ms (nonlinear: Hi8 wow/flutter),
                                                        linear interpolation error < 0.1 ms
  ratio = 1/(1+drift_ppm*1e-6): a device whose clock runs fast (+ppm) stores more samples per
  source second, so it advances less than 1 source second per device second.
  Helpers: map_eval(m, dev_t), map_inverse_eval(m, src_t), shift_map(m, delta).

record() truth:  {kind, seed, src_sr, out_sr, drift_ppm, map, src_start, src_end, dev_duration,
                  params{...}, dropouts[[dev_start, dur], ...]}
  Its map is relative to the source array passed in (source time 0 = first sample).

make_concert() truth: list of {song, start, end, tempo_scale, sections:[[name,start,end],...]}
  in concert time. start/end are song start and end (excluding the reverb tail).

Fixture scenario folder <out>/<scenario>/ holds media files, optional refs/*.wav and truth.json:
  {
   "scenario", "synth_version", "description",
   "events": {event_id: {"type": "concert"|"performance"|"playback"|"unrelated",
                          "duration": s,
                          "songs": [{"song", "start", "end", "song_t0", "rate"}]}},
        song time (canonical = the reference render at tempo_scale 1) = song_t0 + (event_t - start)*rate
   "songs": {song_name: {"duration": canonical seconds, "ref": "refs/x.wav" | null}},
   "files": {relpath: {
        "kind": device kind, "event": event_id, "role": str,
        "map": device->event time map (see above), "dev_duration": s,
        "drift_ppm": float, "start_tc": "HH:MM:SS:FF" | null, "vfr": bool,
        "exact_span": [ev_lo, ev_hi]  event-time range of this file where the shared sound is audible,
        "songs": [{"song", "dev_start", "dev_end", "ev_start", "ev_end",
                   "song_start", "song_end", "partial"}]   song_* in canonical song time,
        "params": {...device parameters}, "dropouts": [...]}},
   "pairs": [{"a","b","event","overlap_s"}]       files that heard the same sound >= 5 s,
   "expected_exact_groups": [[files...]]          connected components of pairs (size >= 2),
   "expected_songs": {song_name: [relpath,...]},  files containing that song (any part),
   "notes": {...}
  }
Top-level manifest.json: {"synth_version", "quick", "scenarios": {name: <that truth>}}.

Known realism limits are listed in docs of the final report; the main ones: instruments are
synthetic (no real vocals), crowd/applause are procedural, mono only, linear-interpolation
resampling, no AAC pre-roll modelling beyond what ffmpeg itself does.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

SYNTH_VERSION = 2
SR = 48000
TAIL_S = 1.2

# ============================================================================ song model

_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]
_PROGS = [[0, 4, 5, 3], [0, 5, 3, 4], [5, 3, 0, 4], [0, 3, 4, 4], [0, 2, 5, 4],
          [0, 4, 3, 4], [5, 4, 3, 4], [0, 3, 0, 4], [0, 5, 1, 4], [0, 2, 3, 4], [3, 4, 0, 5]]
_STRUCTS = ["IVCVCBCO", "IVCVCO", "IVCVCBCCO", "VCVCBCO", "IVCCVCBCO"]
_NAMES = {"I": "intro", "V": "verse", "C": "chorus", "B": "bridge", "O": "outro"}
_DRUMS = {
    "rock":  {"kick": [0, 8, 10], "snare": [4, 12], "hat": list(range(0, 16, 2)), "open": []},
    "disco": {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [0, 4, 8, 12], "open": [2, 6, 10, 14]},
    "sync":  {"kick": [0, 3, 8, 11], "snare": [4, 12], "hat": list(range(16)), "open": []},
    "half":  {"kick": [0, 10], "snare": [8], "hat": [0, 4, 8, 12], "open": [14]},
}
_BASS_PATTERNS = [[0, 8], [0, 6, 8, 14], [0, 4, 8, 12], [0, 3, 6, 8, 11, 14], list(range(0, 16, 2))]
_RHYTHMS = [
    [(0, 4), (4, 2), (6, 2), (8, 4), (12, 4)],
    [(0, 2), (2, 2), (4, 4), (8, 2), (10, 2), (12, 4)],
    [(0, 6), (6, 2), (8, 4), (12, 2), (14, 2)],
    [(2, 2), (4, 2), (6, 2), (8, 8)],
    [(0, 3), (3, 3), (6, 2), (8, 3), (11, 3), (14, 2)],
    [(0, 8), (8, 8)],
    [(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)],
]
_POOLS = {"V": [3, 1, 5, 0], "C": [0, 1, 2, 4], "B": [5, 2, 3]}
_RANGES = {"V": (2, 9), "C": (5, 12), "B": (3, 10)}


@dataclass
class SongSpec:
    name: str
    tempo_bpm: float
    key: int                 # 0-11 (tonic pitch class offset)
    structure: str           # e.g. "IVCVCBCO"
    bars: dict               # section char -> bars
    progs: dict              # section char -> list of scale degrees (one chord per bar, tiled)
    melody_seed: int
    drum_pattern: str
    mode: str = "major"
    bass_pattern: int = 1
    timbre: int = 0


def random_song(seed: int, target_s: float | None = None, name: str | None = None) -> SongSpec:
    """Varied but reproducible song. Verses/choruses repeat within the song."""
    rng = np.random.default_rng([seed, 31337])
    tempo = int(rng.integers(70, 171))
    key = int(rng.integers(12))
    mode = "major" if rng.random() < 0.55 else "minor"
    struct = _STRUCTS[int(rng.integers(len(_STRUCTS)))]
    if target_s is None:
        target_s = float(rng.uniform(150, 210))
    base = {"I": 4, "V": 8, "C": 8, "B": 8, "O": 4}
    bar_s = 240.0 / tempo
    best = None
    for scale in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        bars = {k: max(2, 2 * int(round(v * scale / 2))) for k, v in base.items()}
        err = abs(sum(bars[c] for c in struct) * bar_s - target_s)
        if best is None or err < best[0]:
            best = (err, bars)
    idx = rng.permutation(len(_PROGS))[:3]
    pv, pc, pb = [list(_PROGS[i]) for i in idx]
    progs = {"I": pv, "V": pv, "C": pc, "B": pb, "O": pc}
    return SongSpec(
        name=name or f"song_{seed:03d}", tempo_bpm=float(tempo), key=key, structure=struct,
        bars=best[1], progs=progs, melody_seed=int(rng.integers(1 << 30)),
        drum_pattern=list(_DRUMS)[int(rng.integers(len(_DRUMS)))], mode=mode,
        bass_pattern=int(rng.integers(len(_BASS_PATTERNS))), timbre=int(rng.integers(1 << 30)))


def _layout(spec: SongSpec, tempo_scale: float = 1.0):
    bar = 240.0 / spec.tempo_bpm / tempo_scale
    secs, b = [], 0
    for ch in spec.structure:
        nb = spec.bars[ch]
        secs.append((ch, b, nb))
        b += nb
    return bar, secs, b


def song_duration(spec: SongSpec, tempo_scale: float = 1.0) -> float:
    bar, _, tb = _layout(spec, tempo_scale)
    return bar * tb


def _deg_semi(S, d):
    return S[d % 7] + 12 * (d // 7)


def _melody_phrase(spec: SongSpec, ch: str):
    """4-bar phrase (list of bars, each a list of (step, dur_steps, scale_idx)); same every time."""
    rng = np.random.default_rng([spec.melody_seed, ord(ch)])
    lo, hi = _RANGES[ch]
    pool = _POOLS[ch]
    prog = spec.progs[ch]
    prev = (lo + hi) // 2
    pats, bars = [], []
    for bi in range(4):
        pat = _RHYTHMS[pool[int(rng.integers(len(pool)))]] if bi < 2 else pats[bi - 2]
        pats.append(pat)
        d = prog[bi % len(prog)]
        tones = [d, d + 2, d + 4]
        notes = []
        for k, (st, du) in enumerate(pat):
            if k == 0:
                cands = sorted([t + 7 * o for t in tones for o in range(-3, 4) if lo <= t + 7 * o <= hi],
                               key=lambda c: abs(c - prev))
                idx = cands[int(rng.integers(min(3, len(cands))))]
            else:
                idx = int(np.clip(prev + rng.choice([-2, -1, -1, 0, 1, 1, 2]), lo, hi))
            prev = idx
            notes.append((st, du, idx))
        bars.append(notes)
    return bars


# ============================================================================ instrument bank

_BANKS: dict = {}


class _Bank:
    def __init__(self, spec: SongSpec, sr: int):
        self.sr, self.tb, self.c = sr, spec.timbre, {}
        r = np.random.default_rng([spec.timbre, 5])
        self.kick_f = r.uniform(42, 58)
        self.snare_f = r.uniform(170, 220)
        self.hat_hp = r.uniform(6000, 9000)
        self.lw = (r.uniform(0.15, 0.6), r.uniform(0.05, 0.3))
        self.pad_detune = r.uniform(4, 12) / 1200.0
        self.pad_k = int(r.integers(7, 12))

    def rng(self, *key):
        return np.random.default_rng([self.tb, zlib.crc32(repr(key).encode())])

    def get(self, key, fn):
        v = self.c.get(key)
        if v is None:
            v = self.c[key] = fn().astype(np.float32)
        return v

    def _t(self, n):
        return np.arange(n) / self.sr

    def kick(self, v):
        def mk():
            t = self._t(int(0.3 * self.sr))
            f = self.kick_f + 95 * np.exp(-t / 0.03)
            s = np.sin(2 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t / 0.09)
            s += 0.4 * self.rng("k", v).standard_normal(len(t)) * np.exp(-t / 0.004)
            return s
        return self.get(("kick", v), mk)

    def tom(self, f0):
        def mk():
            t = self._t(int(0.3 * self.sr))
            f = f0 * (1 + 0.6 * np.exp(-t / 0.04))
            return np.sin(2 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t / 0.11)
        return self.get(("tom", f0), mk)

    def snare(self, v):
        def mk():
            t = self._t(int(0.25 * self.sr))
            sos = signal.butter(2, [1500, 9000], "band", fs=self.sr, output="sos")
            n = signal.sosfilt(sos, self.rng("s", v).standard_normal(len(t)))
            n = n / np.max(np.abs(n))
            return 0.8 * n * np.exp(-t / 0.07) + 0.6 * np.sin(2 * np.pi * self.snare_f * t) * np.exp(-t / 0.05)
        return self.get(("snare", v), mk)

    def hat(self, v, open_=False):
        def mk():
            ln = 0.35 if open_ else 0.07
            t = self._t(int(ln * self.sr))
            sos = signal.butter(2, self.hat_hp, "high", fs=self.sr, output="sos")
            n = signal.sosfilt(sos, self.rng("h", v, open_).standard_normal(len(t)))
            return n / np.max(np.abs(n)) * np.exp(-t / (0.12 if open_ else 0.02))
        return self.get(("hat", v, open_), mk)

    def crash(self):
        def mk():
            t = self._t(int(1.6 * self.sr))
            sos = signal.butter(2, 4500, "high", fs=self.sr, output="sos")
            n = signal.sosfilt(sos, self.rng("c").standard_normal(len(t)))
            return n / np.max(np.abs(n)) * np.exp(-t / 0.45)
        return self.get(("crash",), mk)

    def bass(self, midi, ns):
        def mk():
            t = self._t(ns)
            f = 440 * 2 ** ((midi - 69) / 12)
            ph = 2 * np.pi * f * t
            s = np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
            env = np.exp(-t / 0.6)
            a, r = min(int(0.004 * self.sr), ns // 2), min(int(0.015 * self.sr), ns // 2)
            env[:a] *= np.linspace(0, 1, a)
            env[-r:] *= np.linspace(1, 0, r)
            return s * env
        return self.get(("bass", midi, ns), mk)

    def pad(self, midis, ns):
        def mk():
            t = self._t(ns)
            s = np.zeros(ns)
            for m in midis:
                f = 440 * 2 ** ((m - 69) / 12)
                for dt in (-1, 1):
                    th = 2 * np.pi * f * (1 + dt * self.pad_detune) * t
                    for k in range(1, min(self.pad_k, int(4000 / f)) + 1):
                        s += np.sin(k * th) / k ** 1.2
            env = np.ones(ns)
            a, r = min(int(0.08 * self.sr), ns // 2), min(int(0.15 * self.sr), ns // 2)
            env[:a] = np.linspace(0, 1, a)
            env[-r:] = np.linspace(1, 0, r)
            return s * env * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t)) / len(midis)
        return self.get(("pad", tuple(midis), ns), mk)

    def lead(self, midi, ns):
        def mk():
            t = self._t(ns)
            f0 = 440 * 2 ** ((midi - 69) / 12)
            vd = 0.012 * np.clip((t - 0.12) / 0.25, 0, 1)
            f = f0 * (1 + vd * np.sin(2 * np.pi * 5.6 * t))
            ph = 2 * np.pi * np.cumsum(f) / self.sr
            s = np.sin(ph) + self.lw[0] * np.sin(2 * ph) + self.lw[1] * np.sin(3 * ph)
            s += 0.03 * self.rng("l", midi, ns).standard_normal(ns)
            env = (0.85 + 0.15 * np.exp(-t / 0.4))
            a, r = min(int(0.02 * self.sr), ns // 2), min(int(0.06 * self.sr), ns // 2)
            env[:a] *= np.linspace(0, 1, a)
            env[-r:] *= np.linspace(1, 0, r)
            return 0.5 * s * env
        return self.get(("lead", midi, ns), mk)


def _bank(spec, sr):
    k = (spec.name, spec.timbre, sr)
    if k not in _BANKS:
        _BANKS[k] = _Bank(spec, sr)
    return _BANKS[k]


def _add(out, buf, start, g):
    if start < 0:
        buf = buf[-start:]
        start = 0
    e = min(len(out), start + len(buf))
    if e > start:
        out[start:e] += g * buf[:e - start]


# ============================================================================ render

_SEC_GAIN = {"I": 0.6, "V": 0.8, "C": 1.0, "B": 0.85, "O": 0.7}
_FILLS = [
    [(12, "snare", .7), (13, "snare", .6), (14, "snare", .7), (15, "snare", .8)],
    [(12, "tom1", .8), (13, "tom1", .7), (14, "tom2", .8), (15, "tom2", .9)],
    [(8, "snare", .5), (10, "snare", .6), (12, "snare", .7), (14, "snare", .8)],
]


def render(spec: SongSpec, sr: int = SR, seed: int = 0, tempo_scale: float = 1.0,
           humanize_ms: float = 0.0, variation: float = 0.0):
    """Render a song. Returns (float32 audio, [(section_name, start_s, end_s), ...]).

    humanize_ms jitters note timing; variation changes fills, ornaments, dynamics and adds a slow
    tempo wobble (<= ~80 ms) so a re-render with another seed is a different live performance.
    With humanize_ms == 0 and variation == 0 the result does not depend on seed.
    """
    bank = _bank(spec, sr)
    bar, lay, total_bars = _layout(spec, tempo_scale)
    step = bar / 16.0
    n = int((total_bars * bar + TAIL_S) * sr)
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng([seed, 99])
    live = humanize_ms > 0 or variation > 0
    hum = humanize_ms * 1e-3 * sr
    S = _MAJOR if spec.mode == "major" else _MINOR
    drums = _DRUMS[spec.drum_pattern]
    ph1, ph2 = rng.uniform(0, 6.28, 2)
    phrases = {ch: _melody_phrase(spec, ch) for ch in "VCB"}
    phrases["O"] = phrases["C"]
    bpat = _BASS_PATTERNS[spec.bass_pattern]
    fill_base = spec.melody_seed % 3

    def jit():
        return int(np.clip(rng.normal(0, hum), -3 * hum, 3 * hum)) if hum > 0 else 0

    def vel():
        return float(np.clip(1 + (0.25 * variation + (0.06 if hum > 0 else 0)) * rng.normal(), 0.5, 1.4)) if live else 1.0

    def pos(b, k, drift):
        return int(round(((b * 16 + k) * step + drift) * sr)) + jit()

    for si, (ch, b0, nb) in enumerate(lay):
        sg = _SEC_GAIN[ch] * ((1 + variation * rng.normal(0, 0.15)) if variation > 0 else 1.0)
        prog = spec.progs[ch]
        nxt = lay[si + 1][0] if si + 1 < len(lay) else None
        for bi in range(nb):
            b = b0 + bi
            drift = variation * 0.08 * (0.6 * np.sin(2 * np.pi * b / 37 + ph1) + 0.4 * np.sin(2 * np.pi * b / 13 + ph2)) if variation > 0 else 0.0
            d = prog[bi % len(prog)]
            triad = [_deg_semi(S, d), _deg_semi(S, d + 2), _deg_semi(S, d + 4)]
            # pad
            if ch != "O" or bi < nb or True:
                midis = [48 + spec.key + x for x in triad]
                midis = [m - 12 if m > 66 else m for m in midis]
                ns = int((bar + 0.15) * sr)
                _add(out, bank.pad(midis, ns), pos(b, 0, drift) - int(0.03 * sr), 0.20 * sg * (0.6 if ch in "IV" else 1.0))
            # bass
            if ch in "VCBO" or (ch == "I" and bi >= nb // 2):
                root = 28 + (spec.key + _deg_semi(S, d)) % 12
                hits = bpat if ch != "B" else bpat[:2]
                if ch == "O" and bi >= nb - 1:
                    hits = [0]
                for hi_, st in enumerate(hits):
                    nxt_st = hits[hi_ + 1] if hi_ + 1 < len(hits) else 16
                    dur = int((nxt_st - st) * step * sr * 0.92)
                    if variation > 0 and rng.random() < variation * 0.08:
                        continue
                    m = root + (7 if (len(hits) >= 4 and hi_ % 4 == 3) else 0)
                    _add(out, bank.bass(m, max(dur, 200)), pos(b, st, drift), 0.5 * sg * vel())
            # drums
            last_bar = bi == nb - 1
            if ch == "I":
                dl = {"hat": drums["hat"][::2] if bi >= nb // 2 else [], "kick": [], "snare": [], "open": []}
            elif ch == "V":
                dl = {"hat": drums["hat"][::2] if len(drums["hat"]) > 8 else drums["hat"], "kick": drums["kick"],
                      "snare": drums["snare"][:1] if bi % 2 == 0 else drums["snare"], "open": []}
            elif ch == "C":
                dl = drums
            elif ch == "B":
                h = _DRUMS["half"]
                dl = h
            else:
                dl = {"hat": [], "kick": drums["kick"][:1] if bi < nb - 2 else [], "snare": [], "open": []}
            if ch == "O" and last_bar:
                dl = {"hat": [], "kick": [], "snare": [], "open": []}
            if variation > 0 and ch in "VC":
                dl = {k: [s for s in v if not (k == "hat" and rng.random() < variation * 0.12)] for k, v in dl.items()}
                if rng.random() < variation * 0.3:
                    dl["kick"] = dl["kick"] + [int(rng.choice([2, 6, 7, 14, 15]))]
            if bi == 0 and ch == "C":
                _add(out, bank.crash(), pos(b, 0, drift), 0.25 * sg * vel())
            has_fill = last_bar and nxt in ("C", "B") or (bi == 3 and nb >= 8 and ch in "VC")
            fill_style = fill_base
            if variation > 0 and rng.random() < variation:
                fill_style = int(rng.integers(3))
            if variation > 0 and ch in "VC" and bi == 3 and rng.random() < variation * 0.4:
                has_fill = True
            fill_steps = {s for s, _, _ in _FILLS[fill_style]} if has_fill else set()
            vh = int(rng.integers(2)) if live else 0
            for st in dl["kick"]:
                if st not in fill_steps or st < 12:
                    _add(out, bank.kick(vh), pos(b, st, drift), 0.9 * sg * vel())
            for st in dl["snare"]:
                if st not in fill_steps:
                    _add(out, bank.snare(vh), pos(b, st, drift), 0.7 * sg * vel())
            for st in dl["hat"]:
                if st not in fill_steps:
                    _add(out, bank.hat(vh), pos(b, st, drift), (0.28 if st % 4 == 0 else 0.17) * sg * vel())
            for st in dl["open"]:
                _add(out, bank.hat(vh, True), pos(b, st, drift), 0.2 * sg * vel())
            if has_fill:
                for st, ins, g in _FILLS[fill_style]:
                    buf = bank.snare(vh) if ins == "snare" else bank.tom(200 if ins == "tom1" else 140)
                    _add(out, buf, pos(b, st, drift), g * 0.8 * sg * vel())
            # lead
            if ch in "VCBO":
                for st, du, idx in phrases[ch][bi % 4]:
                    if variation > 0 and rng.random() < variation * 0.12:
                        continue
                    midi = 57 + spec.key + _deg_semi(S, idx)
                    dsteps = du
                    if variation > 0 and rng.random() < variation * 0.2:
                        dsteps = max(1, du + int(rng.choice([-1, 1])))
                    if variation > 0 and rng.random() < variation * 0.3 and st > 0:  # grace note
                        gm = 57 + spec.key + _deg_semi(S, idx + int(rng.choice([-1, 1])))
                        _add(out, bank.lead(gm, int(step * sr * 0.9)), pos(b, st, drift) - int(step * sr), 0.28 * sg * vel())
                    ns = max(int(dsteps * step * sr * 0.95), int(0.05 * sr))
                    gl = (0.38 if ch != "O" else 0.3) * sg * vel()
                    _add(out, bank.lead(midi, ns), pos(b, st, drift), gl)

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) + 1e-9
    out *= 0.12 / rms
    out = np.tanh(out).astype(np.float32)
    secs = [(_NAMES[ch], b0 * bar, (b0 + nb) * bar) for ch, b0, nb in lay]
    return out, secs


# ============================================================================ audience, talk, noise

_VOW = [(730, 1090), (270, 2290), (300, 870), (530, 1840), (570, 840)]
_VSOS: dict = {}


def _vowel_sos(sr, v, q1=8.0, q2=10.0):
    k = (sr, v, q1, q2)
    if k not in _VSOS:
        f1, f2 = _VOW[v]
        _VSOS[k] = np.vstack([signal.tf2sos(*signal.iirpeak(f1, q1, fs=sr)),
                              signal.tf2sos(*signal.iirpeak(f2, q2, fs=sr))])
    return _VSOS[k]


def _voice(sr, d, f0a, f0b, v, vib=0.0, rng=None):
    n = max(int(d * sr), 16)
    t = np.arange(n) / sr
    f0 = np.linspace(f0a, f0b, n) * (1 + vib * np.sin(2 * np.pi * 5.5 * t))
    saw = 2 * ((np.cumsum(f0) / sr) % 1.0) - 1
    y = signal.sosfilt(_vowel_sos(sr, v), saw)
    y = y / (np.max(np.abs(y)) + 1e-9)
    env = np.hanning(n) ** 0.6
    return (y * env).astype(np.float32)


def _slow_noise(n, sr, rng, rate, depth):
    k = int(n / sr * rate) + 3
    v = rng.standard_normal(k)
    return np.exp(depth * np.interp(np.arange(n) / sr * rate, np.arange(k), v)).astype(np.float32)


def _unit(x):
    return x / (np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _crowd(n, sr, rng, babble=1.0, voices=1.0, screams=0.1, claps=2.0, bright=False, whistles=0.0):
    """Procedural crowd: babble + voice bursts + screams + claps. RMS-normalised to 1."""
    dur = n / sr
    band = [600, 8000] if bright else [250, 3500]
    sos = signal.butter(2, band, "band", fs=sr, output="sos")
    parts = [babble * _unit(signal.sosfilt(sos, rng.standard_normal(n).astype(np.float32)) * _slow_noise(n, sr, rng, 1.5, 0.35))]
    v = np.zeros(n, np.float32)
    for _ in range(rng.poisson(voices * dur * 1.2)):
        d = rng.uniform(0.2, 0.8)
        f0 = rng.choice([110, 130, 180, 220, 260, 300]) * rng.uniform(0.9, 1.1)
        _add(v, _voice(sr, d, f0, f0 * rng.uniform(0.8, 1.25), int(rng.integers(5))), int(rng.uniform(0, dur) * sr), rng.lognormal(0, 0.5))
    if v.any():
        parts.append(0.8 * _unit(v))
    s = np.zeros(n, np.float32)
    for _ in range(rng.poisson(screams * dur)):
        d = rng.uniform(0.5, 1.2)
        f0 = rng.uniform(650, 1000)
        y = _voice(sr, d, f0, f0 * rng.uniform(1.0, 1.4), 0, vib=0.02)
        _add(s, y, int(rng.uniform(0, dur) * sr), rng.lognormal(0, 0.4))
    if s.any():
        parts.append(0.7 * _unit(s))
    if claps > 0:
        imp = np.zeros(n, np.float32)
        idx = rng.integers(0, n, rng.poisson(claps * dur))
        np.add.at(imp, idx, rng.lognormal(0, 0.6, len(idx)).astype(np.float32))
        kn = int(0.03 * sr)
        ker = signal.sosfilt(signal.butter(2, [1200, 5000], "band", fs=sr, output="sos"), rng.standard_normal(kn)) * np.exp(-np.arange(kn) / sr / 0.006)
        c = signal.fftconvolve(imp, ker.astype(np.float32))[:n]
        parts.append(0.6 * _unit(c))
    if whistles > 0:
        w = np.zeros(n, np.float32)
        for _ in range(rng.poisson(whistles * dur)):
            d = rng.uniform(0.3, 0.9)
            tt = np.arange(int(d * sr)) / sr
            f = rng.uniform(2200, 3200)
            _add(w, (np.sin(2 * np.pi * f * tt + 3 * np.sin(2 * np.pi * 6 * tt)) * np.hanning(len(tt))).astype(np.float32), int(rng.uniform(0, dur) * sr), 1.0)
        if w.any():
            parts.append(0.5 * _unit(w))
    return _unit(sum(parts)).astype(np.float32)


def _applause(dur, sr, rng, rms=0.15):
    n = int(dur * sr)
    c = _crowd(n, sr, rng, babble=0.6, voices=0.5, screams=0.15, claps=35, bright=True, whistles=0.08)
    t = np.arange(n) / sr
    env = np.minimum(1, t / 0.5) * np.where(t < 0.6 * dur, 1.0, np.exp(-(t - 0.6 * dur) / (0.22 * dur)))
    env = env * _slow_noise(n, sr, rng, 1.0, 0.15)
    return (c * env * rms).astype(np.float32)


def _talk(dur, sr, rng, rms=0.1):
    """Speech-like: glottal saw through vowel formants, syllable envelope, consonant noise."""
    n = int(dur * sr)
    out = np.zeros(n, np.float32)
    f0b = rng.choice([105, 120, 140, 200, 230])
    hp = signal.butter(2, 3500, "high", fs=sr, output="sos")
    t = 0.0
    while t < dur - 0.3:
        if rng.random() < 0.15:
            t += rng.uniform(0.4, 0.9)
        for _ in range(int(rng.integers(1, 5))):
            sd = rng.uniform(0.12, 0.26)
            f0a = f0b * rng.uniform(0.9, 1.15)
            cn = signal.sosfilt(hp, rng.standard_normal(int(0.03 * sr))).astype(np.float32) * np.hanning(int(0.03 * sr)).astype(np.float32) * 0.3
            _add(out, cn, int(t * sr), 1.0)
            _add(out, _voice(sr, sd, f0a, f0a * rng.uniform(0.85, 1.15), int(rng.integers(5)), rng=rng), int((t + 0.02) * sr), 1.0)
            t += sd * 0.9
        t += rng.uniform(0.06, 0.2)
    return (_unit(out) * rms).astype(np.float32)


def _room_tone(n, sr, rng, rms=0.001):
    sos = signal.butter(1, 2000, "low", fs=sr, output="sos")
    return (_unit(signal.sosfilt(sos, rng.standard_normal(n))) * rms).astype(np.float32)


def _clapper(sr, rng):
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    sos = signal.butter(2, [800, 9000], "band", fs=sr, output="sos")
    c = signal.sosfilt(sos, rng.standard_normal(n)) * np.exp(-t / 0.012)
    return (c / np.max(np.abs(c)) * 0.8).astype(np.float32)


def _pink(n, rng):
    x = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.arange(len(x)), 1)
    return _unit(np.fft.irfft(x / np.sqrt(f), n)).astype(np.float32)


# ============================================================================ concert

def make_concert(specs, seed, sr: int = SR, gap_scale: float = 1.0):
    """Songs separated by applause / talking / silence, with one segue (no gap).

    Returns (audio, truth) with truth = list of {song, start, end, tempo_scale, sections}.
    """
    rng = np.random.default_rng([seed, 4242])
    gs = gap_scale
    items = []       # (start_s, audio)
    truth = []
    cur = 0.0

    def put(a, at):
        items.append((at, a))

    a = _applause(rng.uniform(5, 7) * gs + 1, sr, rng)
    put(a, cur)
    cur += len(a) / sr * 0.8
    tk = _talk(rng.uniform(5, 8) * gs + 1, sr, rng, 0.09)
    put(tk + _crowd(len(tk), sr, rng, voices=0.4, claps=0.3) * 0.012, cur)
    cur += len(tk) / sr + 1.0 * gs
    nS = len(specs)
    for i, sp in enumerate(specs):
        ts = float(rng.uniform(0.96, 1.04))
        a, secs = render(sp, sr, seed=int(rng.integers(1 << 30)), tempo_scale=ts, humanize_ms=7, variation=0.3)
        end = secs[-1][2]
        put(a, cur)
        truth.append({"song": sp.name, "start": cur, "end": cur + end, "tempo_scale": ts,
                      "sections": [[nm, cur + s, cur + e] for nm, s, e in secs]})
        cur += end
        if i < nS - 1:
            if i == nS - 2 and nS >= 3:
                continue  # segue: next song starts immediately
            ap = _applause(rng.uniform(4, 7) * gs + 1, sr, rng)
            put(ap, cur)
            cur += len(ap) / sr * 0.7
            if i % 2 == 0:
                tk = _talk(rng.uniform(8, 18) * gs + 1, sr, rng, 0.09)
                put(tk + _crowd(len(tk), sr, rng, voices=0.4, claps=0.3) * 0.012, cur)
                cur += len(tk) / sr + 1.0 * gs
            else:
                cur += rng.uniform(2.5, 4.0) * gs + 1
    ap = _applause(rng.uniform(7, 9) * gs + 1, sr, rng)
    put(ap, cur)
    total = cur + len(ap) / sr + 0.5
    audio = _room_tone(int(total * sr), sr, rng, 0.0015)
    for at, a in items:
        _add(audio, a, int(round(at * sr)), 1.0)
    return audio.astype(np.float32), truth


# ============================================================================ time maps

def _pwl_ext(x, xp, fp):
    x = np.asarray(x, dtype=np.float64)
    y = np.interp(x, xp, fp)
    y = np.where(x < xp[0], fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    y = np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y if y.ndim else float(y)


def map_eval(m, dev_t):
    """device time -> source time"""
    if m["kind"] == "linear":
        return m["ratio"] * np.asarray(dev_t, dtype=np.float64) + m["offset"]
    return _pwl_ext(dev_t, np.asarray(m["dev"]), np.asarray(m["src"]))


def map_inverse_eval(m, src_t):
    """source time -> device time"""
    if m["kind"] == "linear":
        return (np.asarray(src_t, dtype=np.float64) - m["offset"]) / m["ratio"]
    return _pwl_ext(src_t, np.asarray(m["src"]), np.asarray(m["dev"]))


def shift_map(m, delta):
    """New map with source time shifted by delta (source -> event conversion)."""
    if m["kind"] == "linear":
        return {"kind": "linear", "ratio": m["ratio"], "offset": m["offset"] + delta}
    return {"kind": "pwl", "dev": list(m["dev"]), "src": [s + delta for s in m["src"]]}


# ============================================================================ device simulation

KINDS = ("pro", "recorder", "phone", "crowd_phone", "hi8", "camera_mic")

_EQ = {
    "pro": [(0, -6), (30, -3), (80, 0), (8000, 0), (14000, -1.5), (20000, -4), (24000, -8)],
    "recorder": [(0, -20), (40, -6), (120, -1), (250, 0), (3000, 2.5), (8000, 1), (14000, -3), (22000, -12), (24000, -14)],
    "phone": [(0, -60), (100, -30), (160, -9), (220, -3), (400, 0), (3000, 2), (6000, 0), (9000, -8), (13000, -25), (24000, -40)],
    "hi8": [(0, -30), (40, -12), (100, -2), (300, 0), (6000, 0), (9000, -2), (10000, -5), (11000, -20), (13000, -45), (24000, -60)],
    "camera_mic": [(0, -30), (40, -8), (80, 3), (200, 6), (500, 2), (2000, -2), (4000, -6), (6000, -12), (9000, -26), (24000, -50)],
}
_EQ["crowd_phone"] = _EQ["phone"]
_REVERB = {"recorder": None, "pro": None, "phone": (0.25, 0.18), "crowd_phone": (0.3, 0.2),
           "camera_mic": (0.55, 0.35), "hi8": (0.2, 0.1)}


def _eq(x, sr, pts, ntaps):
    nyq = sr / 2.0
    f = np.linspace(0, nyq, 513)
    pf = np.array([p[0] for p in pts], float)
    pg = np.array([p[1] for p in pts], float)
    g = 10 ** (np.interp(f, pf, pg) / 20.0)
    h = signal.firwin2(ntaps, f / nyq, g)
    return signal.oaconvolve(x, h.astype(np.float32), mode="same").astype(np.float32)


def _reverb(x, sr, rt60, wet, rng):
    n = int(rt60 * 1.1 * sr)
    t = np.arange(n) / sr
    ir = signal.sosfilt(signal.butter(2, 5000, "low", fs=sr, output="sos"), rng.standard_normal(n)) * np.exp(-6.9 * t / rt60)
    pre = int(0.004 * sr)
    ir[:pre] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    y = signal.oaconvolve(x, ir.astype(np.float32))[:len(x)]
    return (x + wet * y).astype(np.float32)


def _agc(x, sr, target, attack, release, depth=1.0, max_gain=30.0):
    blk = int(0.01 * sr)
    nb = len(x) // blk
    if nb < 2:
        return x
    r = np.sqrt(np.mean(x[:nb * blk].astype(np.float64).reshape(nb, blk) ** 2, axis=1)) + 1e-5
    ac, rc = np.exp(-0.01 / attack), np.exp(-0.01 / release)
    env = np.empty(nb)
    e = r[0]
    for i in range(nb):
        c = ac if r[i] > e else rc
        e = c * e + (1 - c) * r[i]
        env[i] = e
    g = np.clip((target / (env + 1e-4)) ** depth, 0.05, max_gain)
    gs = np.interp(np.arange(len(x)), (np.arange(nb) + 0.5) * blk, g)
    return (x * gs).astype(np.float32)


def _active_rms(x):
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return r if r > 1e-4 else 0.05


def _warp(a, sr, g, n_out, out_sr, fast):
    if fast is not None:
        i0 = fast
        y = np.zeros(n_out, np.float32)
        lo, hi = max(i0, 0), min(len(a), i0 + n_out)
        if hi > lo:
            y[lo - i0:hi - i0] = a[lo:hi]
        return y
    y = np.zeros(n_out, np.float32)
    ch = 1 << 21
    for c0 in range(0, n_out, ch):
        c1 = min(n_out, c0 + ch)
        pos = g(np.arange(c0, c1) / out_sr) * sr
        i = np.floor(pos).astype(np.int64)
        fr = (pos - i).astype(np.float32)
        ok = (i >= 0) & (i < len(a) - 1)
        ii = np.clip(i, 0, len(a) - 2)
        y[c0:c1] = np.where(ok, a[ii] * (1 - fr) + a[ii + 1] * fr, 0)
    return y


def record(audio, sr, kind, seed, start=0.0, length=None, drift_ppm=0.0, out_sr=48000, **opts):
    """Simulate a device capturing a window of `audio` (source). Returns (device_audio, truth).

    start: source seconds where the device starts (may be negative: device starts early).
    length: device seconds (None = until the source ends).
    Extra options: crowd (level re source rms), wow_ms, flutter_ms, speed_var_ppm, dropouts (bool).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    rng = np.random.default_rng([seed, 777, KINDS.index(kind)])
    p = drift_ppm * 1e-6
    ratio = 1.0 / (1.0 + p)
    if length is None:
        D = max((len(audio) / sr - start) * (1 + p), 0.0)
    else:
        D = float(length)
    n_out = int(round(D * out_sr))

    disp, sv = [], []
    params = {}
    if kind == "hi8":
        wow = opts.get("wow_ms", float(rng.uniform(0.5, 1.5)))   # ~0.1-0.3 % speed at 0.5 Hz, like real Hi8
        fl = opts.get("flutter_ms", float(rng.uniform(0.02, 0.06)))
        disp = [(wow * 1e-3, 0.5, rng.uniform(0, 6.28)), (0.3 * wow * 1e-3, 0.37, rng.uniform(0, 6.28)),
                (fl * 1e-3, 6.0, rng.uniform(0, 6.28))]
        svp = opts.get("speed_var_ppm", float(rng.uniform(300, 1500)))  # slow tape-speed wander dominates
        if svp:
            sv = [(svp / 1.7 * a, 1.0 / T, rng.uniform(0, 6.28)) for a, T in ((1.0, 47.0), (0.8, 91.0), (0.5, 23.0))]
        params.update(wow_ms=wow, flutter_ms=fl, speed_var_ppm=svp)

    def g(t):
        t = np.asarray(t, dtype=np.float64)
        s = start + ratio * t
        for A, f, ph in disp:
            s = s + A * np.sin(2 * np.pi * f * t + ph)
        for A, f, ph in sv:
            s = s - ratio * (A * 1e-6) / (2 * np.pi * f) * (np.cos(2 * np.pi * f * t + ph) - np.cos(ph))
        return s

    fast = None
    if not disp and not sv and out_sr == sr and abs(ratio - 1) < 1e-12 and abs(start * sr - round(start * sr)) < 1e-9:
        fast = int(round(start * sr))
    x = _warp(np.asarray(audio, np.float32), sr, g, n_out, out_sr, fast)
    ref = _active_rms(x)
    srr = out_sr

    # --- room / mic chain
    rv = _REVERB[kind]
    if rv and rv[0] > 0:
        x = _reverb(x, srr, rv[0], rv[1], rng)
    crowd_lvl = {"pro": 0.0, "recorder": 0.0, "phone": float(rng.uniform(0.5, 0.9)), "crowd_phone": 2.2,
                 "hi8": 0.05, "camera_mic": 0.15}[kind]
    crowd_lvl = opts.get("crowd", crowd_lvl)
    params["crowd"] = crowd_lvl
    if crowd_lvl > 0 and n_out > 0:
        dens = 3.0 if kind == "crowd_phone" else 1.0
        c = _crowd(n_out, srr, rng, voices=dens, screams=0.4 if kind == "crowd_phone" else 0.1,
                   claps=2 * dens, whistles=0.05)
        sig_db = 5.0 if kind == "crowd_phone" else 2.0
        m = _slow_noise(n_out, srr, rng, 0.12, sig_db * np.log(10) / 20)
        x = x + c * m * ref * crowd_lvl
    if kind == "hi8" and n_out > 0:
        hiss = signal.sosfilt(signal.butter(2, 2500, "high", fs=srr, output="sos"), rng.standard_normal(n_out))
        t = np.arange(n_out) / srr
        x = x + (_unit(hiss) * ref * 0.06 + 0.004 * ref * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    nt = 513 if kind in ("pro", "recorder") else 2049
    if n_out > nt * 2:
        x = _eq(x, srr, _EQ[kind], nt)
    x = x + (rng.standard_normal(n_out) * (3e-4 if kind in ("pro", "recorder") else 1e-3 * ref)).astype(np.float32)

    # --- dynamics / saturation
    if kind in ("phone", "crowd_phone"):
        x = _agc(x, srr, 0.12, 0.04, 0.9 if kind == "phone" else 0.6, depth=1.0)
        drive = 2.5 if kind == "phone" else 3.5
        x = (np.tanh(drive * x) * 0.8).astype(np.float32)
    elif kind == "camera_mic":
        x = _agc(x, srr, 0.12, 0.2, 2.0, depth=0.5, max_gain=6.0)
        x = (np.tanh(1.5 * x) * 0.8).astype(np.float32)
    elif kind == "hi8":
        x = (np.tanh(1.5 * x / max(ref, 1e-3) * 0.12) * 0.8).astype(np.float32)
    else:
        x = (x * (rng.uniform(0.06, 0.14) / _active_rms(x))).astype(np.float32)

    # --- hi8 dropouts
    drops = []
    if kind == "hi8" and opts.get("dropouts", True) and D > 20:
        for _ in range(max(1, rng.poisson(D / 45.0))):
            ds, dd = float(rng.uniform(2, D - 2)), float(rng.uniform(0.05, 0.3))
            i0, i1 = int(ds * srr), int((ds + dd) * srr)
            ramp = int(0.002 * srr)
            lo, hi = max(0, i0 - ramp), min(n_out, i1 + ramp)
            w = np.interp(np.arange(lo, hi), [i0 - ramp, i0, i1, i1 + ramp], [1, 0.03, 0.03, 1])
            x[lo:hi] = x[lo:hi] * w.astype(np.float32) + (rng.standard_normal(hi - lo) * 0.01 * (1 - w)).astype(np.float32)
            drops.append([ds, dd])
    x = np.clip(x, -1.0, 1.0).astype(np.float32)

    if disp or sv:
        dev = np.arange(0, D + 1e-9, 0.02)
        m = {"kind": "pwl", "dev": np.round(dev, 6).tolist(), "src": np.round(g(dev), 7).tolist()}
    else:
        m = {"kind": "linear", "ratio": ratio, "offset": float(start)}
    truth = {"kind": kind, "seed": seed, "src_sr": sr, "out_sr": out_sr, "drift_ppm": drift_ppm,
             "map": m, "src_start": float(g(0.0)), "src_end": float(g(D)), "dev_duration": float(n_out / out_sr),
             "params": params, "dropouts": drops}
    return x, truth


# ============================================================================ file writers

def write_wav(path, audio, sr, fmt="pcm16"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(audio, np.float32)
    if fmt == "float":
        wavfile.write(str(path), int(sr), a)
    else:
        wavfile.write(str(path), int(sr), (np.clip(a, -1, 1) * 32767).astype(np.int16))


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")


def write_video(path, audio, sr, fps=25, vfr=False, timecode=None, size="320x180"):
    """Mux audio with a cheap lavfi testsrc2 video. .mp4/.m4v get AAC, others PCM (sample exact).

    vfr=True drops every 23rd frame and jitters timestamps by up to 6 ms (irregular pts).
    timecode "HH:MM:SS:FF" writes a start timecode tag.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dur = len(audio) / sr
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        write_wav(wav, audio, sr)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
               f"testsrc2=size={size}:rate={fps}", "-i", wav]
        if vfr:
            cmd += ["-vf", "select='not(eq(mod(n,23),5))',setpts='PTS+0.006*sin(N*2.3)/TB'", "-fps_mode", "vfr"]
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "35", "-pix_fmt", "yuv420p", "-g", "50"]
        cmd += ["-c:a", "aac", "-b:a", "160k"] if path.suffix.lower() in (".mp4", ".m4v") else ["-c:a", "pcm_s16le"]
        if timecode:
            cmd += ["-timecode", timecode]
        cmd += ["-t", f"{dur:.6f}", str(path)]
        _run(cmd)


# ============================================================================ fixtures

def _tc(seconds, fps=25, base_h=19):
    fr = int(round(seconds * fps))
    h, r = divmod(fr, 3600 * fps)
    m, r = divmod(r, 60 * fps)
    s, f = divmod(r, fps)
    return f"{base_h + h:02d}:{m:02d}:{s:02d}:{f:02d}"


class _Scn:
    def __init__(self, root, name, description, quick):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t = {"scenario": name, "synth_version": SYNTH_VERSION, "description": description,
                  "events": {}, "songs": {}, "files": {}, "pairs": [], "expected_exact_groups": [],
                  "expected_songs": {}, "notes": {}}

    def event(self, eid, typ, duration, songs):
        self.t["events"][eid] = {"type": typ, "duration": float(duration), "songs": songs}

    def song(self, name, duration, ref=None):
        self.t["songs"][name] = {"duration": float(duration), "ref": ref}

    def concert_event(self, eid, truth, total, typ="concert"):
        self.event(eid, typ, total, [{"song": s["song"], "start": s["start"], "end": s["end"],
                                      "song_t0": 0.0, "rate": s["tempo_scale"]} for s in truth])

    def add(self, rel, audio, tr, event, role="", shared=None, tc=None, vfr=False, shift=0.0, **vk):
        path = self.dir / rel
        if path.suffix.lower() == ".wav":
            write_wav(path, audio, tr["out_sr"])
        else:
            write_video(path, audio, tr["out_sr"], timecode=tc, vfr=vfr, **vk)
        m = shift_map(tr["map"], shift) if shift else tr["map"]
        D = tr["dev_duration"]
        ev = self.t["events"][event]
        lo_w, hi_w = float(map_eval(m, 0.0)), float(map_eval(m, D))
        sh = shared if shared is not None else (0.0, ev["duration"])
        lo, hi = max(lo_w, sh[0]), min(hi_w, sh[1])
        songs = []
        for s in ev["songs"]:
            a, b = max(s["start"], lo), min(s["end"], hi)
            if b - a >= 3.0:
                songs.append({"song": s["song"], "dev_start": float(map_inverse_eval(m, a)),
                              "dev_end": float(map_inverse_eval(m, b)), "ev_start": a, "ev_end": b,
                              "song_start": s["song_t0"] + (a - s["start"]) * s["rate"],
                              "song_end": s["song_t0"] + (b - s["start"]) * s["rate"],
                              "partial": bool((b - a) < 0.9 * (s["end"] - s["start"]))})
        self.t["files"][rel] = {"kind": tr["kind"], "event": event, "role": role, "map": m,
                                "dev_duration": D, "drift_ppm": tr["drift_ppm"], "start_tc": tc, "vfr": vfr,
                                "exact_span": [lo, max(lo, hi)], "songs": songs, "params": tr["params"],
                                "dropouts": tr["dropouts"], "out_sr": tr["out_sr"]}

    def finish(self):
        files = list(self.t["files"])
        parent = {f: f for f in files}

        def find(f):
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f
        for i, a in enumerate(files):
            for b in files[i + 1:]:
                fa, fb = self.t["files"][a], self.t["files"][b]
                if fa["event"] != fb["event"]:
                    continue
                ov = min(fa["exact_span"][1], fb["exact_span"][1]) - max(fa["exact_span"][0], fb["exact_span"][0])
                if ov >= 5.0:
                    self.t["pairs"].append({"a": a, "b": b, "event": fa["event"], "overlap_s": ov})
                    parent[find(a)] = find(b)
        comp = {}
        for f in files:
            comp.setdefault(find(f), []).append(f)
        self.t["expected_exact_groups"] = [v for v in comp.values() if len(v) > 1]
        for f in files:
            for s in self.t["files"][f]["songs"]:
                self.t["expected_songs"].setdefault(s["song"], [])
                if f not in self.t["expected_songs"][s["song"]]:
                    self.t["expected_songs"][s["song"]].append(f)
        (self.dir / "truth.json").write_text(json.dumps(self.t, default=_jd))
        return self.t


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _concert_specs(base_seed, n, quick):
    return [random_song(base_seed + i, 50 if quick else 170) for i in range(n)]


def _scn_concert_full(root, quick):
    sc = _Scn(root, "concert_full", "4-song concert, 2 pro cams (one +40 ppm with timecode), recorder -25 ppm, "
              "Hi8 middle half, 3 phone snippets, crowd phone snippet, VFR phone video", quick)
    specs = _concert_specs(10, 4, quick)
    audio, ct = make_concert(specs, 101, gap_scale=0.4 if quick else 1.0)
    total = len(audio) / SR
    sc.concert_event("show", ct, total)
    for s in specs:
        sc.song(s.name, song_duration(s))
    rng = np.random.default_rng(1001)
    a, t = record(audio, SR, "pro", 1, start=2.0, drift_ppm=40)
    sc.add("cam_a.mov", a, t, "show", "pro cam +40 ppm, timecode", tc=_tc(2.0))
    a, t = record(audio, SR, "pro", 2, start=1.3)
    sc.add("cam_b.mp4", a, t, "show", "pro cam, AAC")
    a, t = record(audio, SR, "recorder", 3, start=0.0, drift_ppm=-25, out_sr=44100)
    sc.add("recorder.wav", a, t, "show", "audio recorder -25 ppm, 44.1 kHz")
    a, t = record(audio, SR, "hi8", 4, start=0.25 * total, length=0.5 * total, speed_var_ppm=300)
    sc.add("hi8.mov", a, t, "show", "Hi8 capture, middle half, wow/flutter/dropouts")
    for i in range(3):
        L = float(rng.uniform(20, 90))
        st = float(rng.uniform(0.05 * total, total - L - 5))
        a, t = record(audio, SR, "phone", 10 + i, start=st, length=L, drift_ppm=float(rng.uniform(-100, 100)))
        sc.add(f"phone_{i + 1}.mp4", a, t, "show", "phone snippet")
    st = float(0.4 * total)
    a, t = record(audio, SR, "crowd_phone", 20, start=st, length=30)
    sc.add("crowd_phone.mp4", a, t, "show", "crowd phone snippet")
    st = float(0.6 * total)
    a, t = record(audio, SR, "phone", 21, start=st, length=45)
    sc.add("phone_vfr.mp4", a, t, "show", "VFR phone video", vfr=True, fps=30)
    return sc.finish()


def _scn_snippets(root, quick):
    sc = _Scn(root, "snippets_only", "6 phone snippets of a concert; 3 overlap, 2 overlap, 1 alone", quick)
    specs = [random_song(30 + i, 90 if quick else 170) for i in range(4)]
    audio, ct = make_concert(specs, 102, gap_scale=0.4 if quick else 1.0)
    total = len(audio) / SR
    sc.concert_event("show", ct, total)
    for s in specs:
        sc.song(s.name, song_duration(s))
    A, B = ct[0], ct[2]
    a0 = A["start"] + 0.02 * (A["end"] - A["start"])
    b0 = B["start"] + 0.05 * (B["end"] - B["start"])
    k = min(1.0, 0.5 * (A["end"] - A["start"]) / 65.0)     # keep the cluster inside the first half of song A
    plan = [(a0, 40 * k), (a0 + 15 * k, 50 * k), (a0 + 30 * k, 25 * k),
            (b0, 30), (b0 + 18, 45),
            (A["start"] + 0.78 * (A["end"] - A["start"]), 18)]
    for i, (st, L) in enumerate(plan):
        a, t = record(audio, SR, "phone", 40 + i, start=st, length=L, drift_ppm=(0, 60, -80, 20, 0, 35)[i])
        sc.add(f"snip_{i + 1}.mp4" if i % 2 == 0 else f"snip_{i + 1}.wav", a, t, "show", "phone snippet")
    return sc.finish()


def _scn_mv(root, quick):
    sc = _Scn(root, "mv_playback", "music video: one song, 6 takes of a speaker playing the track with talk/clapper around", quick)
    spec = random_song(77, 70 if quick else 160)
    ref, _ = render(spec, SR, seed=0)
    D = song_duration(spec)
    ref = ref[:int(D * SR)]
    write_wav(sc.dir / "refs" / f"{spec.name}.wav", ref, SR)
    sc.song(spec.name, D, f"refs/{spec.name}.wav")
    eid = f"playback:{spec.name}"
    sc.event(eid, "playback", D, [{"song": spec.name, "start": 0.0, "end": D, "song_t0": 0.0, "rate": 1.0}])
    takes = [(0.0, D), (0.10 * D, 30.0), (0.30 * D, 45.0), (0.45 * D, 0.5 * D), (0.65 * D, 0.35 * D), (0.0, 35.0)]
    kinds = ["camera_mic", "phone", "camera_mic", "phone", "crowd_phone", "camera_mic"]
    drift = [0, 30, -20, 0, 50, 10]
    rng = np.random.default_rng(1003)
    spk = [(0, -40), (80, -12), (150, 0), (9000, 0), (14000, -18), (24000, -40)]
    for i, (p0, L) in enumerate(takes):
        L = min(L, D - p0)
        seg = _eq(ref[int(p0 * SR):int((p0 + L) * SR)], SR, spk, 513)
        pre, post = float(rng.uniform(4, 10)), float(rng.uniform(4, 8))
        pre_a = _talk(pre, SR, rng, 0.12) + _crowd(int(pre * SR), SR, rng, claps=0.2) * 0.04
        cl = _clapper(SR, rng)
        _add(pre_a, cl, int((pre - 0.8) * SR), 1.0)
        _add(pre_a, cl, int((pre - 0.7) * SR), 0.6)
        post_a = _talk(post, SR, rng, 0.12) + _crowd(int(post * SR), SR, rng, claps=0.5) * 0.05
        _add(post_a, cl, int(0.5 * SR), 0.9)
        seg = seg.copy()
        yell = _talk(2.5, SR, rng, 0.09)
        _add(seg, yell, int(0.2 * SR), 1.0)
        room = np.concatenate([pre_a, seg, post_a]).astype(np.float32)
        a, t = record(room, SR, kinds[i], 50 + i, start=0.0, drift_ppm=drift[i])
        sc.add(f"take_{i + 1}.mp4" if i % 2 == 0 else f"take_{i + 1}.wav", a, t, eid,
               f"take {i + 1} at song pos {p0:.1f}s len {L:.1f}s", shared=(p0, p0 + L), shift=p0 - pre)
    return sc.finish()


def _scn_live_takes(root, quick):
    sc = _Scn(root, "live_takes", "one song performed 4 times, different devices, no reference", quick)
    spec = random_song(55, 70 if quick else 160)
    sc.song(spec.name, song_duration(spec))
    perf = [(1.0, 5, 0.2), (0.94, 12, 0.4), (1.06, 10, 0.5), (0.98, 15, 0.6)]
    kinds = ["pro", "phone", "hi8", "camera_mic"]
    rng = np.random.default_rng(1004)
    for i, (ts, hu, va) in enumerate(perf):
        a, secs = render(spec, SR, seed=300 + i, tempo_scale=ts, humanize_ms=hu, variation=va)
        end = secs[-1][2]
        lead = float(rng.uniform(2, 4))
        ap = _applause(5.0, SR, rng)
        room = np.concatenate([_talk(lead, SR, rng, 0.05) + _room_tone(int(lead * SR), SR, rng, 0.003), a[:int(end * SR)], ap])
        tail = a[int(end * SR):]
        _add(room, tail, int((lead + end) * SR), 1.0)
        eid = f"perf_{i + 1}"
        sc.event(eid, "performance", len(room) / SR, [{"song": spec.name, "start": lead, "end": lead + end,
                                                      "song_t0": 0.0, "rate": ts}])
        a2, t = record(room, SR, kinds[i], 60 + i, start=float(rng.uniform(0, 1.5)),
                       drift_ppm=float(rng.uniform(-50, 50)), **({"speed_var_ppm": 200} if kinds[i] == "hi8" else {}))
        sc.add(f"take_{i + 1}.mov" if i % 2 == 0 else f"take_{i + 1}.wav", a2, t, eid,
               f"performance {i + 1}: tempo {ts}, humanize {hu} ms, variation {va}")
    sc.t["notes"]["song_map"] = "song time is linear in performance time up to ~80 ms wobble * variation"
    return sc.finish()


def _scn_two_nights(root, quick):
    sc = _Scn(root, "two_nights", "same 3-song setlist on two nights; exact sync must not join nights", quick)
    specs = _concert_specs(90, 3, quick)
    for s in specs:
        sc.song(s.name, song_duration(s))
    gs = 0.4 if quick else 1.0
    for night, seed, kinds in (("A", 201, ["pro", "phone", "phone"]), ("B", 202, ["pro", "crowd_phone", "phone"])):
        audio, ct = make_concert(specs, seed, gap_scale=gs)
        total = len(audio) / SR
        eid = f"night_{night}"
        sc.concert_event(eid, ct, total)
        rng = np.random.default_rng(seed)
        a, t = record(audio, SR, kinds[0], seed + 1, start=1.0, drift_ppm=15 if night == "A" else -30)
        sc.add(f"cam_{night.lower()}.mov", a, t, eid, f"night {night} camera")
        for j in range(2):
            L = float(rng.uniform(25, 60))
            st = float(rng.uniform(0.1 * total, total - L - 5))
            a, t = record(audio, SR, kinds[1 + j], seed + 5 + j, start=st, length=L)
            sc.add(f"snip_{night.lower()}{j + 1}.mp4" if j == 0 else f"snip_{night.lower()}{j + 1}.wav", a, t, eid,
                   f"night {night} snippet")
    return sc.finish()


def _scn_negative(root, quick):
    sc = _Scn(root, "negative", "unrelated songs and noise; nothing should match exactly", quick)
    rng = np.random.default_rng(1006)
    L = 50 if quick else 110
    sp = [random_song(400 + i, L, name=f"neg_{i + 1}") for i in range(3)]
    # hard negative: same key/chords/tempo family as neg_1 but different melody, drums, tempo
    hard = random_song(410, L, name="neg_sameprog")
    hard.key, hard.mode, hard.progs, hard.structure, hard.bars = sp[0].key, sp[0].mode, sp[0].progs, sp[0].structure, sp[0].bars
    hard.tempo_bpm = sp[0].tempo_bpm * 1.1
    specs = sp + [hard]
    kinds = ["pro", "phone", "camera_mic", "phone"]
    for i, s in enumerate(specs):
        a, secs = render(s, SR, seed=500 + i)
        sc.song(s.name, song_duration(s))
        eid = f"neg_event_{i + 1}"
        sc.event(eid, "unrelated", len(a) / SR, [{"song": s.name, "start": 0.0, "end": secs[-1][2], "song_t0": 0.0, "rate": 1.0}])
        d, t = record(a, SR, kinds[i], 70 + i)
        sc.add(f"song_{i + 1}.mov" if i == 0 else f"song_{i + 1}.wav", d, t, eid, f"unrelated song {s.name}")
    sc.t["notes"]["hard_negative"] = "neg_sameprog shares key/chords/structure with neg_1 (different melody, drums, tempo): " \
                                     "must not match exactly; a musical (chroma) match is ambiguous"
    for nm, au in (("noise_pink", _pink(30 * SR, rng) * 0.1),
                   ("applause_talk", np.concatenate([_applause(12, SR, rng), _talk(15, SR, rng, 0.1)]))):
        eid = nm
        sc.event(eid, "unrelated", len(au) / SR, [])
        d, t = record(au.astype(np.float32), SR, "phone", 80 + len(sc.t["files"]))
        sc.add(f"{nm}.wav", d, t, eid, "non-music")
    return sc.finish()


def make_fixtures(out_dir, quick=False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = {"synth_version": SYNTH_VERSION, "quick": bool(quick), "scenarios": {}}
    for fn in (_scn_concert_full, _scn_snippets, _scn_mv, _scn_live_takes, _scn_two_nights, _scn_negative):
        t0 = time.time()
        t = fn(out, quick)
        man["scenarios"][t["scenario"]] = t
        print(f"  {t['scenario']}: {len(t['files'])} files in {time.time() - t0:.1f}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, default=_jd))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tests.synth")
    ap.add_argument("out_dir")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    make_fixtures(a.out_dir, quick=a.quick)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
