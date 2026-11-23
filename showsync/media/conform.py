"""VFR -> CFR conform and drift-corrected audio export."""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from typing import Optional

import numpy as np

from ..models import TimeMap
from .probe import probe

STANDARD_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)


def _run_ffmpeg(args: list, label: str = "") -> None:
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y"] + args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}) {label}: "
                           f"{r.stderr.decode('utf-8', 'replace').strip()}")


def nearest_standard_rate(fps: float) -> float:
    return min(STANDARD_RATES, key=lambda s: abs(s - fps) / s)


def conform_cfr(path: str, out_path: str, fps: Optional[float] = None) -> str:
    """Transcode `path` to constant frame rate. Returns out_path.

    Video: H.264 crf 14 / preset medium (or ProRes HQ if the source is ProRes). Audio: PCM
    (s24le), resampled with async=1:first_pts=0 so it stays aligned to t=0. Use a .mov out_path.
    """
    info = probe(path)
    if not info.has_video:
        raise ValueError(f"{path}: no video stream to conform")
    if fps is None:
        nominal = info.fps
        if nominal <= 0 or nominal > 121:   # phones sometimes report 600/1000 as r_frame_rate
            nominal = info.fps_num / info.fps_den if info.fps_den else 30.0
            if nominal <= 0 or nominal > 121:
                nominal = 30.0
        fps = nearest_standard_rate(nominal)
    rate = {23.976: "24000/1001", 29.97: "30000/1001", 59.94: "60000/1001"}.get(round(fps, 3), repr(float(fps)))
    if info.video_codec == "prores":
        vcodec = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    else:
        vcodec = ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
    args = ["-i", path, "-map", "0:v:0", "-fps_mode", "cfr", "-r", rate] + vcodec
    if info.has_audio:
        args += ["-map", "0:a?", "-af", "aresample=async=1:first_pts=0", "-c:a", "pcm_s24le"]
    args += [out_path]
    _run_ffmpeg(args, path)
    return out_path


# ----------------------------------------------------------------------------- audio export

_HALF = 32          # sinc half-width in source samples (64 taps)
_BLOCK = 16384
_CUTOFF = 0.96      # of Nyquist
_BETA = 9.0


_PHASES = 4096      # fractional-position table resolution (timing error <= 1/8192 sample)
_TABLE = None


def _table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        taps = np.arange(-_HALF + 1, _HALF + 1)
        frac = np.arange(_PHASES, dtype=np.float64) / _PHASES
        x = taps[None, :] - frac[:, None]
        _TABLE = (_CUTOFF * np.sinc(_CUTOFF * x)
                  * np.i0(_BETA * np.sqrt(np.clip(1.0 - (x / _HALF) ** 2, 0.0, 1.0))) / np.i0(_BETA)
                  ).astype(np.float32)
    return _TABLE


def _write_wav_header(fh, sr: int, ch: int, n_frames: int) -> None:
    data = n_frames * ch * 4
    fh.write(b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE")
    fh.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, ch, sr, sr * ch * 4, ch * 4, 32))
    fh.write(b"data" + struct.pack("<I", data))


def export_corrected_audio(path: str, out_path: str, m: TimeMap, duration: float, sr: int = 48000) -> str:
    """Write a float32 WAV in the TARGET time base: output time T holds source time m^-1(T).

    `m` maps source clip time -> target time (linear or piecewise). The output is `duration`
    seconds long (zeros where the source does not cover), all source channels kept, at `sr`.

    Method: the source is decoded losslessly at `sr`, then each output sample is computed by
    64-tap Kaiser-windowed-sinc interpolation (beta 9, cutoff 0.96 Nyquist, 4096-phase table,
    timing quantised to 1/8192 sample) at the exactly inverted source time, per sample, so piecewise knots are followed exactly and there is no
    ratio rounding. Timing accuracy is limited only by float64 time arithmetic (<< 1 sample);
    amplitude error is below about -60 dB up to ~0.9 Nyquist. Cost is O(64 * samples * channels).
    """
    info = probe(path)
    if not info.has_audio:
        raise ValueError(f"{path}: file has no audio stream")
    ch = info.audio_channels or 1
    inv = m.inverse()
    n_out = int(round(duration * sr))

    tmp_fd, tmp_raw = tempfile.mkstemp(suffix=".f32")
    os.close(tmp_fd)
    try:
        _run_ffmpeg(["-i", path, "-map", "0:a:0", "-vn", "-ac", str(ch), "-ar", str(sr),
                     "-f", "f32le", tmp_raw], path)
        nbytes = os.path.getsize(tmp_raw)
        n_src = nbytes // (4 * ch)
        src = (np.memmap(tmp_raw, dtype="<f4", mode="r", shape=(n_src, ch))
               if n_src > 0 else np.zeros((0, ch), np.float32))

        taps = np.arange(-_HALF + 1, _HALF + 1)  # 64 offsets from floor(position)
        part = out_path + ".part"
        with open(part, "wb") as fh:
            _write_wav_header(fh, sr, ch, n_out)
            for start in range(0, n_out, _BLOCK):
                b = min(_BLOCK, n_out - start)
                T = (start + np.arange(b, dtype=np.float64)) / sr
                pos = np.asarray(inv(T), dtype=np.float64) * sr       # source sample position
                ph = np.rint(pos * _PHASES).astype(np.int64)
                i0 = ph // _PHASES                                     # floor(position) after rounding
                w = _table()[ph - i0 * _PHASES]                        # (b, 64) float32
                idx = i0[:, None] + taps[None, :]                      # (b, 64)
                valid = (idx >= 0) & (idx < n_src)
                if not valid.any():
                    block = np.zeros((b, ch), np.float32)
                else:
                    lo = max(int(idx.min()), 0)
                    hi = min(int(idx.max()) + 1, n_src)
                    seg = np.asarray(src[lo:hi], dtype=np.float32)
                    g = seg[np.clip(idx, lo, hi - 1) - lo]             # (b, 64, ch)
                    w = np.where(valid, w, np.float32(0))
                    block = np.einsum("bt,btc->bc", w, g).astype(np.float32)
                    # samples with no overlap with the source at all must be exactly zero
                    block[~valid.any(axis=1)] = 0.0
                fh.write(block.astype("<f4").tobytes())
        os.replace(part, out_path)
        del src
    finally:
        try:
            os.unlink(tmp_raw)
        except OSError:
            pass
    return out_path
