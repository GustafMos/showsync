"""Audio decoding through ffmpeg (mono float32, cached)."""
from __future__ import annotations

import subprocess
import threading

import numpy as np

from ..config import ANALYSIS_SR
from . import cache
from .probe import probe


def decode_audio(path: str, sr: int = ANALYSIS_SR, stream: int = 0, *, mmap: bool = False) -> np.ndarray:
    """Decode audio stream `stream` of `path` to mono float32 at `sr` Hz (channels averaged).

    Cached on disk as an uncompressed .npy keyed by path/size/mtime/sr/stream. With mmap=True
    the result is a read-only np.memmap of the cache file (keeps RAM low for 10 x 90 min).
    Raises ValueError if the file has no audio.
    """
    key = cache.cache_key(path, "audio-v2", sr, stream)
    hit = cache.load_npy(key, mmap=mmap)
    if hit is not None:
        return hit
    info = probe(path)
    if not info.has_audio:
        raise ValueError(f"{path}: file has no audio stream")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", f"0:a:{stream}",
           "-vn", "-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "-"]
    audio = run_f32(cmd, path)
    cache.save_npy(key, audio)
    if mmap:
        del audio
        return cache.load_npy(key, mmap=True)
    return audio


def run_f32(cmd: list, label: str = "") -> np.ndarray:
    """Run ffmpeg writing raw f32le to stdout; return float32 array. stderr surfaced on failure."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    chunks = []
    leftover = b""
    while True:
        buf = p.stdout.read(1 << 22)
        if not buf:
            break
        buf = leftover + buf
        n = len(buf) // 4 * 4
        leftover = buf[n:]
        chunks.append(np.frombuffer(buf[:n], dtype="<f4").copy())
    rc = p.wait()
    t.join()
    if rc != 0:
        msg = (err[0] if err else b"").decode("utf-8", "replace").strip()
        if "does not contain any stream" in msg or "matches no streams" in msg:
            raise ValueError(f"{label}: no such audio stream: {msg}")
        raise RuntimeError(f"ffmpeg failed ({rc}) on {label}: {msg}")
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)
