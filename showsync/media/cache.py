"""On-disk cache of decoded arrays (np.savez / plain .npy, atomic writes)."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


def cache_dir() -> Path:
    """%LOCALAPPDATA%/ShowSync/cache (env SHOWSYNC_CACHE overrides). Created on demand."""
    env = os.environ.get("SHOWSYNC_CACHE")
    if env:
        d = Path(env)
    else:
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
        d = Path(base) / "ShowSync" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(path: str, *parts) -> str:
    """Key from absolute path + file size + mtime_ns + extra parts."""
    ap = os.path.normcase(os.path.abspath(path))
    try:
        st = os.stat(ap)
        sig = f"{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        sig = "missing"
    raw = "|".join([ap, sig] + [str(p) for p in parts])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _file(key: str, ext: str = ".npz") -> Path:
    return cache_dir() / f"{key}{ext}"


def load_npz(key: str):
    """Return dict of arrays, or None on miss / unreadable entry."""
    f = _file(key)
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def _atomic_write(final: Path, writer) -> None:
    fd, tmp = tempfile.mkstemp(dir=final.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            writer(fh)
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_npz(key: str, **arrays) -> None:
    """Atomic write: temp file in the cache dir, then os.replace."""
    _atomic_write(_file(key), lambda fh: np.savez(fh, **arrays))


def npy_path(key: str) -> Path:
    return _file(key, ".npy")


def save_npy(key: str, arr: np.ndarray) -> Path:
    """Uncompressed .npy (memory-mappable), atomic write. Returns the path."""
    p = npy_path(key)
    _atomic_write(p, lambda fh: np.save(fh, arr, allow_pickle=False))
    return p


def load_npy(key: str, mmap: bool = False):
    """Return array (or read-only memmap if mmap), or None on miss / unreadable entry."""
    p = npy_path(key)
    if not p.exists():
        return None
    try:
        return np.load(p, mmap_mode="r" if mmap else None, allow_pickle=False)
    except Exception:
        return None
