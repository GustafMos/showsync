"""ffprobe wrappers: MediaInfo, VFR detection, clip ids."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from fractions import Fraction

import numpy as np

from ..models import MediaInfo

VFR_PACKETS = 600


def clip_id(path: str) -> str:
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:12]


def _run(args: list) -> str:
    r = subprocess.run(args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({r.returncode}): {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout.decode("utf-8", "replace")


def _frac(s) -> Fraction:
    try:
        f = Fraction(str(s))
        return f
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def _tags(d: dict) -> dict:
    return {str(k).lower(): str(v) for k, v in (d.get("tags") or {}).items()}


def _first_packet_times(path: str, n: int = VFR_PACKETS) -> np.ndarray:
    """Sorted pts times (s) of the first ~n video packets (B-frame reordering => sort)."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "packet=pts_time", "-read_intervals", "%+30", "-of", "csv=p=0", path])
    ts = []
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            try:
                ts.append(float(line))
            except ValueError:
                pass
        if len(ts) >= n:
            break
    return np.sort(np.asarray(ts, dtype=np.float64))


def _detect_vfr(path: str, nominal: float, avg: float, interlaced: bool = False) -> tuple:
    reasons = []
    if nominal > 0 and avg > 0 and abs(avg - nominal) / nominal > 0.005:
        reasons.append(f"avg {avg:.3f} vs nominal {nominal:.3f} fps")
    try:
        # interlaced streams carry field-coded packets (half-frame steps in decode order), so their
        # packet intervals say nothing about the frame rate; the average check above still applies
        ts = np.zeros(0) if interlaced else _first_packet_times(path)
    except RuntimeError:
        ts = np.zeros(0)
    if len(ts) >= 10 and nominal > 0:
        d = np.diff(ts)
        d = d[d > 1e-9]
        if len(d) >= 9:
            nd = 1.0 / nominal
            dev = np.abs(d - nd)
            jitter_ms = float(np.percentile(dev, 90) * 1000)
            frac_off = float(np.mean(dev > 0.25 * nd))
            if frac_off > 0.05 or float(np.max(d)) > 1.8 * nd:
                reasons.append(f"{frac_off * 100:.0f}% of intervals off nominal, "
                               f"interval {d.min() * 1000:.1f}-{d.max() * 1000:.1f} ms vs {nd * 1000:.1f}, "
                               f"jitter {jitter_ms:.1f} ms")
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def probe(path: str) -> MediaInfo:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    j = json.loads(out)
    fmt = j.get("format", {})
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    alltags = [_tags(fmt)] + [_tags(s) for s in streams]

    def tag(*names):
        for t in alltags:
            for n in names:
                if t.get(n):
                    return t[n]
        return None

    dur = 0.0
    for src in (fmt.get("duration"), (v or {}).get("duration"), (a or {}).get("duration")):
        try:
            dur = max(dur, float(src))
        except (TypeError, ValueError):
            pass

    info = MediaInfo(clip_id=clip_id(path), path=os.path.abspath(path), duration=dur,
                     has_audio=a is not None, has_video=v is not None)
    if a:
        info.audio_sr = int(a.get("sample_rate") or 0)
        info.audio_channels = int(a.get("channels") or 0)
        info.audio_codec = a.get("codec_name", "")
    if v:
        info.width = int(v.get("width") or 0)
        info.height = int(v.get("height") or 0)
        info.video_codec = v.get("codec_name", "")
        r = _frac(v.get("r_frame_rate", "0/1"))
        avg = _frac(v.get("avg_frame_rate", "0/1"))
        if r == 0:
            r = avg
        # Interlaced streams (AVCHD/HDV camcorders, 50i/60i): ffprobe's r_frame_rate is the FIELD
        # rate (50), avg_frame_rate the frame rate (25) that Resolve and every NLE use. Taking 50
        # doubled every frame number for these clips (seen on real AVCHD .MTS: placements past the
        # clip end) and flagged them as variable frame rate.
        interlaced = v.get("field_order", "") in ("tt", "bb", "tb", "bt")
        if interlaced and avg != 0 and r == 2 * avg:
            r = avg
        if r != 0:
            info.fps_num, info.fps_den = r.numerator, r.denominator
            info.fps = float(r)
        info.is_vfr, info.vfr_detail = _detect_vfr(path, info.fps, float(avg), interlaced)

    info.start_tc = tag("timecode")
    info.creation_time = tag("creation_time")

    parts = []
    for names in (("com.apple.quicktime.make", "com.android.manufacturer", "make"),
                  ("com.apple.quicktime.model", "com.android.model", "model"),
                  ("com.apple.quicktime.software", "encoder"),
                  ("serial", "serialnumber", "serial_number", "com.apple.quicktime.serial")):
        val = tag(*names)
        if val and val.startswith(("Lavf", "Lavc", "Lavf")):
            continue  # muxer/encoder library name, not a device
        if val and val not in parts:
            parts.append(val)
    info.device = " ".join(parts)
    return info
