"""Command line: python -m showsync <command> ..."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _progress_printer(jsonl: bool):
    last = [0.0]

    def say(stage, done, total, msg=""):
        if jsonl:
            print(json.dumps({"stage": stage, "done": done, "total": total, "msg": msg}), flush=True)
            return
        now = time.time()
        if done == total or now - last[0] > 0.5:
            last[0] = now
            print(f"\r{stage:<10} {done}/{total} {msg[:50]:<50}", end="" if done < total else "\n",
                  file=sys.stderr, flush=True)
    return say


def cmd_analyze(a):
    from .pipeline import analyze
    say = _progress_printer(a.jsonl_progress)
    t0 = time.time()
    paths = list(a.paths)
    if a.paths_file:
        with open(a.paths_file, encoding="utf-8-sig") as fh:
            paths += [ln.strip() for ln in fh if ln.strip()]
    if not paths:
        sys.exit("no input files")
    res = analyze(paths, refs=[a.refs] if a.refs else None, progress=say, workers=a.workers,
                  hints_path=a.hints)
    out = a.out or "showsync_result.json"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(res.to_json())
    report = a.report
    if report:
        from .report.html import write_report
        write_report(res, report)
    summary = {
        "stage": "done", "result": os.path.abspath(out),
        "report": os.path.abspath(report) if report else None,
        "clips": len(res.clips), "groups": len(res.groups), "loners": len(res.loners),
        "songs": len(res.songs), "warnings": len(res.warnings), "seconds": round(time.time() - t0, 1),
    }
    if a.jsonl_progress:
        print(json.dumps(summary), flush=True)
    else:
        print(f"{summary['clips']} clips, {summary['groups']} synced groups, "
              f"{summary['loners']} unmatched, {summary['songs']} songs, "
              f"{summary['warnings']} warnings in {summary['seconds']} s")
        for w in res.warnings:
            print("  ! " + w)
        print("Result: " + summary["result"] + (f"\nReport: {summary['report']}" if report else ""))


def cmd_conform(a):
    from .media.probe import probe
    from .media.conform import conform_cfr
    root, _ = os.path.splitext(a.file)
    out = a.out or root + "_cfr.mov"
    info = probe(a.file)
    print(conform_cfr(a.file, out, a.fps or None))
    if not info.is_vfr:
        print("note: source was already constant frame rate", file=sys.stderr)


def cmd_fix_audio(a):
    """Write a WAV of <clip> resampled onto the time base of --to (default: its group's anchor)."""
    from .models import AnalysisResult
    from .media.conform import export_corrected_audio
    with open(a.result, encoding="utf-8") as fh:
        res = AnalysisResult.from_json(fh.read())
    by_name = {os.path.basename(m.path).lower(): c for c, m in res.clips.items()}
    cid = res.clips.get(a.clip) and a.clip or by_name.get(os.path.basename(a.clip).lower())
    if not cid:
        sys.exit(f"clip not found in result: {a.clip}")
    g = next((g for g in res.groups if cid in g.maps), None)
    if not g:
        sys.exit("clip is not in a synced group")
    target = g.anchor
    if a.to:
        target = res.clips.get(a.to) and a.to or by_name.get(os.path.basename(a.to).lower())
        if target not in g.maps:
            sys.exit("--to clip is not in the same group")
    # clip time -> target clip time
    m = g.maps[target].inverse().compose(g.maps[cid])
    src = res.clips[cid]
    out = a.out or os.path.splitext(src.path)[0] + f"_synced_to_{os.path.splitext(os.path.basename(res.clips[target].path))[0]}.wav"
    print(export_corrected_audio(src.path, out, m, res.clips[target].duration))
    print(f"ratio {m.ratio:.9f} ({(m.ratio - 1) * 1e6:+.1f} ppm), starts at {m(0.0):+.4f} s of target",
          file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="showsync", description="Sync live-music footage by sound.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("analyze", help="analyse clips and write a result JSON")
    s.add_argument("paths", nargs="*", help="files or folders")
    s.add_argument("--paths-file", help="text file with one file/folder path per line")
    s.add_argument("--refs", help="folder (or file) of reference songs")
    s.add_argument("--out", help="result JSON path")
    s.add_argument("--report", help="HTML report path")
    s.add_argument("--hints", help="hints.json with your sync points, song tags and relations")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--jsonl-progress", action="store_true")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("conform", help="convert variable frame rate video to constant")
    s.add_argument("file")
    s.add_argument("--out")
    s.add_argument("--fps", type=float)
    s.set_defaults(fn=cmd_conform)

    s = sub.add_parser("fix-audio", help="write drift-corrected audio for one clip")
    s.add_argument("result")
    s.add_argument("clip")
    s.add_argument("--to")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_fix_audio)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
