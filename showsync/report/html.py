"""Self-contained HTML report for an AnalysisResult (inline CSS / JS / SVG, no network)."""
from __future__ import annotations

import math
import os
from html import escape as esc
from typing import Optional

from showsync.graph.groups import seconds_to_tc, tc_is_drop, tc_to_seconds
from showsync.models import AnalysisResult, EventGroup, MediaInfo, PairMatch, SongSegment

N_DEVICE_COLORS = 8
N_SONG_COLORS = 6

# --------------------------------------------------------------------------- formatting


def fmt_time(s: float, ms: bool = True) -> str:
    sign = "-" if s < 0 else ""
    s = abs(s)
    total_ms = int(round(s * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    sec, milli = divmod(rem, 1000)
    base = f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
    return sign + base + (f".{milli:03d}" if ms else "")


def _name(r: AnalysisResult, cid: str) -> str:
    i = r.clips.get(cid)
    return os.path.basename(i.path) if i is not None and i.path else cid


def _tc_at(g: EventGroup, r: AnalysisResult, event_t: float) -> Optional[str]:
    """Absolute timecode of an event-time instant, when the anchor carried real TC."""
    ai = r.clips.get(g.anchor)
    if not g.anchor_tc or ai is None or ai.fps <= 0:
        return None
    try:
        return seconds_to_tc(tc_to_seconds(g.anchor_tc, ai.fps) + event_t, ai.fps,
                             drop=tc_is_drop(g.anchor_tc, ai.fps))
    except ValueError:
        return None


def _clip_fps(r: AnalysisResult, g: EventGroup, cid: str) -> float:
    i = r.clips.get(cid)
    if i is not None and i.fps > 0:
        return i.fps
    a = r.clips.get(g.anchor)
    return a.fps if a is not None and a.fps > 0 else 25.0


def drift_stats(g: EventGroup, r: AnalysisResult, cid: str) -> dict:
    """Drift of a clip against the anchor: ppm, accumulated ms over the clip, frames."""
    m = g.maps[cid]
    i = r.clips.get(cid)
    dur = i.duration if i is not None and i.duration > 0 else (g.spans[cid][1] - g.spans[cid][0])
    acc = float(m(dur) - m(0.0)) - dur           # event seconds minus clip seconds
    ppm = acc / dur * 1e6 if dur > 0 else 0.0
    fps = _clip_fps(r, g, cid)
    return {"ppm": ppm, "ms": acc * 1000.0, "frames": acc * fps, "fps": fps, "dur": dur}


def _clip_edges(g: EventGroup, cid: str) -> list[PairMatch]:
    return [e for e in g.edges if cid in (e.a, e.b)]


def _nice_step(span: float, target: int = 8) -> float:
    raw = max(span, 1e-6) / target
    mag = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 5, 10):
        if raw <= k * mag:
            return k * mag
    return 10 * mag


# --------------------------------------------------------------------------- device colours

def _device_index(r: AnalysisResult) -> dict[str, int]:
    """Stable colour slot per device; clips without a device get their own slot by clip."""
    idx: dict[str, int] = {}
    for cid, i in r.clips.items():
        key = i.device or f"clip:{cid}"
        if key not in idx:
            idx[key] = len(idx) % N_DEVICE_COLORS
    return idx


def _kind(i: Optional[MediaInfo]) -> str:
    if i is None:
        return "clip"
    if getattr(i, "role", "clip") == "reference":
        return "reference"
    if not i.has_video:
        return "audio recorder"
    return "camera"


# --------------------------------------------------------------------------- song lookups

def _song_info(r: AnalysisResult):
    songs = {s.id: s for s in r.songs}
    order = {s.id: n for n, s in enumerate(r.songs)}
    return songs, order


def _seg_label(r: AnalysisResult, seg: SongSegment) -> str:
    songs, _ = _song_info(r)
    s = songs.get(seg.song_id) if seg.song_id else None
    return s.name if s else "unassigned music"


# --------------------------------------------------------------------------- group chart

LABEL_W = 190
CHART_W = 1000
PAD_R = 16
LANE_H = 26
LANE_GAP = 6
AXIS_H = 30
BAND_H = 22


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def group_svg(g: EventGroup, r: AnalysisResult, dev_idx: dict[str, int]) -> str:
    members = sorted(g.spans, key=lambda c: (g.spans[c][0], c))
    dur = max(g.duration, 1e-6)
    plot_w = CHART_W - LABEL_W - PAD_R
    sx = lambda t: LABEL_W + t / dur * plot_w
    segs = [s for s in r.segments if s.owner_kind == "group" and s.owner == g.id]
    _, sorder = _song_info(r)
    top = AXIS_H + (BAND_H + 4 if segs else 0)
    h = top + len(members) * (LANE_H + LANE_GAP) + 8
    out = [f'<svg class="chart" viewBox="0 0 {CHART_W} {h}" role="img" '
           f'aria-label="Lane chart of group {esc(g.id)}" preserveAspectRatio="xMinYMin meet">']

    # axis + grid
    step = _nice_step(dur)
    t = 0.0
    while t <= dur + 1e-9:
        x = sx(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{AXIS_H - 6}" x2="{x:.1f}" y2="{h - 4}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{AXIS_H - 12}" text-anchor="middle">'
                   f'{fmt_time(t, ms=False)}</text>')
        t += step

    # song bands behind the bars, label strip on top
    for s in segs:
        c = sorder.get(s.song_id, 0) % N_SONG_COLORS if s.song_id else N_SONG_COLORS
        x0, x1 = sx(max(s.start, 0)), sx(min(s.end, dur))
        tip = (f"{_seg_label(r, s)}\n{fmt_time(s.start)} - {fmt_time(s.end)}"
               f"\nmatch {s.match_score:.2f}" + ("\npartial (snippet)" if s.is_partial else ""))
        out.append(f'<g class="tipper" data-tip="{esc(tip)}">'
                   f'<rect class="band s{c}" x="{x0:.1f}" y="{AXIS_H}" width="{max(x1 - x0, 1):.1f}" '
                   f'height="{h - AXIS_H - 4}"/>'
                   f'<rect class="bandtag s{c}" x="{x0:.1f}" y="{AXIS_H}" width="{max(x1 - x0, 1):.1f}" height="{BAND_H}" rx="3"/>'
                   f'<text class="bandtxt" x="{x0 + 5:.1f}" y="{AXIS_H + 15}">'
                   f'{esc(_trunc(_seg_label(r, s), max(int((x1 - x0) / 6.5), 3)))}</text></g>')

    # lanes
    for n, cid in enumerate(members):
        y = top + n * (LANE_H + LANE_GAP)
        i = r.clips.get(cid)
        s0, s1 = g.spans[cid]
        ds = drift_stats(g, r, cid)
        edges = _clip_edges(g, cid)
        conf = max((e.confidence for e in edges), default=None)
        nonlin = not g.maps[cid].is_linear
        slot = dev_idx.get((i.device if i and i.device else f"clip:{cid}"), 0)
        etc = _tc_at(g, r, s0)
        tip = [_name(r, cid) + ("  (anchor)" if cid == g.anchor else ""),
               f"{_kind(i)}" + (f" - {i.device}" if i and i.device else ""),
               f"event {fmt_time(s0)} - {fmt_time(s1)}" + (f"\ntimecode {etc}" if etc else ""),
               f"drift {ds['ppm']:+.0f} ppm, {ds['ms']:+.1f} ms ({ds['frames']:+.2f} fr) over the clip",
               "confidence " + (f"{conf:.2f}" if conf is not None else "n/a (anchor only)"),
               "NONLINEAR sync (piecewise map)" if nonlin else "linear sync"]
        w = max(sx(s1) - sx(s0), 2)
        out.append(f'<text class="lane" x="{LABEL_W - 8}" y="{y + LANE_H / 2 + 4:.1f}" text-anchor="end">'
                   f'{esc(_trunc(_name(r, cid), 28))}{"  [ref]" if _kind(i) == "reference" else ""}</text>')
        alts = [float(a) for a in (getattr(g, "alt_shifts", None) or {}).get(cid, [])]
        if alts:
            tip.append("also fits at " + ", ".join(f"{a:+.1f} s" for a in alts) + " (repeated material)")
        out.append(f'<g class="tipper" data-tip="{esc(chr(10).join(tip))}">')
        for a in alts:                                   # ghost outlines at alternative positions
            gx0, gx1 = sx(s0 + a), sx(s1 + a)
            gx0c, gx1c = max(gx0, LABEL_W), min(gx1, LABEL_W + plot_w)
            if gx1c > gx0c:
                out.append(f'<rect class="ghost" x="{gx0c:.1f}" y="{y}" width="{gx1c - gx0c:.1f}" '
                           f'height="{LANE_H}" rx="4"/>')
        out.append(f'<rect class="bar d{slot}{" anchor" if cid == g.anchor else ""}{" nl" if nonlin else ""}'
                   f'{" ref" if _kind(i) == "reference" else ""}" '
                   f'x="{sx(s0):.1f}" y="{y}" width="{w:.1f}" height="{LANE_H}" rx="4"/>')
        for s in segs:                                   # song stripes on top of the bar
            a, b = max(s.start, s0), min(s.end, s1)
            if b > a:
                c = sorder.get(s.song_id, 0) % N_SONG_COLORS if s.song_id else N_SONG_COLORS
                out.append(f'<rect class="stripe s{c}" x="{sx(a):.1f}" y="{y + LANE_H - 6}" '
                           f'width="{max(sx(b) - sx(a), 1):.1f}" height="6"/>')
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- sections

def _cards(r: AnalysisResult) -> str:
    refs = [c for c, i in r.clips.items() if getattr(i, "role", "clip") == "reference"]
    loners = [c for c in r.loners if c not in refs]
    total = sum(i.duration for i in r.clips.values() if i.role != "reference")
    items = [("Clips", len(r.clips) - len(refs), f"{fmt_time(total, ms=False)} of footage" + (f", +{len(refs)} reference" if refs else "")),
             ("Groups", len(r.groups), "exact sync"),
             ("Loners", len(loners), "synced to nothing"),
             ("Songs", len(r.songs), f"{len(r.segments)} segments"),
             ("Warnings", len(r.warnings) + sum(len(g.conflicts) for g in r.groups), "incl. conflicts")]
    cls = lambda lab, n: " warn" if lab == "Warnings" and n else ""
    return '<div class="cards">' + "".join(
        f'<div class="card{cls(lab, n)}"><div class="n">{n}</div><div class="l">{lab}</div>'
        f'<div class="s">{esc(sub)}</div></div>' for lab, n, sub in items) + "</div>"


def _group_section(g: EventGroup, r: AnalysisResult, dev_idx) -> str:
    members = sorted(g.spans, key=lambda c: (g.spans[c][0], c))
    head = (f'<h3><span class="gid">{esc(g.id)}</span> {len(members)} clips, '
            f'{fmt_time(g.duration, ms=False)} long</h3>')
    meta = [f"anchor: <b>{esc(_name(r, g.anchor))}</b>"]
    if g.anchor_tc:
        meta.append(f"timecode at event 0: <b>{esc(g.anchor_tc)}</b>")
    meta.append(f"{len(g.edges)} sync links")
    out = [f'<section class="group">{head}<p class="meta">{" &middot; ".join(meta)}</p>']

    # legend
    seen, leg = set(), []
    for c in members:
        i = r.clips.get(c)
        key = i.device if i and i.device else f"clip:{c}"
        if key in seen:
            continue
        seen.add(key)
        label = i.device if i and i.device else _name(r, c)
        leg.append(f'<span class="lg"><i class="sw d{dev_idx.get(key, 0)}"></i>{esc(label)}</span>')
    out.append(f'<div class="legend">{"".join(leg)}'
               f'<span class="lg dim">hover a bar for details &middot; stripes and bands are songs</span></div>')
    out.append(f'<div class="chartwrap">{group_svg(g, r, dev_idx)}</div>')

    # drift table
    rows = []
    for c in members:
        i = r.clips.get(c)
        ds = drift_stats(g, r, c)
        edges = _clip_edges(g, c)
        conf = max((e.confidence for e in edges), default=None)
        nonlin = not g.maps[c].is_linear
        s0 = g.spans[c][0]
        tc = _tc_at(g, r, s0)
        big = abs(ds["frames"]) >= 0.5
        alts = (getattr(g, "alt_shifts", None) or {}).get(c, [])
        alt_html = ("<div class=\"pill-warn\">also fits at " + ", ".join(f"{float(a):+.1f} s" for a in alts) + "</div>") if alts else ""
        rows.append(
            f'<tr><td>{esc(_name(r, c))}{" <span class=pill>anchor</span>" if c == g.anchor else ""}</td>'
            f'<td>{esc(_kind(i))}</td>'
            f'<td class="num">{fmt_time(s0)}</td><td class="num">{esc(tc) if tc else "&ndash;"}</td>'
            f'<td class="num">{ds["ppm"]:+.0f}</td>'
            f'<td class="num{" hot" if big else ""}">{ds["ms"]:+.1f} ms / {ds["frames"]:+.2f} fr</td>'
            f'<td class="num">{f"{conf:.2f}" if conf is not None else "&ndash;"}</td>'
            f'<td>{"<span class=pill-warn>nonlinear</span>" if nonlin else "linear"}{alt_html}</td></tr>')
    out.append('<details open><summary>Drift and offsets</summary><div class="tablewrap"><table>'
               '<thead><tr><th>Clip</th><th>Kind</th><th>Event start</th><th>Timecode</th>'
               '<th>Drift ppm</th><th>Accumulated over clip</th><th>Confidence</th><th>Sync</th></tr></thead>'
               f'<tbody>{"".join(rows)}</tbody></table></div></details>')

    amb = [e for e in g.edges if getattr(e, "ambiguous", False)]
    if amb:
        lis = []
        for e in amb:
            alts = ", ".join(f"{a.offset - e.map.offset:+.1f} s" for a in e.alternatives) if e.alternatives else ""
            lis.append(f"<li>{esc(_name(r, e.a))} / {esc(_name(r, e.b))}: several equally good placements"
                       + (f" (other fits: {esc(alts)})" if alts else "") + (f". {esc(e.note)}" if e.note else "") + "</li>")
        out.append('<div class="conflicts"><h4>Ambiguous matches</h4><ul>' + "".join(lis) + "</ul></div>")
    if g.conflicts:
        out.append('<div class="conflicts"><h4>Conflicts</h4><ul>' +
                   "".join(f"<li>{esc(c)}</li>" for c in g.conflicts) + "</ul></div>")
    out.append("</section>")
    return "".join(out)


def _owner_text(r: AnalysisResult, seg: SongSegment) -> str:
    if seg.owner_kind == "group":
        return f"{seg.owner} (all angles)"
    return _name(r, seg.owner)


def _songs_section(r: AnalysisResult) -> str:
    by_seg = {s.id: s for s in r.segments}
    parts = ['<section><h2>Songs</h2>']
    if not r.songs:
        parts.append('<p class="dim">No songs were found.</p></section>')
        return "".join(parts)
    _, sorder = _song_info(r)
    rows = []
    for s in r.songs:
        segs = [by_seg[i] for i in s.segments if i in by_seg]
        c = sorder[s.id] % N_SONG_COLORS
        for n, seg in enumerate(segs):
            first = (f'<td rowspan="{len(segs)}"><i class="sw s{c}"></i><b>{esc(s.name)}</b>'
                     f'<div class="dim">{fmt_time(s.duration, ms=False)} &middot; '
                     f'{esc("reference file" if s.reference.startswith("file:") else "master take")}</div></td>') if n == 0 else ""
            rows.append(
                f"<tr>{first}<td>{esc(_owner_text(r, seg))}</td>"
                f'<td class="num">{fmt_time(seg.start, ms=False)} - {fmt_time(seg.end, ms=False)}</td>'
                f'<td class="num">{seg.match_score:.2f}</td>'
                f'<td>{"<span class=pill-warn>partial</span>" if seg.is_partial else "full"}</td></tr>')
        if not segs:
            rows.append(f'<tr><td><b>{esc(s.name)}</b></td><td colspan="4" class="dim">no takes</td></tr>')
    parts.append('<div class="tablewrap"><table><thead><tr><th>Song</th><th>Take (owner)</th>'
                 '<th>Span</th><th>Match</th><th>Coverage</th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
    un = [s for s in r.segments if not s.song_id]
    if un:
        parts.append(f'<p class="dim">{len(un)} music segment(s) matched no song.</p>')
    parts.append("</section>")
    return "".join(parts)


def _loner_reason(r: AnalysisResult, cid: str) -> str:
    i = r.clips.get(cid)
    if i is None:
        return "Unknown clip."
    if not i.has_audio:
        return "No audio track, so it cannot be matched by sound."
    if i.duration < 8:
        return "Very short clip and no match found; try a clearer snippet or sync it by hand."
    return ("No exact match with any other clip. It is probably another event, or the sound was "
            "too different (distant mic, heavy clipping) to line up.")


def _loners_section(r: AnalysisResult) -> str:
    loners = [c for c in r.loners if getattr(r.clips.get(c), "role", "clip") != "reference"]
    if not loners:
        return ""
    by_clip: dict[str, list[SongSegment]] = {}
    for s in r.segments:
        if s.owner_kind == "clip":
            by_clip.setdefault(s.owner, []).append(s)
    rows = []
    for cid in loners:
        i = r.clips.get(cid)
        songs = sorted({_seg_label(r, s) for s in by_clip.get(cid, []) if s.song_id})
        rows.append(f"<tr><td>{esc(_name(r, cid))}</td>"
                    f'<td class="num">{fmt_time(i.duration, ms=False) if i else "&ndash;"}</td>'
                    f"<td>{esc(_loner_reason(r, cid))}</td>"
                    f'<td>{esc(", ".join(songs)) if songs else "&ndash;"}</td></tr>')
    return ('<section><h2>Loners</h2><p class="dim">Clips that line up with nothing exactly. '
            'They may still belong to a song.</p><div class="tablewrap"><table><thead><tr>'
            '<th>Clip</th><th>Length</th><th>Why</th><th>Songs</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></section>')


# --------------------------------------------------------------------------- page

_CSS = """
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1c1b19;--dim:#6b6860;--line:#e3e0d8;--grid:#e8e5de;
--accent:#c2410c;--warnbg:#fff4e5;--warnink:#8a4b00;--shadow:0 1px 2px rgba(0,0,0,.06);
--d0:#2f6fb0;--d1:#d1692b;--d2:#3d9a6b;--d3:#8a58b8;--d4:#c0425e;--d5:#b09a1e;--d6:#2a9aa6;--d7:#7a7f87;
--s0:#e0b13a;--s1:#4aa3a3;--s2:#c76f9a;--s3:#7d8fd6;--s4:#8cad3f;--s5:#a8a49b}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--panel:#201f1d;--ink:#ecebe6;--dim:#9b978d;
--line:#34322e;--grid:#2c2a27;--accent:#f08a4b;--warnbg:#3a2a12;--warnink:#f2c27a;--shadow:none;
--d0:#5b9bd8;--d1:#ef8f4f;--d2:#5fc08d;--d3:#b189de;--d4:#e5738c;--d5:#d8c24a;--d6:#55c0cc;--d7:#a1a6ad;
--s0:#f0c657;--s1:#62c0c0;--s2:#e58ab5;--s3:#9aabee;--s4:#a6c85a;--s5:#bdb9b0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 12px}
h3{font-size:16px;margin:0 0 4px}h4{margin:0 0 6px}
.sub{color:var(--dim);margin:0 0 22px}.dim{color:var(--dim);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;box-shadow:var(--shadow)}
.card .n{font-size:30px;font-weight:650;line-height:1.1}.card .l{font-weight:600}.card .s{color:var(--dim);font-size:12.5px}
.card.warn{border-color:var(--accent)}.card.warn .n{color:var(--accent)}
.group{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 18px 14px;margin:18px 0;box-shadow:var(--shadow)}
.gid{background:var(--ink);color:var(--bg);border-radius:6px;padding:1px 8px;font-size:13px;margin-right:6px}
.meta{margin:2px 0 10px;color:var(--dim);font-size:13.5px}.meta b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:6px 0 8px;font-size:13px}
.lg{display:inline-flex;align-items:center;gap:6px}.lg.dim{margin-left:auto}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.lg .sw{margin-right:0}
.chartwrap{overflow-x:auto}.chart{width:100%;min-width:640px;height:auto;display:block}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .tick{fill:var(--dim);font-size:11px}
.chart .lane{fill:var(--ink);font-size:12px}
.chart .bar{stroke:rgba(0,0,0,.25);stroke-width:1}.chart .bar.anchor{stroke:var(--ink);stroke-width:2}
.chart .bar.nl{stroke-dasharray:4 2;stroke:var(--accent);stroke-width:2}
.chart .band{opacity:.13}.chart .bandtag{opacity:.85}.chart .bandtxt{fill:#111;font-size:11.5px;font-weight:600;pointer-events:none}
.chart .stripe{opacity:.95}
.chart .ghost{fill:none;stroke:var(--ink);stroke-width:1.5;stroke-dasharray:3 3;opacity:.55}
.chart .bar.ref{fill-opacity:.55;stroke-dasharray:1 3;stroke:var(--ink)}
.tipper{cursor:default}.tipper:hover .bar{filter:brightness(1.12)}
.d0{fill:var(--d0);background:var(--d0)}.d1{fill:var(--d1);background:var(--d1)}.d2{fill:var(--d2);background:var(--d2)}
.d3{fill:var(--d3);background:var(--d3)}.d4{fill:var(--d4);background:var(--d4)}.d5{fill:var(--d5);background:var(--d5)}
.d6{fill:var(--d6);background:var(--d6)}.d7{fill:var(--d7);background:var(--d7)}
.s0{fill:var(--s0);background:var(--s0)}.s1{fill:var(--s1);background:var(--s1)}.s2{fill:var(--s2);background:var(--s2)}
.s3{fill:var(--s3);background:var(--s3)}.s4{fill:var(--s4);background:var(--s4)}.s5{fill:var(--s5);background:var(--s5)}
.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
border-bottom:1px solid var(--line);padding:6px 10px;white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.hot{color:var(--accent);font-weight:600}
.pill{background:var(--line);border-radius:10px;padding:0 7px;font-size:11px;margin-left:4px}
.pill-warn{background:var(--warnbg);color:var(--warnink);border-radius:10px;padding:0 8px;font-size:12px}
details{margin-top:10px}summary{cursor:pointer;font-weight:600;color:var(--dim);margin-bottom:4px}
.conflicts,.warnings{background:var(--warnbg);color:var(--warnink);border-radius:8px;padding:10px 14px;margin-top:12px}
.conflicts ul,.warnings ul{margin:4px 0 0;padding-left:18px}
#tip{position:fixed;z-index:9;pointer-events:none;background:#111;color:#f2f2f2;padding:8px 10px;border-radius:7px;
font-size:12.5px;line-height:1.45;white-space:pre-line;max-width:360px;display:none;box-shadow:0 4px 14px rgba(0,0,0,.35)}
@media (max-width:640px){main{padding:18px 14px 60px}.lg.dim{margin-left:0}}
"""

_JS = """
(function(){var t=document.getElementById('tip');
document.addEventListener('mousemove',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
if(!el){t.style.display='none';return;}t.textContent=el.getAttribute('data-tip');t.style.display='block';
var x=e.clientX+14,y=e.clientY+16;var w=t.offsetWidth,h=t.offsetHeight;
if(x+w>innerWidth-8)x=e.clientX-w-14;if(y+h>innerHeight-8)y=e.clientY-h-14;
t.style.left=Math.max(4,x)+'px';t.style.top=Math.max(4,y)+'px';});
document.addEventListener('mouseleave',function(){t.style.display='none';});})();
"""


def write_report(result: AnalysisResult, out_path: str) -> str:
    r = result
    dev_idx = _device_index(r)
    body = [f"<h1>ShowSync report</h1>",
            f'<p class="sub">{len(r.clips)} clips, {len(r.groups)} exact-sync groups, '
            f'{len(r.songs)} songs</p>', _cards(r)]
    if r.warnings:
        body.append('<div class="warnings"><h4>Warnings</h4><ul>' +
                    "".join(f"<li>{esc(w)}</li>" for w in r.warnings) + "</ul></div>")
    body.append("<h2>Event groups</h2>")
    if r.groups:
        body.extend(_group_section(g, r, dev_idx) for g in r.groups)
    else:
        body.append('<p class="dim">No clips lined up exactly.</p>')
    body.append(_songs_section(r))
    body.append(_loners_section(r))
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark"><title>ShowSync report</title>'
            f"<style>{_CSS}</style></head><body><main>{''.join(body)}</main>"
            f'<div id="tip"></div><script>{_JS}</script></body></html>')
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path
