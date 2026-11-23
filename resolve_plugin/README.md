# ShowSync for DaVinci Resolve Studio

Lines up concert, live-session and music-video footage inside Resolve. The analysis runs in the
ShowSync engine (separate Python with numpy); this plugin only picks clips, starts the engine and
applies the result, using Resolve's own Python and the standard library.

## Install

    python resolve_plugin/install.py            (python resolve_plugin/install.py --uninstall to remove)

This copies `ShowSync.py` to `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility\`,
the helper modules to `%APPDATA%\ShowSync\plugin\`, and writes `%APPDATA%\ShowSync\settings.json`:

    {"engine_python": "...\\.venv\\Scripts\\python.exe", "engine_root": "...\\showsync", "plugin_dir": "..."}

Edit `engine_python` / `engine_root` if the engine lives elsewhere. Restart Resolve, then
Workspace > Scripts > Utility > ShowSync. Re-run the installer after updating the plugin files.

## Use

1. Choose the source: selected media-pool clips, the current bin (recursive), or the clips on the
   current timeline. Optionally pick a folder of reference songs.
2. Analyze. Progress shows in the window; the result table lists clip, group, offset, drift (ppm),
   songs, confidence and warnings. Open report shows the engine's HTML report.
3. Tick what to apply and press Apply:
   - Event timelines: one timeline per exact-sync group (`ShowSync G01 ...`), a track pair per clip.
     A clip with clock drift (tape captures, phones) is placed once with a speed change
     (SetSpeed, pitch correction off); only a clip whose timing wanders (Hi8) is cut into a few
     pieces, each with its own speed, so the error stays under 0.5 frame. The plan log lists
     speed, piece count and residual per clip.
   - Song timelines: `ShowSync Song - <name>`. Every camera of an exact-sync group that covers the song
     gets its own track pair (anchor first, then by coverage), with the same speed placement as the
     event timelines and exact relative offsets; separate takes (other nights, snippets) follow
     below. The placement of a group on the song timeline comes from the song map, so it is
     musical (approximate), the cameras within a group are exact. A marker at song time 0 carries
     the song name. Clips missing from the media pool get no tracks.
   - Master-night song timelines: `ShowSync Master - <song>`. Timeline time is the master take's
     time (master = a group id or loner clip id, blank = the song's longest take). The master night's
     cameras sit on the top tracks (video and audio, as in event timelines). Other nights and takes
     contribute picture only: mapped through the song maps, with speed changes within +-max warp %
     (default 5), cut and re-placed where the structure differs (extra jam bars, skipped section).
     Their audio is left out, or kept on disabled tracks ("other audio: muted"). The plan log lists
     pieces, speed range, structure cuts and residual per take. A multicam clip is not created:
     MediaPool.CreateMulticamClip only accepts media-pool clips, not a timeline.
   - Song markers on media-pool clips and group timelines (one colour per song).
   - Timecode: none / event (the anchor's time of day if it had one, else group N at hour N) / song
     (song N at hour N; only clips lying entirely inside one song are changed).
   - Propagate markers: select one clip that carries your markers; they are copied to every other
     clip exactly synced with it, or in the same song.
4. Undo last reverses the most recent apply that has not been undone.

## What gets changed, and the backup

Every apply writes `<result>.backup-<time>.json` next to the result JSON (in
`%APPDATA%\ShowSync\runs`). It records created timelines, markers added (customData `showsync:...`),
old and new Start TC, and the Comments metadata. Changing Start TC appends a line
`ShowSync orig TC=<old>` to the clip's Comments (existing text is kept). Undo deletes the timelines,
removes only markers whose customData starts with `showsync`, and restores Start TC and Comments.

Clips are matched to the media pool by their normalised file path, so every analysed file must be
imported in the current project.

## Known limits

- Needs Resolve Studio (external scripting plus the UI scripting API).
- See the notes in `resolve_ops.py` for API details that depend on live behaviour (AppendToTimeline
  end-frame convention, marker frame origin).
