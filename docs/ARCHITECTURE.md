# ShowSync architecture (the contract)

ShowSync lines up footage of live music: concerts, live sessions and music-video takes, from
any mix of cameras (pro cameras, phones, Hi8 captures, audio recorders). It works **with or
without** reference songs, and with full-length recordings or short snippets.

Read this before touching any module. `showsync/models.py` and `showsync/config.py` are part of
the contract; change them only through the architect.

## Two levels of "together"

1. **Exact sync (event time).** Two clips heard the *same sound*: the same show, or the same
   playback track on a music-video set. Alignment is sample-accurate and includes clock drift
   (`PairMatch`, `EventGroup`). Clips that overlap exactly form an *event group* with one
   shared event-time line.
2. **Musical match (song time).** Two clips contain the *same song* but different performances
   (another take, another night). Alignment is approximate and tempo-tolerant (`SongSegment`,
   `Song`, `song_map`). This is how takes of one song are stacked.

Snippets are first class: a 20-second phone clip can join an event group (exact) or a song
(musical) without anything else being complete.

No reference needed: songs are discovered by clustering segments against each other. When the
user gives reference audio files, they become the song's time base and name.

## Pipeline (`showsync/pipeline.py`)

```
paths ──probe──> MediaInfo ──decode(8 kHz mono, cached)──> audio ──features──> Features
          │
          ├─ pairwise exact sync (dsp.align.match_pair) over all clip pairs ─> PairMatch[]
          ├─ graph.groups.build_groups ─> EventGroup[] + loners
          ├─ music.songs.find_segments  (per group on event time, per loner on clip time)
          ├─ music.songs.assign_songs   (references if given, else clustering) ─> Song[]
          └─ AnalysisResult ─> JSON (+ report/html)
```

The Resolve side (`resolve_plugin/`) only reads `AnalysisResult` JSON and applies it. It never
imports numpy, so it runs in Resolve's own Python.

## Module contracts

### media/ (ffmpeg wrappers, stdlib + numpy)
```python
# media/probe.py
def clip_id(path: str) -> str            # sha1 of normcase(abspath) [:12]
def probe(path: str) -> MediaInfo        # ffprobe JSON; VFR: scan packet pts of first 600 video frames
# media/decode.py
def decode_audio(path: str, sr: int = ANALYSIS_SR, stream: int = 0) -> np.ndarray  # mono float32, cached
# media/cache.py
def cache_dir() -> Path                  # %LOCALAPPDATA%/ShowSync/cache (env SHOWSYNC_CACHE overrides)
def cache_key(path: str, *parts) -> str  # path + size + mtime + parts
def load_npz(key) -> dict | None ; def save_npz(key, **arrays) -> None
# media/conform.py
def conform_cfr(path, out_path, fps: float | None = None) -> str     # VFR -> CFR intermediate
def export_corrected_audio(path, out_path, m: TimeMap, duration: float, sr: int = 48000) -> str
    # writes a WAV whose sample t equals the source at m^-1... see docstring: output time = m(src time)
```
Decode multichannel by averaging channels. Never shell=True; pass argument lists.

### dsp/features.py (numpy/scipy)
```python
def compute_features(clip_id: str, audio: np.ndarray, sr: int = ANALYSIS_SR) -> Features
def onset_envelope(audio, sr) -> np.ndarray   # ENV_RATE Hz, 32 mel bands 80-1500 Hz (crowd noise lives above 1 kHz;
                                              # measured 19/20 vs 3/20 crowd-phone hits), spectral flux, whitened
def chroma(audio, sr) -> np.ndarray           # (12, n) CHROMA_RATE Hz
def musicness(chroma, energy_db_at_chroma_rate, tonal=None) -> np.ndarray  # tonal peak ratio x chroma stability x gate
```

### dsp/align.py (exact sync)
```python
def match_pair(fa: Features, fb: Features, audio_a, audio_b, sr=ANALYSIS_SR) -> PairMatch | None
```
Algorithm (must handle snippets, drift up to MAX_DRIFT, repeated choruses, different mics):
1. **Coarse:** slice b's onset envelope into COARSE_WIN_S windows every COARSE_HOP_S (one window
   if b is short). Correlate each window against all of a's envelope with FFT; normalise each
   lag by the local energy of a under the window (sliding sums) so the score is a true
   normalised cross-correlation. Keep the top 3 peaks per window (lag, score).
2. **Line fit:** RANSAC over the (t_b window centre, t_a) points with slope in
   [1-MAX_DRIFT, 1+MAX_DRIFT]. The line with most support (score-weighted) wins. Repeated
   musical material gives isolated wrong peaks; the line rejects them.
3. **Fine:** every FINE_HOP_S along the overlap, GCC-PHAT (band FINE_BAND) on raw audio, window
   FINE_WIN_S, searching ±FINE_SEARCH_S around the coarse prediction, parabolic sub-sample peak.
4. **Fit:** robust line through fine points (iteratively reweighted, inliers within INLIER_MS).
   If RMS residual > NONLINEAR_RESIDUAL_MS and enough points, fit a smoothed piecewise map with
   knots every PIECEWISE_KNOT_S (set `nonlinear=True`).
5. **Confidence** in 0..1 from coarse peak saliency, inlier fraction, fine GCC peak sharpness.
   Return None if no line is supported.

### graph/groups.py
```python
def match_all(feats: dict[str, Features], audio: dict[str, np.ndarray], infos: dict[str, MediaInfo],
              progress=None) -> list[PairMatch]
def build_groups(matches: list[PairMatch], infos: dict[str, MediaInfo]) -> tuple[list[EventGroup], list[str]]
```
Groups = connected components of accepted matches. Maximum spanning tree by confidence from the
anchor (longest clip; prefer one with real start_tc), maps composed along the tree. Every
non-tree edge is checked: disagreement > 1 video frame (or 20 ms) is reported in `conflicts`.
Same-device clips that overlap in event time are a conflict. Event time 0 = earliest covered
moment (shift all maps). If the anchor has start_tc, set `anchor_tc` to the TC at event time 0.

### music/songs.py
```python
def find_segments(owner: str, owner_kind: str, feats: Features, start_offset: float = 0.0) -> list[SongSegment]
def group_features(group: EventGroup, feats: dict[str, Features]) -> Features   # event-time fused features
def crop_features(feats, start, end, start_offset=0.0) -> Features             # seg_feats are built with this
def assign_songs(segments, seg_feats, references: dict[path, Features] | None,
                 allow_transpose=False, similar=None, ref_accept=None) -> list[Song]  # fills song_id/song_map
def propagate_points(times, from_owner, to_owner, segments, groups)             # exact via groups, else song time
def propagate_points_json(result_dict, times, from_owner, to_owner)            # stdlib twin for the plugin
```
* Segmentation: songs are stretches of high musicness; split at non-music ≥ SONG_GAP_S, and at
  strong chroma novelty (segues). Snippets shorter than SONG_MIN_S still become one partial
  segment if they hold music.
* Similarity: subsequence DTW on smoothed chroma (CENS-like, 2–5 Hz, 20 s per-pitch mean removed),
  score = 1 - cost / cost against the time-reversed reference (null model), best of 12 key transpositions only if `allow_transpose`. Score in 0..1.
* With references: each segment goes to the best reference above SONG_REF_ACCEPT; song_map from
  the DTW path (smoothed, monotonic). Without: agglomerative clustering at SONG_SIMILAR; the
  longest segment of each cluster is the master take, others map onto it.

### report/html.py
`def write_report(result: AnalysisResult, out_path: str) -> str` — one self-contained HTML file:
groups as lanes on an event-time line, drift per match (ppm and ms over the clip), conflicts and
warnings, songs with their takes. No external resources.

### resolve_plugin/ (stdlib only, runs inside Resolve)
* `ShowSync.py` — goes into `%APPDATA%/Blackmagic Design/DaVinci Resolve/Support/Fusion/Scripts/Utility/`.
  UIManager window: pick source (selected media-pool clips / current bin / current timeline),
  optional reference folder, Analyze (runs the CLI as a subprocess, shows progress), review
  table, Apply.
* `apply.py` — operations, each undoable from a backup JSON:
  - **Timelines per event group** (all angles stacked, drift handled by re-cutting an angle
    whenever accumulated error would exceed DRIFT_RESYNC_FRAC frames) and **per song** (all takes
    stacked on song time).
  - **Song markers** on media-pool clips and timelines.
  - **Timecode**: set clip Start TC to event TC (time-of-day when the anchor has it) or song TC
    (song N → hour N), original TC kept in the backup file and in clip metadata.
  - **Marker propagation**: markers the user puts on one clip are copied to every clip that is
    exactly synced (event time) or in the same song (song time).

## User hints (`models.Hints`, hints.json)

The user can always overrule the audio. Hints are hard rules, applied on every re-analysis.
Clips are named by file path in hints.json; the pipeline converts paths to clip ids.

| hint | effect |
|---|---|
| `sync_points` (a, ta, b, tb) | Pair placement fixed to that moment. If an audio match exists, the placement (map or alternative) nearest the point is chosen; otherwise `align.match_pair(..., hint=(ta, tb))` refines it sample-accurately when the clips share sound, else a `kind="manual"` edge is made from the points (ratio 1, or fitted from 2+ points; residual_ms 20 = marker precision). Clips **without audio** join groups only this way. Hint edges outrank all others in the group builder. |
| `song_tags` (clip, song, [start, end]) | Segments of that clip overlapping the range get that song; the title names the song (a title equal to a reference file's stem selects that reference). Forces a song even on weak music evidence. |
| `same_song` / `different_song` | must-link / cannot-link constraints in song clustering (segment level, via the clips' segments). |
| `different_event` | never exact-sync these two clips (their matches are discarded). |
| `ignore` | clip left out of the analysis. |

Sources in the Resolve plugin: same-named markers starting with `#` on two or more clips
(e.g. `#kick` on the drum hit) become sync points; a marker named `song: Title` (its duration
is the range) or the Tag button becomes a song tag; picking among "also fits" placements in the
review becomes a sync point with source "pick".

## Cross-show sync (musical, Nov 2026 phase)

Different nights are different performances, so they are aligned musically: the same beat of the
same bar lines up.
* `SongSegment.song_map` is beat-accurate: beats/downbeats are tracked per segment and the map
  is snapped to them (target: same downbeat within ~20 ms across nights), monotonic, tempo
  ratio limited to 0.7–1.4.
* **Musical anchors** (`assign_songs(..., anchors=[(seg_a, owner_t_a, seg_b, owner_t_b)])`):
  a `#` sync point between two clips that both have audio but don't share sound at that moment
  is routed here by `hints.apply_sync_points` instead of becoming an exact edge, so nights are
  never merged. The song maps must pass through every anchor (within one beat).
* **Master-night timeline** (plugin): the user picks a master take/night; its audio is the
  timeline audio, other nights' picture is time-warped onto it with piecewise speed changes
  within ±MAX_WARP (default 5 %), changing speed at bar/section boundaries; where structures
  differ (extra jam bars, cut chorus) the take is cut and re-placed at the next section instead
  of stretched. Optional Resolve multicam clip per song.

## CLI (`python -m showsync`)
```
showsync analyze <files/folders...> [--paths-file F] [--refs DIR] [--hints hints.json] [--out result.json] [--report report.html] [--jsonl-progress]
showsync conform <file> [--out ...]          # VFR -> CFR
showsync fix-audio <result.json> <clip> [--to <clip>] [--out ...]   # drift-corrected WAV
```
`--jsonl-progress` prints `{"stage":..., "done":n, "total":m, "msg":...}` lines for the plugin.

## Measured limits (synthetic, Nov 2026)
* Crowd-dominated phone snippets sync exactly (envelope 80-1500 Hz) but cannot be matched *musically*.
* A segue between two songs in the same key is not split.
* A chorus snippet with no exact partner may be placed on the wrong (harmonically identical) repeat.
* Snippet-vs-snippet song matching with no full take present is unreliable.

## Testing
`tests/synth.py` makes ground-truth material: songs with verse/chorus repeats, live takes at
other tempos, concerts (songs + applause + talk), camera simulations (EQ, reverb, crowd, phone
clipping, clock drift, Hi8 wow/flutter, snippets). Every module is tested against it.
Rule: report what was verified versus only written.
