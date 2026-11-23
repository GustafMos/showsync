"""Tunable constants. Modules import from here instead of hard-coding numbers.

Thresholds marked CALIBRATE are first guesses; tests/test_calibration.py measures them on
synthetic material and the values here should be updated from its output.
"""

# ---- analysis audio
ANALYSIS_SR = 8000          # mono float32 decode rate for everything (timing needs < 4 kHz)
ENV_RATE = 100              # onset / energy envelope frames per second (hop 80 samples)
CHROMA_RATE = 10            # chroma / musicness frames per second
CHROMA_NFFT = 4096          # 2 Hz bins at 8 kHz
CHROMA_FMIN = 55.0
CHROMA_FMAX = 2000.0

# ---- exact sync (dsp.align)
COARSE_WIN_S = 30.0         # window of clip b correlated against all of clip a
COARSE_HOP_S = 15.0
COARSE_MIN_OVERLAP_S = 8.0  # clips/snippets shorter than this are matched with one window
MAX_DRIFT = 0.006           # 6000 ppm: real SD tape captures measured at +2150 ppm (Nov 2026 tour footage)
FINE_WIN_S = 3.0            # GCC-PHAT window
FINE_HOP_S = 10.0           # fine measurements every N seconds of overlap
FINE_SEARCH_S = 0.15        # +- search around the coarse prediction
FINE_BAND = (150.0, 3500.0)
INLIER_MS = 2.0             # fine measurement within this of the fit counts as inlier
NONLINEAR_RESIDUAL_MS = 4.0 # above this RMS a piecewise map is fitted (Hi8 wow, VFR)
PIECEWISE_KNOT_S = 5.0      # knot spacing of piecewise maps (Hi8 wow needs dense knots)

# ---- acceptance (CALIBRATE)
MATCH_ACCEPT = 0.5          # PairMatch.confidence needed to use a match
MIN_INLIERS = 2

# ---- songs (music.songs)
SONG_MIN_S = 25.0           # shorter music stretches are not songs on their own
SONG_GAP_S = 2.5            # non-music this long splits songs
SONG_SIMILAR = 0.35        # calibrated on synthetic degraded devices (false pairs p99 0.24)
SONG_REF_ACCEPT = 0.35     # calibrated (wrong-reference max 0.335 on clean; keep watching)

# ---- timeline building (resolve side)
DRIFT_RESYNC_FRAC = 0.5     # re-cut a drifting angle when error would exceed this many frames

ANALYSIS_VERSION = 2          # 2: Features.perc added (feature cache key)
