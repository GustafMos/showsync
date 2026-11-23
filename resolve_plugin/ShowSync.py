"""ShowSync - DaVinci Resolve Studio menu script (Workspace > Scripts > Utility > ShowSync).

Runs the analysis engine as a subprocess (stdlib only here), shows the results and applies them.
The pure logic lives in showsync_core.py / resolve_ops.py; this file is only the window.
"""
import os
import queue
import subprocess
import sys
import threading
import time
import traceback


# ---- make the sibling modules importable (Resolve may exec this file without __file__) ----------
def _bootstrap_path():
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    cands.append(os.path.join(appdata, "ShowSync", "plugin"))
    try:
        import json
        with open(os.path.join(appdata, "ShowSync", "settings.json"), "r", encoding="utf-8") as f:
            pd = json.load(f).get("plugin_dir")
        if pd:
            cands.insert(0, pd)
    except Exception:
        pass
    for c in cands:
        if os.path.isfile(os.path.join(c, "showsync_core.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


_bootstrap_path()

import showsync_core as core            # noqa: E402
import resolve_ops as ops               # noqa: E402

TITLE = "ShowSync"


# ---- source collection -------------------------------------------------------------------------
def collect_paths(layer, project, mode):
    """Return (paths, skipped_count) for the chosen source mode."""
    clips, paths, skipped = collect_clips(layer, project, mode)
    return paths, skipped


def collect_clips(layer, project, mode):
    """Return (clip objects, paths, skipped_count); video-only and audio-less clips are included."""
    pool = project.GetMediaPool()
    clips = []
    if mode == 0:
        clips = pool.GetSelectedClips() or []
    elif mode == 1:
        clips = list(layer._walk(pool.GetCurrentFolder()))
    else:
        tl = project.GetCurrentTimeline()
        if tl is not None:
            for ttype in ("video", "audio"):
                for i in range(1, tl.GetTrackCount(ttype) + 1):
                    for it in tl.GetItemListInTrack(ttype, i) or []:
                        mp = it.GetMediaPoolItem()
                        if mp is not None:
                            clips.append(mp)
    seen, paths, skipped, objs = set(), [], 0, []
    for c in clips:
        p = c.GetClipProperty("File Path")
        if not p or not os.path.exists(p):
            skipped += 1
            continue
        k = core.norm_path(p)
        if k not in seen:
            seen.add(k)
            paths.append(p)
            objs.append(c)
    return objs, paths, skipped


class Analysis:
    """Engine subprocess + reader thread; polled from the UI timer."""

    def __init__(self, settings, paths, refs, out_json, report_html, hints_file=""):
        py, root = settings["engine_python"], settings["engine_root"]
        if not os.path.exists(py):
            raise RuntimeError("engine python not found: %s (edit %s)" % (py, core.settings_path()))
        # clip list goes through a file: no command-line length limit for big shows
        list_path = os.path.splitext(out_json)[0] + "_clips.txt"
        with open(list_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(paths))
        cmd = [py, "-m", "showsync", "analyze", "--paths-file", list_path]
        if refs:
            cmd += ["--refs", refs]
        if hints_file and os.path.exists(hints_file):
            cmd += ["--hints", hints_file]
        cmd += ["--out", out_json, "--report", report_html, "--jsonl-progress"]
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        self.q = queue.Queue()
        self.tail = []
        self.out_json, self.report_html = out_json, report_html
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.q.put(raw.decode("utf-8", "replace").rstrip())
        self.q.put(None)

    def poll(self):
        """Returns (progress_text or None, finished bool, returncode)."""
        text, finished = None, False
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                finished = True
                break
            pr = core.parse_progress(line)
            if pr:
                total = pr.get("total") or 0
                frac = " %d/%d" % (pr.get("done", 0), total) if total else ""
                text = "%s%s %s" % (pr["stage"], frac, pr.get("msg", ""))
            elif line:
                self.tail = (self.tail + [line])[-12:]
        rc = self.proc.poll() if finished else None
        if finished and rc is None:
            rc = self.proc.wait()
        return text, finished, rc

    def cancel(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


# ---- the window --------------------------------------------------------------------------------
def main():
    res_obj = globals().get("resolve")
    fu = globals().get("fusion") or globals().get("fu")
    bmd_ = globals().get("bmd")
    if res_obj is None or fu is None or bmd_ is None:
        print("ShowSync must be started from DaVinci Resolve (Workspace > Scripts).")
        return
    project = res_obj.GetProjectManager().GetCurrentProject()
    if project is None:
        print("ShowSync: no project open")
        return

    ui = fu.UIManager
    disp = bmd_.UIDispatcher(ui)
    layer = ops.ResolveLayer(res_obj, project)
    settings = core.load_settings()
    state = {"analysis": None, "result": None, "result_path": "", "report": "", "pick_cid": None,
             "pick_opts": [], "rowmap": {}}

    win = disp.AddWindow({"ID": "ShowSyncWin", "WindowTitle": TITLE, "Geometry": [200, 120, 980, 700]},
        ui.VGroup([
            ui.HGroup({"Weight": 0}, [
                ui.Label({"Text": "Source", "Weight": 0}),
                ui.ComboBox({"ID": "Source", "Weight": 1}),
                ui.Label({"ID": "SrcCount", "Text": "", "Weight": 1}),
            ]),
            ui.HGroup({"Weight": 0}, [
                ui.Label({"Text": "Reference songs folder (optional)", "Weight": 0}),
                ui.LineEdit({"ID": "Refs", "Text": settings.get("last_refs_dir", ""), "Weight": 1}),
                ui.Button({"ID": "Browse", "Text": "Browse...", "Weight": 0}),
            ]),
            ui.Label({"ID": "EngineInfo", "Weight": 0}),
            ui.HGroup({"Weight": 0}, [
                ui.Button({"ID": "Analyze", "Text": "Analyze", "Weight": 0}),
                ui.Button({"ID": "Cancel", "Text": "Cancel", "Weight": 0, "Enabled": False}),
                ui.Button({"ID": "Refresh", "Text": "Refresh", "Weight": 0}),
                ui.Label({"ID": "Progress", "Text": "idle", "Weight": 1}),
            ]),
            ui.Tree({"ID": "Results", "Weight": 1, "SortingEnabled": False, "UniformRowHeights": True}),
            ui.HGroup({"Weight": 0}, [
                ui.CheckBox({"ID": "ChkEvent", "Text": "Event timelines", "Checked": True}),
                ui.CheckBox({"ID": "ChkSong", "Text": "Song timelines", "Checked": True}),
                ui.CheckBox({"ID": "ChkMarkers", "Text": "Song markers", "Checked": True}),
                ui.Label({"Text": "Timecode", "Weight": 0}),
                ui.ComboBox({"ID": "TcMode", "Weight": 0}),
                ui.CheckBox({"ID": "ChkProp", "Text": "Propagate markers from selected clip", "Checked": False}),
            ]),
            ui.HGroup({"Weight": 0}, [
                ui.Label({"Text": "Song timelines: Handles (s)", "Weight": 0}),
                ui.LineEdit({"ID": "Handles", "Text": "10", "Weight": 0, "MinimumSize": [50, 0]}),
                ui.CheckBox({"ID": "ChkWhole", "Text": "Whole clips", "Checked": False}),
            ]),
            ui.HGroup({"Weight": 0}, [
                ui.CheckBox({"ID": "ChkMaster", "Text": "Master-night song timelines", "Checked": False}),
                ui.Label({"Text": "master (group id / clip id, blank = longest)", "Weight": 0}),
                ui.LineEdit({"ID": "MasterId", "Text": "", "Weight": 1}),
                ui.Label({"Text": "max warp %", "Weight": 0}),
                ui.LineEdit({"ID": "MaxWarp", "Text": "5", "Weight": 0, "MinimumSize": [40, 0]}),
                ui.Label({"Text": "other audio", "Weight": 0}),
                ui.ComboBox({"ID": "OtherAudio", "Weight": 0}),
                ui.CheckBox({"ID": "ChkMulticam", "Text": "Multicam clip", "Checked": False}),
            ]),
            ui.HGroup({"Weight": 0}, [
                ui.Label({"ID": "HintCount", "Text": "Hints: 0", "Weight": 0}),
                ui.Button({"ID": "SameSong", "Text": "Same song"}),
                ui.Button({"ID": "DiffSong", "Text": "Different songs"}),
                ui.Button({"ID": "DiffShow", "Text": "Different shows"}),
                ui.Button({"ID": "IgnoreClip", "Text": "Ignore clip"}),
                ui.LineEdit({"ID": "SongTitle", "PlaceholderText": "song title", "Weight": 1}),
                ui.Button({"ID": "TagSong", "Text": "Tag selected clips as song"}),
                ui.Button({"ID": "ClearHints", "Text": "Clear hints"}),
            ]),
            ui.HGroup({"Weight": 0}, [
                ui.Label({"Text": "Repeat-ambiguous take, placement", "Weight": 0}),
                ui.ComboBox({"ID": "PickOpt", "Weight": 1}),
                ui.Button({"ID": "Preview", "Text": "Preview"}),
                ui.Button({"ID": "UsePick", "Text": "Use this placement"}),
            ]),
            ui.HGroup({"Weight": 0}, [
                ui.Button({"ID": "OpenReport", "Text": "Open report", "Enabled": False}),
                ui.Button({"ID": "Apply", "Text": "Apply", "Enabled": False}),
                ui.Button({"ID": "Undo", "Text": "Undo last"}),
            ]),
            ui.TextEdit({"ID": "Status", "ReadOnly": True, "Weight": 0, "MinimumSize": [100, 110]}),
        ]))
    it = win.GetItems()
    it["Source"].AddItem("Selected media-pool clips")
    it["Source"].AddItem("Current bin (recursive)")
    it["Source"].AddItem("Clips on current timeline")
    it["OtherAudio"].AddItem("off")
    it["OtherAudio"].AddItem("muted")
    for t in ("None", "Event", "Song"):
        it["TcMode"].AddItem(t)
    it["EngineInfo"].Text = "Engine: %s   (settings: %s)" % (settings["engine_python"], core.settings_path())
    tree = it["Results"]
    cols = ["Clip", "Group", "Offset TC", "Drift ppm", "Songs", "Confidence", "Repeat-ambiguous", "Warnings"]
    tree.ColumnCount = len(cols)
    tree.SetHeaderLabels(cols)
    for i, w in enumerate((230, 60, 90, 80, 200, 80, 110, 300)):
        tree.ColumnWidth[i] = w

    def say(msg):
        try:
            it["Status"].Append(str(msg))
        except Exception:
            try:
                it["Status"].PlainText = str(msg)
            except Exception:
                pass
        print("ShowSync: " + str(msg))

    def fail(where, exc=None):
        say("%s: %s" % (where, exc))
        if exc is not None:
            print(traceback.format_exc())

    def hints_file():
        d = os.path.join(core.appdata_dir(), "runs")
        return core.hints_path_for_key(project.GetName(), d)

    def load_hints():
        return core.Hints.load(hints_file())

    def refresh_hint_count():
        it["HintCount"].Text = "Hints: %d" % load_hints().count()

    def selected_clips():
        return project.GetMediaPool().GetSelectedClips() or []

    def selected_paths():
        return [p for p in (c.GetClipProperty("File Path") for c in selected_clips()) if p]

    def show_results():
        tree.Clear()
        res = state["result"]
        fps = layer.project_fps()[0]
        state["rowmap"] = {}
        for row in core.summarize_result(res, fps):
            itm = tree.NewItem()
            for i, k in enumerate(("clip", "group", "offset_tc", "drift_ppm", "songs", "confidence", "repeat_ambiguous", "warnings")):
                itm.Text[i] = row[k]
            tree.AddTopLevelItem(itm)
            state["rowmap"][row["clip"]] = row["clip_id"]
            if row["repeat_ambiguous"]:
                try:
                    for o in core.alt_options(res, row["clip_id"], fps):
                        ch = tree.NewItem()
                        label = "   %s: %s" % (row["clip"], o["label"])
                        ch.Text[0] = label
                        itm.AddChild(ch)
                        state["rowmap"][label] = row["clip_id"]
                except Exception:
                    pass
        it["Apply"].Enabled = True
        it["OpenReport"].Enabled = bool(state["report"]) and os.path.exists(state["report"])
        r = state["result"]
        say("%d clips, %d groups, %d songs, %d loners. %s" % (
            len(r.clips), len(r.groups), len(r.songs), len(r.loners),
            ("Warnings: " + "; ".join(r.warnings)) if r.warnings else ""))

    def poll(ev=None):
        a = state["analysis"]
        if a is None:
            return
        try:
            text, finished, rc = a.poll()
            if text:
                it["Progress"].Text = text
            if finished:
                state["analysis"] = None
                it["Analyze"].Enabled = True
                it["Cancel"].Enabled = False
                if rc != 0:
                    it["Progress"].Text = "failed (exit %s)" % rc
                    say("Engine failed (exit %s):\n%s" % (rc, "\n".join(a.tail)))
                    return
                state["result"] = core.Result.load(a.out_json)
                state["result_path"], state["report"] = a.out_json, a.report_html
                it["Progress"].Text = "done"
                show_results()
        except Exception as e:
            state["analysis"] = None
            it["Analyze"].Enabled = True
            fail("poll", e)

    def on_analyze(ev):
        try:
            if state["analysis"] is not None:
                return
            clips, paths, skipped = collect_clips(layer, project, it["Source"].CurrentIndex)
            it["SrcCount"].Text = "%d clips%s" % (len(paths), (", %d without a file path skipped" % skipped) if skipped else "")
            if not paths:
                say("No usable clips: pick clips with media files first.")
                return
            refs = it["Refs"].Text.strip()
            if refs and not os.path.isdir(refs):
                say("Reference folder does not exist: " + refs)
                return
            settings["last_refs_dir"] = refs
            try:
                core.save_settings(settings)
            except OSError:
                pass
            outdir = os.path.join(core.appdata_dir(), "runs")
            os.makedirs(outdir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out_json = os.path.join(outdir, "result-%s.json" % stamp)
            report = os.path.join(outdir, "report-%s.html" % stamp)
            # hints: refresh what comes from markers, keep what the user made
            fps0 = layer.project_fps()[0]
            dumps = [layer.clip_marker_dump(c, fps0) for c in clips]
            hints = load_hints()
            tags = []
            for d in dumps:
                tags += core.song_tags_from_markers(d["path"], d["fps"], d["markers"])
            hints.merge_marker_derived(core.marker_sync_points(dumps), tags)
            hints.save(hints_file())
            refresh_hint_count()
            state["analysis"] = Analysis(settings, paths, refs, out_json, report, hints_file())
            it["Analyze"].Enabled = False
            it["Cancel"].Enabled = True
            it["Progress"].Text = "starting..."
            say("Analyzing %d clips..." % len(paths))
            if timer is not None:
                timer.Start()
        except Exception as e:
            state["analysis"] = None
            fail("Analyze", e)

    def on_cancel(ev):
        if state["analysis"]:
            state["analysis"].cancel()
            it["Progress"].Text = "cancelling..."

    def on_browse(ev):
        try:
            d = fu.RequestDir(it["Refs"].Text or os.path.expanduser("~"))
            if d:
                it["Refs"].Text = str(d)
        except Exception as e:
            fail("Browse", e)

    def on_open_report(ev):
        try:
            if state["report"] and os.path.exists(state["report"]):
                os.startfile(state["report"])
        except Exception as e:
            fail("Open report", e)

    def on_apply(ev):
        try:
            res = state["result"]
            if res is None:
                return
            fps = core.footage_fps(res, layer.project_fps()[0])   # timelines follow the footage, not the project default
            mode = ("none", "event", "song")[it["TcMode"].CurrentIndex]
            try:
                handles = max(0.0, float(it["Handles"].Text))
            except ValueError:
                say("Handles must be a number of seconds.")
                return
            try:
                warp = max(0.0, float(it["MaxWarp"].Text))
            except ValueError:
                say("Max warp must be a percentage.")
                return
            opts = core.ApplyOptions(handles_s=handles, whole_clips=it["ChkWhole"].Checked,
                                     master_mode=it["ChkMaster"].Checked,
                                     master_group=it["MasterId"].Text.strip() or None, max_warp_pct=warp,
                                     other_audio=("off", "muted")[it["OtherAudio"].CurrentIndex],
                                     multicam=it["ChkMulticam"].Checked, event_timelines=it["ChkEvent"].Checked, song_timelines=it["ChkSong"].Checked,
                                     song_markers=it["ChkMarkers"].Checked, tc_mode=mode)
            if it["ChkProp"].Checked:
                sel = project.GetMediaPool().GetSelectedClips() or []
                if len(sel) != 1:
                    say("Propagate: select exactly one clip in the media pool.")
                    return
                cid = res.clip_ids_by_path().get(core.norm_path(sel[0].GetClipProperty("File Path") or ""))
                if cid is None:
                    say("Propagate: the selected clip is not part of this analysis.")
                    return
                try:
                    sfps = float(sel[0].GetClipProperty("FPS"))
                except (TypeError, ValueError):
                    sfps = fps
                opts.propagate_source, opts.propagate_fps = cid, sfps
                opts.propagate_markers = layer.clip_markers(sel[0])
                if not opts.propagate_markers:
                    say("Propagate: the selected clip has no (non-ShowSync) markers.")
            plan = core.build_apply_plan(res, fps, opts)
            layer.refresh_index()
            say("Applying: %d timelines, %d markers, %d timecode changes (%d clips untouched)..." % (
                len(plan.timelines), len(plan.clip_markers), len(plan.tc_ops), len(plan.tc_untouched)))
            backup = core.Backup.new(state["result_path"])
            errs = ops.apply_plan(layer, plan, backup, log=say)
            say("Done. Backup: %s%s" % (backup.file, ("  (%d errors)" % len(errs)) if errs else ""))
        except Exception as e:
            fail("Apply", e)

    def on_undo(ev):
        try:
            rp = state["result_path"]
            path = core.find_latest_backup(rp) if rp else None
            if not path:
                say("No backup to undo (analyze or load a result first).")
                return
            layer.refresh_index()
            b = core.Backup.load(path)
            errs = ops.undo_backup(layer, b, log=say)
            say("Undid %s%s" % (path, ("  (%d errors)" % len(errs)) if errs else ""))
        except Exception as e:
            fail("Undo", e)

    def relate(kind, need):
        try:
            paths = selected_paths()
            if len(paths) < need:
                say("Select %s clip(s) in the media pool first." % ("at least 2" if need == 2 else "1+"))
                return
            h = load_hints()
            n = h.relate(kind, paths)
            h.save(hints_file())
            refresh_hint_count()
            say("%s: %d new hint(s). Analyze again to use them." % (kind, n))
        except Exception as e:
            fail(kind, e)

    def on_tag(ev):
        try:
            title, paths = it["SongTitle"].Text.strip(), selected_paths()
            if not title or not paths:
                say("Type a song title and select clips.")
                return
            h = load_hints()
            h.tag_song(paths, title)
            h.save(hints_file())
            refresh_hint_count()
            say("Tagged %d clip(s) as song %r. Analyze again to use it." % (len(paths), title))
        except Exception as e:
            fail("Tag", e)

    def on_clear(ev):
        try:
            h = load_hints()
            h.clear()
            h.save(hints_file())
            refresh_hint_count()
            say("Hints cleared.")
        except Exception as e:
            fail("Clear hints", e)

    def on_row(ev):
        try:
            item = ev["item"]
            cid = state["rowmap"].get(str(item.Text[0]))
            res = state["result"]
            it["PickOpt"].Clear()
            state["pick_cid"], state["pick_opts"] = None, []
            if cid and res is not None:
                opts_ = core.alt_options(res, cid, layer.project_fps()[0])
                if opts_:
                    state["pick_cid"], state["pick_opts"] = cid, opts_
                    for o in opts_:
                        it["PickOpt"].AddItem(o["label"])
        except Exception as e:
            fail("select", e)

    def chosen():
        i = it["PickOpt"].CurrentIndex
        if state["pick_cid"] is None or not (0 <= i < len(state["pick_opts"])):
            say("Select a repeat-ambiguous clip in the table first.")
            return None
        return state["pick_cid"], state["pick_opts"][i]

    def on_preview(ev):
        try:
            c = chosen()
            if not c:
                return
            cid, o = c
            res = state["result"]
            g = res.group_of(cid)
            title = core.group_title(res, res.groups.index(g))
            msg = layer.open_timeline_at(title, o["tc"])
            say(msg or "Playhead at %s on %s" % (o["tc"], title))
        except Exception as e:
            fail("Preview", e)

    def on_use_pick(ev):
        try:
            c = chosen()
            if not c:
                return
            cid, o = c
            h = load_hints()
            h.add_pick(core.pick_sync_point(state["result"], cid, o["shift"]))
            h.save(hints_file())
            refresh_hint_count()
            say("Stored placement %r for %s. Analyze again to apply it." % (o["label"], state["result"].clip_name(cid)))
        except Exception as e:
            fail("Use placement", e)

    def on_close(ev):
        a = state["analysis"]
        if a:
            a.cancel()
        disp.ExitLoop()

    timer = None
    try:
        timer = ui.Timer({"ID": "PollTimer", "Interval": 400, "Events": {"Timeout": True}})
        win.On.PollTimer.Timeout = poll
        disp.On.PollTimer.Timeout = poll
    except Exception:
        try:
            timer = ui.Timer({"ID": "PollTimer", "Interval": 400, "Events": {"Timeout": True}})
        except Exception:
            timer = None

    win.On.Analyze.Clicked = on_analyze
    win.On.Cancel.Clicked = on_cancel
    win.On.Refresh.Clicked = poll
    win.On.Browse.Clicked = on_browse
    win.On.OpenReport.Clicked = on_open_report
    win.On.Apply.Clicked = on_apply
    win.On.Undo.Clicked = on_undo
    win.On.SameSong.Clicked = lambda ev: relate("same_song", 2)
    win.On.DiffSong.Clicked = lambda ev: relate("different_song", 2)
    win.On.DiffShow.Clicked = lambda ev: relate("different_event", 2)
    win.On.IgnoreClip.Clicked = lambda ev: relate("ignore", 1)
    win.On.TagSong.Clicked = on_tag
    win.On.ClearHints.Clicked = on_clear
    win.On.Preview.Clicked = on_preview
    win.On.UsePick.Clicked = on_use_pick
    win.On.Results.ItemClicked = on_row
    refresh_hint_count()
    win.On.ShowSyncWin.Close = on_close
    win.Show()
    disp.RunLoop()
    win.Hide()


main()
