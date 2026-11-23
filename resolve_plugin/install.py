"""Install / uninstall the ShowSync Resolve script.

  python resolve_plugin/install.py              install
  python resolve_plugin/install.py --uninstall  remove the script and the plugin modules (keeps settings)

ShowSync.py goes into Resolve's Utility scripts folder (so it shows in Workspace > Scripts). The helper
modules go to %APPDATA%\\ShowSync\\plugin, NOT next to it, because every .py in Utility would otherwise
appear as a menu entry. ShowSync.py finds the modules through its own folder, then settings.json
"plugin_dir", then that default folder.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ("showsync_core.py", "resolve_ops.py")
SCRIPT = "ShowSync.py"


def utility_dir(override=None):
    if override:
        return override
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "Blackmagic Design", "DaVinci Resolve", "Support", "Fusion", "Scripts", "Utility")


def appdata_dir():
    return os.environ.get("SHOWSYNC_APPDATA") or os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "ShowSync")


def install(util, force_settings=False):
    plugin = os.path.join(appdata_dir(), "plugin")
    os.makedirs(util, exist_ok=True)
    os.makedirs(plugin, exist_ok=True)
    shutil.copy2(os.path.join(HERE, SCRIPT), os.path.join(util, SCRIPT))
    for m in MODULES:
        shutil.copy2(os.path.join(HERE, m), os.path.join(plugin, m))
    root = os.path.dirname(HERE)
    py = os.path.join(root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = os.path.join(root, ".venv", "bin", "python")
    sp = os.path.join(appdata_dir(), "settings.json")
    settings = {}
    if os.path.exists(sp) and not force_settings:
        with open(sp, "r", encoding="utf-8") as f:
            settings = json.load(f)
    settings.setdefault("engine_python", py)
    settings.setdefault("engine_root", root)
    settings["plugin_dir"] = plugin
    settings.setdefault("last_refs_dir", "")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    print("script   ->", os.path.join(util, SCRIPT))
    print("modules  ->", plugin)
    print("settings ->", sp)
    if not os.path.exists(settings["engine_python"]):
        print("WARNING: engine python not found: %s (edit settings.json)" % settings["engine_python"])
    print("Restart Resolve (or reopen Workspace > Scripts) to see ShowSync.")


def uninstall(util):
    plugin = os.path.join(appdata_dir(), "plugin")
    for path in [os.path.join(util, SCRIPT)] + [os.path.join(plugin, m) for m in MODULES]:
        if os.path.exists(path):
            os.remove(path)
            print("removed", path)
    if os.path.isdir(plugin) and not os.listdir(plugin):
        os.rmdir(plugin)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--utility-dir", help="override Resolve's Utility scripts folder")
    ap.add_argument("--reset-settings", action="store_true", help="overwrite existing settings.json values")
    a = ap.parse_args()
    util = utility_dir(a.utility_dir)
    uninstall(util) if a.uninstall else install(util, a.reset_settings)
