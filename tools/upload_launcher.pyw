"""ShowSync Upload — a small window for starting phone_upload.py.

Pick the folder the files go to, tick the machines that may send (devices found on the local network and
your Tailscale devices), press Start. The window shows the link to open on the other machine, the
server's log, and offers to let in a device that tries to connect but isn't ticked yet.
Settings are remembered in %APPDATA%\\ShowSync\\upload_launcher.json.
"""
import ipaddress, json, os, queue, re, secrets, socket, subprocess, sys, threading, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "phone_upload.py")
CFG = os.path.join(os.environ.get("APPDATA", HERE), "ShowSync", "upload_launcher.json")
NOWIN = 0x08000000  # CREATE_NO_WINDOW
TAILSCALE = r"C:\Program Files\Tailscale\tailscale.exe"
_LIB = os.path.join(os.path.expanduser("~"), "Videos", "ShowSync")
DEFAULTS = {"dest": os.path.join(_LIB, "phone_uploads"), "library": _LIB,
            "port": 8765, "key": None, "allowed": {}, "whole_lan": False}


def load_cfg():
    cfg = dict(DEFAULTS)
    try:
        with open(CFG, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except (OSError, ValueError):
        pass
    if not cfg.get("key"):
        cfg["key"] = secrets.token_urlsafe(9)
    return cfg


def save_cfg(cfg):
    try:
        os.makedirs(os.path.dirname(CFG), exist_ok=True)
        tmp = CFG + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        os.replace(tmp, CFG)
    except OSError:
        pass


def run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NOWIN).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def local_addresses():
    """[(address, label)] for this PC's IPv4 addresses (LAN and Tailscale)."""
    out = []
    try:
        for *_, sa in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = sa[0]
            if ip.startswith("127.") or ip.startswith("169.254.") or ip in [a for a, _ in out]:
                continue
            out.append((ip, "Tailscale" if ipaddress.ip_address(ip) in ipaddress.ip_network("100.64.0.0/10")
                        else "local network"))
    except OSError:
        pass
    return out


def lan_of(ip):
    return ipaddress.ip_network(ip + "/24", strict=False)


# ---- discovery -------------------------------------------------------------------------------------------
NBSTAT = b"\x13\x37\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x20CK" + b"AA" * 15 + b"\x00\x00\x21\x00\x01"


def netbios_sweep(net, wait=2.0):
    """Ask every address in `net` for its Windows machine name. Returns {ip: name} for those that answer."""
    names = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(0.2)
        for host in net.hosts():
            try:
                s.sendto(NBSTAT, (str(host), 137))
            except OSError:
                pass
        end = time.time() + wait
        while time.time() < end:
            try:
                data, (ip, _) = s.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                count = data[56]
                for i in range(count):
                    rec = data[57 + 18 * i: 57 + 18 * (i + 1)]
                    if rec[15] == 0 and not (rec[16] & 0x80):
                        names[ip] = rec[:15].decode("ascii", "replace").strip()
                        break
            except IndexError:
                pass
    finally:
        s.close()
    return names


def arp_neighbours(own_ip):
    out = run(["arp", "-a", "-N", own_ip])
    ips = []
    for m in re.finditer(r"^\s*(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f-]{17})\s+dynamic", out, re.M | re.I):
        ips.append(m.group(1))
    return ips


def reverse_name(ip):
    try:
        return socket.gethostbyaddr(ip)[0].split(".")[0]
    except OSError:
        return ""


def tailscale_peers():
    """[(ip, name, os, online)] from `tailscale status --json`."""
    if not os.path.exists(TAILSCALE):
        return []
    try:
        st = json.loads(run([TAILSCALE, "status", "--json"]) or "{}")
    except ValueError:
        return []
    out = []
    for p in (st.get("Peer") or {}).values():
        v4 = [a for a in p.get("TailscaleIPs", []) if "." in a]
        if v4:
            name = p.get("HostName", "")
            if not name or name == "localhost":
                name = (p.get("DNSName") or "").split(".")[0] or name
            out.append((v4[0], name, p.get("OS", ""), bool(p.get("Online"))))
    return out


def discover(own):
    """All devices we can see: [(ip, name, where)]."""
    found = {}
    for ip, label in own:
        if label != "local network":
            continue
        net = lan_of(ip)
        names = netbios_sweep(net)
        neigh = arp_neighbours(ip)  # the sweep above also fills the ARP table
        for n in set(neigh) | set(names):
            if ipaddress.ip_address(n) in net and n != ip and not n.endswith(".255") and n != str(net[1]):
                found[n] = names.get(n, "")
        missing = [n for n, v in found.items() if not v]
        threads = []
        res = {}
        for n in missing:
            t = threading.Thread(target=lambda n=n: res.__setitem__(n, reverse_name(n)), daemon=True)
            t.start(); threads.append(t)
        for t in threads:
            t.join(3)
        for n in missing:
            found[n] = res.get(n, "") or ""
    rows = [(ip, name or "(unnamed device)", "local network") for ip, name in found.items()]
    for ip, name, osname, online in tailscale_peers():
        rows.append((ip, name, "Tailscale, %s%s" % (osname, "" if online else ", offline")))
    rows.sort(key=lambda r: (r[2].startswith("Tailscale"), ipaddress.ip_address(r[0])))
    return rows


# ---- window ----------------------------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root, self.cfg = root, load_cfg()
        self.proc, self.q, self.asked = None, queue.Queue(), set()
        self.own = local_addresses()
        root.title("ShowSync Upload")
        root.geometry("760x640")
        root.minsize(640, 520)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        pad = dict(padx=8, pady=4)

        f = ttk.LabelFrame(root, text="1. Where should the files go?")
        f.pack(fill="x", **pad)
        self.dest = tk.StringVar(value=self.cfg["dest"])
        self.lib = tk.StringVar(value=self.cfg["library"])
        self._folder_row(f, "Save into:", self.dest, 0)
        self._folder_row(f, "Skip files already in:", self.lib, 1)
        ttk.Label(f, text="Files the PC already has anywhere in that folder are marked \"uploaded\" instead of "
                          "being sent again.", foreground="#666").grid(row=2, column=1, columnspan=2, sticky="w")
        f.columnconfigure(1, weight=1)

        f = ttk.LabelFrame(root, text="2. Which machines may send files?")
        f.pack(fill="both", expand=True, **pad)
        cols = ("allow", "name", "ip", "where")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=8, selectmode="none")
        for c, t, w in zip(cols, ("Allow", "Name", "Address", "Network"), (50, 220, 130, 200)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center" if c == "allow" else "w", stretch=c != "allow")
        self.tree.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=4, pady=4)
        self.tree.bind("<Button-1>", self.toggle)
        sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=4, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.scan_btn = ttk.Button(f, text="Scan network", command=self.scan)
        self.scan_btn.grid(row=1, column=0, sticky="w", padx=4)
        ttk.Label(f, text="Add address:").grid(row=1, column=1, sticky="e")
        self.manual = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.manual, width=18)
        e.grid(row=1, column=2, sticky="w")
        e.bind("<Return>", lambda _: self.add_manual())
        ttk.Button(f, text="Add", command=self.add_manual).grid(row=1, column=3, sticky="w", padx=4)
        lans = sorted({str(lan_of(ip)) for ip, lab in self.own if lab == "local network"})
        self.whole = tk.BooleanVar(value=self.cfg.get("whole_lan", False))
        ttk.Checkbutton(f, variable=self.whole, text="Allow any device on my home network (%s) - it still needs "
                        "the link" % ", ".join(lans or ["none found"])).grid(row=2, column=0, columnspan=4,
                                                                               sticky="w", padx=4)
        self.status = ttk.Label(f, text="Click a row to tick or untick it.", foreground="#666")
        self.status.grid(row=3, column=0, columnspan=4, sticky="w", padx=4)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        self.rows = {}
        for ip, (name, where) in self.cfg["allowed"].items():
            self.put_row(ip, name, where, True)

        f = ttk.Frame(root)
        f.pack(fill="x", **pad)
        self.go = ttk.Button(f, text="Start server", command=self.start_stop)
        self.go.pack(side="left")
        ttk.Label(f, text="  Port:").pack(side="left")
        self.port = tk.StringVar(value=str(self.cfg["port"]))
        ttk.Entry(f, textvariable=self.port, width=6).pack(side="left")
        ttk.Button(f, text="New link", command=self.new_key).pack(side="left", padx=8)
        self.state = ttk.Label(f, text="Stopped", foreground="#a00")
        self.state.pack(side="left", padx=8)

        f = ttk.LabelFrame(root, text="3. Open this link on the other machine")
        f.pack(fill="x", **pad)
        self.links = ttk.Frame(f)
        self.links.pack(fill="x")
        ttk.Label(self.links, text="(press Start server)", foreground="#666").pack(anchor="w", padx=4)

        f = ttk.LabelFrame(root, text="Log")
        f.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(f, height=8, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(f, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

        root.after(200, self.pump)
        self.scan()

    # -- helpers
    def _folder_row(self, f, label, var, r):
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=var).grid(row=r, column=1, sticky="ew", pady=2)
        ttk.Button(f, text="Browse...", command=lambda: self._browse(var)).grid(row=r, column=2, padx=4)

    def _browse(self, var):
        d = filedialog.askdirectory(initialdir=var.get() if os.path.isdir(var.get()) else None)
        if d:
            var.set(os.path.normpath(d))

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip("\n") + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def put_row(self, ip, name, where, allowed=None):
        if ip in self.rows:
            iid = self.rows[ip]
            cur = self.tree.set(iid, "allow")
            if name and name != "(unnamed device)":
                self.tree.set(iid, "name", name)
            self.tree.set(iid, "where", where)
            if allowed is not None:
                self.tree.set(iid, "allow", "\u2714" if allowed else "")
            return
        iid = self.tree.insert("", "end", values=("\u2714" if allowed else "", name, ip, where))
        self.rows[ip] = iid

    def allowed(self):
        return {self.tree.set(i, "ip"): (self.tree.set(i, "name"), self.tree.set(i, "where"))
                for i in self.tree.get_children() if self.tree.set(i, "allow")}

    def toggle(self, ev):
        iid = self.tree.identify_row(ev.y)
        if iid:
            self.tree.set(iid, "allow", "" if self.tree.set(iid, "allow") else "\u2714")
            self.changed()

    def changed(self):
        if self.proc:
            self.status.configure(text="Changed - restarting the server (uploads in progress resume by themselves).")
            self.restart()

    def add_manual(self):
        txt = self.manual.get().strip()
        try:
            ipaddress.ip_network(txt, strict=False)
        except ValueError:
            messagebox.showerror("ShowSync Upload", "'%s' is not an IP address like 192.168.1.50" % txt)
            return
        self.put_row(txt, "(added by hand)", "", True)
        self.manual.set("")
        self.changed()

    def new_key(self):
        if messagebox.askyesno("ShowSync Upload", "Make a new link? The old link stops working."):
            self.cfg["key"] = secrets.token_urlsafe(9)
            self.changed()

    # -- discovery
    def scan(self):
        self.scan_btn.configure(state="disabled")
        self.status.configure(text="Looking for devices on the network...")

        def work():
            rows = discover(self.own)
            self.q.put(("scan", rows))
        threading.Thread(target=work, daemon=True).start()

    # -- server
    def settings(self):
        self.cfg.update(dest=self.dest.get().strip(), library=self.lib.get().strip(),
                        allowed={ip: list(v) for ip, v in self.allowed().items()}, whole_lan=self.whole.get())
        try:
            self.cfg["port"] = int(self.port.get())
        except ValueError:
            pass
        save_cfg(self.cfg)

    def start_stop(self):
        if self.proc:
            self.stop()
        else:
            self.start()

    def start(self):
        self.settings()
        allow = list(self.cfg["allowed"])
        if self.whole.get():
            allow += sorted({str(lan_of(ip)) for ip, lab in self.own if lab == "local network"})
        if not allow:
            messagebox.showinfo("ShowSync Upload", "Tick at least one machine (or the home-network box) first.\n\n"
                                "Can't find the machine in the list? Tick any row, start, then open the link on "
                                "it - this window will ask whether to let it in.")
            return
        if not self.cfg["dest"]:
            messagebox.showerror("ShowSync Upload", "Choose the folder the files should go to.")
            return
        binds = [ip for ip, _ in self.own] or ["0.0.0.0"]
        cmd = [sys.executable.replace("pythonw.exe", "python.exe"), "-I", "-u", SERVER, "--port",
               str(self.cfg["port"]), "--dest", self.cfg["dest"], "--key", self.cfg["key"]]
        if self.cfg["library"]:
            cmd += ["--library", self.cfg["library"]]
        for b in binds:
            cmd += ["--bind", b]
        for a in allow:
            cmd += ["--allow", a]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         creationflags=NOWIN, bufsize=1, encoding="utf-8", errors="replace")
        except OSError as e:
            messagebox.showerror("ShowSync Upload", "Could not start the server:\n%s" % e)
            return
        self.write("--- started %s" % time.strftime("%H:%M:%S"))
        threading.Thread(target=self.reader, args=(self.proc,), daemon=True).start()
        self.go.configure(text="Stop server")
        self.state.configure(text="Running", foreground="#080")
        for w in self.links.winfo_children():
            w.destroy()

    def reader(self, proc):
        for line in proc.stdout:
            self.q.put(("line", line, proc))
        self.q.put(("exit", proc.wait(), proc))

    def stop(self):
        p, self.proc = self.proc, None
        if p:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.go.configure(text="Start server")
        self.state.configure(text="Stopped", foreground="#a00")

    def restart(self):
        self.stop()
        self.start()

    def link_row(self, url, where):
        row = ttk.Frame(self.links)
        row.pack(fill="x", padx=4, pady=1)
        e = ttk.Entry(row, font=("Consolas", 10))
        e.insert(0, url)
        e.configure(state="readonly")
        e.pack(side="left", fill="x", expand=True)
        ttk.Label(row, text=" " + where, width=24).pack(side="left")

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(url)
            self.status.configure(text="Link copied.")
        ttk.Button(row, text="Copy", command=copy).pack(side="left")

    def pump(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "scan":
                    for ip, name, where in msg[1]:
                        self.put_row(ip, name, where)
                    n = sum(1 for r in msg[1] if r[2] == "local network")
                    self.status.configure(text="Found %d device(s) on the local network. Click a row to tick it. "
                                               "Don't see the machine? Open the link on it anyway - this window "
                                               "will ask." % n)
                    self.scan_btn.configure(state="normal")
                elif msg[0] == "line":
                    line, proc = msg[1], msg[2]
                    if proc is not self.proc:
                        continue
                    self.write(line)
                    m = re.match(r"URL: (http://([\d.]+):\d+/\S+)", line)
                    if m:
                        where = dict(self.own).get(m.group(2), "")
                        self.link_row(m.group(1), "(%s)" % where if where else "")
                    m = re.search(r"refused client ([\d.]+)", line)
                    if m and m.group(1) not in self.asked:
                        self.asked.add(m.group(1))
                        self.root.after(10, self.ask_allow, m.group(1))
                elif msg[0] == "exit":
                    if msg[2] is self.proc:
                        self.write("--- server stopped (exit code %s) - is the port already in use? Try another "
                                   "port." % msg[1])
                        self.proc = None
                        self.go.configure(text="Start server")
                        self.state.configure(text="Stopped", foreground="#a00")
        except queue.Empty:
            pass
        self.root.after(200, self.pump)

    def ask_allow(self, ip):
        name = reverse_name(ip) or "an unknown device"
        if messagebox.askyesno("ShowSync Upload", "%s (%s) opened the upload link but isn't allowed yet.\n\n"
                               "Let it send files?" % (name, ip)):
            self.put_row(ip, name if name != "an unknown device" else "(unnamed device)", "", True)
            self.changed()
        else:
            self.asked.add(ip)

    def quit(self):
        if self.proc and not messagebox.askyesno("ShowSync Upload", "Stop the server and close? Unfinished uploads "
                                                 "can be resumed later."):
            return
        self.settings()
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
