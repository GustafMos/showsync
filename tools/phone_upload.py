"""Resumable phone -> PC upload page (stdlib only).

    python phone_upload.py --bind <this PC's address> --allow <phone's address> --dest <folder for uploads>

Open the printed URL on the phone, pick files, and they upload as 16 MB pieces, 4 at a time, over kept-alive
connections. If the connection drops or the tab reloads, pick the same files again and only the missing pieces
are sent. Partial data lives in <dest>\\.partial\\<name>.<size>.part (+ .part.json listing the pieces that
arrived) and is moved to <dest>\\<name> only when every piece is there. Before uploading, the page asks whether
the PC already has the file (same size + 12 sampled blocks) and skips it if so.
Only clients in --allow (IPs or networks) may connect, and every request needs the secret key in the URL.
"""
import argparse
import ipaddress
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CHUNK = 16 << 20
LOCK = threading.Lock()

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Upload to PC</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--acc:#4caf50;--bar:#333;--err:#ff6b6b}
body{background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;margin:0;padding:16px;max-width:640px}
h1{font-size:20px;margin:0 0 4px}p{color:var(--mut);margin:4px 0 14px}
label.pick{display:block;background:var(--acc);color:#000;text-align:center;padding:16px;border-radius:10px;font-weight:600}
input{display:none}.f{margin:14px 0;padding:10px;border:1px solid var(--bar);border-radius:8px}
.n{word-break:break-all;font-weight:600}.s{color:var(--mut);font-size:14px;margin-top:4px}
.b{height:8px;background:var(--bar);border-radius:4px;margin-top:8px;overflow:hidden}
.b i{display:block;height:100%;width:0;background:var(--acc)}.e{color:var(--err)}
</style></head><body>
<h1>Upload to PC</h1>
<p>Keep this tab open and the screen on while it runs. If it stops, pick the same files again and it continues where it left off.</p>
<label class="pick">Choose files<input id="p" type="file" multiple></label>
<div id="list"></div>
<script>
const KEY = location.pathname.split('/')[1];
// report what happens on the phone to the PC's console (helps when the phone screen misbehaves)
const log = (m) => { try { navigator.sendBeacon('/' + KEY + '/log', String(m).slice(0, 1000)); } catch (e) {} };
window.onerror = (m, src, line) => log('JS error: ' + m + ' line ' + line);
window.onunhandledrejection = (e) => log('JS rejection: ' + (e.reason && e.reason.message || e.reason));
document.addEventListener('visibilitychange', () => log('page ' + document.visibilityState));
log('page loaded: ' + navigator.userAgent);
if (/Firefox/.test(navigator.userAgent) && /Android/.test(navigator.userAgent)) {
  const w = document.createElement('p'); w.className = 'e';
  w.textContent = 'Firefox copies every picked file on the phone before uploading, so big files freeze it. Open this link in Chrome or Samsung Internet instead.';
  document.querySelector('h1').after(w);
}
document.getElementById('p').onclick = () => log('file picker opened');
const PAR = 4;  // pieces in flight at once
const q = (o) => Object.entries(o).map(([k, v]) => k + '=' + encodeURIComponent(v)).join('&');
const fmt = (b) => b > 1e9 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(0) + ' MB';
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let queue = [], running = false, checking = Promise.resolve();
// Same sample positions as the server: N blocks of L bytes spread evenly over the file (whole file if small).
const N = 12, L = 65536;
function sampleBlob(f) {
  if (f.size <= N * L) return f.slice(0);
  const parts = [];
  for (let i = 0; i < N; i++) { const o = Math.floor(i * (f.size - L) / (N - 1)); parts.push(f.slice(o, o + L)); }
  return new Blob(parts);
}
document.getElementById('p').onchange = (e) => {
  const picked = [...e.target.files];
  log('picked ' + picked.length + ' file(s): ' + picked.map(f => f.name + ' ' + fmt(f.size)).join(', '));
  e.target.value = '';
  const jobs = picked.map(f => {
    const d = document.createElement('div'); d.className = 'f';
    d.innerHTML = '<div class="n"></div><div class="s">checking if the PC already has it...</div><div class="b"><i></i></div>';
    d.querySelector('.n').textContent = f.name;
    document.getElementById('list').appendChild(d);
    return {f, d};
  });
  // check every picked file first (fast), then upload only the ones the PC doesn't have
  checking = checking.then(async () => {
    for (const j of jobs) {
      const st = j.d.querySelector('.s');
      let r = null;
      for (let t = 0; t < 5 && !r; t++) {
        try { r = await call('/check?' + q({name: j.f.name, size: j.f.size}), {method: 'POST', body: sampleBlob(j.f)}); }
        catch (err) { st.innerHTML = '<span class="e">check failed, retrying...</span>'; await sleep(2000); }
      }
      if (r && r.dup) {
        j.d.querySelector('i').style.width = '100%';
        st.textContent = 'already uploaded - on PC as ' + r.dup + ' (skipped) ';
        const b = document.createElement('button'); b.textContent = 'upload anyway';
        b.onclick = () => { b.remove(); st.textContent = 'waiting'; queue.push(j); if (!running) run(); };
        st.appendChild(b);
      } else {
        st.textContent = 'waiting';
        queue.push(j);
        if (!running) run();
      }
    }
  });
};
async function call(path, opts) {
  const r = await fetch('/' + KEY + path, opts);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
async function run() {
  running = true;
  while (queue.length) { const j = queue.shift(); await send(j.f, j.d); }
  running = false;
}
async function send(f, d) {
  const st = d.querySelector('.s'), bar = d.querySelector('i');
  const id = {name: f.name, size: f.size};
  const finish = (note) => { bar.style.width = '100%'; st.textContent = 'done - ' + fmt(f.size) + ' saved on PC' + (note ? ' (' + note + ')' : ''); };
  let fails = 0;
  while (true) {
    let s;
    try { s = await call('/status?' + q(id)); fails = 0; }
    catch (err) {
      fails++;
      st.innerHTML = '<span class="e">PC not reachable, retrying (' + fails + ')...</span>';
      await sleep(Math.min(30000, 1000 * fails));
      continue;
    }
    if (s.done) return finish(s.note);
    const CH = s.chunk, total = s.total, todo = s.missing.slice();
    let have = total - todo.length, errs = 0, result = null;
    const base = have, t0 = Date.now();
    const show = () => {
      const pct = have / total * 100, rate = (have - base) * CH / Math.max((Date.now() - t0) / 1000, 0.001);
      bar.style.width = pct.toFixed(1) + '%';
      st.textContent = fmt(Math.min(have * CH, f.size)) + ' / ' + fmt(f.size) + ' - ' + (rate / 1e6).toFixed(1) + ' MB/s' +
        (rate > 0 ? ' - ' + Math.ceil((total - have) * CH / rate / 60) + ' min left' : '');
    };
    show();
    const worker = async () => {
      while (todo.length && !result) {
        const i = todo.shift();
        try {
          const r = await call('/put?' + q({...id, index: i}), {method: 'PUT', body: f.slice(i * CH, Math.min((i + 1) * CH, f.size))});
          errs = 0; have = Math.max(have, r.have_chunks ?? have + 1); show();
          if (r.done) result = r;
        } catch (err) {
          todo.push(i); errs++;
          st.innerHTML = '<span class="e">connection lost, retrying (' + errs + ')...</span>';
          await sleep(Math.min(30000, 1000 * errs));
        }
      }
    };
    await Promise.all(Array.from({length: PAR}, worker));
    if (result) return finish(result.note);
    // otherwise ask the PC again what is still missing
  }
}
</script></body></html>"""


SAMPLE_N, SAMPLE_L = 12, 65536
_sample_cache = {}


def sample_bytes(path, size):
    """The same N evenly spaced L-byte blocks the page sends (whole file if small). Cached per mtime."""
    key = (path, size, os.path.getmtime(path))
    if key not in _sample_cache:
        with open(path, "rb") as fh:
            if size <= SAMPLE_N * SAMPLE_L:
                _sample_cache[key] = fh.read()
            else:
                out = []
                for i in range(SAMPLE_N):
                    fh.seek(i * (size - SAMPLE_L) // (SAMPLE_N - 1))
                    out.append(fh.read(SAMPLE_L))
                _sample_cache[key] = b"".join(out)
    return _sample_cache[key]


def find_duplicate(library, size, samples):
    """A file in the library with the same size and identical sample blocks, or None."""
    for d, dirs, fs in os.walk(library):
        dirs[:] = [x for x in dirs if x != ".partial"]
        for f in fs:
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) == size and sample_bytes(p, size) == samples:
                    return p
            except OSError:
                continue
    return None


def safe_name(n):
    n = os.path.basename(n.replace("\\", "/")).strip()
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", n)
    if not n or n in (".", ".."):
        raise ValueError("bad name")
    return n


def n_chunks(size):
    return max(1, -(-size // CHUNK))


def chunk_len(size, i):
    return max(0, min(CHUNK, size - i * CHUNK))


_states = {}      # part path -> set of piece indices already on disk
_rate = {}        # part path -> [bytes since last report, time of last report]


def load_state(part, size):
    """Pieces already received. A .part without a .json sidecar is the old sequential format:
    a contiguous prefix written in CHUNK-sized steps from offset 0."""
    if part in _states:
        return _states[part]
    done = set()
    side = part + ".json"
    if os.path.exists(side):
        try:
            with open(side) as fh:
                st = json.load(fh)
            if st.get("chunk") == CHUNK:
                done = set(st["done"])
        except (OSError, ValueError, KeyError):
            done = set()
    elif os.path.exists(part):
        have = os.path.getsize(part)
        done = set(range(n_chunks(size))) if have >= size else set(range(have // CHUNK))
    _states[part] = done
    return done


def save_state(part, done):
    tmp = part + ".json.tmp"
    with open(tmp, "w") as fh:
        json.dump({"chunk": CHUNK, "done": sorted(done)}, fh)
    os.replace(tmp, part + ".json")


class H(BaseHTTPRequestHandler):
    server_version = "PhoneUpload"
    protocol_version = "HTTP/1.1"  # keep connections open between pieces
    timeout = 120                  # a stalled phone connection must not hang a worker forever

    def log_message(self, fmt, *a):
        pass

    def _json(self, obj, code=200, close=False):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(b)

    def _allowed(self, ip):
        a = ipaddress.ip_address(ip)
        return any(a in net for net in self.server.allow)

    def _gate(self):
        if not self._allowed(self.client_address[0]):
            print(time.strftime("%H:%M:%S"), "refused client", self.client_address[0], flush=True)
            self.send_error(403)
            return None
        u = urlparse(self.path)
        parts = u.path.split("/")
        if len(parts) < 2 or not secrets.compare_digest(parts[1], self.server.key):
            self.send_error(404)
            return None
        return "/" + "/".join(parts[2:]), {k: v[0] for k, v in parse_qs(u.query).items()}

    def _paths(self, qs):
        name, size = safe_name(qs["name"]), int(qs["size"])
        if size < 0:
            raise ValueError("bad size")
        part = os.path.join(self.server.partdir, f"{name}.{size}.part")
        return name, size, part, os.path.join(self.server.dest, name)

    def _finish(self, name, size, part, final):
        """Move a complete .part into place. Same name + same size already there -> treat as done."""
        _states.pop(part, None)
        _rate.pop(part, None)
        if os.path.exists(part + ".json"):
            os.remove(part + ".json")
        with open(part, "r+b") as fh:
            fh.truncate(size)
        if os.path.exists(final):
            if os.path.getsize(final) == size:
                os.remove(part)
                return "already there"
            base, ext = os.path.splitext(name)
            i = 2
            while os.path.exists(os.path.join(self.server.dest, f"{base} ({i}){ext}")):
                i += 1
            final = os.path.join(self.server.dest, f"{base} ({i}){ext}")
        os.replace(part, final)
        print(time.strftime("%H:%M:%S"), f"complete: {final} ({size / 1e9:.2f} GB)", flush=True)
        return ""

    def _drain(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            while n > 0:
                b = self.rfile.read(min(n, 1 << 20))
                if not b:
                    break
                n -= len(b)
        except (OSError, ValueError):
            self.close_connection = True

    def do_GET(self):
        g = self._gate()
        if not g:
            return
        path, qs = g
        if path in ("/", ""):
            print(time.strftime("%H:%M:%S"), "page requested by", self.client_address[0], flush=True)
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return
        if path == "/status":
            try:
                name, size, part, final = self._paths(qs)
            except (KeyError, ValueError):
                return self._json({"error": "bad request"}, 400)
            with LOCK:
                total = n_chunks(size)
                if not os.path.exists(part) and os.path.exists(final) and os.path.getsize(final) == size:
                    return self._json({"done": True, "note": "already there"})
                done = load_state(part, size)
                if size == 0 or len(done) == total and os.path.exists(part):
                    if not os.path.exists(part):
                        open(part, "wb").close()
                    return self._json({"done": True, "note": self._finish(name, size, part, final)})
                missing = [i for i in range(total) if i not in done]
            return self._json({"done": False, "chunk": CHUNK, "total": total, "missing": missing})
        self.send_error(404)

    def do_POST(self):
        g = self._gate()
        if not g:
            return self._drain()
        path, qs = g
        if path == "/log":
            n = int(self.headers.get("Content-Length") or 0)
            msg = self.rfile.read(n)[:4096].decode("utf-8", "replace")
            print(time.strftime("%H:%M:%S"), "phone:", msg, flush=True)
            self.send_response(204)
            self.end_headers()
            return
        if path != "/check":
            self._drain()
            return self.send_error(404)
        print(time.strftime("%H:%M:%S"), "check:", qs.get("name"), qs.get("size"), flush=True)
        try:
            safe_name(qs["name"])
            size, length = int(qs["size"]), int(self.headers["Content-Length"])
        except (KeyError, ValueError, TypeError):
            return self._json({"error": "bad request"}, 400, close=True)
        if length != min(size, SAMPLE_N * SAMPLE_L):
            return self._json({"error": "bad sample"}, 400, close=True)
        samples = self.rfile.read(length)
        dup = find_duplicate(self.server.library, size, samples)
        rel = os.path.relpath(dup, self.server.library) if dup else None
        if dup:
            print(f"skip (already have): {qs['name']} == {rel}", flush=True)
        self._json({"dup": rel})

    def do_PUT(self):
        g = self._gate()
        if not g:
            return self._drain()
        path, qs = g
        if path != "/put":
            self._drain()
            return self.send_error(404)
        try:
            name, size, part, final = self._paths(qs)
            index, length = int(qs["index"]), int(self.headers["Content-Length"])
        except (KeyError, ValueError, TypeError):
            return self._json({"error": "bad request"}, 400, close=True)
        if not 0 <= index < n_chunks(size) or length != chunk_len(size, index):
            return self._json({"error": "bad piece"}, 400, close=True)
        data, left = [], length          # read the whole piece first, outside the lock
        try:
            while left:
                b = self.rfile.read(min(left, 1 << 20))
                if not b:
                    break
                data.append(b)
                left -= len(b)
        except OSError:
            left = -1
        if left:  # connection dropped mid-piece: keep nothing from it
            self.close_connection = True
            return
        with LOCK:
            if not os.path.exists(part) and os.path.exists(final) and os.path.getsize(final) == size:
                return self._json({"done": True, "note": "already there"})
            done = load_state(part, size)
            if index not in done:
                with open(part, "r+b" if os.path.exists(part) else "w+b") as fh:
                    fh.seek(index * CHUNK)
                    for b in data:
                        fh.write(b)
                done.add(index)
                save_state(part, done)
                r = _rate.setdefault(part, [0, time.time()])
                r[0] += length
                if r[0] >= 1 << 30:
                    dt = time.time() - r[1]
                    print(time.strftime("%H:%M:%S"), f"upload {name}: {len(done) / n_chunks(size) * 100:.0f}%"
                          f" ({len(done) * CHUNK / 1e9:.1f} / {size / 1e9:.1f} GB), {r[0] / 1e6 / max(dt, 1e-3):.1f} MB/s",
                          flush=True)
                    _rate[part] = [0, time.time()]
            have = len(done)
            all_done = have == n_chunks(size)
            note = self._finish(name, size, part, final) if all_done else ""
        self._json({"done": all_done, "have_chunks": have, "note": note})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", action="append", required=True, help="address to listen on (repeatable)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow", action="append", required=True,
                    help="client IP or network (e.g. 192.168.1.50 or 192.168.1.0/24) allowed to connect (repeatable)")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--library", default=None, help="folder searched for files the PC already has (default: --dest)")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    partdir = os.path.join(a.dest, ".partial")
    os.makedirs(partdir, exist_ok=True)
    free = shutil.disk_usage(a.dest).free
    key = a.key or secrets.token_urlsafe(9)
    allow = [ipaddress.ip_network(x, strict=False) for x in a.allow]
    servers = []
    for b in a.bind:  # one process for every address, so all uploads share one write lock
        srv = ThreadingHTTPServer((b, a.port), H)
        srv.allow, srv.dest, srv.partdir = allow, a.dest, partdir
        srv.library = a.library or a.dest
        srv.key = key
        servers.append(srv)
        print(f"URL: http://{b}:{a.port}/{key}/", flush=True)
    print(f"saving to {a.dest}  ({free / 1e9:.0f} GB free), allowed clients: {', '.join(a.allow)}", flush=True)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
