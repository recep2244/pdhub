"""
Embedded PyMOL HTTP server for interactive structure visualization.

Architecture:
  - Singleton PyMOLViewerServer runs on a background daemon thread
  - Serves an interactive HTML viewer at GET /
  - REST endpoints for rotate/zoom/style/load
  - Embedded in Streamlit via <iframe src="localhost:{PORT}">

Usage:
    from protein_design_hub.web.pymol_server import get_pymol_server, PYMOL_SERVER_PORT
    server = get_pymol_server()          # creates once, reuses
    server.load_structure(pdb_data=..., highlight_residues=[5, 10])
    # Then render <iframe src="http://localhost:{PORT}/"> in Streamlit
"""

import json
import os
import queue
import threading
import tempfile
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Optional, List

log = logging.getLogger(__name__)

# Allow overriding via environment variable (e.g. when port 8592 is in use)
PYMOL_SERVER_PORT: int = int(os.environ.get("PYMOL_PORT", "8592"))


# ---------------------------------------------------------------------------
# HTML viewer template
# ---------------------------------------------------------------------------
def _viewer_html(port: int = PYMOL_SERVER_PORT) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0a0f; color: #e2e8f0; font-family: system-ui, sans-serif; overflow: hidden; }}
  #root {{ display: flex; flex-direction: column; height: 100vh; }}

  #toolbar {{
    display: flex; align-items: center; gap: 6px; padding: 6px 10px;
    background: #111827; border-bottom: 1px solid rgba(255,255,255,0.07);
    flex-wrap: wrap; flex-shrink: 0;
  }}
  .tbtn {{
    background: rgba(255,255,255,0.06); color: #cbd5e1; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 5px; padding: 4px 10px; font-size: 12px; cursor: pointer;
    transition: background 0.15s;
  }}
  .tbtn:hover {{ background: rgba(255,255,255,0.14); color: #f1f5f9; }}
  .tbtn.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
  .sep {{ width: 1px; height: 20px; background: rgba(255,255,255,0.1); margin: 0 2px; }}
  #status {{ margin-left: auto; font-size: 11px; color: #64748b; }}
  #btn-fs {{ margin-left: 6px; }}
  :fullscreen #root, :-webkit-full-screen #root, :-moz-full-screen #root {{
    height: 100vh !important;
  }}

  #viewer-wrap {{
    position: relative; flex: 1; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    contain: strict;
    touch-action: none;
  }}
  #mol-img {{
    max-width: 100%; max-height: 100%;
    cursor: grab; display: block; user-select: none;
    will-change: opacity;
    transform-origin: center center;
    object-fit: contain;
  }}
  #mol-img.dragging {{ cursor: grabbing; opacity: 0.82; }}

  #spinner {{
    position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; background: rgba(10,10,15,0.7);
    font-size: 13px; color: #94a3b8; display: none;
  }}

  #rotation-btns {{
    position: absolute; bottom: 12px; right: 12px;
    display: grid; grid-template-columns: repeat(3, 32px);
    grid-template-rows: repeat(3, 32px); gap: 2px;
  }}
  .rbtn {{
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 4px; color: #cbd5e1; font-size: 14px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.1s;
  }}
  .rbtn:hover {{ background: rgba(255,255,255,0.18); }}

  #legend {{
    position: absolute; top: 10px; left: 10px;
    background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 8px 10px; font-size: 11px; line-height: 1.6;
    display: none;
  }}
  #legend.visible {{ display: block; }}
  .leg-mut {{ color: #f59e0b; }}
  .leg-res {{ color: #10b981; }}

  #score-overlay {{
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,0,0,0.65); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 8px 12px; font-size: 11px; line-height: 1.7;
    min-width: 140px; display: none;
  }}
  #score-overlay.visible {{ display: block; }}
  .score-row {{ display: flex; justify-content: space-between; gap: 16px; }}
  .score-k {{ color: #94a3b8; }}
  .score-v {{ color: #e2e8f0; font-weight: 600; }}
</style>
</head>
<body>
<div id="root">
  <div id="toolbar">
    <button class="tbtn active" id="btn-cartoon" onclick="setStyle('cartoon')">Cartoon</button>
    <button class="tbtn" id="btn-sticks" onclick="setStyle('sticks')">Sticks</button>
    <button class="tbtn" id="btn-spheres" onclick="setStyle('spheres')">Spheres</button>
    <button class="tbtn" id="btn-surface" onclick="setStyle('surface')">Surface</button>
    <div class="sep"></div>
    <button class="tbtn active" id="btn-chain" onclick="setColor('chain')">By Chain</button>
    <button class="tbtn" id="btn-plddt" onclick="setColor('plddt')">pLDDT</button>
    <button class="tbtn" id="btn-spectrum" onclick="setColor('spectrum')">Spectrum</button>
    <div class="sep"></div>
    <button class="tbtn" onclick="toggleSurface()">+Surface</button>
    <button class="tbtn" onclick="resetView()">⟳ Reset</button>
    <button class="tbtn" id="btn-fs" onclick="toggleFullscreen()" title="Expand to full screen">⛶ Expand</button>
    <span id="status">Loading…</span>
  </div>

  <div id="viewer-wrap">
    <img id="mol-img" src="" alt="PyMOL viewer" draggable="false"/>
    <div id="spinner">⟳ Rendering…</div>

    <div id="rotation-btns">
      <div></div>
      <div class="rbtn" onclick="rotBtn(0,-15)" title="Tilt up">▲</div>
      <div></div>
      <div class="rbtn" onclick="rotBtn(-15,0)" title="Rotate left">◀</div>
      <div class="rbtn" onclick="resetView()" title="Reset">⌂</div>
      <div class="rbtn" onclick="rotBtn(15,0)" title="Rotate right">▶</div>
      <div></div>
      <div class="rbtn" onclick="rotBtn(0,15)" title="Tilt down">▼</div>
      <div></div>
    </div>

    <div id="legend"></div>
    <div id="score-overlay"></div>
  </div>
</div>

<script>
const BASE = 'http://localhost:{port}';
// ── State ──────────────────────────────────────────────────────────────────
let dragging = false;
let lastX = 0, lastY = 0;
let accX = 0, accY = 0;     // rotation delta accumulated since last PyMOL sync
let cssRx = 0, cssRy = 0;   // CSS prediction angles from mousedown
let highBusy = false;        // high-quality render in flight
let lowBusy = false;         // state-sync request in flight
let lowTimer = null;         // throttle timer for drag sync
const img = document.getElementById('mol-img');
const spinner = document.getElementById('spinner');
const status = document.getElementById('status');
const legend = document.getElementById('legend');
const scoreOverlay = document.getElementById('score-overlay');

// ── High-quality render (toolbar, mouseup, zoom) ───────────────────────────
// Shows spinner; resets CSS prediction and swaps image when done.
async function highRender(endpoint, body) {{
  if (highBusy) return;
  highBusy = true;
  spinner.style.display = 'flex';
  try {{
    const r = await fetch(BASE + endpoint, {{
      method: body !== null ? 'POST' : 'GET',
      headers: body !== null ? {{'Content-Type': 'application/json'}} : undefined,
      body: body !== null ? JSON.stringify(body) : undefined,
    }});
    if (r.ok) {{
      const ct = r.headers.get('Content-Type') || '';
      if (ct.includes('image')) {{
        const blob = await r.blob();
        // Reset CSS prediction before swapping image → seamless transition
        img.style.transform = '';
        cssRx = 0; cssRy = 0;
        const old = img.src;
        img.src = URL.createObjectURL(blob);
        if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
        status.textContent = 'Ready';
      }} else {{
        const d = await r.json();
        status.textContent = d.status || 'OK';
      }}
    }}
  }} catch(e) {{
    img.style.transform = '';
    cssRx = 0; cssRy = 0;
    status.textContent = 'Error: ' + e.message;
  }}
  spinner.style.display = 'none';
  highBusy = false;
}}

// ── Low-latency state sync (during drag, no render returned) ──────────────
// Just tells PyMOL to rotate — no image download — so PyMOL stays in sync.
function lowRender(x, y) {{
  if (lowBusy) return;
  lowBusy = true;
  fetch(BASE + '/api/rotate_fast', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{x, y}}),
  }}).then(() => {{ lowBusy = false; }}).catch(() => {{ lowBusy = false; }});
}}

// ── Drag-to-rotate with CSS 3D prediction ─────────────────────────────────
img.addEventListener('mousedown', e => {{
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  accX = 0; accY = 0; cssRx = 0; cssRy = 0;
  img.classList.add('dragging'); e.preventDefault();
}});

document.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;

  // Accumulate drag delta — no 3D transform so the viewer stays fixed on screen
  cssRx += dy * 0.5;
  cssRy += dx * 0.5;

  // Accumulate for periodic PyMOL state sync (keeps server in sync during long drags)
  accX += dx; accY += dy;
  if (!lowTimer) {{
    lowTimer = setTimeout(() => {{
      const ax = accX, ay = accY; accX = 0; accY = 0; lowTimer = null;
      lowRender(ay * 0.5, ax * 0.5);
    }}, 100);
  }}
}});

document.addEventListener('mouseup', () => {{
  if (!dragging) return;
  dragging = false;
  img.classList.remove('dragging');
  img.style.transform = '';  // clear any residual transform
  if (lowTimer) {{ clearTimeout(lowTimer); lowTimer = null; }}

  // Send remaining unsynced rotation + request authoritative high-res render
  const ax = accX, ay = accY; accX = 0; accY = 0;
  if (ax !== 0 || ay !== 0) {{
    highRender('/api/rotate', {{x: ay * 0.5, y: ax * 0.5}});
  }} else {{
    highRender('/api/frame', null);
  }}
}});

// Touch support (mobile / tablet)
img.addEventListener('touchstart', e => {{
  if (e.touches.length === 1) {{
    dragging = true;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    accX = 0; accY = 0; cssRx = 0; cssRy = 0;
    e.preventDefault();
  }}
}}, {{passive: false}});
document.addEventListener('touchmove', e => {{
  if (!dragging || e.touches.length !== 1) return;
  const dx = e.touches[0].clientX - lastX, dy = e.touches[0].clientY - lastY;
  lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  cssRx += dy * 0.5; cssRy += dx * 0.5;
  accX += dx; accY += dy;
  if (!lowTimer) {{
    lowTimer = setTimeout(() => {{
      const ax = accX, ay = accY; accX = 0; accY = 0; lowTimer = null;
      lowRender(ay * 0.5, ax * 0.5);
    }}, 100);
  }}
  e.preventDefault();
}}, {{passive: false}});
document.addEventListener('touchend', () => {{
  if (!dragging) return;
  dragging = false;
  if (lowTimer) {{ clearTimeout(lowTimer); lowTimer = null; }}
  const ax = accX, ay = accY; accX = 0; accY = 0;
  if (ax !== 0 || ay !== 0) highRender('/api/rotate', {{x: ay * 0.5, y: ax * 0.5}});
  else highRender('/api/frame', null);
}});

// ── Scroll zoom ────────────────────────────────────────────────────────────
document.getElementById('viewer-wrap').addEventListener('wheel', e => {{
  e.preventDefault();
  if (!highBusy) highRender('/api/zoom', {{factor: e.deltaY > 0 ? 0.85 : 1.18}});
}}, {{passive: false}});

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  const k = e.key;
  if (k === 'r') resetView();
  else if (k === 'c') setStyle('cartoon');
  else if (k === 's') setStyle('sticks');
  else if (k === 'f') setStyle('surface');
  else if (k === 'ArrowLeft') rotBtn(-20, 0);
  else if (k === 'ArrowRight') rotBtn(20, 0);
  else if (k === 'ArrowUp') rotBtn(0, -20);
  else if (k === 'ArrowDown') rotBtn(0, 20);
  else if (k === '+' || k === '=') highRender('/api/zoom', {{factor: 1.15}});
  else if (k === '-') highRender('/api/zoom', {{factor: 0.87}});
}});

// ── Toolbar ────────────────────────────────────────────────────────────────
function setStyle(s) {{
  ['cartoon','sticks','spheres','surface'].forEach(n => {{
    const b = document.getElementById('btn-'+n);
    if (b) b.classList.toggle('active', n === s);
  }});
  highRender('/api/style', {{style: s}});
}}

function setColor(s) {{
  ['chain','plddt','spectrum'].forEach(n => {{
    const b = document.getElementById('btn-'+n);
    if (b) b.classList.toggle('active', n === s);
  }});
  highRender('/api/color', {{scheme: s}});
}}

function toggleSurface() {{ highRender('/api/surface', {{opacity: 0.5}}); }}
function resetView() {{ highRender('/api/reset', {{}}); }}
function rotBtn(dy, dx) {{ highRender('/api/rotate', {{x: dx, y: dy}}); }}

// ── Parent frame postMessage ───────────────────────────────────────────────
window.addEventListener('message', e => {{
  if (e.data && e.data.type === 'pdh-load') {{
    const d = e.data;
    if (d.legend_html) {{
      legend.innerHTML = d.legend_html;
      legend.classList.add('visible');
    }}
    if (d.scores_html) {{
      scoreOverlay.innerHTML = d.scores_html;
      scoreOverlay.classList.add('visible');
    }}
  }}
}});

// ── Fullscreen toggle ──────────────────────────────────────────────────────
function toggleFullscreen() {{
  const el = document.documentElement;
  const btn = document.getElementById('btn-fs');
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement);
  if (isFs) {{
    (document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen).call(document);
    btn.textContent = '⛶ Expand';
  }} else {{
    const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
    if (req) req.call(el);
    btn.textContent = '✕ Exit';
  }}
}}
document.addEventListener('fullscreenchange', () => {{
  const btn = document.getElementById('btn-fs');
  if (!document.fullscreenElement) btn.textContent = '⛶ Expand';
}});

// ── Initial frame load ─────────────────────────────────────────────────────
highRender('/api/frame', null);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _image(self, data: bytes):
        # Auto-detect JPEG (FF D8) vs PNG
        ct = "image/jpeg" if len(data) >= 2 and data[:2] == b"\xff\xd8" else "image/png"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _html(self, text: str):
        b = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(b))
        self._cors()
        self.end_headers()
        self.wfile.write(b)

    def _json(self, data: dict, status: int = 200):
        b = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(b))
        self._cors()
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        srv: PyMOLViewerServer = self.server  # type: ignore[assignment]
        # Strip query string (?t=...) for routing
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._html(_viewer_html(port=srv.server_address[1]))
        elif path == "/api/frame":
            self._image(srv.get_frame())
        elif path == "/api/health":
            self._json({"status": "ok", "loaded": srv.structure_loaded})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        srv: PyMOLViewerServer = self.server  # type: ignore[assignment]
        body = self._body()
        path = self.path

        if path == "/api/load":
            ok = srv.load_structure(
                pdb_data=body.get("pdb_data", ""),
                name=body.get("name", "structure"),
                highlight_residues=body.get("highlight_residues"),
                restraint_residues=body.get("restraint_residues"),
                mutation_label=body.get("mutation_label", ""),
            )
            self._image(srv.get_frame()) if ok else self._json({"error": "load failed"}, 500)
        elif path == "/api/rotate":
            srv.rotate(body.get("x", 0), body.get("y", 0))
            self._image(srv.get_frame())
        elif path == "/api/rotate_fast":
            # State-sync during drag: rotate without rendering (fast, ~1 ms)
            srv.rotate(body.get("x", 0), body.get("y", 0))
            self._json({"ok": True})
        elif path == "/api/run_command":
            # Execute arbitrary PyMOL command (used by agent tools)
            result = srv.run_command(body.get("command", ""))
            self._image(srv.get_frame()) if result == "OK" else self._json(
                {"result": result, "ok": False}
            )
        elif path == "/api/zoom":
            srv.zoom(body.get("factor", 1.0))
            self._image(srv.get_frame())
        elif path == "/api/style":
            srv.set_style(body.get("style", "cartoon"))
            self._image(srv.get_frame())
        elif path == "/api/color":
            srv.set_color(body.get("scheme", "chain"))
            self._image(srv.get_frame())
        elif path == "/api/surface":
            srv.toggle_surface(body.get("opacity", 0.5))
            self._image(srv.get_frame())
        elif path == "/api/reset":
            srv.reset_view()
            self._image(srv.get_frame())
        elif path == "/api/select":
            srv.select_residues(body.get("resi", []))
            self._image(srv.get_frame())
        else:
            self._json({"error": "not found"}, 404)


# ---------------------------------------------------------------------------
# PyMOL server
# ---------------------------------------------------------------------------
class PyMOLViewerServer(ThreadingMixIn, HTTPServer):
    """Persistent PyMOL headless instance wrapped in a threaded HTTP server.

    PyMOL is not thread-safe (thread-local C bindings), so ALL PyMOL calls are
    dispatched through a dedicated worker thread via _run_on_pymol_thread().
    The HTTP server accepts connections on multiple threads, but PyMOL work
    serialises through a single-consumer queue.
    """
    daemon_threads = True  # don't block shutdown on in-flight requests

    def __init__(self, port: int = PYMOL_SERVER_PORT):
        super().__init__(("localhost", port), _Handler)
        self._cmd = None  # set by _init_pymol on the worker thread
        self.structure_loaded = False
        self._surface_on = False
        self._dirty = True
        self._frame_cache: Optional[bytes] = None
        self._render_w = 720
        self._render_h = 450
        # Work queue: (callable, result_list, done_event)
        self._work_q: queue.Queue = queue.Queue()
        # Start the PyMOL worker thread — PyMOL MUST be initialised here
        self._pymol_thread = threading.Thread(
            target=self._pymol_worker, daemon=True, name="pymol-worker"
        )
        self._pymol_thread.start()
        # Block until PyMOL is initialised
        init_done = threading.Event()
        self._work_q.put((self._init_pymol, [], init_done))
        if not init_done.wait(timeout=30):
            raise RuntimeError("PyMOL failed to initialise within 30 s")

    # ------------------------------------------------------------------ worker
    def _pymol_worker(self):
        """Single consumer — all PyMOL operations run here."""
        while True:
            func, result_box, done = self._work_q.get()
            try:
                result_box.append(func())
            except Exception as e:
                result_box.append(e)
            finally:
                done.set()

    def _run_on_pymol_thread(self, func):
        """Dispatch *func* to the PyMOL worker and return its result."""
        result_box: list = []
        done = threading.Event()
        self._work_q.put((func, result_box, done))
        done.wait(timeout=60)
        val = result_box[0] if result_box else None
        if isinstance(val, Exception):
            raise val
        return val

    def _init_pymol(self):
        import ctypes
        libpath = _find_libstdcpp()
        if libpath:
            try:
                ctypes.CDLL(libpath, ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        import pymol2
        self._pymol = pymol2.PyMOL()
        self._pymol.start()
        cmd = self._pymol.cmd
        cmd.set("cartoon_fancy_helices", 1)
        cmd.set("cartoon_side_chain_helper", 1)
        cmd.set("ray_shadows", 0)
        cmd.set("depth_cue", 0)
        cmd.set("antialias", 1)   # was 2 — 4× render cost with no real benefit at this size
        cmd.set("hash_max", 100)
        cmd.bg_color("black")
        self._cmd = cmd

    # ------------------------------------------------------------------ frame
    def get_frame(self, fast: bool = False) -> bytes:
        """Render current view.

        fast=True renders at half resolution with no antialiasing and lower
        JPEG quality (~45ms vs ~185ms).  Used internally; the browser only
        receives full-quality images — fast renders are fired-and-forgotten
        for state-sync purposes, so their output is currently discarded.
        """
        def _work():
            if not fast and not self._dirty and self._frame_cache:
                return self._frame_cache
            w = 360 if fast else self._render_w
            h = 225 if fast else self._render_h
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                out = Path(f.name)
            try:
                if fast:
                    self._cmd.set("antialias", 0)
                self._cmd.png(str(out), width=w, height=h, quiet=1)
                if fast:
                    self._cmd.set("antialias", 1)
                # Convert PNG → JPEG: ~5× smaller file = faster browser transfer
                try:
                    from PIL import Image
                    import io as _io
                    with Image.open(str(out)) as im:
                        buf = _io.BytesIO()
                        quality = 45 if fast else 82
                        im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
                        data = buf.getvalue()
                except Exception:
                    data = out.read_bytes()
            finally:
                out.unlink(missing_ok=True)
            if not fast:
                self._frame_cache = data
                self._dirty = False
            return data
        return self._run_on_pymol_thread(_work)

    # ------------------------------------------------------------------ load
    def load_structure(
        self,
        pdb_data: str,
        name: str = "structure",
        highlight_residues: Optional[List[int]] = None,
        restraint_residues: Optional[List[int]] = None,
        mutation_label: str = "",
    ) -> bool:
        def _work():
            try:
                self._cmd.delete("all")
                with tempfile.NamedTemporaryFile(
                    suffix=".pdb", mode="w", delete=False
                ) as f:
                    f.write(pdb_data)
                    pdb_path = f.name
                self._cmd.load(pdb_path, name)
                Path(pdb_path).unlink(missing_ok=True)

                cmd = self._cmd
                cmd.hide("everything", "all")
                cmd.show("cartoon", "all")
                cmd.color("0x3b82f6", "all")

                if highlight_residues:
                    resi = "+".join(str(r) for r in highlight_residues)
                    cmd.select("mutations", f"resi {resi}")
                    cmd.show("sticks", "mutations")
                    cmd.color("0xf59e0b", "mutations")
                    cmd.set("stick_radius", 0.25, "mutations")
                    if mutation_label:
                        cmd.label("mutations and name CA", f'"{mutation_label}"')
                        cmd.set("label_color", "0xf59e0b")
                        cmd.set("label_size", 14)
                        cmd.set("label_font_id", 7)

                if restraint_residues:
                    resi = "+".join(str(r) for r in restraint_residues)
                    cmd.select("restraints", f"resi {resi}")
                    cmd.show("sticks", "restraints")
                    cmd.color("0x10b981", "restraints")
                    cmd.set("stick_radius", 0.20, "restraints")

                cmd.orient("all")    # auto-orient to principal axes for best view
                cmd.zoom("all", -3)  # negative buffer = zoom in past tight fit
                cmd.deselect()
                self.structure_loaded = True
                self._surface_on = False
                self._dirty = True
                return True
            except Exception as e:
                log.error("PyMOL load failed: %s", e)
                return False
        return self._run_on_pymol_thread(_work)

    # ------------------------------------------------------------------ controls
    def rotate(self, x: float = 0, y: float = 0):
        def _work():
            if x:
                self._cmd.turn("x", x)
            if y:
                self._cmd.turn("y", y)
            self._dirty = True
        self._run_on_pymol_thread(_work)

    def zoom(self, factor: float = 1.0):
        def _work():
            delta = (factor - 1.0) * 8
            self._cmd.move("z", delta)
            self._dirty = True
        self._run_on_pymol_thread(_work)

    def set_style(self, style: str = "cartoon"):
        styles = {
            "cartoon": "cartoon",
            "sticks": "sticks",
            "spheres": "spheres",
            "surface": "surface",
        }
        def _work():
            self._cmd.hide("everything", "all")
            self._cmd.show(styles.get(style, "cartoon"), "all")
            for sel in ("mutations", "restraints"):
                try:
                    if self._cmd.count_atoms(sel):
                        self._cmd.show("sticks", sel)
                except Exception:
                    pass
            self._dirty = True
        self._run_on_pymol_thread(_work)

    def set_color(self, scheme: str = "chain"):
        def _work():
            cmd = self._cmd
            if scheme == "chain":
                # cmd.util.cbc has a default _self=cmd that points at the global
                # module-level cmd (not the pymol2 instance).  Must pass explicitly.
                cmd.util.cbc("all", _self=cmd)
            elif scheme == "spectrum":
                cmd.spectrum("count", "rainbow", "all")
            elif scheme == "plddt":
                cmd.spectrum("b", "red_yellow_green", "all", minimum=50, maximum=100)
            self._dirty = True
        self._run_on_pymol_thread(_work)

    def toggle_surface(self, opacity: float = 0.5):
        def _work():
            if self._surface_on:
                self._cmd.hide("surface", "all")
                self._surface_on = False
            else:
                self._cmd.show("surface", "all")
                self._cmd.set("transparency", opacity, "all")
                self._surface_on = True
            self._dirty = True
        self._run_on_pymol_thread(_work)

    def reset_view(self):
        def _work():
            self._cmd.orient("all")
            self._cmd.zoom("all", -3)
            self._dirty = True
        self._run_on_pymol_thread(_work)

    def run_command(self, command: str) -> str:
        """Execute an arbitrary PyMOL command string. Returns 'OK' or 'Error: ...'."""
        def _work():
            try:
                self._cmd.do(command)
                self._dirty = True
                return "OK"
            except Exception as e:
                return f"Error: {e}"
        return self._run_on_pymol_thread(_work)

    def select_residues(self, resi: List[int]):
        def _work():
            if resi:
                # "resi " prefix required — bare "1+5+10" is not a valid expression
                self._cmd.select("sele", "resi " + "+".join(str(r) for r in resi))
            else:
                self._cmd.deselect()
            self._dirty = True
        self._run_on_pymol_thread(_work)

    # ------------------------------------------------------------------ lifecycle
    def start_background(self):
        t = threading.Thread(
            target=self.serve_forever,
            daemon=True,
            name="pymol-viewer-server",
        )
        t.start()
        log.info("PyMOL viewer server running on port %d", self.server_address[1])


# ---------------------------------------------------------------------------
# Lightweight proxy — used by Streamlit to talk to the subprocess server
# ---------------------------------------------------------------------------
class PyMOLViewerProxy:
    """HTTP proxy to a separately-running PyMOL server subprocess."""

    def __init__(self, port: int):
        self._port = port
        self.structure_loaded = False
        self.server_address = ("localhost", port)

    def _request(self, endpoint: str, data: Optional[dict] = None) -> bytes:
        import urllib.request as _ur, json as _j
        url = f"http://localhost:{self._port}{endpoint}"
        if data is None:
            req = _ur.Request(url)
        else:
            body = _j.dumps(data).encode()
            req = _ur.Request(url, data=body, headers={"Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=30) as resp:
            return resp.read()

    def load_structure(
        self,
        pdb_data: str,
        name: str = "structure",
        highlight_residues: Optional[List[int]] = None,
        restraint_residues: Optional[List[int]] = None,
        mutation_label: str = "",
    ) -> bool:
        try:
            self._request("/api/load", {
                "pdb_data": pdb_data,
                "name": name,
                "highlight_residues": highlight_residues or [],
                "restraint_residues": restraint_residues or [],
                "mutation_label": mutation_label,
            })
            self.structure_loaded = True
            return True
        except Exception as e:
            log.error("load_structure failed: %s", e)
            return False

    def get_frame(self) -> bytes:
        return self._request("/api/frame")

    def reset_view(self):
        self._request("/api/reset", {})

    def set_style(self, style: str):
        self._request("/api/style", {"style": style})

    def set_color(self, scheme: str):
        self._request("/api/color", {"scheme": scheme})


# ---------------------------------------------------------------------------
# Singleton accessor — launches server subprocess on first call
# ---------------------------------------------------------------------------
def _find_pymol_python() -> str:
    """Return the Python interpreter that has pymol2 installed.

    Search order:
    1. $PYMOL_PYTHON env var (allows explicit override)
    2. sys.executable (current interpreter — usually correct when pymol2 is
       installed in the same conda/venv environment as the app)
    3. Common conda environment names on the local machine
    4. shutil.which("python3") / "python"
    """
    import os, shutil, subprocess, sys
    candidate = os.environ.get("PYMOL_PYTHON")
    if candidate and Path(candidate).exists():
        return candidate
    for py in [sys.executable, shutil.which("python3"), shutil.which("python")]:
        if not py or not Path(py).exists():
            continue
        try:
            r = subprocess.run(
                [py, "-c", "import pymol2"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                return py
        except Exception:
            pass
    # Last-resort: return sys.executable and let the subprocess fail with a
    # meaningful ImportError rather than a "file not found" error.
    return sys.executable


def _find_libstdcpp() -> str:
    """Return the path of the system libstdc++.so.6 (or empty string)."""
    import glob as _g
    patterns = [
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6",
        "/usr/lib/aarch64-linux-gnu/libstdc++.so.6",
        "/usr/lib/arm-linux-gnueabihf/libstdc++.so.6",
    ]
    for pat in patterns:
        matches = sorted(_g.glob(pat + "*"), reverse=True)
        if matches:
            return matches[0]
    return ""


_PYMOL_PYTHON: str = ""   # resolved lazily on first use
_LIBSTDCPP: str = ""      # resolved lazily on first use
_proxy_instance: Optional[PyMOLViewerProxy] = None
_proxy_lock = threading.Lock()


def _check_alive(port: int) -> bool:
    """Return True if the server on *port* responds to /api/health."""
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout=1)
        return True
    except Exception:
        return False


def get_pymol_server(port: int = PYMOL_SERVER_PORT) -> Optional[PyMOLViewerProxy]:
    """Get (or create) a proxy to the singleton PyMOL HTTP server subprocess.

    Launches the server as a separate process with LD_PRELOAD so the system
    libstdc++ is loaded before Anaconda's version (fixes GLIBCXX_3.4.30 error).

    Returns None if the server cannot be started.
    """
    global _proxy_instance
    with _proxy_lock:
        # Return cached proxy if server is still alive
        if _proxy_instance is not None:
            if _check_alive(_proxy_instance._port):
                return _proxy_instance
            log.warning("PyMOL server not responding; relaunching…")
            _proxy_instance = None

        # Server might be running from a previous Streamlit session
        for p in [port, port + 1, port + 2, port + 10]:
            if _check_alive(p):
                _proxy_instance = PyMOLViewerProxy(p)
                log.warning("Reattached to existing PyMOL server on port %d", p)
                return _proxy_instance

        # Launch a fresh subprocess
        import subprocess, os, time
        global _PYMOL_PYTHON, _LIBSTDCPP
        if not _PYMOL_PYTHON:
            _PYMOL_PYTHON = _find_pymol_python()
        if not _LIBSTDCPP:
            _LIBSTDCPP = _find_libstdcpp()

        python_bin = _PYMOL_PYTHON
        if not Path(python_bin).exists():
            log.error("Python not found at %s", python_bin)
            return None

        env = dict(os.environ)
        env["DISPLAY"] = env.get("DISPLAY") or ":1"
        if _LIBSTDCPP:
            env["LD_PRELOAD"] = _LIBSTDCPP

        server_script = str(Path(__file__).resolve())

        for p in [port, port + 1, port + 2, port + 10]:
            try:
                proc = subprocess.Popen(
                    [python_bin, server_script, "--serve", "--port", str(p)],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                log.error("Failed to launch PyMOL subprocess: %s", e)
                continue

            # Poll until healthy (max 15 s)
            for _ in range(30):
                time.sleep(0.5)
                if _check_alive(p):
                    _proxy_instance = PyMOLViewerProxy(p)
                    log.warning("PyMOL server subprocess started (pid=%d, port=%d)", proc.pid, p)
                    return _proxy_instance

            log.warning("PyMOL server subprocess did not start on port %d", p)
            try:
                proc.terminate()
            except Exception:
                pass

        log.error("Could not start PyMOL server on any port")
        return None


# ---------------------------------------------------------------------------
# Entry point — run as standalone server subprocess
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, ctypes

    # Ensure system libstdc++ symbols are loaded (LD_PRELOAD already handles it,
    # but be explicit for safety)
    _libpath = _find_libstdcpp()
    if _libpath:
        try:
            ctypes.CDLL(_libpath, ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    _parser = argparse.ArgumentParser(description="PyMOL HTTP viewer server")
    _parser.add_argument("--serve", action="store_true", required=True)
    _parser.add_argument("--port", type=int, default=PYMOL_SERVER_PORT)
    _args = _parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    srv = PyMOLViewerServer(port=_args.port)
    log.info("PyMOL HTTP server listening on port %d", _args.port)
    srv.serve_forever()
