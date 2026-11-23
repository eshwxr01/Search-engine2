"""Shared test helpers (synthetic images and a tiny local web server)."""
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


def make_structured_image(size=(400, 300)):
    """A gradient with shapes: has enough structure for perceptual hashing."""
    w, h = size
    img = Image.new("RGB", size, (30, 60, 120))
    d = ImageDraw.Draw(img)
    for i in range(h):
        d.line([(0, i), (w, i)], fill=(i * 255 // h, 80, 255 - i * 255 // h))
    d.ellipse((w * 0.3, h * 0.1, w * 0.7, h * 0.6), fill=(240, 200, 170))
    d.rectangle((w * 0.1, h * 0.65, w * 0.9, h * 0.95), fill=(20, 20, 20))
    return img


def to_bytes(img, fmt="JPEG", **kwargs):
    buf = BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def structured_bytes(size=(400, 300)):
    return to_bytes(make_structured_image(size), quality=92)


def noise_bytes(size=(400, 300), seed=7):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return to_bytes(Image.fromarray(arr), quality=92)


class QuietHandler(SimpleHTTPRequestHandler):
    """Serves files from a directory; /redir -> 302 to /page.html; /loop -> redirect loop."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", "/page.html")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        else:
            super().do_GET()


def start_server(directory):
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
