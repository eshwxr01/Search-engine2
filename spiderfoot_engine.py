"""
spiderfoot_engine.py - Stage 5 OSINT engine (v2)

Given an image that earlier forensic stages flagged as suspicious, this module
gathers *leads* about where visually similar content appears online:

  1. fingerprints the evidence (SHA-256, MD5, pHash, dHash)
  2. slices it (full / subject / background) using real face detection when
     OpenCV is available, with a clearly-labelled fixed-crop fallback
  3. reverse-searches each slice (Playwright -> Yandex Images)
  4. pivots on usernames found in the hits and runs multi-platform dorks
  5. scrapes page metadata, downloads each page's primary image and compares
     it to the evidence with perceptual hashes, then ranks the leads
  6. resolves IPs and WHOIS data for the domains involved

IMPORTANT: everything here produces LEADS for a human analyst. A reverse-image
hit shows where similar content appears, not who created or owns it.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import random
import re
import socket
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import NamedTuple
from urllib.parse import urljoin, urlparse

import imagehash
import requests
import whois
from bs4 import BeautifulSoup
from ddgs import DDGS
from PIL import Image, ImageFilter, ImageOps
from playwright.sync_api import sync_playwright

from osint_utils import (
    domain_matches,
    host_of,
    is_blocked_domain,
    normalize_whois_date,
    normalize_whois_text,
    parse_timestamp_utc,
    registrable_domain,
)

try:  # optional: enables real face detection for the subject/background slices
    import cv2
    import numpy as np
except ImportError:  # tests simulate this by monkeypatching `cv2` to None
    cv2 = None
    np = None

logger = logging.getLogger(__name__)

ENGINE_VERSION = "2.0.0"

YANDEX_IMAGES_URL = "https://yandex.com/images/"
MAX_MATCHES_PER_SLICE = 10
MAX_DIAGNOSTICS = 300
FACE_DETECT_MAX_SIDE = 1280
MAX_SUBJECT_AREA_RATIO = 0.90

# Perceptual-hash Hamming distances (64-bit hashes). These are heuristics.
NEAR_DUPLICATE_MAX_DISTANCE = 6
SIMILAR_MAX_DISTANCE = 12

# Handles auto-extracted from URLs must look like real usernames.
HANDLE_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")

PLATFORMS = {
    "instagram.com": "INSTAGRAM",
    "x.com": "X (TWITTER)",
    "twitter.com": "X (TWITTER)",
    "facebook.com": "FACEBOOK",
    "reddit.com": "REDDIT",
    "youtube.com": "YOUTUBE",
    "youtu.be": "YOUTUBE",
    "t.me": "TELEGRAM",
    "pinterest.com": "PINTEREST",
    "linkedin.com": "LINKEDIN",
    "github.com": "GITHUB",
    "wikipedia.org": "WIKIPEDIA",
    "wikimedia.org": "WIKIMEDIA COMMONS",
    "espncricinfo.com": "ESPN CRICINFO",
    "ndtv.com": "NDTV",
    "indiatoday.in": "INDIA TODAY",
    "indiatimes.com": "TIMES OF INDIA",
    "hindustantimes.com": "HINDUSTAN TIMES",
}

# URL path segments that look like usernames but are not.
IGNORED_PATH_SEGMENTS = {
    "p", "reel", "reels", "stories", "explore", "status", "watch", "shorts",
    "hashtag", "search", "pin", "r", "user", "u", "i", "home", "intent",
    "share", "login", "accounts", "settings", "topics", "orgs", "features",
    "about", "joinchat", "channel", "c", "tv", "direct",
}


def _short(exc: Exception) -> str:
    """One-line, length-capped description of an exception (Playwright errors are multi-line)."""
    lines = str(exc).strip().splitlines()
    first_line = lines[0].strip() if lines else ""
    return f"{type(exc).__name__}: {first_line[:160]}"


class FetchError(Exception):
    """Raised for blocked, failed, or oversized fetches (never a silent skip)."""


class FetchResult(NamedTuple):
    url: str
    content_type: str
    content: bytes


class MiniSpiderFoot:
    def __init__(self, allow_private_hosts: bool = False, use_face_detection: bool = True,
                 delay_scale: float = 1.0):
        """
        allow_private_hosts: disable the SSRF guard (only for local tests).
        use_face_detection:  use OpenCV face detection for the subject slice.
        delay_scale:         multiplier for the polite random delays (0 in tests).
        """
        self.allow_private_hosts = allow_private_hosts
        self.use_face_detection = use_face_detection
        self.delay_scale = delay_scale
        self.diagnostics: list[dict] = []
        self._face_cascade = None  # None = not loaded, False = unavailable
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
        ]

    # ------------------------------------------------------------------ utils
    def _note(self, level: str, phase: str, message: str, **extra):
        """Record a diagnostic so failures are visible instead of silent."""
        entry = {"level": level, "phase": phase, "message": message}
        entry.update({k: v for k, v in extra.items() if v is not None})
        if len(self.diagnostics) < MAX_DIAGNOSTICS:
            self.diagnostics.append(entry)
        log_level = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}.get(level, logging.INFO)
        logger.log(log_level, "[%s] %s %s", phase, message, extra or "")

    def _sleep(self, low: float, high: float):
        delay = random.uniform(low, high) * self.delay_scale
        if delay > 0:
            time.sleep(delay)

    def _get_headers(self):
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    @staticmethod
    def _load_image(image_bytes: bytes) -> Image.Image:
        try:
            img = Image.open(BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

    # --------------------------------------------------------- safe networking
    def _is_public_host(self, host: str) -> bool:
        """SSRF guard: refuse loopback / private / link-local / reserved targets."""
        if self.allow_private_hosts:
            return True
        try:
            infos = socket.getaddrinfo(host, None)
        except (OSError, UnicodeError):
            return False
        if not infos:
            return False
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if getattr(ip, "ipv4_mapped", None):
                ip = ip.ipv4_mapped
            if not ip.is_global:
                return False
        return True

    def _safe_get(self, url: str, max_bytes: int = 2_000_000, timeout: int = 7,
                  max_redirects: int = 3, on_overflow: str = "truncate") -> FetchResult:
        """GET with redirect-by-redirect SSRF checks and a hard size cap.

        Note: a DNS-rebinding race between this check and the request is not
        fully closed; this guard is defence in depth for a local analyst tool.
        """
        current = url
        for _ in range(max_redirects + 1):
            parsed = urlparse(current)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise FetchError("unsupported or malformed URL")
            if not self._is_public_host(parsed.hostname):
                raise FetchError("blocked non-public or unresolvable host")
            resp = requests.get(current, headers=self._get_headers(), timeout=timeout,
                                stream=True, allow_redirects=False)
            try:
                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        raise FetchError("redirect without Location header")
                    current = urljoin(current, location)
                    continue
                if resp.status_code != 200:
                    raise FetchError(f"HTTP {resp.status_code}")
                content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                data = bytearray()
                for chunk in resp.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        if on_overflow == "error":
                            raise FetchError(f"response larger than {max_bytes} bytes")
                        del data[max_bytes:]
                        break
                return FetchResult(current, content_type, bytes(data))
            finally:
                resp.close()
        raise FetchError("too many redirects")

    # -------------------------------------------------------- URL understanding
    def identify_platform(self, url):
        host = host_of(url)
        for domain, name in PLATFORMS.items():
            if domain_matches(host, domain):
                return name
        return host.upper() if host else "WEB SOURCE"

    def extract_handle_from_url(self, url):
        host = host_of(url)
        parts = [p for p in urlparse(url).path.split("/") if p]
        if not host or not parts:
            return None

        if domain_matches(host, "reddit.com"):
            # /r/<subreddit> is a community, not a person: only /u/ and /user/ count.
            if parts[0].lower() in {"u", "user"} and len(parts) > 1:
                candidate = parts[1]
            else:
                return None
        elif domain_matches(host, "youtube.com") or domain_matches(host, "youtu.be"):
            first = parts[0]
            if first.startswith("@"):
                candidate = first
            elif first.lower() in {"c", "user"} and len(parts) > 1:
                candidate = parts[1]
            else:  # /watch, /shorts, /channel/<opaque id>, youtu.be/<video id>
                return None
        elif domain_matches(host, "t.me"):
            candidate = parts[1] if parts[0].lower() == "s" and len(parts) > 1 else parts[0]
        elif any(domain_matches(host, d) for d in ("instagram.com", "x.com", "twitter.com", "github.com", "pinterest.com")):
            candidate = parts[0]
        else:
            return None

        candidate = candidate.lstrip("@")
        if candidate.lower() in IGNORED_PATH_SEGMENTS or not HANDLE_RE.match(candidate):
            return None
        return candidate

    # ------------------------------------------------------------- fingerprint
    def fingerprint_image(self, image_bytes, filename):
        img = self._load_image(image_bytes)
        w, h = img.size
        return {
            "filename": filename,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "md5": hashlib.md5(image_bytes).hexdigest(),
            "phash": str(imagehash.phash(img)),
            "dhash": str(imagehash.dhash(img)),
            "resolution": f"{w}x{h}",
            "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ slicing
    def _get_face_cascade(self):
        if self._face_cascade is None:
            self._face_cascade = False
            if cv2 is None:
                self._note("info", "slicing", "OpenCV not installed; using fixed-heuristic subject crop.")
            else:
                haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
                path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml") if haar_dir else None
                cascade = cv2.CascadeClassifier(path) if path and os.path.exists(path) else None
                if cascade is not None and not cascade.empty():
                    self._face_cascade = cascade
                else:
                    self._note(
                        "warning", "slicing",
                        "OpenCV Haar face cascade not found (OpenCV 5.x wheels no longer bundle it). "
                        "Install opencv-python-headless<5 for face-based slicing; using fixed-heuristic crop.",
                    )
        return self._face_cascade or None

    def _detect_face_subject_box(self, img: Image.Image):
        """Union of detected faces, expanded to cover head, shoulders and torso."""
        cascade = self._get_face_cascade()
        if cascade is None:
            return None
        w, h = img.size
        scale = min(1.0, FACE_DETECT_MAX_SIDE / max(w, h))
        work = img if scale == 1.0 else img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2GRAY))
        min_side = max(24, int(min(work.size) * 0.05))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        if len(faces) == 0:
            return None

        inv = 1.0 / scale
        boxes = [(x * inv, y * inv, (x + fw) * inv, (y + fh) * inv) for x, y, fw, fh in faces]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        avg_w = sum(b[2] - b[0] for b in boxes) / len(boxes)
        avg_h = sum(b[3] - b[1] for b in boxes) / len(boxes)

        left = max(0, int(x0 - 0.6 * avg_w))
        right = min(w, int(x1 + 0.6 * avg_w))
        top = max(0, int(y0 - 0.6 * avg_h))
        bottom = min(h, int(y1 + 3.0 * avg_h))
        if right - left < 8 or bottom - top < 8:
            return None
        return (left, top, right, bottom)

    def _find_subject_box(self, img: Image.Image):
        if self.use_face_detection:
            box = self._detect_face_subject_box(img)
            if box:
                return box, "face_detection"
            self._note("info", "slicing", "No face detected; using fixed-heuristic subject region.")
        w, h = img.size
        return (int(w * 0.15), int(h * 0.05), int(w * 0.85), int(h * 0.65)), "fixed_heuristic"

    @staticmethod
    def _blur_out_region(img: Image.Image, box) -> Image.Image:
        """Return the image with the subject region heavily blurred, so a reverse
        search matches the *scenery* instead of the person."""
        w, h = img.size
        radius = max(8, max(w, h) // 25)
        region = img.crop(box).filter(ImageFilter.GaussianBlur(radius))
        out = img.copy()
        out.paste(region, (box[0], box[1]))
        return out

    @staticmethod
    def _make_slice(name, img, method, box):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return {
            "name": name,
            "bytes": buf.getvalue(),
            "method": method,
            "box": list(box) if box else None,
            "phash": str(imagehash.phash(img)),
            "dhash": str(imagehash.dhash(img)),
        }

    def generate_image_slices(self, image_bytes, enable_crops=True):
        """FULL image + (optionally) SUBJECT crop + BACKGROUND with subject masked."""
        img = self._load_image(image_bytes)
        w, h = img.size
        slices = [self._make_slice("FULL_IMAGE", img, "whole_image", None)]
        if not enable_crops:
            return slices

        box, method = self._find_subject_box(img)
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
        if area_ratio > MAX_SUBJECT_AREA_RATIO:
            self._note("info", "slicing", "Subject fills most of the frame; crop/background slices skipped.")
            return slices

        slices.append(self._make_slice("SUBJECT_FOCUS_CROP", img.crop(box), method, box))
        slices.append(self._make_slice("CONTEXT_BACKGROUND_CROP", self._blur_out_region(img, box), method, box))
        return slices

    # ------------------------------------------------------ visual reverse search
    def _browser_channels(self):
        """Browser launch order. OSINT_BROWSER_CHANNEL=msedge|chrome|chromium overrides."""
        forced = os.environ.get("OSINT_BROWSER_CHANNEL", "").strip().lower()
        if forced:
            return [None if forced == "chromium" else forced]
        return ["msedge", "chrome", None]  # None = Playwright's bundled Chromium

    def _launch_browser(self, p):
        last_error = None
        for channel in self._browser_channels():
            label = channel or "bundled-chromium"
            try:
                if channel is None:
                    return p.chromium.launch(headless=True), label
                return p.chromium.launch(headless=True, channel=channel), label
            except Exception as exc:
                last_error = exc
                self._note("info", "visual_search", f"Browser '{label}' unavailable: {_short(exc)}")
        raise RuntimeError(
            "No usable browser found. Run `playwright install chromium` "
            "(or install Microsoft Edge / Google Chrome)."
        ) from last_error

    @staticmethod
    def _looks_like_captcha(page) -> bool:
        try:
            if "captcha" in page.url.lower():
                return True
            return page.get_by_text(re.compile(r"not a robot|verify you are human", re.I)).count() > 0
        except Exception:
            return False

    def _search_single_slice(self, browser, slice_info, seen_urls):
        slice_name = slice_info["name"]
        matches = []
        tmp_path = None
        context = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(slice_info["bytes"])
                tmp_path = tmp.name

            context = browser.new_context(
                user_agent=random.choice(self.user_agents),
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
            page.goto(YANDEX_IMAGES_URL, timeout=25000)
            page.wait_for_timeout(random.randint(1500, 2800))
            if self._looks_like_captcha(page):
                self._note("warning", "visual_search",
                           "Yandex showed a CAPTCHA before upload; results for this slice are missing, not empty.",
                           slice=slice_name)
                return matches

            file_input = page.locator("input.cbir-panel__file-input, input[type='file']").first
            if file_input.count() == 0:
                page.locator(".cbir-button, .HeaderDesktopActions-CbirButton").first.click(timeout=5000)
                page.wait_for_timeout(1000)
                file_input = page.locator("input[type='file']").last

            file_input.set_input_files(tmp_path, timeout=10000)
            page.wait_for_timeout(random.randint(6000, 8000))
            if self._looks_like_captcha(page):
                self._note("warning", "visual_search",
                           "Yandex showed a CAPTCHA after upload; results for this slice are missing, not empty.",
                           slice=slice_name)
                return matches

            sites_tab = page.get_by_text("Sites", exact=True)
            if sites_tab.count() > 0:
                sites_tab.first.click()
                page.wait_for_timeout(random.randint(2500, 3500))

            anchors = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => ({href: e.href || '', text: (e.innerText || '').trim(),"
                " title: e.getAttribute('title') || e.getAttribute('aria-label') || ''}))",
            )
            for anchor in anchors:
                href = anchor.get("href", "")
                if not href.startswith(("http://", "https://")) or href in seen_urls:
                    continue
                host = host_of(href)
                if not host or is_blocked_domain(host):
                    continue
                title_text = " ".join((anchor.get("text") or "").split())
                if len(title_text) < 4:
                    title_text = anchor.get("title") or host
                seen_urls.add(href)
                matches.append({
                    "platform": self.identify_platform(href),
                    "url": href,
                    "page_title": title_text[:100],
                    "search_slice": slice_name,
                    "match_type": f"VISUAL MATCH ({slice_name})",
                })
                if len(matches) >= MAX_MATCHES_PER_SLICE:
                    break

            if not matches:
                self._note("warning", "visual_search",
                           "No result links extracted: Yandex may have changed its layout, blocked the "
                           "request, or truly has no matches. Do not read this as proof of absence.",
                           slice=slice_name)
        except Exception as exc:
            self._note("error", "visual_search", _short(exc), slice=slice_name)
        finally:
            if context is not None:
                try:
                    context.close()
                except Exception:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        return matches

    def multi_crop_visual_reverse_search(self, image_bytes, enable_crops=True, slices=None):
        if slices is None:
            slices = self.generate_image_slices(image_bytes, enable_crops=enable_crops)
        elif not enable_crops:
            slices = slices[:1]

        all_matches, seen_urls = [], set()
        try:
            with sync_playwright() as p:
                browser, label = self._launch_browser(p)
                self._note("info", "visual_search", f"Using browser: {label}")
                try:
                    for idx, slice_info in enumerate(slices):
                        all_matches.extend(self._search_single_slice(browser, slice_info, seen_urls))
                        if idx < len(slices) - 1:
                            self._sleep(1.0, 2.2)
                finally:
                    browser.close()
        except Exception as exc:
            self._note("error", "visual_search", f"Visual reverse search aborted: {_short(exc)}")
        return all_matches

    # --------------------------------------------------------- page intelligence
    @staticmethod
    def _meta_content(soup, *keys):
        for key in keys:
            for attr in ("property", "name", "itemprop"):
                tag = soup.find("meta", attrs={attr: key})
                if tag and tag.get("content"):
                    return tag["content"].strip()
        return ""

    def _extract_timestamp(self, soup):
        """Returns (raw_value, kind). Only kind == 'published' should be used for
        'earliest seen' reasoning; 'updated' and bare <time> tags are weaker."""
        raw = self._meta_content(soup, "article:published_time", "datePublished")
        if raw:
            return raw, "published"
        raw = self._meta_content(soup, "og:updated_time", "article:modified_time", "dateModified")
        if raw:
            return raw, "updated"
        time_tag = soup.find("time")
        if time_tag:
            raw = time_tag.get("datetime") or time_tag.get_text(strip=True)
            if raw:
                return raw, "time_tag_unverified"
        return None, None

    @staticmethod
    def _strength(phash_distance: int) -> str:
        if phash_distance <= NEAR_DUPLICATE_MAX_DISTANCE:
            return "NEAR_DUPLICATE"
        if phash_distance <= SIMILAR_MAX_DISTANCE:
            return "SIMILAR"
        return "WEAK"

    def _compare_candidate_image(self, image_url, evidence_hashes):
        """Download a page's primary image and compare it to every evidence slice."""
        fetched = self._safe_get(image_url, max_bytes=5_000_000, on_overflow="error")
        if not fetched.content_type.startswith("image/"):
            raise FetchError(f"not an image ({fetched.content_type or 'unknown type'})")
        img = self._load_image(fetched.content)
        cand_p, cand_d = imagehash.phash(img), imagehash.dhash(img)

        best = None
        for ev in evidence_hashes:
            p_dist = cand_p - imagehash.hex_to_hash(ev["phash"])
            d_dist = cand_d - imagehash.hex_to_hash(ev["dhash"])
            if best is None or (p_dist, d_dist) < (best["phash_distance"], best["dhash_distance"]):
                best = {"best_matching_slice": ev["name"], "phash_distance": int(p_dist), "dhash_distance": int(d_dist)}
        best["strength"] = self._strength(best["phash_distance"])
        best["note"] = "Perceptual-hash similarity of the page's primary image; a lead, not proof."
        return best

    def scrape_page_metadata(self, urls, max_pages=6, evidence_hashes=None):
        extracted = []
        for url in urls[:max_pages]:
            try:
                fetched = self._safe_get(url, max_bytes=2_000_000, on_overflow="truncate")
            except FetchError as exc:
                self._note("info", "page_metadata", f"Skipped: {exc}", url=url)
                continue
            except requests.RequestException as exc:
                self._note("info", "page_metadata", f"Request failed: {_short(exc)}", url=url)
                continue

            if "html" not in fetched.content_type:
                self._note("info", "page_metadata", f"Skipped non-HTML content ({fetched.content_type or 'unknown'}).", url=url)
                continue

            try:
                soup = BeautifulSoup(fetched.content, "html.parser")
                title = self._meta_content(soup, "og:title") or (soup.title.get_text(strip=True) if soup.title else "")
                description = self._meta_content(soup, "og:description", "description")
                raw_ts, ts_kind = self._extract_timestamp(soup)

                base_host = host_of(fetched.url)
                outbound = []
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if not href.lower().startswith(("http://", "https://")):
                        continue
                    link_host = host_of(href)
                    if (not link_host or registrable_domain(link_host) == registrable_domain(base_host)
                            or is_blocked_domain(link_host)):
                        continue
                    if href not in outbound:
                        outbound.append(href)
                    if len(outbound) >= 4:
                        break

                image_ref = self._meta_content(soup, "og:image", "twitter:image")
                image_url = urljoin(fetched.url, image_ref) if image_ref else None

                comparison = None
                if image_url and evidence_hashes:
                    try:
                        comparison = self._compare_candidate_image(image_url, evidence_hashes)
                    except (FetchError, requests.RequestException, ValueError) as exc:
                        self._note("info", "image_compare", f"Could not compare page image: {exc}", url=url)

                extracted.append({
                    "source_url": url,
                    "platform": self.identify_platform(url),
                    "meta_title": title[:100],
                    "meta_description": description[:180],
                    "detected_timestamp": raw_ts,
                    "detected_timestamp_utc": parse_timestamp_utc(raw_ts),
                    "timestamp_kind": ts_kind,
                    "outbound_links": outbound,
                    "candidate_image_url": image_url,
                    "image_comparison": comparison,
                })
            except Exception as exc:  # parsing must never kill the whole scan, but it must be reported
                self._note("warning", "page_metadata", f"Parse error: {_short(exc)}", url=url)
            self._sleep(0.4, 1.1)
        return extracted

    # --------------------------------------------------------------- dorking
    @staticmethod
    def _clean_seed_username(username: str) -> str:
        name = (username or "").strip().lstrip("@")
        name = re.sub(r'["\\\r\n\t]', "", name)
        return name[:64]

    def scrambled_deep_osint_search(self, username="", extra_keywords=None):
        queries = []
        clean_user = self._clean_seed_username(username)

        if clean_user:
            variants = list(dict.fromkeys([clean_user, clean_user.replace(".", "")]))
            for v in variants:
                queries.extend([
                    ("INSTAGRAM", f'site:instagram.com "{v}"'),
                    ("X (TWITTER)", f'site:x.com "{v}"'),
                    ("YOUTUBE", f'site:youtube.com "@{v}"'),
                    ("REDDIT", f'site:reddit.com "{v}"'),
                    ("FACEBOOK", f'site:facebook.com "{v}"'),
                    ("LINKEDIN", f'site:linkedin.com/in "{v}"'),
                    ("GITHUB", f'site:github.com "{v}"'),
                    ("TELEGRAM", f'site:t.me "{v}"'),
                    ("PINTEREST", f'site:pinterest.com "{v}"'),
                    ("WEB_MENTION", f'"{v}" -site:instagram.com -site:facebook.com'),
                ])

        for kw in extra_keywords or []:
            kw = (kw or "").replace('"', "").strip()
            if len(kw) > 5:
                queries.append(("CONTEXT_LEAD", f'"{kw[:60]}"'))

        if not queries:
            return []
        random.shuffle(queries)

        found_accounts, seen_urls = [], set()
        failures, first_error = 0, None
        needle = clean_user.lower().replace(".", "")
        try:
            with DDGS() as ddgs:
                for category, query in queries:
                    try:
                        results = list(ddgs.text(query, max_results=3))
                    except Exception as exc:
                        failures += 1
                        first_error = first_error or _short(exc)
                        continue
                    for r in results:
                        href = r.get("href") or ""
                        if not href or href in seen_urls:
                            continue
                        host = host_of(href)
                        if is_blocked_domain(host):
                            continue
                        seen_urls.add(href)
                        title = (r.get("title") or "")[:95]
                        body = (r.get("body") or "")[:200]
                        haystack = f"{href} {title} {body}".lower().replace(".", "")
                        found_accounts.append({
                            "platform": self.identify_platform(href),
                            "query_category": category,
                            "matched_handle": clean_user or "CONTEXT_MATCH",
                            "handle_in_result": bool(needle) and needle in haystack,
                            "query_used": query,
                            "url": href,
                            "display_title": title,
                            "bio_summary": body,
                        })
                    self._sleep(0.35, 0.95)
        except Exception as exc:
            self._note("error", "dorking", f"Search client failed: {_short(exc)}")

        if failures:
            level = "warning"
            msg = f"{failures} of {len(queries)} dork queries failed (first error: {first_error})."
            if failures == len(queries):
                msg += " ALL queries failed (likely rate-limited): zero results here is NOT evidence of absence."
            self._note(level, "dorking", msg)
        return found_accounts

    # ----------------------------------------------------------- infrastructure
    def inspect_multiple_domains(self, domains, max_domains=8):
        reports = []
        unique = [d for d in dict.fromkeys(domains) if d and not is_blocked_domain(d)]
        for domain in unique[:max_domains]:
            try:
                ip = socket.gethostbyname(domain)
            except (OSError, UnicodeError):
                ip = "Unresolved"

            registrar, created = "Unknown", "Unknown"
            reg_domain = registrable_domain(domain)
            try:
                w = whois.whois(reg_domain)
                registrar = normalize_whois_text(w.registrar) or "Unknown"
                created = normalize_whois_date(w.creation_date) or "Unknown"
            except Exception as exc:
                self._note("info", "infrastructure", f"WHOIS failed for {reg_domain}: {_short(exc)}")

            reports.append({
                "domain": domain,
                "registrable_domain": reg_domain,
                "server_ip": ip,
                "registrar": registrar,
                "creation_date": created,
            })
        return reports

    # -------------------------------------------------------------- lead ranking
    @staticmethod
    def _strength_rank(comparison):
        order = {"NEAR_DUPLICATE": 0, "SIMILAR": 1, "WEAK": 2}
        return order.get((comparison or {}).get("strength"), 3)

    def rank_leads(self, urls, visual_sources, page_metadata):
        meta_by_url = {m["source_url"]: m for m in page_metadata}
        title_by_url = {s["url"]: s.get("page_title", "") for s in visual_sources}
        leads = []
        for url in dict.fromkeys(urls):
            meta = meta_by_url.get(url)
            leads.append({
                "source_url": url,
                "platform": self.identify_platform(url),
                "page_title": (meta or {}).get("meta_title") or title_by_url.get(url, ""),
                "analyzed": meta is not None,
                "image_comparison": (meta or {}).get("image_comparison"),
                "published_timestamp_utc": (meta or {}).get("detected_timestamp_utc"),
                "timestamp_kind": (meta or {}).get("timestamp_kind"),
            })

        def sort_key(lead):
            comparison = lead["image_comparison"]
            distance = comparison["phash_distance"] if comparison else 64
            return (self._strength_rank(comparison), distance, lead["published_timestamp_utc"] or "9999")

        leads.sort(key=sort_key)

        strong = [l for l in leads if self._strength_rank(l["image_comparison"]) <= 1]
        published = [
            (datetime.fromisoformat(l["published_timestamp_utc"]), l["source_url"])
            for l in strong
            if l["timestamp_kind"] == "published" and l["published_timestamp_utc"]
        ]
        summary = {
            "leads_total": len(leads),
            "leads_with_image_comparison": sum(1 for l in leads if l["image_comparison"]),
            "best_visual_lead": strong[0]["source_url"] if strong else None,
            "earliest_published_lead": min(published)[1] if published else None,
        }
        return leads, summary

    # ------------------------------------------------------------------ pipeline
    def run_full_recon(self, image_bytes, filename, seed_url="", seed_username="",
                       enable_crops=True, auto_pivot=True):
        self.diagnostics = []
        img_meta = self.fingerprint_image(image_bytes, filename)
        slices = self.generate_image_slices(image_bytes, enable_crops=enable_crops)
        evidence_hashes = [{"name": s["name"], "phash": s["phash"], "dhash": s["dhash"]} for s in slices]

        seed_url = (seed_url or "").strip()
        if seed_url and not seed_url.lower().startswith(("http://", "https://")):
            self._note("warning", "input", "Seed URL ignored: it must start with http:// or https://")
            seed_url = ""

        visual_sources = self.multi_crop_visual_reverse_search(image_bytes, enable_crops=enable_crops, slices=slices)

        discovered_handles, title_keywords = [], []
        for src in visual_sources:
            handle = self.extract_handle_from_url(src.get("url", ""))
            if handle and handle not in discovered_handles:
                discovered_handles.append(handle)
            title = src.get("page_title", "")
            if len(title) > 12 and "yandex" not in title.lower():
                title_keywords.append(title)

        pivot_username, pivot_origin = None, None
        analyst_user = self._clean_seed_username(seed_username)
        if analyst_user:
            pivot_username, pivot_origin = analyst_user, "analyst_supplied"
        elif auto_pivot and discovered_handles:
            pivot_username, pivot_origin = discovered_handles[0], "auto_first_visual_hit"

        linked_accounts = self.scrambled_deep_osint_search(
            username=pivot_username or "", extra_keywords=title_keywords[:3]
        )

        page_urls = ([seed_url] if seed_url else []) + [s["url"] for s in visual_sources]
        page_urls = list(dict.fromkeys(page_urls))
        page_metadata = self.scrape_page_metadata(page_urls, max_pages=6, evidence_hashes=evidence_hashes)

        candidate_hosts = []
        if seed_url:
            candidate_hosts.append(host_of(seed_url))
        candidate_hosts += [host_of(s["url"]) for s in visual_sources]
        candidate_hosts += [host_of(a["url"]) for a in linked_accounts]
        infra = self.inspect_multiple_domains([h for h in candidate_hosts if h])

        ranked_leads, lead_summary = self.rank_leads(page_urls, visual_sources, page_metadata)

        caveats = [
            "A reverse-image hit shows where similar content appears, not who created, owns or posted it. "
            "The account may be the victim, a repost, or the original photographer.",
            "Handles and profiles are associated leads. Verify manually before attributing anything to a person.",
            "Image similarity compares perceptual hashes of each page's primary (og:image) picture to the "
            "evidence slices. It prioritises leads; it does not prove a donor/original relationship.",
            "'earliest_published_lead' only uses explicit published-time metadata, which pages can omit or fake.",
        ]
        if pivot_origin == "auto_first_visual_hit":
            caveats.append(
                f"Pivot handle '@{pivot_username}' was auto-selected from the first visual hit and is UNVERIFIED. "
                "Supply a handle yourself or disable auto-pivot to avoid anchoring on the wrong account."
            )

        return {
            "engine_version": ENGINE_VERSION,
            "evidence_profile": img_meta,
            "seed_inputs": {
                "provided_url": seed_url or None,
                "pivot_username": pivot_username,
                "pivot_username_origin": pivot_origin,
                "auto_discovered_handles": discovered_handles,
            },
            "search_slices": [{k: v for k, v in s.items() if k != "bytes"} for s in slices],
            "osint_findings": {
                "linked_social_accounts": linked_accounts,
                "other_web_sources": visual_sources,
                "page_metadata_intel": page_metadata,
                "infrastructure_intel": infra,
                "ranked_leads": ranked_leads,
                "lead_summary": lead_summary,
            },
            "analysis_caveats": caveats,
            "engine_diagnostics": list(self.diagnostics),
        }
