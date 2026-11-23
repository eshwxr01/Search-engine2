import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

import spiderfoot_engine as se
from helpers import make_structured_image, noise_bytes, structured_bytes, to_bytes
from spiderfoot_engine import FetchError, MiniSpiderFoot


# ----------------------------------------------------------------- platforms
@pytest.mark.parametrize("url,expected", [
    ("https://www.instagram.com/someone/", "INSTAGRAM"),
    ("https://x.com/someone", "X (TWITTER)"),
    ("https://mobile.twitter.com/someone", "X (TWITTER)"),
    ("https://t.me/somechannel", "TELEGRAM"),
    ("https://youtu.be/abc123", "YOUTUBE"),
    ("https://en.wikipedia.org/wiki/X", "WIKIPEDIA"),
    # the old substring matching mislabelled these
    ("https://box.com/s/abc", "BOX.COM"),
    ("https://smart.me/page", "SMART.ME"),
    ("https://notgithub.com/x", "NOTGITHUB.COM"),
    ("", "WEB SOURCE"),
])
def test_identify_platform(engine, url, expected):
    assert engine.identify_platform(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("https://instagram.com/virat.kohli/", "virat.kohli"),
    ("https://instagram.com/p/Cabc123/", None),
    ("https://instagram.com/reel/Cabc123/", None),
    ("https://x.com/someone/status/123", "someone"),
    ("https://x.com/home", None),
    ("https://twitter.com/@someone", "someone"),
    ("https://github.com/eshwxr01", "eshwxr01"),
    ("https://github.com/orgs/foo/people", None),
    ("https://t.me/s/channelname", "channelname"),
    ("https://t.me/joinchat/AAAA", None),
    ("https://www.youtube.com/@creator", "creator"),
    ("https://www.youtube.com/c/CreatorName", "CreatorName"),
    ("https://www.youtube.com/channel/UCxxxxxxxxxxxx", None),   # opaque id, not a handle
    ("https://www.youtube.com/watch?v=abc", None),
    ("https://youtu.be/dQw4w9WgXcQ", None),
    ("https://reddit.com/r/python", None),                      # subreddit != person
    ("https://reddit.com/u/some_user", "some_user"),
    ("https://reddit.com/user/some_user/comments", "some_user"),
    ("https://box.com/someone", None),                          # not a social domain
    ("https://instagram.com/ab", None),                         # too short
    ("https://instagram.com/bad%20name", None),                 # invalid characters
])
def test_extract_handle_from_url(engine, url, expected):
    assert engine.extract_handle_from_url(url) == expected


# --------------------------------------------------------------- fingerprint
def test_fingerprint_is_correct_and_stable(engine, evidence_bytes):
    meta = engine.fingerprint_image(evidence_bytes, "evidence.jpg")
    assert meta["sha256"] == hashlib.sha256(evidence_bytes).hexdigest()
    assert meta["md5"] == hashlib.md5(evidence_bytes).hexdigest()
    assert meta["resolution"] == "400x300"
    assert len(meta["phash"]) == 16 and len(meta["dhash"]) == 16
    again = engine.fingerprint_image(evidence_bytes, "evidence.jpg")
    assert (meta["sha256"], meta["phash"], meta["dhash"]) == (again["sha256"], again["phash"], again["dhash"])


def test_undecodable_image_raises_clear_error(engine):
    with pytest.raises(ValueError, match="Could not decode image"):
        engine.fingerprint_image(b"this is not an image", "x.jpg")
    with pytest.raises(ValueError):
        engine.generate_image_slices(b"nope")


def test_exif_orientation_is_applied():
    img = make_structured_image((400, 300))
    exif = img.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise when displayed
    data = to_bytes(img, exif=exif)
    loaded = MiniSpiderFoot._load_image(data)
    assert loaded.size == (300, 400)


# ------------------------------------------------------------------- slicing
def test_crops_disabled_gives_single_full_slice(engine, evidence_bytes):
    slices = engine.generate_image_slices(evidence_bytes, enable_crops=False)
    assert [s["name"] for s in slices] == ["FULL_IMAGE"]
    assert slices[0]["method"] == "whole_image"


def test_heuristic_slices_when_face_detection_off(evidence_bytes):
    engine = MiniSpiderFoot(use_face_detection=False, delay_scale=0)
    slices = engine.generate_image_slices(evidence_bytes)
    assert [s["name"] for s in slices] == ["FULL_IMAGE", "SUBJECT_FOCUS_CROP", "CONTEXT_BACKGROUND_CROP"]
    subject = slices[1]
    assert subject["method"] == "fixed_heuristic"
    assert subject["box"] == [60, 15, 340, 195]          # 400x300 image
    assert all(len(s["phash"]) == 16 and s["bytes"][:2] == b"\xff\xd8" for s in slices)


def test_blur_out_region_only_changes_the_subject_box():
    img = make_structured_image((400, 300))
    box = (100, 50, 300, 250)
    out = MiniSpiderFoot._blur_out_region(img, box)
    a, b = np.asarray(img).astype(int), np.asarray(out).astype(int)
    assert np.array_equal(a[:50], b[:50])                # above the box: untouched
    assert np.array_equal(a[:, :100], b[:, :100])        # left of the box: untouched
    assert np.abs(a[50:250, 100:300] - b[50:250, 100:300]).sum() > 0   # inside: blurred


class FakeCascade:
    def __init__(self, faces):
        self.faces = np.array(faces, dtype=int).reshape(-1, 4)

    def detectMultiScale(self, gray, **kwargs):
        return self.faces


def test_face_box_is_expanded_to_head_shoulders_and_torso(evidence_bytes):
    pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    engine._face_cascade = FakeCascade([[100, 80, 60, 60]])
    slices = engine.generate_image_slices(evidence_bytes)
    subject = slices[1]
    assert subject["method"] == "face_detection"
    assert subject["box"] == [64, 44, 196, 300]          # padded sideways/up, clamped at the bottom


def test_subject_filling_the_frame_skips_extra_slices(evidence_bytes):
    pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    engine._face_cascade = FakeCascade([[0, 0, 400, 300]])
    slices = engine.generate_image_slices(evidence_bytes)
    assert [s["name"] for s in slices] == ["FULL_IMAGE"]
    assert any("fills most of the frame" in d["message"] for d in engine.diagnostics)


def test_missing_opencv_falls_back_with_a_note(monkeypatch, evidence_bytes):
    monkeypatch.setattr(se, "cv2", None)
    engine = MiniSpiderFoot(delay_scale=0)
    slices = engine.generate_image_slices(evidence_bytes)
    assert slices[1]["method"] == "fixed_heuristic"
    assert any("OpenCV not installed" in d["message"] for d in engine.diagnostics)


def test_real_haar_cascade_loads_with_opencv_4():
    cv2 = pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    assert engine._get_face_cascade() is not None, (
        f"Haar cascade missing in OpenCV {cv2.__version__}: install opencv-python-headless<5"
    )


# ------------------------------------------------------- SSRF / safe fetching
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.0.0.5", "192.168.1.10",
                                  "169.254.169.254", "::1", "0.0.0.0"])
def test_private_hosts_are_refused_by_default(strict_engine, host):
    assert strict_engine._is_public_host(host) is False


def test_public_ip_literal_is_allowed(strict_engine):
    assert strict_engine._is_public_host("8.8.8.8") is True


def test_safe_get_refuses_loopback_server(strict_engine, web):
    folder, base = web
    (folder / "page.html").write_text("<html></html>")
    with pytest.raises(FetchError, match="non-public"):
        strict_engine._safe_get(f"{base}/page.html")


def test_safe_get_rejects_non_http_schemes(engine):
    for url in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"):
        with pytest.raises(FetchError):
            engine._safe_get(url)


def test_safe_get_follows_redirects_and_caps_loops(engine, web):
    folder, base = web
    (folder / "page.html").write_text("<html><title>hi</title></html>")
    result = engine._safe_get(f"{base}/redir")
    assert result.url.endswith("/page.html") and b"<title>hi" in result.content
    with pytest.raises(FetchError, match="too many redirects"):
        engine._safe_get(f"{base}/loop")


def test_safe_get_size_limits(engine, web):
    folder, base = web
    (folder / "big.html").write_bytes(b"<html>" + b"a" * 10_000)
    assert len(engine._safe_get(f"{base}/big.html", max_bytes=1000, on_overflow="truncate").content) == 1000
    with pytest.raises(FetchError, match="larger than"):
        engine._safe_get(f"{base}/big.html", max_bytes=1000, on_overflow="error")


def test_safe_get_reports_http_errors(engine, web):
    _, base = web
    with pytest.raises(FetchError, match="HTTP 404"):
        engine._safe_get(f"{base}/missing.html")


# ----------------------------------------------- page metadata + image compare
def _evidence_hashes(engine, data):
    return [{"name": s["name"], "phash": s["phash"], "dhash": s["dhash"]}
            for s in engine.generate_image_slices(data)]


def test_scrape_page_metadata_end_to_end(engine, web, evidence_bytes):
    folder, base = web
    (folder / "evidence.jpg").write_bytes(evidence_bytes)
    (folder / "noise.jpg").write_bytes(noise_bytes())
    (folder / "dup.html").write_text(f"""<html><head>
        <title>fallback title</title>
        <meta property="og:title" content="Original &lt;b&gt;Post&lt;/b&gt;">
        <meta property="og:description" content="desc here">
        <meta property="article:published_time" content="2024-03-01T10:00:00Z">
        <meta property="og:image" content="/evidence.jpg">
        </head><body>
        <a href="{base}/same-site">same</a>
        <a href="http://other.example/x">external</a>
        <a href="https://www.google.com/search?q=x">blocked brand</a>
        <a href="/relative">relative</a>
        </body></html>""")
    (folder / "weak.html").write_text("""<html><head>
        <meta property="og:title">
        <meta property="og:updated_time" content="2025-01-01T00:00:00+00:00">
        <meta property="og:image" content="/noise.jpg"></head><title></title></html>""")
    (folder / "bare.html").write_text("<html><head><title>  Just a title  </title></head><body><time datetime='2020-05-05'>x</time></body></html>")

    hashes = _evidence_hashes(engine, evidence_bytes)
    urls = [f"{base}/dup.html", f"{base}/weak.html", f"{base}/bare.html", f"{base}/missing.html"]
    result = engine.scrape_page_metadata(urls, max_pages=10, evidence_hashes=hashes)

    assert len(result) == 3                              # the 404 is skipped, but reported below
    dup, weak, bare = result

    assert dup["meta_title"] == "Original <b>Post</b>"
    assert dup["meta_description"] == "desc here"
    assert dup["detected_timestamp_utc"] == "2024-03-01T10:00:00+00:00"
    assert dup["timestamp_kind"] == "published"
    assert dup["outbound_links"] == ["http://other.example/x"]   # no same-site, no blocked brand, no relative
    assert dup["image_comparison"]["strength"] == "NEAR_DUPLICATE"
    assert dup["image_comparison"]["best_matching_slice"] == "FULL_IMAGE"

    # og:title without a content attribute used to raise TypeError and silently drop the whole page
    assert weak["meta_title"] == ""
    assert weak["timestamp_kind"] == "updated"
    assert weak["image_comparison"]["strength"] == "WEAK"

    assert bare["meta_title"] == "Just a title"
    assert bare["timestamp_kind"] == "time_tag_unverified"
    assert bare["image_comparison"] is None

    assert any("HTTP 404" in d["message"] for d in engine.diagnostics)


def test_scrape_skips_non_html_and_reports_it(engine, web, evidence_bytes):
    folder, base = web
    (folder / "pic.jpg").write_bytes(evidence_bytes)
    assert engine.scrape_page_metadata([f"{base}/pic.jpg"]) == []
    assert any("non-HTML" in d["message"] for d in engine.diagnostics)


def test_scrape_blocks_private_hosts_in_production_mode(strict_engine, web):
    folder, base = web
    (folder / "page.html").write_text("<html><title>x</title></html>")
    assert strict_engine.scrape_page_metadata([f"{base}/page.html"]) == []
    assert any("non-public" in d["message"] for d in strict_engine.diagnostics)


# ------------------------------------------------------------------- dorking
class FakeDDGS:
    """Stand-in for ddgs.DDGS. `behaviour` maps a substring of the query to results or an Exception."""
    behaviour = {}
    default = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=3):
        for needle, outcome in self.behaviour.items():
            if needle in query:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        return list(self.default)


def test_dorking_flags_relevance_and_skips_blocked(monkeypatch, engine):
    FakeDDGS.behaviour = {}
    FakeDDGS.default = [
        {"href": "https://github.com/some.user", "title": "some.user (Some User)", "body": "Developer"},
        {"href": "https://www.google.com/search?q=x", "title": "noise", "body": ""},
        {"href": "https://example.org/unrelated", "title": "Totally unrelated", "body": None},
        {"href": "https://example.org/blank", "title": None, "body": None},
    ]
    monkeypatch.setattr(se, "DDGS", FakeDDGS)
    found = engine.scrambled_deep_osint_search(username="@some.user")
    by_url = {f["url"]: f for f in found}
    assert "https://www.google.com/search?q=x" not in by_url
    assert by_url["https://github.com/some.user"]["handle_in_result"] is True
    assert by_url["https://github.com/some.user"]["platform"] == "GITHUB"
    assert by_url["https://example.org/unrelated"]["handle_in_result"] is False
    assert by_url["https://example.org/blank"]["display_title"] == ""     # None handled, no TypeError


def test_dorking_total_failure_is_loud(monkeypatch, engine):
    FakeDDGS.behaviour = {"": RuntimeError("202 Ratelimit")}
    monkeypatch.setattr(se, "DDGS", FakeDDGS)
    assert engine.scrambled_deep_osint_search(username="someone") == []
    messages = [d["message"] for d in engine.diagnostics if d["phase"] == "dorking"]
    assert messages and "ALL queries failed" in messages[0] and "Ratelimit" in messages[0]


def test_dorking_partial_failure_is_reported(monkeypatch, engine):
    FakeDDGS.behaviour = {"site:reddit.com": RuntimeError("boom")}
    FakeDDGS.default = []
    monkeypatch.setattr(se, "DDGS", FakeDDGS)
    engine.scrambled_deep_osint_search(username="someone")
    messages = [d["message"] for d in engine.diagnostics if d["phase"] == "dorking"]
    assert messages and "ALL" not in messages[0] and "failed" in messages[0]


def test_dorking_without_username_or_keywords_does_nothing(monkeypatch, engine):
    monkeypatch.setattr(se, "DDGS", FakeDDGS)
    assert engine.scrambled_deep_osint_search(username="", extra_keywords=["abc"]) == []


# ----------------------------------------------------------- infrastructure
def test_inspect_domains_normalises_list_valued_whois(monkeypatch, engine):
    class FakeWhois:
        registrar = ["MarkMonitor Inc.", "Other"]
        creation_date = [datetime(2005, 5, 5), datetime(1999, 1, 1, tzinfo=timezone.utc)]

    seen = []
    monkeypatch.setattr(se.whois, "whois", lambda d: seen.append(d) or FakeWhois())
    monkeypatch.setattr(se.socket, "gethostbyname", lambda d: "93.184.216.34")
    out = engine.inspect_multiple_domains(["m.instagram.com", "m.instagram.com", "google.com", "olive.com"])
    assert [r["domain"] for r in out] == ["m.instagram.com", "olive.com"]      # deduped + blocked removed
    assert seen == ["instagram.com", "olive.com"]                              # WHOIS on registrable domain
    assert out[0]["registrar"] == "MarkMonitor Inc."
    assert out[0]["creation_date"] == "1999-01-01T00:00:00+00:00"              # earliest, not a raw list repr
    assert out[0]["server_ip"] == "93.184.216.34"


def test_inspect_domains_survives_failures_and_says_so(monkeypatch, engine):
    def boom(_):
        raise RuntimeError("whois down")

    def unresolved(_):
        raise OSError("no dns")

    monkeypatch.setattr(se.whois, "whois", boom)
    monkeypatch.setattr(se.socket, "gethostbyname", unresolved)
    out = engine.inspect_multiple_domains(["example.org"])
    assert out[0]["server_ip"] == "Unresolved" and out[0]["registrar"] == "Unknown"
    assert any("WHOIS failed" in d["message"] for d in engine.diagnostics)


# ------------------------------------------------- Playwright flow (faked)
class FakeLocator:
    def __init__(self, n=1):
        self._n = n
        self.first = self
        self.last = self

    def count(self):
        return self._n

    def set_input_files(self, *a, **k):
        pass

    def click(self, *a, **k):
        pass


class FakePage:
    def __init__(self, anchors, captcha=False):
        self.anchors = anchors
        self.captcha = captcha
        self.url = "https://yandex.com/images/"

    def goto(self, *a, **k):
        if self.captcha:
            self.url = "https://yandex.com/showcaptcha?retpath=x"

    def wait_for_timeout(self, *a):
        pass

    def locator(self, selector):
        return FakeLocator(1)

    def get_by_text(self, *a, **k):
        return FakeLocator(0)

    def eval_on_selector_all(self, selector, js):
        return self.anchors


class FakeContext:
    def __init__(self, page):
        self.page = page
        self.closed = False

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, page):
        self.page = page
        self.contexts = []
        self.closed = False

    def new_context(self, **kwargs):
        ctx = FakeContext(self.page)
        self.contexts.append(ctx)
        return ctx

    def close(self):
        self.closed = True


ANCHORS = [
    {"href": "https://www.instagram.com/someone/", "text": "Someone on Instagram", "title": ""},
    {"href": "https://www.instagram.com/someone/", "text": "duplicate", "title": ""},
    {"href": "https://www.google.com/intl/en/about", "text": "Google junk", "title": ""},
    {"href": "https://yandex.com/support", "text": "Yandex junk", "title": ""},
    {"href": "/relative/link", "text": "relative", "title": ""},
    {"href": "javascript:void(0)", "text": "js", "title": ""},
    {"href": "https://olive.example/photo", "text": "x", "title": "Photo title from attribute"},
    {"href": "https://news.example/story", "text": "", "title": ""},
]


def test_single_slice_extraction_filters_and_labels(engine):
    browser = FakeBrowser(FakePage(ANCHORS))
    slice_info = {"name": "FULL_IMAGE", "bytes": structured_bytes()}
    matches = engine._search_single_slice(browser, slice_info, seen_urls=set())
    urls = [m["url"] for m in matches]
    assert urls == ["https://www.instagram.com/someone/", "https://olive.example/photo", "https://news.example/story"]
    assert matches[0]["platform"] == "INSTAGRAM" and matches[0]["match_type"] == "VISUAL MATCH (FULL_IMAGE)"
    assert matches[1]["page_title"] == "Photo title from attribute"
    assert matches[2]["page_title"] == "news.example"
    assert all("direct_image_url" not in m for m in matches)       # the misleading field is gone
    assert browser.contexts[0].closed


def test_single_slice_caps_matches(engine):
    many = [{"href": f"https://site{i}.example/p", "text": f"Result number {i}", "title": ""} for i in range(40)]
    matches = engine._search_single_slice(FakeBrowser(FakePage(many)), {"name": "X", "bytes": b"0"}, set())
    assert len(matches) == se.MAX_MATCHES_PER_SLICE


def test_captcha_is_reported_not_silent(engine):
    browser = FakeBrowser(FakePage(ANCHORS, captcha=True))
    matches = engine._search_single_slice(browser, {"name": "FULL_IMAGE", "bytes": b"0"}, set())
    assert matches == []
    assert any("CAPTCHA" in d["message"] and d["level"] == "warning" for d in engine.diagnostics)


def test_empty_results_are_reported_as_warning(engine):
    matches = engine._search_single_slice(FakeBrowser(FakePage([])), {"name": "FULL_IMAGE", "bytes": b"0"}, set())
    assert matches == []
    assert any("No result links extracted" in d["message"] for d in engine.diagnostics)


class FakeChromium:
    def __init__(self, fail_channels, page):
        self.fail_channels = fail_channels
        self.page = page
        self.launched = []
        self.browser = None

    def launch(self, headless=True, channel=None):
        self.launched.append(channel)
        if channel in self.fail_channels:
            raise RuntimeError(f"Chromium distribution '{channel}' is not found")
        self.browser = FakeBrowser(self.page)
        return self.browser


class FakePlaywright:
    def __init__(self, chromium):
        self.chromium = chromium

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_browser_falls_back_to_bundled_chromium(monkeypatch, engine):
    monkeypatch.delenv("OSINT_BROWSER_CHANNEL", raising=False)
    chromium = FakeChromium({"msedge", "chrome"}, FakePage([]))
    browser, label = engine._launch_browser(FakePlaywright(chromium))
    assert label == "bundled-chromium" and chromium.launched == ["msedge", "chrome", None]


def test_browser_channel_env_override(monkeypatch, engine):
    monkeypatch.setenv("OSINT_BROWSER_CHANNEL", "chrome")
    chromium = FakeChromium(set(), FakePage([]))
    _, label = engine._launch_browser(FakePlaywright(chromium))
    assert label == "chrome" and chromium.launched == ["chrome"]
    monkeypatch.setenv("OSINT_BROWSER_CHANNEL", "chromium")
    chromium = FakeChromium(set(), FakePage([]))
    assert engine._launch_browser(FakePlaywright(chromium))[1] == "bundled-chromium"


def test_no_browser_gives_actionable_error(monkeypatch, engine):
    monkeypatch.delenv("OSINT_BROWSER_CHANNEL", raising=False)
    chromium = FakeChromium({"msedge", "chrome", None}, FakePage([]))
    with pytest.raises(RuntimeError, match="playwright install chromium"):
        engine._launch_browser(FakePlaywright(chromium))


def test_multi_crop_search_dedupes_across_slices_and_closes_browser(monkeypatch, evidence_bytes):
    monkeypatch.delenv("OSINT_BROWSER_CHANNEL", raising=False)
    engine = MiniSpiderFoot(use_face_detection=False, delay_scale=0)
    chromium = FakeChromium(set(), FakePage(ANCHORS))
    monkeypatch.setattr(se, "sync_playwright", lambda: FakePlaywright(chromium))
    matches = engine.multi_crop_visual_reverse_search(evidence_bytes, enable_crops=True)
    # 3 slices see the same anchors, but each URL is reported only once overall
    assert len(matches) == 3 and chromium.browser.closed
    assert len(chromium.browser.contexts) == 3 and all(c.closed for c in chromium.browser.contexts)


def test_browser_failure_becomes_a_diagnostic(monkeypatch, evidence_bytes):
    monkeypatch.delenv("OSINT_BROWSER_CHANNEL", raising=False)
    engine = MiniSpiderFoot(delay_scale=0)
    chromium = FakeChromium({"msedge", "chrome", None}, FakePage([]))
    monkeypatch.setattr(se, "sync_playwright", lambda: FakePlaywright(chromium))
    assert engine.multi_crop_visual_reverse_search(evidence_bytes) == []
    assert any(d["level"] == "error" and "aborted" in d["message"] for d in engine.diagnostics)


# ----------------------------------------------------------- lead ranking
def test_rank_leads_orders_by_similarity_then_date(engine):
    def meta(url, strength, dist, ts=None, kind=None):
        comparison = {"strength": strength, "phash_distance": dist, "dhash_distance": dist,
                      "best_matching_slice": "FULL_IMAGE"} if strength else None
        return {"source_url": url, "meta_title": url, "detected_timestamp_utc": ts,
                "timestamp_kind": kind, "image_comparison": comparison}

    metas = [
        meta("https://a.example/weak", "WEAK", 30),
        meta("https://b.example/late", "NEAR_DUPLICATE", 2, "2024-06-01T00:00:00+00:00", "published"),
        meta("https://c.example/early", "NEAR_DUPLICATE", 2, "2023-01-01T00:00:00+00:00", "published"),
        meta("https://d.example/similar", "SIMILAR", 9, "2020-01-01T00:00:00+00:00", "updated"),
        meta("https://e.example/none", None, 0),
    ]
    urls = [m["source_url"] for m in metas] + ["https://f.example/not-scraped"]
    leads, summary = engine.rank_leads(urls, [], metas)
    order = [l["source_url"] for l in leads]
    assert order[:3] == ["https://c.example/early", "https://b.example/late", "https://d.example/similar"]
    assert order[-1] == "https://f.example/not-scraped" or order[-2:] == ["https://e.example/none", "https://f.example/not-scraped"]
    assert summary["best_visual_lead"] == "https://c.example/early"
    # the 'similar' page has an older but merely 'updated' timestamp: it must not count as earliest-published
    assert summary["earliest_published_lead"] == "https://c.example/early"
    assert summary["leads_total"] == 6 and summary["leads_with_image_comparison"] == 4


# ----------------------------------------------------------- full pipeline
def _prepare_pipeline(monkeypatch, engine, web, evidence_bytes):
    folder, base = web
    (folder / "evidence.jpg").write_bytes(evidence_bytes)
    (folder / "noise.jpg").write_bytes(noise_bytes())
    (folder / "orig.html").write_text("""<html><head><title>Original photo shoot</title>
        <meta property="article:published_time" content="2023-02-02T08:00:00Z">
        <meta property="og:image" content="/evidence.jpg"></head><body></body></html>""")
    (folder / "other.html").write_text("""<html><head><title>Unrelated page</title>
        <meta property="og:image" content="/noise.jpg"></head><body></body></html>""")

    monkeypatch.setattr(engine, "multi_crop_visual_reverse_search", lambda image_bytes, enable_crops=True, slices=None: [
        {"platform": "INSTAGRAM", "url": "https://www.instagram.com/first.hit/", "page_title": "A post by first.hit",
         "search_slice": "FULL_IMAGE", "match_type": "VISUAL MATCH (FULL_IMAGE)"},
        {"platform": "WEB", "url": f"{base}/orig.html", "page_title": "Original photo shoot",
         "search_slice": "SUBJECT_FOCUS_CROP", "match_type": "VISUAL MATCH (SUBJECT_FOCUS_CROP)"},
        {"platform": "WEB", "url": f"{base}/other.html", "page_title": "Unrelated page content here",
         "search_slice": "CONTEXT_BACKGROUND_CROP", "match_type": "VISUAL MATCH (CONTEXT_BACKGROUND_CROP)"},
    ])
    FakeDDGS.behaviour = {}
    FakeDDGS.default = [{"href": "https://github.com/first.hit", "title": "first.hit", "body": "bio"}]
    monkeypatch.setattr(se, "DDGS", FakeDDGS)
    monkeypatch.setattr(se.whois, "whois", lambda d: type("W", (), {"registrar": "Reg", "creation_date": None})())
    monkeypatch.setattr(se.socket, "gethostbyname", lambda d: "93.184.216.34")
    return base


def test_full_recon_auto_pivot_report(monkeypatch, engine, web, evidence_bytes):
    base = _prepare_pipeline(monkeypatch, engine, web, evidence_bytes)
    report = engine.run_full_recon(evidence_bytes, "evidence.jpg")

    assert report["engine_version"] == se.ENGINE_VERSION
    assert report["seed_inputs"]["pivot_username"] == "first.hit"
    assert report["seed_inputs"]["pivot_username_origin"] == "auto_first_visual_hit"
    assert any("UNVERIFIED" in c for c in report["analysis_caveats"])
    assert [s["name"] for s in report["search_slices"]][0] == "FULL_IMAGE"
    assert all("bytes" not in s for s in report["search_slices"])

    summary = report["osint_findings"]["lead_summary"]
    assert summary["best_visual_lead"] == f"{base}/orig.html"
    assert summary["earliest_published_lead"] == f"{base}/orig.html"
    assert report["osint_findings"]["ranked_leads"][0]["image_comparison"]["strength"] == "NEAR_DUPLICATE"
    assert report["osint_findings"]["linked_social_accounts"][0]["platform"] == "GITHUB"
    assert report["osint_findings"]["infrastructure_intel"]
    json.dumps(report)                                   # must be fully serialisable


def test_full_recon_respects_analyst_handle_and_auto_pivot_switch(monkeypatch, engine, web, evidence_bytes):
    _prepare_pipeline(monkeypatch, engine, web, evidence_bytes)
    analyst = engine.run_full_recon(evidence_bytes, "e.jpg", seed_username="@my.handle")
    assert analyst["seed_inputs"]["pivot_username"] == "my.handle"
    assert analyst["seed_inputs"]["pivot_username_origin"] == "analyst_supplied"
    assert not any("UNVERIFIED" in c for c in analyst["analysis_caveats"])

    off = engine.run_full_recon(evidence_bytes, "e.jpg", auto_pivot=False)
    assert off["seed_inputs"]["pivot_username"] is None
    assert off["seed_inputs"]["auto_discovered_handles"] == ["first.hit"]     # still listed, just not used


def test_full_recon_ignores_non_http_seed_url_and_resets_diagnostics(monkeypatch, engine, web, evidence_bytes):
    _prepare_pipeline(monkeypatch, engine, web, evidence_bytes)
    first = engine.run_full_recon(evidence_bytes, "e.jpg", seed_url="file:///etc/passwd")
    assert first["seed_inputs"]["provided_url"] is None
    assert any("Seed URL ignored" in d["message"] for d in first["engine_diagnostics"])
    second = engine.run_full_recon(evidence_bytes, "e.jpg")
    assert not any("Seed URL ignored" in d["message"] for d in second["engine_diagnostics"])


def test_short_exception_text_is_one_line():
    exc = RuntimeError("Page.goto: net::ERR_NAME_NOT_RESOLVED\nCall log:\n  - navigating to ...")
    text = se._short(exc)
    assert "\n" not in text and text == "RuntimeError: Page.goto: net::ERR_NAME_NOT_RESOLVED"
    assert se._short(ValueError("")) == "ValueError: "
