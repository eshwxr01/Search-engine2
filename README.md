# OSINT Attribution & Spread Mapping Engine
**Stage 5 Module — Deepfake Detection & Media Provenance Pipeline**

[![tests](https://github.com/eshwxr01/Search-engine2/actions/workflows/tests.yml/badge.svg)](https://github.com/eshwxr01/Search-engine2/actions/workflows/tests.yml)

## Overview
This repository contains the **OSINT, Scraping & Entity Attribution Module**: a standalone Python engine designed and developed independently as part of a multi-stage **Deepfake Detection & Original Source Identification** system.

Once upstream forensic stages flag a suspect image as manipulated or synthetic, this module **gathers leads** about where visually similar content appears online, which accounts and domains are associated with those appearances, and how the content spreads. It ranks those leads by how closely each page's image matches the evidence, and presents everything as a forensic JSON report plus an interactive link graph.

> **What the output is, and is not.** A reverse-image hit shows *where similar content appears*, not *who created, owns or posted it*. An associated account may be a victim, a repost, or the original photographer. Everything here is a **lead for a human analyst to verify**, and the report says so in an `analysis_caveats` section.

---

## End-to-End System Pipeline

```text
[1. Intake & Hashing]             ──▶ MD5 / SHA-256 chain-of-custody fingerprinting
         │
[2. Metadata & Payload]           ──▶ ExifTool & Binwalk static inspection
         │
[3. Pixel-Level Manipulation]     ──▶ ELA, noise & clone analysis, ML & frequency detection
         │
[4. Provenance & Signatures]      ──▶ C2PA manifests, YARA rules, GAN/Diffusion artifact checks
         │
╔══════════════════════════════════════════════════════════════════════════════════╗
║ [5. OSINT & Scraping (THIS REPO)] ──▶ Multi-crop lead search, social dorking,    ║
║                                       image-similarity ranking, WHOIS/IP recon   ║
║                                       & Maltego-style link-graphing              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
         │
[6. Legal / Reporting Channel]    ──▶ Forensic JSON & visual graph handoff (External)
```

---

## What the Engine Does

1. **Fingerprint** the evidence: SHA-256, MD5, pHash, dHash and resolution (EXIF orientation is applied before perceptual hashing and slicing).
2. **Slice** the image into three views:
   - `FULL_IMAGE`
   - `SUBJECT_FOCUS_CROP`: located with **OpenCV face detection**, expanded to head, shoulders and torso. If no face is found (or OpenCV is missing) it falls back to a fixed region, and the report records which method was used per slice.
   - `CONTEXT_BACKGROUND_CROP`: the full image with the subject region **blurred out**, so the reverse search matches the scenery rather than the person.
3. **Reverse-search** each slice on Yandex Images with Playwright, using one browser session. CAPTCHA pages are detected and reported.
4. **Pivot on handles:** extracts usernames from social URLs in the hits, then runs randomized multi-platform dorks (ddgs) on Instagram, X, YouTube, Reddit, Facebook, LinkedIn, GitHub, Telegram and Pinterest. Each result records whether the handle actually appears in it (`handle_in_result`).
5. **Scrape page metadata:** OpenGraph tags, publication timestamps (classified as `published`, `updated` or `time_tag_unverified`) and outbound links (BeautifulSoup4).
6. **Rank leads by image similarity:** downloads each page's primary image (`og:image`), computes pHash/dHash, and compares it with every evidence slice (Hamming distance, see below). Leads are sorted best-match first; `lead_summary` names the closest lead and the earliest *explicitly published* similar lead.
7. **Infrastructure recon:** server IPs and WHOIS registrar / earliest creation date (python-whois) on the registrable domain.
8. **Graph:** interactive physics-enabled entity network (NetworkX + PyVis) with degree-centrality pivot-hub highlighting, near-duplicate sightings emphasised, and handles labelled as *unverified associations*.

### Image-similarity strength
Computed on 64-bit perceptual hashes (smaller distance = more similar). These thresholds are heuristics, not proof:

| Strength | pHash distance |
|---|---|
| `NEAR_DUPLICATE` | ≤ 6 |
| `SIMILAR` | ≤ 12 |
| `WEAK` | > 12 |

Thresholds live at the top of `spiderfoot_engine.py` (`NEAR_DUPLICATE_MAX_DISTANCE`, `SIMILAR_MAX_DISTANCE`).

---

## Repository Structure
- `spiderfoot_engine.py` — core engine: hashing, face-aware slicing, Playwright reverse search, dorking, safe page fetching, image-similarity ranking, WHOIS/DNS recon.
- `maltego_visualizer.py` — converts the forensic JSON into an interactive PyVis entity map.
- `osint_utils.py` — small pure helpers (domain matching, timestamp / WHOIS normalisation, text sanitising).
- `app.py` — Streamlit testbench: run scans, browse ranked leads, diagnostics, graph and raw JSON.
- `tests/` — offline pytest suite (no browser or internet needed).
- `requirements.txt` / `requirements-dev.txt` — runtime / development dependencies.
- `.github/workflows/tests.yml` — CI running the tests on every push.

---

## Local Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eshwxr01/Search-engine2.git
   cd Search-engine2
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies and a browser for Playwright:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
   The engine tries **Microsoft Edge, then Google Chrome, then Playwright's bundled Chromium**, so any one of them is enough. To force one, set `OSINT_BROWSER_CHANNEL` to `msedge`, `chrome` or `chromium`.

4. **Launch the Streamlit testbench:**
   ```bash
   streamlit run app.py
   ```

### Using the engine from Python
```python
from spiderfoot_engine import MiniSpiderFoot
from maltego_visualizer import MiniMaltego

with open("evidence.jpg", "rb") as f:
    image_bytes = f.read()

engine = MiniSpiderFoot()
report = engine.run_full_recon(
    image_bytes,
    "evidence.jpg",
    seed_username="",      # optional: a handle YOU supply
    seed_url="",           # optional: a known page URL
    enable_crops=True,
    auto_pivot=False,      # True = dork on the first handle found (unverified)
)
html_path = MiniMaltego().build_graph_from_intel(report)   # writes outputs/maltego_graph_<sha>.html
```

### Running the tests
```bash
pip install -r requirements-dev.txt
pytest
```
The suite uses a local web server and fake browser objects, so it needs no internet and no Playwright browser.

---

## Report Format
`run_full_recon()` returns a JSON-serialisable dict:

| Key | Contents |
|---|---|
| `engine_version` | Version of the engine that produced the report |
| `evidence_profile` | Filename, SHA-256, MD5, pHash, dHash, resolution, scan time (UTC) |
| `seed_inputs` | `provided_url`, `pivot_username`, `pivot_username_origin` (`analyst_supplied` / `auto_first_visual_hit` / `null`), `auto_discovered_handles` |
| `search_slices` | Per slice: name, `method` (`whole_image` / `face_detection` / `fixed_heuristic`), crop box, hashes |
| `osint_findings` | `linked_social_accounts`, `other_web_sources`, `page_metadata_intel`, `infrastructure_intel`, `ranked_leads`, `lead_summary` |
| `analysis_caveats` | Plain-language limits of the findings (always present) |
| `engine_diagnostics` | Every skipped page, failed query, CAPTCHA, missing browser or fallback, as `{level, phase, message}` |

**Read `engine_diagnostics` before trusting an empty result.** An empty section can mean "blocked / rate-limited / layout changed" rather than "nothing exists".

---

## Known Limitations
- **Search backends are scraped, not official APIs.** The Yandex Images flow depends on its page layout and can be CAPTCHA-blocked; DuckDuckGo-style search can rate-limit. Both are reported in diagnostics, not hidden. Use low volumes and respect each service's terms.
- **Face detection is Haar-cascade based** (classic OpenCV). It misses side profiles, small or occluded faces, and non-human subjects, in which case the fixed-region fallback is used and labelled.
- **Similarity is measured on each page's primary (`og:image`) picture only**, and pages without one cannot be compared.
- **Timestamps come from page metadata**, which can be absent, edited or wrong. Only explicit published-time tags feed `earliest_published_lead`.
- **Outbound fetches are SSRF-guarded** (public hosts only, redirects re-checked, size-capped), but the check is not immune to DNS-rebinding races. Run it as a local analyst tool, not as a public service.
- OpenCV 5.x wheels no longer ship the Haar cascade file, so `requirements.txt` pins `opencv-python-headless<5`.

---

## What's New in 2.0
- Subject/background slices use real face detection; the background slice masks the subject instead of cropping an overlapping region.
- Reverse-image leads are **verified and ranked** by perceptual-hash similarity to the evidence.
- Handles are presented as **associated / unverified**, never as a "suspect" or "target"; auto-pivot is optional and flagged.
- Failures are surfaced in `engine_diagnostics` and the UI instead of being swallowed.
- Browser launch no longer hard-codes Microsoft Edge.
- Fixed: substring-based platform and domain matching (`box.com` was labelled X/Twitter; `olive.com` was blocked), subreddit and YouTube channel-ID names read as usernames, pages silently dropped on a missing `og:title` content attribute, list-valued WHOIS fields printed raw, WHOIS creation dates missing from graph tooltips, a misleading `direct_image_url` field, and results vanishing from the UI after clicking Download.
- Moved from the deprecated `st.components.v1.html` to `st.iframe`; renamed `test_app.py` to `app.py`; added a real test suite and CI.

---

## Disclaimer
Developed for academic research, digital media forensics, and authorized OSINT provenance verification. Results are investigative leads and must be independently verified before any attribution or legal action.
