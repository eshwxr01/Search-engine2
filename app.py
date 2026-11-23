"""
app.py - Streamlit testbench for the Stage 5 OSINT engine.

Run with:  streamlit run app.py
"""
import json
import os
import re

import streamlit as st

from maltego_visualizer import MiniMaltego
from spiderfoot_engine import MiniSpiderFoot

st.set_page_config(page_title="OSINT Attribution & Spread Mapping Engine", layout="wide")


# ---------------------------------------------------------------- helpers
def md_escape(text) -> str:
    """Escape Markdown control characters in scraped text."""
    return re.sub(r"([\\`*_{}\[\]()#+!|<>~])", r"\\\1", str(text or ""))


def safe_link(label, url) -> str:
    """Markdown link for http(s) URLs only; anything else is shown as inert code."""
    url = str(url or "")
    if url.lower().startswith(("http://", "https://")):
        encoded = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
        return f"[{md_escape(label or url)}]({encoded})"
    return f"`{md_escape(url)}`"


def render_html(html: str, height: int):
    """Embed the graph page. st.components.v1.html is deprecated in favour of st.iframe
    (Streamlit >= 1.5x); fall back to it only on older versions.

    The embedded page gets same-origin access to the app, so everything scraped from the
    web that ends up in it is sanitised first (see osint_utils.clean_text).
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # pragma: no cover - older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def lead_rows(leads):
    rows = []
    for rank, lead in enumerate(leads, start=1):
        comparison = lead.get("image_comparison") or {}
        rows.append({
            "rank": rank,
            "platform": lead.get("platform"),
            "image_similarity": comparison.get("strength", "not compared"),
            "phash_distance": comparison.get("phash_distance"),
            "published_utc": lead.get("published_timestamp_utc"),
            "timestamp_kind": lead.get("timestamp_kind"),
            "title": lead.get("page_title"),
            "url": lead.get("source_url"),
        })
    return rows


# ------------------------------------------------------------------ header
st.title("OSINT Attribution & Spread Mapping Engine - Testbench")
st.markdown(
    "**Stage 5** of the deepfake-detection pipeline. Sub-steps: evidence fingerprint -> multi-crop reverse "
    "search -> handle pivot & dorking -> page / infrastructure recon -> lead ranking & link graph.  \n"
    "The output is a set of **leads for a human analyst**, not proof of who created or owns an image."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Upload Evidence")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

    st.header("2. Optional Pivots")
    seed_username = st.text_input("Associated username / handle (optional)", placeholder="some.username")
    seed_url = st.text_input("Known page URL (optional)", placeholder="https://example.com/post/123")

    st.header("3. Scan Options")
    enable_crops = st.checkbox(
        "Multi-crop search (full + subject + background)",
        value=True,
        help="Subject is located with face detection when OpenCV is installed, otherwise a fixed region. "
             "The background slice blurs the subject out so the search matches the scenery.",
    )
    auto_pivot = st.checkbox(
        "Auto-pivot on the first handle found (unverified)",
        value=True,
        help="If you gave no username, the first handle seen in a visual hit is used for dorking. "
             "That account may be a victim or a repost, so turn this off to avoid anchoring on it.",
    )
    run_btn = st.button("Run OSINT scan & build graph", type="primary")

if uploaded_file is None:
    st.info("Upload an image in the left sidebar to begin.")
    st.stop()

# Results live in session_state so that clicking a download button (which
# reruns the script) does not make them disappear.
file_key = f"{uploaded_file.name}:{uploaded_file.size}"
if st.session_state.get("file_key") != file_key:
    for key in ("report", "graph_path", "short_hash"):
        st.session_state.pop(key, None)
    st.session_state["file_key"] = file_key

if run_btn:
    with st.spinner("Running reverse search, dorking, page analysis, WHOIS/DNS and building the graph..."):
        try:
            engine = MiniSpiderFoot()
            report = engine.run_full_recon(
                image_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                seed_url=seed_url,
                seed_username=seed_username,
                enable_crops=enable_crops,
                auto_pivot=auto_pivot,
            )
            short_hash = report["evidence_profile"]["sha256"][:10]
            os.makedirs("outputs", exist_ok=True)
            with open(f"outputs/spiderfoot_intel_{short_hash}.json", "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=4)
            graph_path = MiniMaltego().build_graph_from_intel(report)
        except ValueError as exc:  # e.g. undecodable image
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Scan failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state.update(report=report, graph_path=graph_path, short_hash=short_hash)

report = st.session_state.get("report")
col_img, col_graph = st.columns([1, 1.4])
with col_img:
    st.subheader("Uploaded Evidence")
    st.image(uploaded_file, caption=uploaded_file.name)

if not report:
    st.info("Set your options in the sidebar and click **Run OSINT scan & build graph**.")
    st.stop()

findings = report.get("osint_findings", {})
diagnostics = report.get("engine_diagnostics", [])
short_hash = st.session_state["short_hash"]

with col_graph:
    st.subheader("Entity & Spread Graph")
    with open(st.session_state["graph_path"], "r", encoding="utf-8") as hf:
        render_html(hf.read(), height=670)

with st.expander("Read before using these results", expanded=True):
    for caveat in report.get("analysis_caveats", []):
        st.markdown(f"- {caveat}")

problems = [d for d in diagnostics if d.get("level") in ("warning", "error")]
if problems:
    st.warning(
        f"{len(problems)} warning(s)/error(s) occurred during the scan. Missing results may mean a blocked "
        "or failed step, not an absence of evidence. See the **Diagnostics** tab."
    )

st.divider()
tab_leads, tab_visual, tab_profiles, tab_infra, tab_diag, tab_json = st.tabs([
    "Ranked Leads",
    "Visual Sightings",
    "Profiles (Dorking)",
    "Page Metadata & Infrastructure",
    "Diagnostics",
    "Raw Forensic JSON",
])

with tab_leads:
    summary = findings.get("lead_summary", {})
    st.markdown(
        f"**Leads:** {summary.get('leads_total', 0)} "
        f"({summary.get('leads_with_image_comparison', 0)} compared by image similarity)"
    )
    if summary.get("best_visual_lead"):
        st.markdown(f"**Closest visual lead:** {safe_link(summary['best_visual_lead'], summary['best_visual_lead'])}")
    if summary.get("earliest_published_lead"):
        st.markdown(
            f"**Earliest published similar lead:** "
            f"{safe_link(summary['earliest_published_lead'], summary['earliest_published_lead'])}"
        )
    rows = lead_rows(findings.get("ranked_leads", []))
    if rows:
        st.dataframe(rows, column_config={"url": st.column_config.LinkColumn("url")})
    else:
        st.info("No leads were produced. Check the Diagnostics tab.")

with tab_visual:
    sources = findings.get("other_web_sources", [])
    slices = {s["name"]: s for s in report.get("search_slices", [])}
    if sources:
        for src in sources:
            st.markdown(
                f"### [{md_escape(src.get('platform', 'WEB'))}] ({md_escape(src.get('search_slice', 'FULL'))}) - "
                f"{safe_link(src.get('page_title') or 'Link', src.get('url'))}"
            )
            method = slices.get(src.get("search_slice"), {}).get("method", "n/a")
            st.caption(f"Match type: `{src.get('match_type', '')}` | slice method: `{method}`")
            st.divider()
    else:
        st.info("No visual matches returned. See Diagnostics for CAPTCHA / layout / browser problems.")

with tab_profiles:
    accounts = findings.get("linked_social_accounts", [])
    if accounts:
        for acc in accounts:
            weak = "" if acc.get("handle_in_result", True) else " (weak: handle not in snippet)"
            with st.expander(
                f"{acc.get('platform', '')} | @{acc.get('matched_handle', '')}{weak}", expanded=False
            ):
                st.markdown(f"**Title:** {md_escape(acc.get('display_title', ''))}")
                st.markdown(f"**URL:** {safe_link(acc.get('url'), acc.get('url'))}")
                st.markdown(f"**Snippet:** {md_escape(acc.get('bio_summary', ''))}")
                st.caption(f"Dork used: `{acc.get('query_used', '')}`")
    else:
        st.info("No handle was entered or discovered, or no public profiles matched.")

with tab_infra:
    col_meta, col_infra = st.columns(2)
    with col_meta:
        st.markdown("#### Page metadata & timestamps")
        metas = findings.get("page_metadata_intel", [])
        if metas:
            for m in metas:
                st.markdown(
                    f"**[{md_escape(m.get('platform', 'WEB'))}]** "
                    f"{safe_link(m.get('meta_title') or m.get('source_url'), m.get('source_url'))}"
                )
                st.caption(
                    f"Timestamp: `{m.get('detected_timestamp') or 'not exposed'}` "
                    f"(kind: {m.get('timestamp_kind') or 'n/a'})"
                )
                comparison = m.get("image_comparison")
                if comparison:
                    st.caption(
                        f"Image similarity: **{comparison['strength']}** "
                        f"(pHash d={comparison['phash_distance']}, vs {comparison['best_matching_slice']})"
                    )
                if m.get("outbound_links"):
                    st.write("Outbound links:", m["outbound_links"])
                st.divider()
        else:
            st.info("No page metadata extracted.")
    with col_infra:
        st.markdown("#### Domain, server IP & WHOIS")
        infras = findings.get("infrastructure_intel", [])
        if infras:
            st.dataframe(infras)
        else:
            st.info("No external domains resolved.")

with tab_diag:
    if diagnostics:
        st.dataframe(diagnostics)
    else:
        st.success("No diagnostics were recorded.")

with tab_json:
    st.download_button(
        label="Download forensic JSON",
        data=json.dumps(report, indent=4),
        file_name=f"osint_report_{short_hash}.json",
        mime="application/json",
    )
    st.json(report)
