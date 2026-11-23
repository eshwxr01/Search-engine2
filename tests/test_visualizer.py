import pytest

from maltego_visualizer import MiniMaltego

XSS = "</script><img src=x onerror=alert(1)>"


def sample_intel(**overrides):
    intel = {
        "evidence_profile": {
            "filename": "evidence.jpg",
            "sha256": "ab" * 32,
            "md5": "cd" * 16,
            "phash": "f0f0f0f0f0f0f0f0",
            "resolution": "400x300",
        },
        "seed_inputs": {
            "provided_url": None,
            "pivot_username": "first.hit",
            "pivot_username_origin": "auto_first_visual_hit",
            "auto_discovered_handles": ["first.hit", "second.hit"],
        },
        "osint_findings": {
            "other_web_sources": [
                {"platform": "WEB", "url": "https://example.com/a", "page_title": "Page A",
                 "search_slice": "FULL_IMAGE", "match_type": "VISUAL MATCH (FULL_IMAGE)"},
                {"platform": "WEB", "url": "https://example.com/b", "page_title": XSS,
                 "search_slice": "SUBJECT_FOCUS_CROP", "match_type": "VISUAL MATCH (SUBJECT_FOCUS_CROP)"},
                {"platform": "WEB", "url": "https://example.com/c", "page_title": "Page C",
                 "search_slice": "CONTEXT_BACKGROUND_CROP", "match_type": "VISUAL MATCH (CONTEXT_BACKGROUND_CROP)"},
            ],
            "linked_social_accounts": [
                {"platform": "GITHUB", "matched_handle": "first.hit", "handle_in_result": True,
                 "query_used": 'site:github.com "first.hit"', "url": "https://github.com/first.hit",
                 "display_title": XSS, "bio_summary": XSS},
                {"platform": "WEB", "matched_handle": "CONTEXT_MATCH", "handle_in_result": False,
                 "query_used": '"some title"', "url": "https://blog.example.net/x",
                 "display_title": "Blog", "bio_summary": "text"},
            ],
            "page_metadata_intel": [
                {"source_url": "https://example.com/a", "outbound_links": ["https://other.example/x"]},
            ],
            "infrastructure_intel": [
                {"domain": "example.com", "server_ip": "93.184.216.34",
                 "registrar": "Example Registrar", "creation_date": "1997-03-04T00:00:00+00:00"},
            ],
            "ranked_leads": [
                {"source_url": "https://example.com/a",
                 "image_comparison": {"strength": "NEAR_DUPLICATE", "phash_distance": 2,
                                      "dhash_distance": 3, "best_matching_slice": "FULL_IMAGE"}},
            ],
        },
    }
    intel.update(overrides)
    return intel


@pytest.fixture
def built(tmp_path):
    mm = MiniMaltego()
    path = mm.build_graph_from_intel(sample_intel(), output_dir=str(tmp_path))
    return mm, path


def test_html_file_is_written(built):
    _, path = built
    html = open(path, encoding="utf-8").read()
    assert path.endswith(".html") and "vis" in html.lower() and "EVIDENCE IMAGE" in html


def test_scraped_markup_is_neutralised_in_the_output(built):
    _, path = built
    html = open(path, encoding="utf-8").read()
    assert XSS not in html
    assert "<img src=x onerror" not in html


def test_whois_creation_date_reaches_domain_node_created_by_a_sighting(built):
    mm, _ = built
    # the sighting loop creates DOM_example.com first; the infra loop must still enrich it
    assert "Created: 1997-03-04" in mm.graph.nodes["DOM_example.com"]["title"]
    assert "Domain: blog.example.net" == mm.graph.nodes["DOM_blog.example.net"]["title"]


def test_handles_are_labelled_as_unverified_associations_not_targets(built):
    mm, _ = built
    labels = {n: d.get("label", "") for n, d in mm.graph.nodes(data=True)}
    assert "AUTO-PIVOT HANDLE (unverified)" in labels["HANDLE_first.hit"]
    assert "CANDIDATE HANDLE (unverified)" in labels["HANDLE_second.hit"]
    assert not any("TARGET" in label.upper() for label in labels.values())
    edge_labels = {d["label"] for _, _, d in mm.graph.edges(data=True)}
    assert "associated_handle" in edge_labels and "linked_handle" not in edge_labels


def test_analyst_supplied_pivot_is_labelled_as_such(tmp_path):
    intel = sample_intel()
    intel["seed_inputs"].update(pivot_username="my.handle", pivot_username_origin="analyst_supplied",
                                auto_discovered_handles=[])
    mm = MiniMaltego()
    mm.build_graph_from_intel(intel, output_dir=str(tmp_path))
    assert "PIVOT HANDLE (analyst)" in mm.graph.nodes["HANDLE_my.handle"]["label"]


def test_near_duplicate_sightings_are_annotated_and_enlarged(built):
    mm, _ = built
    node = mm.graph.nodes["SRC_0"]
    assert "NEAR-DUPLICATE" in node["label"] and "d=2" in node["label"]
    assert "pHash d=2" in node["title"]
    assert node["size"] >= 24
    assert "NEAR-DUPLICATE" not in mm.graph.nodes["SRC_2"]["label"]


def test_weak_dork_results_are_visibly_weaker(built):
    mm, _ = built
    strong, weak = mm.graph.nodes["ACC_0"], mm.graph.nodes["ACC_1"]
    assert weak["size"] < strong["size"]
    assert "Weak lead" in weak["title"]
    assert "(context lead)" in weak["label"]


def test_pivot_hub_detection_marks_well_connected_nodes(built):
    mm, _ = built
    # example.com is linked from 2 sightings + 1 account? at least: SRC_0, SRC_1, SRC_2, outbound parent, IP, registrar
    hub = mm.graph.nodes["DOM_example.com"]
    assert mm.graph.degree("DOM_example.com") >= 3
    assert hub["borderWidth"] == 3 and hub["size"] > 20


def test_empty_intel_still_builds(tmp_path):
    mm = MiniMaltego()
    path = mm.build_graph_from_intel({}, output_dir=str(tmp_path))
    assert mm.graph.number_of_nodes() == 1 and path.endswith(".html")


def test_rebuilding_clears_the_previous_graph(tmp_path):
    mm = MiniMaltego()
    mm.build_graph_from_intel(sample_intel(), output_dir=str(tmp_path))
    first = mm.graph.number_of_nodes()
    mm.build_graph_from_intel({}, output_dir=str(tmp_path))
    assert mm.graph.number_of_nodes() == 1 < first
