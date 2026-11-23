"""
maltego_visualizer.py - turns the engine's forensic JSON into an interactive
entity-relationship graph (NetworkX + PyVis).

Wording is deliberately cautious: handles are "associated" / "candidate"
leads, never "targets" or "suspects" - a reverse-image hit does not prove
ownership or authorship.
"""
from __future__ import annotations

import json
import os

import networkx as nx
from pyvis.network import Network

from osint_utils import clean_text, host_of

# Node colours by entity type
COLOR_EVIDENCE = "#FF4B4B"
COLOR_HANDLE = "#FFA500"
COLOR_SIGHTING = "#00D4FF"
COLOR_ACCOUNT = "#00CC96"
COLOR_DOMAIN = "#AB63FA"
COLOR_OUTBOUND = "#FF6692"
COLOR_IP = "#FECB52"
COLOR_REGISTRAR = "#B6E880"

STRENGTH_LABEL = {"NEAR_DUPLICATE": "NEAR-DUPLICATE", "SIMILAR": "SIMILAR"}
STRENGTH_SIZE = {"NEAR_DUPLICATE": 24, "SIMILAR": 21}


class MiniMaltego:
    def __init__(self):
        self.graph = nx.DiGraph()

    # ----------------------------------------------------------------- helpers
    def _add_node_safe(self, node_id, **kwargs):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **kwargs)

    def _add_domain_node(self, host, created=None):
        """Create a domain node, or enrich an existing one (e.g. with the WHOIS date).

        The old code used add-if-missing for every node, so WHOIS creation dates
        were silently dropped whenever a sighting had already created the node.
        """
        host = clean_text(host, 120)
        node_id = f"DOM_{host}"
        known_created = created and created != "Unknown"
        title = f"Domain: {host}" + (f"\nCreated: {clean_text(created, 60)}" if known_created else "")
        if self.graph.has_node(node_id):
            if known_created:
                self.graph.nodes[node_id]["title"] = title
        else:
            self.graph.add_node(
                node_id,
                label=f"DOMAIN\n{host}",
                color=COLOR_DOMAIN,
                size=20,
                shape="diamond",
                title=title,
                font={"color": "white", "size": 11},
            )
        return node_id

    # ------------------------------------------------------------------- build
    def build_graph_from_intel(self, intel_data, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        self.graph.clear()

        ev = intel_data.get("evidence_profile", {})
        seeds = intel_data.get("seed_inputs", {})
        findings = intel_data.get("osint_findings", {})

        short_sha = (ev.get("sha256") or "0000000000")[:10]
        root_id = "EVIDENCE_ROOT"
        self._add_node_safe(
            root_id,
            label=(
                f"EVIDENCE IMAGE\n{clean_text(ev.get('filename', 'image'), 60)}\n"
                f"SHA256: {short_sha}...\npHash: {clean_text(ev.get('phash', ''), 20)}"
            ),
            color=COLOR_EVIDENCE,
            size=36,
            shape="box",
            title=(
                f"SHA-256: {clean_text(ev.get('sha256', ''), 80)}\n"
                f"MD5: {clean_text(ev.get('md5', ''), 40)}\n"
                f"Resolution: {clean_text(ev.get('resolution', ''), 30)}"
            ),
            font={"color": "white", "size": 13, "face": "monospace"},
        )

        # Similarity per URL (from the engine's image comparison), used to
        # emphasise sightings that look like near-duplicates of the evidence.
        comparison_by_url = {
            lead.get("source_url"): lead.get("image_comparison")
            for lead in findings.get("ranked_leads", [])
            if lead.get("image_comparison")
        }

        # 1. Handles: pivot (analyst-supplied or auto) + other candidates
        pivot_user = seeds.get("pivot_username")
        pivot_origin = seeds.get("pivot_username_origin")
        all_handles = list(dict.fromkeys(
            ([pivot_user] if pivot_user else []) + list(seeds.get("auto_discovered_handles", []))
        ))

        handle_node_by_name = {}
        for idx, handle in enumerate(all_handles):
            h_id = f"HANDLE_{handle.lower()}"
            handle_node_by_name[handle.lower()] = h_id
            is_pivot = bool(pivot_user) and handle.lower() == pivot_user.lower()
            if is_pivot and pivot_origin == "analyst_supplied":
                kind = "PIVOT HANDLE (analyst)"
            elif is_pivot:
                kind = "AUTO-PIVOT HANDLE (unverified)"
            else:
                kind = "CANDIDATE HANDLE (unverified)"
            self._add_node_safe(
                h_id,
                label=f"{kind}\n@{clean_text(handle, 64)}",
                color=COLOR_HANDLE,
                size=28 if is_pivot else 22,
                shape="dot",
                title=f"Associated handle: @{clean_text(handle, 64)}\nNot verified as the owner or author.",
                font={"color": "white", "size": 13, "bold": True},
            )
            self.graph.add_edge(root_id, h_id, label="associated_handle", color=COLOR_HANDLE)

        primary_handle_node = handle_node_by_name.get(pivot_user.lower()) if pivot_user else None

        # 2. Visual reverse-search sightings (full image + crops)
        for idx, src in enumerate(findings.get("other_web_sources", [])):
            src_url = src.get("url", "")
            src_node_id = f"SRC_{idx}"
            platform = clean_text(src.get("platform", "WEB"), 40)
            slice_tag = clean_text(src.get("search_slice", "FULL_IMAGE"), 40)

            label = f"SIGHTING [{platform}]\n({slice_tag})\n{clean_text(src.get('page_title', ''), 34)}..."
            comparison = comparison_by_url.get(src_url)
            strength = (comparison or {}).get("strength")
            if strength in STRENGTH_LABEL:
                label += f"\n~ {STRENGTH_LABEL[strength]} (d={comparison['phash_distance']})"

            title = f"URL: {clean_text(src_url, 300)}\nMatch: {clean_text(src.get('match_type', ''), 80)}"
            if comparison:
                title += (
                    f"\nImage similarity: {strength} "
                    f"(pHash d={comparison['phash_distance']}, dHash d={comparison['dhash_distance']}, "
                    f"vs {clean_text(comparison.get('best_matching_slice', ''), 40)})"
                )
            self._add_node_safe(
                src_node_id,
                label=label,
                color=COLOR_SIGHTING,
                size=STRENGTH_SIZE.get(strength, 18),
                shape="box",
                title=title,
                font={"color": "#0E1117", "size": 11},
            )
            self.graph.add_edge(root_id, src_node_id, label=f"visual_{slice_tag.lower()}", color=COLOR_SIGHTING)

            dom = host_of(src_url)
            if dom:
                dom_id = self._add_domain_node(dom)
                self.graph.add_edge(src_node_id, dom_id, label="hosted_on", color=COLOR_DOMAIN)

        # 3. Profiles found by dorking
        for idx, acc in enumerate(findings.get("linked_social_accounts", [])):
            acc_node_id = f"ACC_{idx}"
            acc_url = acc.get("url", "")
            bio_text = acc.get("bio_summary", "")
            bio_snip = clean_text(bio_text, 45) + ("..." if len(bio_text) > 45 else "")
            matched = acc.get("matched_handle", "")
            handle_line = "(context lead)" if matched == "CONTEXT_MATCH" else f"@{clean_text(matched, 64)}"
            handle_seen = acc.get("handle_in_result", True)

            title = (
                f"URL: {clean_text(acc_url, 300)}\nQuery: {clean_text(acc.get('query_used', ''), 150)}\n"
                f"Bio: {clean_text(bio_text, 200)}"
            )
            if not handle_seen:
                title += "\nWeak lead: the handle does not appear in this result's text."
            self._add_node_safe(
                acc_node_id,
                label=(
                    f"PROFILE: {clean_text(acc.get('platform', 'WEB'), 30)}\n{handle_line}\n"
                    f"{clean_text(acc.get('display_title', ''), 32)}\n{bio_snip}"
                ),
                color=COLOR_ACCOUNT,
                size=21 if handle_seen else 15,
                shape="box",
                title=title,
                font={"color": "#0E1117", "size": 11},
            )
            parent = primary_handle_node or root_id
            self.graph.add_edge(
                parent, acc_node_id,
                label=f"found_on_{clean_text(acc.get('platform', 'WEB'), 30)}",
                color=COLOR_ACCOUNT,
            )

            dom = host_of(acc_url)
            if dom:
                dom_id = self._add_domain_node(dom)
                self.graph.add_edge(acc_node_id, dom_id, label="resolves_domain", color=COLOR_DOMAIN)

        # 4. Outbound bio / webpage links
        for p_idx, page_meta in enumerate(findings.get("page_metadata_intel", [])):
            src_host = host_of(page_meta.get("source_url", ""))
            for o_idx, out_url in enumerate(page_meta.get("outbound_links", [])):
                out_host = host_of(out_url)
                if not out_host:
                    continue
                out_id = f"OUT_{p_idx}_{o_idx}"
                self._add_node_safe(
                    out_id,
                    label=f"OUTBOUND LINK\n{clean_text(out_host, 60)}",
                    color=COLOR_OUTBOUND,
                    size=16,
                    shape="triangle",
                    title=(
                        f"Outbound URL: {clean_text(out_url, 300)}\n"
                        f"Found on: {clean_text(page_meta.get('source_url', ''), 300)}"
                    ),
                    font={"color": "white", "size": 10},
                )
                parent = f"DOM_{clean_text(src_host, 120)}" if self.graph.has_node(f"DOM_{clean_text(src_host, 120)}") else root_id
                self.graph.add_edge(parent, out_id, label="links_to_external", color=COLOR_OUTBOUND)

        # 5. Infrastructure: server IPs and WHOIS registrars
        for infra in findings.get("infrastructure_intel", []):
            dom = (infra.get("domain") or "").lower()
            if not dom:
                continue
            ip = infra.get("server_ip", "Unresolved")
            registrar = infra.get("registrar", "Unknown")
            created = infra.get("creation_date", "Unknown")
            dom_id = self._add_domain_node(dom, created=created)

            if ip and ip != "Unresolved":
                ip_id = f"IP_{ip}"
                self._add_node_safe(
                    ip_id,
                    label=f"SERVER IP\n{clean_text(ip, 45)}",
                    color=COLOR_IP,
                    size=22,
                    shape="hexagon",
                    title=f"IP Address: {clean_text(ip, 45)}",
                    font={"color": "#0E1117", "size": 11, "bold": True},
                )
                self.graph.add_edge(dom_id, ip_id, label="A_record_ip", color=COLOR_IP)

            if registrar and registrar != "Unknown":
                reg_id = f"REG_{clean_text(registrar, 25)}"
                self._add_node_safe(
                    reg_id,
                    label=f"REGISTRAR\n{clean_text(registrar, 24)}",
                    color=COLOR_REGISTRAR,
                    size=18,
                    shape="ellipse",
                    title=f"Registrar: {clean_text(registrar, 120)}\nRegistered On: {clean_text(created, 60)}",
                    font={"color": "#0E1117", "size": 10},
                )
                self.graph.add_edge(dom_id, reg_id, label="whois_registrar", color=COLOR_REGISTRAR)

        # 6. Pivot-hub detection: well-connected nodes are drawn larger with a border
        centrality = nx.degree_centrality(self.graph)
        for node_id, score in centrality.items():
            if node_id != root_id and self.graph.degree(node_id) >= 3:
                current_size = self.graph.nodes[node_id].get("size", 20)
                self.graph.nodes[node_id]["size"] = min(42, current_size + int(score * 30))
                self.graph.nodes[node_id]["borderWidth"] = 3

        net = Network(
            height="660px",
            width="100%",
            bgcolor="#0E1117",
            font_color="white",
            directed=True,
            cdn_resources="in_line",
        )
        net.from_nx(self.graph)
        net.set_options(json.dumps({
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -11000,
                    "centralGravity": 0.22,
                    "springLength": 230,
                    "springConstant": 0.03,
                },
                "minVelocity": 0.75,
            }
        }))

        html_path = os.path.join(output_dir, f"maltego_graph_{short_sha}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(net.generate_html())
        return html_path
