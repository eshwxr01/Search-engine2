"""
osint_utils.py - small, dependency-free helpers shared by the engine and the
graph builder. Everything here is pure (no network, no browser), so it is
cheap to unit-test.
"""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

# --- domain filtering -------------------------------------------------------
# Search-engine / infrastructure noise we never want as "findings".
# Matching is done on the registrable label, NOT on substrings, so that
# "olive.com" or "rubbing.net" are not caught by "live." or "bing.".
BLOCKED_BRANDS = {
    "yandex", "google", "gstatic", "googleapis",
    "bing", "msn", "microsoft", "live",
}
BLOCKED_EXACT_DOMAINS = {"ya.ru"}
BLOCKED_TLDS = {"google"}  # e.g. about.google, policies.google

_GENERIC_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac"}


def host_of(url: str) -> str:
    """Lower-cased hostname of a URL without port, userinfo or a leading www."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def domain_matches(host: str, domain: str) -> bool:
    """True when host is `domain` or one of its subdomains (no substring tricks)."""
    return host == domain or host.endswith("." + domain)


def registrable_domain(host: str) -> str:
    """Approximate registrable domain (a.b.example.co.in -> example.co.in).

    A real Public Suffix List would be more exact; this is good enough for
    grouping and WHOIS lookups without another dependency.
    """
    labels = [label for label in host.lower().strip(".").split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _GENERIC_SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def is_blocked_domain(host: str) -> bool:
    host = (host or "").lower().split(":")[0].strip(".")
    if not host:
        return False
    if any(domain_matches(host, d) for d in BLOCKED_EXACT_DOMAINS):
        return True
    labels = host.split(".")
    if labels[-1] in BLOCKED_TLDS:
        return True
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _GENERIC_SECOND_LEVEL:
        registrable_label = labels[-3]
    elif len(labels) >= 2:
        registrable_label = labels[-2]
    else:
        registrable_label = labels[0]
    return registrable_label in BLOCKED_BRANDS


# --- timestamps / WHOIS -----------------------------------------------------
def parse_timestamp_utc(raw):
    """Best-effort ISO-8601 -> UTC ISO string. Returns None when unparseable."""
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def normalize_whois_text(value):
    """python-whois may return a str, a list of str, or None."""
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if v), None)
    return str(value).strip() if value else None


def normalize_whois_date(value):
    """python-whois may return a datetime, a list of datetimes/strings, or None.

    Returns the EARLIEST date as an ISO string (the registration date is what
    matters for provenance), tolerating naive and timezone-aware values.
    """
    if isinstance(value, (list, tuple)):
        candidates = [v for v in value if v]
    else:
        candidates = [value] if value else []
    if not candidates:
        return None
    datetimes = [
        d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        for d in candidates
        if isinstance(d, datetime)
    ]
    if datetimes:
        return min(datetimes).isoformat()
    return str(candidates[0])


# --- untrusted text ---------------------------------------------------------
def clean_text(value, max_len: int = 200) -> str:
    """Neutralise scraped text before it is placed in the HTML graph.

    Page titles, bios and URLs are attacker-controlled. With the pinned
    pyvis / vis-network versions this was already safe (pyvis escapes the
    embedded JSON and tooltips render as plain text), so this is defence in
    depth: if a future renderer or template treats the text as HTML, angle
    brackets can no longer open a tag or close a <script> block.
    """
    text = " ".join(str(value or "").split())
    text = text.replace("<", "\u2039").replace(">", "\u203a")
    return text[:max_len]
