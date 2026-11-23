import pytest
from datetime import datetime, timezone

from osint_utils import (
    clean_text,
    domain_matches,
    host_of,
    is_blocked_domain,
    normalize_whois_date,
    normalize_whois_text,
    parse_timestamp_utc,
    registrable_domain,
)


@pytest.mark.parametrize("host,blocked", [
    ("yandex.com", True),
    ("yandex.ru", True),
    ("ya.ru", True),
    ("www.google.com", True),
    ("mail.google.co.in", True),
    ("accounts.google.com", True),
    ("about.google", True),
    ("policies.google", True),
    ("www.bing.com", True),
    ("login.live.com", True),
    ("msn.com", True),
    ("gstatic.com", True),
    ("fonts.googleapis.com", True),
    # false positives the old substring matching produced
    ("olive.com", False),
    ("alive.org", False),
    ("rubbing.net", False),
    ("live.staticflickr.com", False),
    ("instagram.com", False),
    ("example.com", False),
    ("", False),
])
def test_is_blocked_domain(host, blocked):
    assert is_blocked_domain(host) is blocked


def test_host_of_strips_www_port_and_userinfo():
    assert host_of("https://www.Example.com:8443/path") == "example.com"
    assert host_of("https://user:pw@example.com/x") == "example.com"
    assert host_of("not a url") == ""


def test_domain_matches_is_not_a_substring_check():
    assert domain_matches("x.com", "x.com")
    assert domain_matches("m.x.com", "x.com")
    assert not domain_matches("box.com", "x.com")
    assert not domain_matches("smart.me", "t.me")


@pytest.mark.parametrize("host,expected", [
    ("example.com", "example.com"),
    ("a.b.example.com", "example.com"),
    ("www.example.co.in", "example.co.in"),
    ("news.bbc.co.uk", "bbc.co.uk"),
    ("localhost", "localhost"),
])
def test_registrable_domain(host, expected):
    assert registrable_domain(host) == expected


def test_parse_timestamp_utc():
    assert parse_timestamp_utc("2024-03-01T10:00:00Z") == "2024-03-01T10:00:00+00:00"
    assert parse_timestamp_utc("2024-03-01T15:30:00+05:30") == "2024-03-01T10:00:00+00:00"
    assert parse_timestamp_utc("2024-03-01T10:00:00") == "2024-03-01T10:00:00+00:00"  # naive -> UTC
    assert parse_timestamp_utc("yesterday-ish") is None
    assert parse_timestamp_utc(None) is None


def test_normalize_whois_text():
    assert normalize_whois_text("GoDaddy") == "GoDaddy"
    assert normalize_whois_text(["MarkMonitor", "Other"]) == "MarkMonitor"
    assert normalize_whois_text([None, "Second"]) == "Second"
    assert normalize_whois_text(None) is None
    assert normalize_whois_text([]) is None


def test_normalize_whois_date_picks_earliest_and_tolerates_mixed_tz():
    naive = datetime(2005, 5, 5, 12, 0)
    aware_early = datetime(1999, 1, 1, tzinfo=timezone.utc)
    assert normalize_whois_date([naive, aware_early]) == "1999-01-01T00:00:00+00:00"
    assert normalize_whois_date(naive) == "2005-05-05T12:00:00+00:00"
    assert normalize_whois_date(["2001-02-03", "2002-02-03"]) == "2001-02-03"
    assert normalize_whois_date(None) is None
    assert normalize_whois_date([]) is None


def test_clean_text_defuses_html_and_script_breakouts():
    payload = "</script><img src=x onerror=alert(1)>"
    cleaned = clean_text(payload)
    assert "<" not in cleaned and ">" not in cleaned
    assert clean_text("a\n\n  b\tc") == "a b c"
    assert len(clean_text("x" * 500, 50)) == 50
    assert clean_text(None) == ""
