import pytest

from helpers import start_server, structured_bytes
from spiderfoot_engine import MiniSpiderFoot


@pytest.fixture
def evidence_bytes():
    return structured_bytes()


@pytest.fixture
def engine():
    """Engine with the SSRF guard relaxed (tests talk to 127.0.0.1) and no sleeping."""
    return MiniSpiderFoot(allow_private_hosts=True, delay_scale=0)


@pytest.fixture
def strict_engine():
    """Engine with production network safety on."""
    return MiniSpiderFoot(delay_scale=0)


@pytest.fixture
def web(tmp_path):
    server, base = start_server(tmp_path)
    yield tmp_path, base
    server.shutdown()
    server.server_close()
