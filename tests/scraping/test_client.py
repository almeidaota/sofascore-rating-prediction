"""Tests for the HTTP client: headers, retries and endpoint helpers."""
import httpx
import pytest

from src import config
from src.scraping.client import SofaScoreClient, SofaScoreError

from conftest import build_handler, load_fixture, make_client


def test_default_headers_sent():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={}, request=request)

    client = SofaScoreClient(transport=httpx.MockTransport(handler))
    client.get_json("/probe")
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["user-agent"].startswith("Mozilla/5.0")
    assert headers["referer"] == "https://www.sofascore.com/"
    assert headers["origin"] == "https://www.sofascore.com"


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "RETRY_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0.001)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    client = SofaScoreClient(transport=httpx.MockTransport(handler))
    assert client.get_json("/flaky") == {"ok": True}
    assert calls["n"] == 3


def test_exhausts_retries_then_raises(monkeypatch):
    monkeypatch.setattr(config, "RETRY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0.001)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={}, request=request)

    client = SofaScoreClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SofaScoreError):
        client.get_json("/always-blocked")


def test_get_seasons():
    client = make_client()
    seasons = client.get_seasons(325)
    assert any(s["id"] == 72034 and s["year"] == "2025" for s in seasons)


def test_get_round_events():
    client = make_client()
    events = client.get_round_events(325, 72034, 1)
    assert len(events) == 10
    assert {e["id"] for e in events} == {
        13473338, 13473339, 13473340, 13473341, 13473342,
        13473343, 13473344, 13473345, 13473346, 13473347,
    }


def test_get_event_raises_on_unexpected_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"weird": True}, request=request)

    client = SofaScoreClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SofaScoreError):
        client.get_event(1)


def test_non_json_body_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>oops</html>", request=request)

    client = SofaScoreClient(transport=httpx.MockTransport(handler))
    with pytest.raises(SofaScoreError):
        client.get_json("/not-json")


def test_fixture_payloads_are_valid():
    assert "seasons" in load_fixture("tournament_325_seasons.json")
    assert "events" in load_fixture("round_1_events.json")
    assert "event" in load_fixture("event_13473343.json")
    assert {"home", "away", "confirmed"} <= set(load_fixture("event_13473343_lineups.json"))
    assert "incidents" in load_fixture("event_13473343_incidents.json")
