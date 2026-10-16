"""Shared test scaffolding: offline client backed by real SofaScore payloads."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.scraping.client import SofaScoreClient

FIXTURES = Path(__file__).parent / "fixtures"

ROUTES: dict[str, str] = {
    "unique-tournament/325/seasons": "tournament_325_seasons.json",
    "unique-tournament/325/season/72034/events/round/1": "round_1_events.json",
    "event/13473343": "event_13473343.json",
    "event/13473343/lineups": "event_13473343_lineups.json",
    "event/13473343/incidents": "event_13473343_incidents.json",
}


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def build_handler(overrides: dict[str, str] | None = None):
    routes = dict(ROUTES)
    if overrides:
        routes.update(overrides)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for suffix, file_name in sorted(routes.items(), key=lambda item: len(item[0]), reverse=True):
            if path.endswith(suffix):
                return httpx.Response(200, json=load_fixture(file_name), request=request)
        return httpx.Response(404, json={"error": "fixture not found"}, request=request)

    return handler


def make_client(overrides: dict[str, str] | None = None) -> SofaScoreClient:
    return SofaScoreClient(transport=httpx.MockTransport(build_handler(overrides)))


@pytest.fixture
def client() -> SofaScoreClient:
    with make_client() as client:
        yield client
