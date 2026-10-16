"""Integration tests: crawl flow + CSV persistence using real fixtures."""
import csv
import json
from pathlib import Path

import httpx
import pytest

from src.scraping.csv_writer import FIELD_NAMES
from src.scraping.crawler import SeasonCrawler

from conftest import make_client

SINGLE_EVENT_ROUND = "unique-tournament/325/season/72034/events/round/1"


def _single_event_fixture(tmp_path: Path, fixture_path: Path) -> Path:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    event = json.loads((fixture_path.parent / "event_13473343.json").read_text())[
        "event"
    ]
    single = {"events": [event], "hasNextPage": False}
    override = tmp_path / "round_1_single.json"
    override.write_text(json.dumps(single), encoding="utf-8")
    return override


def _new_crawler(tmp_path: Path, *, resume: bool, client=None):
    if client is None:
        client = make_client()
    return SeasonCrawler(
        client,
        season_id=72034,
        rounds=(1, 1),
        data_dir=tmp_path / "data",
        resume=resume,
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def test_full_round_crawl_writes_expected_files(tmp_path, fixtures_dir):
    override = _single_event_fixture(tmp_path, fixtures_dir / "round_1_events.json")
    client = make_client(overrides={SINGLE_EVENT_ROUND: str(override)})
    result = _new_crawler(tmp_path, resume=False, client=client).crawl()

    assert result.season_id == 72034
    assert result.rows_written == 32

    consolidated = tmp_path / "data" / "brasileirao_2025_jogadores_partida_a_partida.csv"
    rodada = tmp_path / "data" / "rodada_01.csv"
    assert result.consolidated_file == consolidated
    assert consolidated.exists()
    assert rodada.exists()

    text = consolidated.read_text(encoding="utf-8-sig")
    assert text.startswith("match_id,")  # BOM consumed by utf-8-sig
    assert text.count("\n") == 33  # header + 32 rows
    assert "Cássio" in text
    assert "Brasileirão Betano" in text

    rows = list(csv.DictReader(consolidated.open(encoding="utf-8-sig")))
    assert len(rows) == 32
    assert set(rows[0].keys()) == set(FIELD_NAMES)
    assert list(rows[0].keys()) == FIELD_NAMES


def test_crawl_is_deterministic(tmp_path, fixtures_dir):
    override = _single_event_fixture(tmp_path, fixtures_dir / "round_1_events.json")
    results = []
    for _ in range(2):
        client = make_client(overrides={SINGLE_EVENT_ROUND: str(override)})
        result = _new_crawler(tmp_path, resume=False, client=client).crawl()
        results.append(result)
    out = tmp_path / "data" / "brasileirao_2025_jogadores_partida_a_partida.csv"
    assert out.read_bytes() == out.read_bytes()
    assert results[0].rows_written == results[1].rows_written == 32


def test_resume_skips_completed_rounds_without_requests(tmp_path, fixtures_dir):
    override = _single_event_fixture(tmp_path, fixtures_dir / "round_1_events.json")
    client = make_client(overrides={SINGLE_EVENT_ROUND: str(override)})
    _new_crawler(tmp_path, resume=False, client=client).crawl()

    def exploding(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be touched on a resumed round")

    resumed_client = make_client()
    resumed_client._http = httpx.Client(transport=httpx.MockTransport(exploding))
    result = _new_crawler(tmp_path, resume=True, client=resumed_client).crawl()

    assert result.completed_rounds == [1]
    assert result.rows_written == 0


def test_checkpoint_file_created(tmp_path, fixtures_dir):
    override = _single_event_fixture(tmp_path, fixtures_dir / "round_1_events.json")
    client = make_client(overrides={SINGLE_EVENT_ROUND: str(override)})
    _new_crawler(tmp_path, resume=False, client=client).crawl()
    checkpoint = tmp_path / "data" / "checkpoint.json"
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert data["season_id"] == 72034
    assert data["completed_rounds"] == [1]
    assert data["rows_written"] == 32
