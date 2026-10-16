"""Tests for payload normalization into typed PlayerMatchStat records."""
from src.scraping import parsers as P
from src.scraping.models import PlayerMatchStat

from conftest import load_fixture

SEASON_ID = 72034
SEASON_NAME = "Brasileiro Serie A 2025"


def _parse():
    event = load_fixture("event_13473343.json")["event"]
    lineups = load_fixture("event_13473343_lineups.json")
    incidents = load_fixture("event_13473343_incidents.json")
    return P.parse_match(
        event, lineups, incidents, season_id=SEASON_ID, season_name=SEASON_NAME
    )


def test_row_count_matches_fielded_players():
    rows = _parse()
    assert len(rows) == 32  # 16 fielded players per side (starters + used subs)


def test_only_players_who_played_are_kept():
    rows = _parse()
    assert all(r.minutes_played and r.minutes_played > 0 for r in rows)
    unused_substitutes = [r for r in rows if r.is_substitute and (r.minutes_played or 0) == 0]
    assert unused_substitutes == []


def test_typed_model_fields():
    rows = _parse()
    assert all(isinstance(r, PlayerMatchStat) for r in rows)
    row = rows[0]
    assert isinstance(row.match_id, int)
    assert isinstance(row.is_substitute, bool)


def test_match_context_fields():
    rows = _parse()
    row = rows[0]
    assert row.match_id == 13473343
    assert row.round == 1
    assert row.match_date == "2025-03-29"
    assert row.home_team == "Cruzeiro"
    assert row.away_team == "Mirassol"
    assert row.home_score == 2
    assert row.away_score == 1
    assert row.season_id == SEASON_ID
    assert row.season_name == SEASON_NAME
    assert row.tournament_name == "Brasileirão Betano"


def test_player_statistics_cassio():
    rows = _parse()
    cassio = [r for r in rows if r.player_name == "Cássio"]
    assert len(cassio) == 1
    row = cassio[0]
    assert row.player_id == 27756
    assert row.side == "home"
    assert row.team_name == "Cruzeiro"
    assert row.shirt_number == 1
    assert row.position == "G"
    assert not row.is_substitute
    assert row.minutes_played == 90
    assert row.rating == 7.8
    assert row.touches == 29
    assert row.passes_total == 23
    assert row.passes_accurate == 15
    assert row.pass_accuracy_pct == 65.2
    assert row.yellow_cards == 1
    assert row.red_cards == 0
    assert row.card_minutes == "90"


def test_goals_and_assists_recorded():
    rows = _parse()
    with_goals = [r for r in rows if (r.goals or 0) > 0]
    assert any(r.player_name for r in with_goals)  # fixture match had 3 goals
    assert sum(r.goals or 0 for r in rows) == 3


def test_missing_ratings_are_none_not_crash():
    rows = _parse()
    assert all(r.rating is None or isinstance(r.rating, float) for r in rows)


def test_empty_lineup_side_is_tolerated():
    event = load_fixture("event_13473343.json")["event"]
    incidents = load_fixture("event_13473343_incidents.json")
    lineups = load_fixture("event_13473343_lineups.json")
    lineups["away"] = {}
    rows = P.parse_match(
        event, lineups, incidents, season_id=SEASON_ID, season_name=SEASON_NAME
    )
    assert len(rows) == 16
    assert all(r.side == "home" for r in rows)


def test_aggregate_cards_tally():
    incidents = load_fixture("event_13473343_incidents.json")
    cards = P.aggregate_cards(incidents)
    assert cards[27756] == {"yellow": 1, "red": 0, "minutes": [90]}


def test_is_finished_detection():
    finished = {"status": {"type": "finished", "code": 100}}
    upcoming = {"status": {"type": "notstarted", "code": 0}}
    assert P.is_finished(finished)
    assert not P.is_finished(upcoming)
    assert not P.is_finished({})


def test_match_date_uses_brazil_timezone():
    assert P.match_date_from_timestamp(1743283800) == "2025-03-29"
    assert P.match_date_from_timestamp(1743296400) == "2025-03-29"
    assert P.match_date_from_timestamp(None) == ""
