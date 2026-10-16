"""Tests for the preprocessing steps that build the modelling dataset."""
import json

import pandas as pd
import pytest

from src.cleaning import cleaners as C
from src.cleaning.pipeline import run_cleaning
from src.scraping.csv_writer import FIELD_NAMES

DAY_KICKOFF = 1743283800
NIGHT_KICKOFF = 1743296400


def _row(**overrides):
    row = {name: None for name in FIELD_NAMES}
    row.update(
        match_id=1,
        season_id=72034,
        round=1,
        match_date="",
        kickoff_timestamp=DAY_KICKOFF,
        player_id=10,
        player_name="Player",
        side="home",
        position="F",
        minutes_played=90,
        rating=7.0,
        yellow_cards=0,
        red_cards=0,
        stats_raw_json="{}",
    )
    row.update(overrides)
    return row


def _stats(**stats):
    return json.dumps(stats)


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(player_id=10, position="F", stats_raw_json=_stats(topSpeed=31.5, numberOfSprints=12)),
            _row(player_id=11, position="G", side="away", stats_raw_json=_stats(punches=2)),
            _row(player_id=12, position="M", rating=None, minutes_played=5),
            _row(match_id=2, player_id=10, kickoff_timestamp=NIGHT_KICKOFF, goals=1),
            _row(match_id=2, player_id=10, kickoff_timestamp=NIGHT_KICKOFF, goals=2),
        ]
    )


def _player(df, match_id, player_id):
    return df[(df.match_id == match_id) & (df.player_id == player_id)].iloc[0]


def test_duplicates_keep_last(raw):
    out = C.clean(raw)
    assert len(out[(out.match_id == 2) & (out.player_id == 10)]) == 1
    assert _player(out, 2, 10).goals == 2


def test_match_date_uses_local_timezone(raw):
    out = C.clean(raw)
    assert _player(out, 2, 10).match_date == "2025-03-29"


def test_raw_stats_expanded_into_named_columns(raw):
    out = C.clean(raw)
    assert "stats_raw_json" not in out.columns
    assert set(C.RAW_FIELDS.values()) <= set(out.columns)
    assert _player(out, 1, 10).top_speed == 31.5


def test_goalkeeper_stats_not_applicable_for_outfield_players(raw):
    out = C.clean(raw)
    assert _player(out, 1, 10).saves == C.NOT_APPLICABLE
    goalkeeper = _player(out, 1, 11)
    assert goalkeeper.punches == 2
    assert goalkeeper.saves == 0


def test_tracking_groups_missing_only_when_not_computed(raw):
    out = C.clean(raw)
    tracked = _player(out, 1, 10)
    assert tracked.sprints == 12
    assert tracked.km_covered == 0
    assert pd.isna(_player(out, 2, 10).top_speed)
    assert pd.isna(tracked.carries)


def test_omitted_counts_become_zero(raw):
    out = C.clean(raw)
    row = _player(out, 1, 10)
    assert row.goals == 0
    assert row.expected_goals == 0
    assert row.crosses_total == 0
    assert pd.isna(row.pass_accuracy_pct)


def test_unrated_rows_dropped(raw):
    out = C.clean(raw)
    assert 12 not in set(out.player_id)
    assert out.rating.notna().all()


def test_categoricals_encoded(raw):
    out = C.clean(raw)
    assert _player(out, 1, 10).position == C.POSITION_CODES["F"]
    assert _player(out, 1, 11).position == C.POSITION_CODES["G"]
    assert set(out.side) == {C.SIDE_CODES["home"], C.SIDE_CODES["away"]}


def test_run_cleaning_writes_csv(tmp_path, raw):
    source = tmp_path / "raw.csv"
    target = tmp_path / "processed" / "clean.csv"
    raw.to_csv(source, index=False, encoding="utf-8-sig")

    assert run_cleaning(source, target) == 3
    written = pd.read_csv(target, encoding="utf-8-sig")
    assert len(written) == 3
    assert "player_name" in written.columns
