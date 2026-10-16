"""Normalization of raw SofaScore JSON payloads into typed models."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .. import config
from .models import PlayerMatchStat

# SofaScore stat key -> PlayerMatchStat int field
_INT_FIELDS: dict[str, str] = {
    "minutesPlayed": "minutes_played",
    "goals": "goals",
    "goalAssist": "assists",
    "totalShots": "shots_total",
    "onTargetScoringAttempt": "shots_on_target",
    "shotOffTarget": "shots_off_target",
    "touches": "touches",
    "totalPass": "passes_total",
    "accuratePass": "passes_accurate",
    "keyPass": "key_passes",
    "bigChanceCreated": "big_chance_created",
    "bigChanceMissed": "big_chance_missed",
    "duelWon": "duels_won",
    "duelLost": "duels_lost",
    "aerialWon": "aerial_won",
    "aerialLost": "aerial_lost",
    "totalTackle": "tackles_total",
    "wonTackle": "tackles_won",
    "interceptionWon": "interceptions",
    "totalClearance": "clearances",
    "outfielderBlock": "blocks",
    "blockedScoringAttempt": "blocked_shots",
    "ballRecovery": "ball_recovery",
    "fouls": "fouls_committed",
    "wasFouled": "fouls_suffered",
    "totalOffside": "offsides",
    "possessionLostCtrl": "possession_lost",
}
COUNT_FIELDS: tuple[str, ...] = tuple(v for v in _INT_FIELDS.values() if v != "minutes_played")
_FLOAT_FIELDS: dict[str, str] = {
    "rating": "rating",
    "expectedGoals": "expected_goals",
    "expectedAssists": "expected_assists",
}

_STATUS_FINISHED = "finished"
_LOCAL_TZ = ZoneInfo(config.TIMEZONE)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def match_date_from_timestamp(timestamp: int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, tz=_LOCAL_TZ).strftime("%Y-%m-%d")


def is_finished(event: dict[str, Any]) -> bool:
    return (event.get("status") or {}).get("type") == _STATUS_FINISHED


def aggregate_cards(incidents: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map player_id -> {"yellow": n, "red": n, "minutes": [..]} from incidents."""
    tallies: dict[int, dict[str, Any]] = {}
    for incident in incidents.get("incidents", []):
        if incident.get("incidentType") != "card":
            continue
        player = incident.get("player") or {}
        player_id = _as_int(player.get("id"))
        if player_id is None:
            continue
        incident_class = incident.get("incidentClass")
        entry = tallies.setdefault(player_id, {"yellow": 0, "red": 0, "minutes": []})
        minute = _as_int(incident.get("time"))
        if minute is not None:
            entry["minutes"].append(minute)
        if incident_class in {"yellow", "secondyellow"}:
            entry["yellow"] += 1
        if incident_class in {"red", "secondyellow", "yellowred"}:
            entry["red"] += 1
    return tallies


def parse_match(
    event: dict[str, Any],
    lineups: dict[str, Any],
    incidents: dict[str, Any],
    *,
    season_id: int,
    season_name: str | None,
    tournament_name: str | None = None,
    require_played: bool = True,
) -> list[PlayerMatchStat]:
    """Turn the raw event + lineups + incidents payloads into per-player rows.

    Only fielded players are kept (starters plus used substitutes): unused
    substitutes appear in the SofaScore payload with empty statistics.
    """
    cards = aggregate_cards(incidents)
    context = _match_context(event)
    context["season_id"] = season_id
    context["season_name"] = season_name
    context["tournament_name"] = tournament_name or context["tournament_name"]
    context["kickoff_timestamp"] = _as_int(event.get("startTimestamp")) or 0
    context["match_date"] = match_date_from_timestamp(context["kickoff_timestamp"])

    rows: list[PlayerMatchStat] = []
    for side in ("home", "away"):
        side_payload = lineups.get(side) or {}
        context["side"] = side
        context["team_id"] = context[f"{side}_team_id"]
        context["team_name"] = context[f"{side}_team"]
        for slot in side_payload.get("players", []):
            row = _row_from_slot(slot, cards, context)
            if row is None:
                continue
            if require_played and (row.minutes_played or 0) <= 0:
                continue
            rows.append(row)
    return rows


def _match_context(event: dict[str, Any]) -> dict[str, Any]:
    round_info = event.get("roundInfo") or {}
    home_score = event.get("homeScore") or {}
    away_score = event.get("awayScore") or {}
    return {
        "match_id": event["id"],
        "round": round_info.get("round"),
        "tournament_name": (event.get("tournament") or {}).get("name"),
        "home_team_id": _team_id(event.get("homeTeam")),
        "home_team": _team_name(event.get("homeTeam")),
        "away_team_id": _team_id(event.get("awayTeam")),
        "away_team": _team_name(event.get("awayTeam")),
        "home_score": _as_int(home_score.get("current")) or _as_int(home_score.get("display")),
        "away_score": _as_int(away_score.get("current")) or _as_int(away_score.get("display")),
    }


def _team_id(team: Any) -> int | None:
    return _as_int(team.get("id")) if isinstance(team, dict) else None


def _team_name(team: Any) -> str | None:
    return team.get("name") if isinstance(team, dict) else None


def _row_from_slot(
    slot: dict[str, Any],
    cards: dict[int, dict[str, Any]],
    context: dict[str, Any],
) -> PlayerMatchStat | None:
    player = slot.get("player") or {}
    player_id = _as_int(player.get("id"))
    if player_id is None:
        return None
    raw_stats = slot.get("statistics") or {}
    stats = raw_stats if isinstance(raw_stats, dict) else {}

    data: dict[str, Any] = {key: value for key, value in context.items()}
    data.update(
        {
            "player_id": player_id,
            "player_name": player.get("name"),
            "player_short_name": player.get("shortName") or player.get("name"),
            "shirt_number": _as_int(slot.get("shirtNumber") or player.get("jerseyNumber")),
            "position": slot.get("position") or player.get("position"),
            "is_substitute": bool(slot.get("substitute", False)),
        }
    )
    for soffa_key, attr in _INT_FIELDS.items():
        data[attr] = _as_int(stats.get(soffa_key))
    for soffa_key, attr in _FLOAT_FIELDS.items():
        data[attr] = _as_float(stats.get(soffa_key))

    if data["passes_total"]:
        data["pass_accuracy_pct"] = round(
            100.0 * (data["passes_accurate"] or 0) / data["passes_total"], 1
        )

    card = cards.get(player_id)
    if card:
        data["yellow_cards"] = card["yellow"]
        data["red_cards"] = card["red"]
        data["card_minutes"] = ", ".join(str(m) for m in sorted(card["minutes"])) or None

    data["stats_raw_json"] = json.dumps(stats, ensure_ascii=False)
    return PlayerMatchStat(**data)
