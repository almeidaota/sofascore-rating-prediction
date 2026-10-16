"""Pydantic v2 models: strict, typed representation of every scraped record."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlayerStatistics(BaseModel):
    """Typed subset of SofaScore per-player in-match statistics."""

    model_config = ConfigDict(extra="ignore")

    minutesPlayed: int | None = None
    rating: float | None = None
    goals: int | None = None
    goalAssist: int | None = None
    totalShots: int | None = None
    onTargetScoringAttempt: int | None = None
    shotOffTarget: int | None = None
    touches: int | None = None
    totalPass: int | None = None
    accuratePass: int | None = None
    keyPass: int | None = None
    expectedGoals: float | None = None
    expectedAssists: float | None = None
    bigChanceCreated: int | None = None
    bigChanceMissed: int | None = None
    duelWon: int | None = None
    duelLost: int | None = None
    aerialWon: int | None = None
    aerialLost: int | None = None
    totalTackle: int | None = None
    wonTackle: int | None = None
    interceptionWon: int | None = None
    totalClearance: int | None = None
    outfielderBlock: int | None = None
    blockedScoringAttempt: int | None = None
    ballRecovery: int | None = None
    fouls: int | None = None
    wasFouled: int | None = None
    totalOffside: int | None = None
    possessionLostCtrl: int | None = None


class LineupSlot(BaseModel):
    """One player entry from the SofaScore lineups payload."""

    model_config = ConfigDict(extra="ignore")

    player: dict[str, Any]
    shirtNumber: int | None = None
    position: str | None = None
    substitute: bool = False
    statistics: PlayerStatistics | dict[str, Any] = Field(default_factory=dict)


class PlayerMatchStat(BaseModel):
    """A single output row: one player's performance in one match.

    Field order defines the consolidated CSV column order.
    """

    match_id: int
    season_id: int
    season_name: str | None = None
    round: int
    match_date: str
    kickoff_timestamp: int
    tournament_name: str | None = None
    home_team_id: int | None = None
    home_team: str | None = None
    away_team_id: int | None = None
    away_team: str | None = None
    home_score: int | None = None
    away_score: int | None = None

    player_id: int
    player_name: str
    player_short_name: str | None = None
    side: str
    team_id: int | None = None
    team_name: str | None = None
    shirt_number: int | None = None
    position: str | None = None
    is_substitute: bool = False

    minutes_played: int | None = None
    rating: float | None = None
    goals: int | None = None
    assists: int | None = None
    shots_total: int | None = None
    shots_on_target: int | None = None
    shots_off_target: int | None = None
    touches: int | None = None
    passes_total: int | None = None
    passes_accurate: int | None = None
    pass_accuracy_pct: float | None = None
    key_passes: int | None = None
    expected_goals: float | None = None
    expected_assists: float | None = None
    big_chance_created: int | None = None
    big_chance_missed: int | None = None
    duels_won: int | None = None
    duels_lost: int | None = None
    aerial_won: int | None = None
    aerial_lost: int | None = None
    tackles_total: int | None = None
    tackles_won: int | None = None
    interceptions: int | None = None
    clearances: int | None = None
    blocks: int | None = None
    blocked_shots: int | None = None
    ball_recovery: int | None = None
    fouls_committed: int | None = None
    fouls_suffered: int | None = None
    yellow_cards: int = 0
    red_cards: int = 0
    card_minutes: str | None = None
    offsides: int | None = None
    possession_lost: int | None = None
    stats_raw_json: str | None = Field(default=None, repr=False)
