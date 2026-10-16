"""Pure DataFrame -> DataFrame preprocessing steps for the modelling dataset."""
from __future__ import annotations

import json

import pandas as pd

from ..scraping import parsers

PHYSICAL: dict[str, str] = {
    "topSpeed": "top_speed",
    "kilometersCovered": "km_covered",
    "metersCoveredRunningKm": "km_running",
    "metersCoveredHighSpeedRunningKm": "km_high_speed_running",
    "metersCoveredSprintingKm": "km_sprinting",
    "numberOfSprints": "sprints",
}
CARRIES: dict[str, str] = {
    "ballCarriesCount": "carries",
    "progressiveBallCarriesCount": "progressive_carries",
    "totalBallCarriesDistance": "carries_distance",
    "totalProgressiveBallCarriesDistance": "progressive_carries_distance",
    "bestBallCarryProgression": "best_carry_progression",
    "totalProgression": "total_progression",
}
PASS_ZONES: dict[str, str] = {
    "accurateOwnHalfPasses": "own_half_passes_accurate",
    "totalOwnHalfPasses": "own_half_passes_total",
    "accurateOppositionHalfPasses": "opp_half_passes_accurate",
    "totalOppositionHalfPasses": "opp_half_passes_total",
}
LONG_BALLS_CROSSES: dict[str, str] = {
    "accurateLongBalls": "long_balls_accurate",
    "totalLongBalls": "long_balls_total",
    "accurateCross": "crosses_accurate",
    "totalCross": "crosses_total",
}
CONTESTS: dict[str, str] = {
    "totalContest": "contests_total",
    "wonContest": "contests_won",
}
GOALKEEPER: dict[str, str] = {
    "saves": "saves",
    "savedShotsFromInsideTheBox": "saves_inside_box",
    "punches": "punches",
    "goodHighClaim": "high_claims",
    "accurateKeeperSweeper": "keeper_sweeper_accurate",
    "totalKeeperSweeper": "keeper_sweeper_total",
    "goalsPrevented": "goals_prevented",
}
RAW_FIELDS: dict[str, str] = (
    PHYSICAL | CARRIES | PASS_ZONES | LONG_BALLS_CROSSES | CONTESTS | GOALKEEPER
)

TRACKING_GROUPS: tuple[list[str], ...] = (list(PHYSICAL.values()), list(CARRIES.values()))
ZERO_WHEN_MISSING: list[str] = [
    *parsers.COUNT_FIELDS,
    "expected_goals",
    "expected_assists",
    *PASS_ZONES.values(),
    *LONG_BALLS_CROSSES.values(),
    *CONTESTS.values(),
]
NOT_APPLICABLE = -1
GOALKEEPER_POSITION = "G"

POSITION_CODES: dict[str, int] = {"G": 0, "D": 1, "M": 2, "F": 3}
SIDE_CODES: dict[str, int] = {"away": 0, "home": 1}

ROW_KEY = ["match_id", "player_id"]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=ROW_KEY, keep="last")


def fix_match_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(match_date=df["kickoff_timestamp"].map(parsers.match_date_from_timestamp))


def expand_raw_stats(df: pd.DataFrame) -> pd.DataFrame:
    payloads = df["stats_raw_json"].fillna("{}").map(json.loads).tolist()
    raw = (
        pd.DataFrame(payloads, index=df.index)
        .reindex(columns=list(RAW_FIELDS))
        .rename(columns=RAW_FIELDS)
    )
    return df.drop(columns="stats_raw_json").join(raw)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[ZERO_WHEN_MISSING] = df[ZERO_WHEN_MISSING].fillna(0)

    for group in TRACKING_GROUPS:
        tracked = df[group].notna().any(axis=1)
        df.loc[tracked, group] = df.loc[tracked, group].fillna(0)

    goalkeeper = list(GOALKEEPER.values())
    is_goalkeeper = df["position"] == GOALKEEPER_POSITION
    df.loc[is_goalkeeper, goalkeeper] = df.loc[is_goalkeeper, goalkeeper].fillna(0)
    df.loc[~is_goalkeeper, goalkeeper] = NOT_APPLICABLE
    return df


def drop_unrated(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["rating"])


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        position=df["position"].map(POSITION_CODES),
        side=df["side"].map(SIDE_CODES),
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(drop_duplicates)
        .pipe(fix_match_date)
        .pipe(expand_raw_stats)
        .pipe(fill_missing)
        .pipe(drop_unrated)
        .pipe(encode_categoricals)
        .reset_index(drop=True)
        .convert_dtypes()
    )
