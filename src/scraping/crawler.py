"""Season orchestrator: rounds -> matches -> lineups/incidents -> CSV rows.

Resumable: after every round the completed set is persisted as a checkpoint,
so an interrupted run can continue from the last finished round.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from .client import SofaScoreClient, SofaScoreError
from .rate_limiter import RateLimiter
from . import parsers
from .models import PlayerMatchStat
from .csv_writer import MatchStatsCsvWriter

log = logging.getLogger("sofascore_scraper")

ROUND_RANGE = (config.FIRST_ROUND, config.LAST_ROUND)


@dataclass
class CrawlResult:
    season_id: int
    season_name: str | None
    completed_rounds: list[int] = field(default_factory=list)
    rows_written: int = 0
    round_files: list[Path] = field(default_factory=list)
    consolidated_file: Path | None = None


class SeasonCrawler:
    def __init__(
        self,
        client: SofaScoreClient,
        *,
        tournament_id: int = config.TOURNAMENT_ID,
        season_year: str = config.SEASON_YEAR,
        season_id: int | None = config.SEASON_ID,
        rounds: tuple[int, int] = ROUND_RANGE,
        data_dir: Path = config.DATA_DIR,
        partition: bool = True,
        resume: bool = True,
    ) -> None:
        self.client = client
        self.tournament_id = tournament_id
        self.season_year = season_year
        self.season_id = season_id
        self.rounds = rounds
        self.partition = partition
        self.resume = resume
        self.writer = MatchStatsCsvWriter(
            data_dir,
            consolidated_name=config.CONSOLIDATED_CSV_NAME,
            round_template=config.ROUND_CSV_TEMPLATE,
        )
        self.checkpoint_path = data_dir / config.PROGRESS_FILE

    # -- public -------------------------------------------------------------
    def crawl(self) -> CrawlResult:
        season_id, season_name = self._resolve_season()
        completed = set(self._load_checkpoint(season_id)) if self.resume else set()
        rounds_done: list[int] = []
        total_rows = 0

        for round_number in range(self.rounds[0], self.rounds[1] + 1):
            if round_number in completed:
                log.info("round %s already done, skipping (resume)", round_number)
                rounds_done.append(round_number)
                continue
            rows = self._crawl_round(round_number, season_id, season_name)
            total_rows += self.writer.append_to_consolidated(rows)
            round_files: list[Path] = []
            if self.partition:
                round_files.append(self.writer.write_round(round_number, rows))
            self._save_checkpoint(season_id, rounds_done + [round_number], total_rows)
            log.info("round %s/%-2s done: %s rows", round_number, self.rounds[1], len(rows))
            rounds_done.append(round_number)

        return CrawlResult(
            season_id=season_id,
            season_name=season_name,
            completed_rounds=sorted(rounds_done),
            rows_written=total_rows,
            consolidated_file=self.writer.consolidated_path,
        )

    # -- steps ---------------------------------------------------------------
    def _resolve_season(self) -> tuple[int, str | None]:
        if self.season_id is not None:
            return self.season_id, None
        for season in self.client.get_seasons(self.tournament_id):
            if str(season.get("year")) == self.season_year:
                return int(season["id"]), season.get("name")
        raise RuntimeError(
            f"season year {self.season_year!r} not found for tournament {self.tournament_id}"
        )

    def _crawl_round(
        self, round_number: int, season_id: int, season_name: str | None
    ) -> list[PlayerMatchStat]:
        events = self.client.get_round_events(self.tournament_id, season_id, round_number)
        finished = [e for e in events if parsers.is_finished(e)]
        rows: list[PlayerMatchStat] = []
        for event in finished:
            try:
                event_id = int(event["id"])
                lineups = self.client.get_lineups(event_id)
                incidents = self.client.get_incidents(event_id)
            except (KeyError, TypeError, SofaScoreError) as exc:
                log.warning("skipping event %s: %s", event.get("id"), exc)
                continue
            rows.extend(
                parsers.parse_match(
                    event,
                    lineups,
                    incidents,
                    season_id=season_id,
                    season_name=season_name,
                )
            )
        return rows

    # -- checkpoint ----------------------------------------------------------
    def _load_checkpoint(self, season_id: int) -> list[int]:
        if not self.checkpoint_path.exists():
            return []
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if data.get("season_id") != season_id:
                return []
            return [int(r) for r in data.get("completed_rounds", [])]
        except (ValueError, OSError):
            return []

    def _save_checkpoint(self, season_id: int, rounds_done: list[int], rows_written: int) -> None:
        payload = {
            "season_id": season_id,
            "tournament_id": self.tournament_id,
            "completed_rounds": sorted(set(int(r) for r in rounds_done)),
            "rows_written": rows_written,
        }
        self.checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Brasileirao Serie A player match stats from SofaScore."
    )
    parser.add_argument("--tournament", type=int, default=config.TOURNAMENT_ID)
    parser.add_argument("--year", default=config.SEASON_YEAR)
    parser.add_argument("--season-id", type=int, default=None, help="auto-resolved when omitted")
    parser.add_argument("--from-round", type=int, default=config.FIRST_ROUND)
    parser.add_argument("--to-round", type=int, default=config.LAST_ROUND)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--no-partition", action="store_true", help="skip per-round CSV files")
    parser.add_argument("--no-resume", action="store_true", help="restart from scratch")
    parser.add_argument(
        "--delay-min",
        type=float,
        default=config.REQUEST_DELAY_MIN_SECONDS,
        help="min seconds between requests",
    )
    parser.add_argument("--delay-max", type=float, default=config.REQUEST_DELAY_MAX_SECONDS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    limiter = RateLimiter(min_interval=args.delay_min, max_interval=args.delay_max)
    with SofaScoreClient(rate_limiter=limiter) as client:
        crawler = SeasonCrawler(
            client,
            tournament_id=args.tournament,
            season_year=args.year,
            season_id=args.season_id,
            rounds=(args.from_round, args.to_round),
            data_dir=args.data_dir,
            partition=not args.no_partition,
            resume=not args.no_resume,
        )
        result = crawler.crawl()
    log.info(
        "done: season=%s rounds=%s rows=%s file=%s",
        result.season_id,
        len(result.completed_rounds),
        result.rows_written,
        result.consolidated_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
