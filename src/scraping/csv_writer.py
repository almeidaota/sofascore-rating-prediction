"""CSV persistence: per-round partition files + one consolidated dataset.

All files live under ``data/`` and are written with a BOM (``utf-8`` plus a
single ``\\ufeff`` marker) so Excel/Sheets decode Brazilian accents correctly.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .models import PlayerMatchStat

FIELD_NAMES = list(PlayerMatchStat.model_fields)


def _row_dict(row: PlayerMatchStat) -> dict[str, object]:
    return {name: ("" if value is None else value) for name, value in row.model_dump().items()}


class MatchStatsCsvWriter:
    def __init__(self, data_dir: Path, consolidated_name: str, round_template: str) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.consolidated_path = data_dir / consolidated_name
        self.round_template = round_template

    # -- consolidated -------------------------------------------------------
    def append_to_consolidated(self, rows: list[PlayerMatchStat]) -> int:
        if not rows:
            return 0
        created = not self.consolidated_path.exists()
        with self.consolidated_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if created:
                handle.write("\ufeff")
                writer.writerow(FIELD_NAMES)
            writer.writerows([_row_dict(row).values() for row in _sorted(rows)])
        return len(rows)

    # -- per-round partition ------------------------------------------------
    def write_round(self, round_number: int, rows: list[PlayerMatchStat]) -> Path:
        path = self.data_dir / self.round_template.format(round_number=round_number)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(FIELD_NAMES)
            writer.writerows([_row_dict(row).values() for row in _sorted(rows)])
        return path


def _sorted(rows: list[PlayerMatchStat]) -> list[PlayerMatchStat]:
    return sorted(rows, key=lambda r: (r.match_date, r.match_id, r.team_id or 0, r.player_id))
