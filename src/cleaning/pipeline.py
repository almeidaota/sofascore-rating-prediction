"""Preprocessing step: raw consolidated CSV -> cleaned modelling dataset."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .. import config
from . import cleaners

log = logging.getLogger("sofascore_scraper")


def run_cleaning(source: Path, target: Path) -> int:
    df = cleaners.clean(pd.read_csv(source, encoding="utf-8-sig"))
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return len(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean the scraped player match stats into a modelling dataset."
    )
    parser.add_argument("--source", type=Path, default=config.DATA_DIR / config.CONSOLIDATED_CSV_NAME)
    parser.add_argument("--target", type=Path, default=config.PROCESSED_DIR / config.CLEAN_CSV_NAME)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = run_cleaning(args.source, args.target)
    log.info("done: rows=%s file=%s", rows, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
