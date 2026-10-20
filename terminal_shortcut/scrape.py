"""Entry point: python terminal_shortcut/scrape.py [options]

Example:
    python terminal_shortcut/scrape.py --from-round 1 --to-round 38 --delay-min 1.2 --delay-max 2.4
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraping.crawler import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
