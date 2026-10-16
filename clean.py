"""Entry point: python clean.py [--source RAW_CSV] [--target CLEAN_CSV]"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cleaning.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
