"""Entry point: python train_history.py [--source CLEAN_CSV] [--results-dir DIR]"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.modeling.training_history import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
