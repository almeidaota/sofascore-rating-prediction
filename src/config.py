"""Central configuration: constants, base URLs, delays and data paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("data")
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONSOLIDATED_CSV_NAME = "brasileirao_2025_jogadores_partida_a_partida.csv"
ROUND_CSV_TEMPLATE = "rodada_{round_number:02d}.csv"
PROGRESS_FILE = "checkpoint.json"

PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV_NAME = "brasileirao_2025_dataset_modelo.csv"
TIMEZONE = "America/Sao_Paulo"

# Modelagem (compartilhado)
RANDOM_STATE = 42
N_BOOTSTRAP = 2000

# Segundo experimento: previsão pré-jogo (docs/modelagem.md, seção 12)
RESULTS_HISTORY_DIR = DATA_DIR / "results_history"
HISTORY_TEST_N_MATCHES = 80
WALK_FORWARD_STEP_MATCHES = 10
# Busca em grade: validação cruzada temporal dentro do treino, com 5 partições
# de 30 partidas cada, sempre posteriores ao treino da partição.
CV_N_SPLITS = 5
CV_TEST_N_MATCHES = 30

# SofaScore endpoints (api.sofascore.com/api/v1).  "unique-tournament" is the
# league code: 325 = Campeonato Brasileiro Serie A.
BASE_URL = "https://api.sofascore.com/api/v1"
TOURNAMENT_ID = 325
SEASON_YEAR = "2025"
SEASON_ID = 72034  # resolved automatically if None; pre-set from 2025-06 probe
FIRST_ROUND = 1
LAST_ROUND = 38

EVENTS_ROUND_PATH = "/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_number}"
SEASONS_PATH = "/unique-tournament/{tournament_id}/seasons"
EVENT_PATH = "/event/{event_id}"
LINEUPS_PATH = "/event/{event_id}/lineups"
INCIDENTS_PATH = "/event/{event_id}/incidents"

# Request policy: friendly crawl-delay, bounded by our own conscience.
REQUEST_TIMEOUT_SECONDS = 20.0
REQUEST_DELAY_MIN_SECONDS = 1.2
REQUEST_DELAY_MAX_SECONDS = 2.4
RETRY_MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 1.0
HTTP_MAX_RETRIES_CODES = {429, 500, 502, 503, 504}
