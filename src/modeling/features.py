"""Pure feature-building helpers for the modelling experiment (docs/modelagem.md)."""
from __future__ import annotations

import pandas as pd

from ..cleaning import cleaners as C

TARGET = "rating"
POSITION_COLUMN = "position"

# Identificadores/descritivas: fora do X, usadas só para split,
# agrupamento e análise de erros.
ID_COLUMNS: tuple[str, ...] = (
    "match_id",
    "season_id",
    "season_name",
    "round",
    "match_date",
    "kickoff_timestamp",
    "tournament_name",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "player_id",
    "player_name",
    "player_short_name",
    "team_id",
    "team_name",
    "shirt_number",
    "card_minutes",
)

POSITION_LABELS: dict[int, str] = {code: name for name, code in C.POSITION_CODES.items()}


def feature_columns(df: pd.DataFrame, position: int | None = None) -> list[str]:
    """Colunas de X.

    `df` deve ser o subconjunto de TREINO (geral, se `position=None`; já filtrado
    pela posição, se `position` for dado) -- a decisão de quais colunas são
    constantes é feita só no treino, e a mesma lista deve ser reaproveitada
    depois para fatiar o X de teste correspondente (evita treino/teste com
    colunas diferentes).
    """
    candidates = [
        c for c in df.columns if c not in ID_COLUMNS and c not in (TARGET, POSITION_COLUMN)
    ]
    if position is None:
        return candidates + [POSITION_COLUMN]

    # dropna=False: uma coluna inteiramente NaN também conta como "constante"
    # (não há o que aprender dela).
    constant = {c for c in candidates if df[c].nunique(dropna=False) <= 1}
    return [c for c in candidates if c not in constant]

