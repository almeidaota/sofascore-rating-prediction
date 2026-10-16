"""Pre-match features: opponent strength and player form, using only matches
strictly before the one being predicted (docs/modelagem.md, seções 12-14).

Chronology is always by `kickoff_timestamp`, never by `round` -- `round` does
not reliably track real match order (postponed matches), so sorting by it
would leak future information into a player's or team's "prior" history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import features as F

# Colunas já tratadas de outra forma pelo segundo experimento, então não
# entram como "estatística a ter histórico": `position` é feature separada,
# `side` é contexto (não histórico) e `minutes_played` já vira
# `player_minutes_avg_prior_season` explicitamente.
NON_STAT_COLUMNS: tuple[str, ...] = (F.POSITION_COLUMN, "side", "minutes_played")


def raw_stat_columns(df: pd.DataFrame) -> list[str]:
    """Todas as colunas de estatística bruta da partida (`F.feature_columns`),
    menos as de `NON_STAT_COLUMNS`.
    Reaproveitar essa lista em vez de curar um subconjunto à mão evita viés de
    escolha e deixa a importância por permutação decidir o que importa.

    Chame isto sobre o dataframe ORIGINAL (antes de `add_opponent_strength`),
    nunca sobre um já mesclado com colunas de adversário -- senão
    `opponent_goals_for_avg_prior` etc. seriam tratadas como "estatística" e
    ganhariam sua própria média histórica (nome duplicado, sem sentido). Como
    proteção extra, colunas de contexto/histórico já derivadas também são
    excluídas mesmo se `df` já vier mesclado."""
    candidates = [c for c in F.feature_columns(df, position=None) if c not in NON_STAT_COLUMNS]
    return [
        c for c in candidates
        if c != "opponent_team_id" and not c.startswith("opponent_") and not _is_player_form_column(c)
    ]


def _prior_expanding_mean(s: pd.Series) -> pd.Series:
    """Média histórica até (sem incluir) a linha atual. Requer que `s` já
    esteja ordenada cronologicamente dentro do grupo antes do `.transform()`.
    """
    return s.expanding().mean().shift(1)


def _prior_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Média das últimas `window` observações anteriores (exclui a atual)."""
    return s.rolling(window=window, min_periods=1).mean().shift(1)


def team_match_results(df: pd.DataFrame) -> pd.DataFrame:
    """1 linha por (match_id, team_id): gols marcados/sofridos daquele time
    naquela partida, do seu próprio ponto de vista, mais o `team_id` do
    adversário. Dedup por `match_id` primeiro -- essas colunas repetem por
    jogador no df de entrada."""
    matches = df.drop_duplicates(subset="match_id")[
        ["match_id", "kickoff_timestamp", "home_team_id", "away_team_id", "home_score", "away_score"]
    ]
    home = matches.rename(
        columns={
            "home_team_id": "team_id",
            "away_team_id": "opponent_team_id",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    away = matches.rename(
        columns={
            "away_team_id": "team_id",
            "home_team_id": "opponent_team_id",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )
    cols = ["match_id", "kickoff_timestamp", "team_id", "opponent_team_id", "goals_for", "goals_against"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def team_prior_form(team_results: pd.DataFrame) -> pd.DataFrame:
    """Para cada (team_id, match_id), a força do time ANTES daquela partida:
    média de gols marcados/sofridos nas partidas anteriores (por
    `kickoff_timestamp`, nunca por `round`) e quantas partidas anteriores ele
    tem na temporada."""
    out = team_results.sort_values(["team_id", "kickoff_timestamp", "match_id"]).copy()
    grouped = out.groupby("team_id")

    out["team_goals_for_avg_prior"] = grouped["goals_for"].transform(_prior_expanding_mean)
    out["team_goals_against_avg_prior"] = grouped["goals_against"].transform(_prior_expanding_mean)
    out["team_matches_played_prior"] = grouped.cumcount()

    return out[
        ["match_id", "team_id", "team_goals_for_avg_prior", "team_goals_against_avg_prior", "team_matches_played_prior"]
    ]


def add_opponent_strength(df: pd.DataFrame, team_prior: pd.DataFrame) -> pd.DataFrame:
    """A força "do adversário" de uma linha jogador×partida usa o `team_id` do
    time CONTRÁRIO ao da própria linha -- nunca o `team_id` da própria linha."""
    out = df.copy()
    out["opponent_team_id"] = np.where(out["side"] == 1, out["away_team_id"], out["home_team_id"])

    opponent = team_prior.rename(
        columns={
            "team_id": "opponent_team_id",
            "team_goals_for_avg_prior": "opponent_goals_for_avg_prior",
            "team_goals_against_avg_prior": "opponent_goals_against_avg_prior",
            "team_matches_played_prior": "opponent_matches_played_prior",
        }
    )
    return out.merge(opponent, on=["match_id", "opponent_team_id"], how="left", validate="many_to_one")


def add_player_history(
    df: pd.DataFrame,
    stats: list[str] | None = None,
    last_n: int = 5,
) -> pd.DataFrame:
    """Histórico/forma do próprio jogador, calculado só com as partidas dele
    ANTERIORES à atual (nunca a atual). Agnóstico à posição atual: o jogador
    pode ter alternado entre posições de linha durante a temporada.

    `stats=None` (padrão) usa `raw_stat_columns(df)` -- todas as estatísticas
    brutas do primeiro experimento, não um subconjunto curado à mão."""
    if stats is None:
        stats = raw_stat_columns(df)
    out = df.sort_values(["player_id", "kickoff_timestamp", "match_id"]).copy()
    grouped = out.groupby("player_id")

    out["player_matches_played_prior"] = grouped.cumcount()
    out["player_rating_avg_prior_season"] = grouped["rating"].transform(_prior_expanding_mean)
    out["player_rating_avg_prior_last5"] = grouped["rating"].transform(
        lambda s: _prior_rolling_mean(s, window=last_n)
    )
    out["player_minutes_avg_prior_season"] = grouped["minutes_played"].transform(_prior_expanding_mean)
    for stat in stats:
        out[f"player_{stat}_avg_prior_season"] = grouped[stat].transform(_prior_expanding_mean)

    return out.sort_index()


def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orquestra o cálculo de força do adversário + histórico do jogador.

    Deve ser chamado sobre o dataset inteiro (treino + teste), antes do
    split: cada linha só usa partidas estritamente anteriores dela mesma por
    `kickoff_timestamp`, então uma linha de teste pode (e deve) usar histórico
    de partidas de treino -- exatamente como aconteceria em produção.
    """
    stats = raw_stat_columns(df)  # calculado sobre o df ORIGINAL, antes do merge com o adversário
    team_results = team_match_results(df)
    team_prior = team_prior_form(team_results)
    with_opponent = add_opponent_strength(df, team_prior)
    return add_player_history(with_opponent, stats=stats)


CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "side",
    "opponent_goals_for_avg_prior",
    "opponent_goals_against_avg_prior",
    "opponent_matches_played_prior",
)


def _is_player_form_column(name: str) -> bool:
    return name == "player_matches_played_prior" or (
        name.startswith("player_") and (name.endswith("_avg_prior_season") or name.endswith("_avg_prior_last5"))
    )


def pre_match_feature_columns(df: pd.DataFrame, position: int | None = None) -> list[str]:
    """Colunas de X do segundo experimento: allowlist por padrão de nome (ao
    contrário de `F.feature_columns`, que é por exclusão) -- nenhuma das
    colunas de estatística bruta da partida (`goals`, `touches` etc.) pode
    entrar aqui, só as versões `player_*_avg_prior_*`/`opponent_*`/`side`.
    Detecta as colunas dinamicamente (em vez de uma lista fixa) porque
    `add_player_history` gera uma coluna `player_<stat>_avg_prior_season` para
    cada estatística de `raw_stat_columns(df)`, que pode variar conforme o
    dataset. Mesma assinatura de `F.feature_columns`, para reaproveitar
    `fit_generic`/`fit_specialists`.
    """
    candidates = [c for c in df.columns if c in CONTEXT_FEATURE_COLUMNS or _is_player_form_column(c)]
    if position is None:
        return candidates + [F.POSITION_COLUMN]
    constant = {c for c in candidates if df[c].nunique(dropna=False) <= 1}
    return [c for c in candidates if c not in constant]


def walk_forward_splits(
    df: pd.DataFrame,
    n_test_matches: int,
    step_matches: int = 10,
    match_id_col: str = "match_id",
    timestamp_col: str = "kickoff_timestamp",
):
    """Gera (train_df, test_batch_df) andando pra frente no tempo: treina com
    tudo estritamente ANTERIOR ao lote (por `kickoff_timestamp`, nunca por
    `round`), prevê um lote de `step_matches` partidas, e no próximo passo
    esse lote já faz parte do treino -- exatamente a ideia de "prevê a
    rodada, incorpora ao treino, prevê a próxima" (docs/modelagem.md, seção
    18), só que os lotes são cronológicos, não por número de rodada."""
    matches = (
        df[[match_id_col, timestamp_col]]
        .drop_duplicates(subset=match_id_col)
        .sort_values([timestamp_col, match_id_col])
        .reset_index(drop=True)
    )
    n_total = len(matches)
    first_test_idx = n_total - n_test_matches
    for batch_start in range(first_test_idx, n_total, step_matches):
        batch_end = min(batch_start + step_matches, n_total)
        train_ids = set(matches[match_id_col].iloc[:batch_start])
        test_ids = set(matches[match_id_col].iloc[batch_start:batch_end])
        yield df[df[match_id_col].isin(train_ids)].copy(), df[df[match_id_col].isin(test_ids)].copy()


def temporal_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 30,
    match_id_col: str = "match_id",
    timestamp_col: str = "kickoff_timestamp",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partições da validação cruzada da busca em grade, em ordem cronológica:
    `TimeSeriesSplit` aplicado sobre a lista de partidas (por
    `kickoff_timestamp`, nunca por `round`), e não sobre as linhas, então cada
    partida fica inteira de um lado só. A validação é sempre posterior ao
    treino da partição -- com `GroupKFold`, partidas posteriores entravam no
    treino, e o histórico delas (`*_avg_prior_*`) carregava a nota das
    partidas de validação. Devolve índices posicionais de `df`, no formato
    que `GridSearchCV(cv=...)` aceita."""
    matches = (
        df[[match_id_col, timestamp_col]]
        .drop_duplicates(subset=match_id_col)
        .sort_values([timestamp_col, match_id_col])
        .reset_index(drop=True)
    )
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits = []
    for train_idx, val_idx in splitter.split(matches):
        train_ids = matches[match_id_col].iloc[train_idx]
        val_ids = matches[match_id_col].iloc[val_idx]
        splits.append((
            np.flatnonzero(df[match_id_col].isin(train_ids).to_numpy()),
            np.flatnonzero(df[match_id_col].isin(val_ids).to_numpy()),
        ))
    return splits
