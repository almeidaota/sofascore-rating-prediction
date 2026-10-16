"""Shared training helpers for the pre-match experiment (docs/modelagem.md, seção 12+)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import estimators as E
from . import evaluation as V
from . import features as F
from . import history as H
from .. import config

POSITION_CODES = tuple(sorted(F.POSITION_LABELS))

FeatureColumnsFn = Callable[..., list[str]]

def fit_generic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns_fn: FeatureColumnsFn = F.feature_columns,
    fixed_params: dict | None = None,
    n_splits: int = config.CV_N_SPLITS,
    cv_test_size: int = config.CV_TEST_N_MATCHES,
    **tune_kwargs,
):
    """`fixed_params`, se dado, pula `E.tune` (sem GridSearchCV) e só reajusta
    a floresta com esses hiperparâmetros -- usado no walk-forward pra não
    re-tunar a cada rodada nova incorporada ao treino (ver `history.py`)."""
    cols = feature_columns_fn(train_df, position=None)
    if fixed_params is not None:
        result = E.refit_with_params(train_df[cols], train_df[F.TARGET], fixed_params)
    else:
        cv_splits = H.temporal_cv_splits(train_df, n_splits=n_splits, test_size=cv_test_size)
        result = E.tune(train_df[cols], train_df[F.TARGET], cv_splits, **tune_kwargs)
    preds = pd.Series(result.model.predict(test_df[cols]), index=test_df.index)
    return cols, result, preds

def fit_specialists(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns_fn: FeatureColumnsFn = F.feature_columns,
    fixed_params_by_position: dict[int, dict] | None = None,
    n_splits: int = config.CV_N_SPLITS,
    cv_test_size: int = config.CV_TEST_N_MATCHES,
    **tune_kwargs,
):
    """`fixed_params_by_position`, se dado, pula `E.tune` pra cada posição e
    só reajusta com os hiperparâmetros já escolhidos daquela posição."""
    fitted: dict[int, tuple[list[str], E.TuneResult]] = {}
    preds_parts = []
    for code in POSITION_CODES:
        tr = train_df[train_df[F.POSITION_COLUMN] == code]
        te = test_df[test_df[F.POSITION_COLUMN] == code]
        cols = feature_columns_fn(tr, position=code)
        if fixed_params_by_position is not None:
            result = E.refit_with_params(tr[cols], tr[F.TARGET], fixed_params_by_position[code])
        else:
            cv_splits = H.temporal_cv_splits(tr, n_splits=n_splits, test_size=cv_test_size)
            result = E.tune(tr[cols], tr[F.TARGET], cv_splits, **tune_kwargs)
        fitted[code] = (cols, result)
        preds_parts.append(pd.Series(result.model.predict(te[cols]), index=te.index))
    combined = pd.concat(preds_parts).reindex(test_df.index)
    assert combined.notna().all(), "specialist predictions must cover every test row exactly once"
    return fitted, combined

def build_predictions(test_df, baseline_pred, generic_pred, specialist_pred) -> pd.DataFrame:
    out = test_df[["match_id", "round", "match_date", "player_id", "player_name", "team_name",
                   "minutes_played", F.POSITION_COLUMN, F.TARGET]].copy()
    out[F.POSITION_COLUMN] = out[F.POSITION_COLUMN].map(F.POSITION_LABELS)
    out["pred_baseline"] = baseline_pred
    out["pred_generic"] = generic_pred
    out["pred_specialist"] = specialist_pred
    return out

def build_importances(generic_cols, generic_result, specialists, test_df) -> pd.DataFrame:
    """Importâncias dos modelos de UM lote, sobre o teste desse lote. A coluna
    `n` (atuações usadas) serve de peso em `aggregate_importances`."""
    tables = []

    generic_table = V.importances(generic_result.model, test_df[generic_cols], test_df[F.TARGET])
    generic_table.insert(0, "model", "generic")
    generic_table["n"] = len(test_df)
    tables.append(generic_table)

    for code, (cols, result) in specialists.items():
        te = test_df[test_df[F.POSITION_COLUMN] == code]
        if te.empty:
            continue
        table = V.importances(result.model, te[cols], te[F.TARGET])
        table.insert(0, "model", f"specialist_{F.POSITION_LABELS[code]}")
        table["n"] = len(te)
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def aggregate_importances(per_batch: list[pd.DataFrame]) -> pd.DataFrame:
    """Junta as importâncias dos lotes do walk-forward: média de cada
    variável ponderada pelo número de atuações de teste de cada lote.
    `permutation_importance_std` passa a ser o desvio ENTRE os lotes (também
    ponderado), ou seja, o quanto a importância oscilou ao longo do teste."""
    df = pd.concat(per_batch, ignore_index=True)

    def combine(g: pd.DataFrame) -> pd.Series:
        w = g["n"].to_numpy(dtype=float)
        mean = np.average(g["permutation_importance_mean"], weights=w)
        std = np.sqrt(np.average((g["permutation_importance_mean"] - mean) ** 2, weights=w))
        return pd.Series({
            "permutation_importance_mean": mean,
            "permutation_importance_std": std,
            "native_importance": np.average(g["native_importance"], weights=w),
            "n_batches": len(g),
            "n": int(w.sum()),
        })

    out = df.groupby(["model", "feature"], sort=False).apply(combine, include_groups=False).reset_index()
    out["n_batches"] = out["n_batches"].astype(int)
    out["n"] = out["n"].astype(int)
    return out


def build_best_params(first_generic, first_specialists, last_generic, last_specialists) -> pd.DataFrame:
    """Hiperparâmetros de cada modelo. No walk-forward, a busca (e portanto o
    `cv_mae`) só existe no 1º lote; os lotes seguintes só reajustam com os
    mesmos hiperparâmetros. Por isso `cv_mae_first`/`oob_mae_first` vêm do 1º
    lote (o mesmo modelo, comparáveis entre si) e `oob_mae_last`, do último."""
    def row(model, first, last):
        return {"model": model, **first.best_params, "n_estimators": 500,
                "cv_mae_first": first.cv_mae, "oob_mae_first": first.oob_mae, "oob_mae_last": last.oob_mae}

    rows = [row("generic", first_generic, last_generic)]
    for code, (_, first) in first_specialists.items():
        _, last = last_specialists[code]
        rows.append(row(f"specialist_{F.POSITION_LABELS[code]}", first, last))
    return pd.DataFrame(rows)
