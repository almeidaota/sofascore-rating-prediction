"""Metrics, bootstrap comparison and feature importance (docs/modelagem.md, seção 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .. import config
from . import features as F


def metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n": len(y_true),
    }


def baseline_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = F.TARGET,
    position_col: str = F.POSITION_COLUMN,
) -> pd.Series:
    """Baseline: prevê a nota média da posição, calculada no treino."""
    means = train_df.groupby(position_col)[target].mean()
    return test_df[position_col].map(means)


def athlete_baseline_predict(
    test_df: pd.DataFrame,
    history_col: str,
    position_baseline: pd.Series,
) -> pd.Series:
    """Baseline do atleta: prevê a média das notas anteriores do próprio jogador
    (`history_col`); sem histórico, usa a baseline da posição."""
    return test_df[history_col].fillna(position_baseline)


def bootstrap_mae_diff(
    df: pd.DataFrame,
    y_col: str,
    pred_a_col: str,
    pred_b_col: str,
    group_col: str = "match_id",
    n_bootstrap: int = config.N_BOOTSTRAP,
    random_state: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Bootstrap pareado por `group_col`: array de deltas
    MAE(pred_a) - MAE(pred_b), reamostrando grupos (partidas) com reposição.
    """
    rng = np.random.default_rng(random_state)
    groups = df[group_col].to_numpy()
    unique_groups = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    y = df[y_col].to_numpy()
    pred_a = df[pred_a_col].to_numpy()
    pred_b = df[pred_b_col].to_numpy()

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in sampled])
        deltas[i] = np.abs(y[rows] - pred_a[rows]).mean() - np.abs(y[rows] - pred_b[rows]).mean()
    return deltas


def decide_winner(ci_lower: float, ci_upper: float, reference_label: str, candidate_label: str) -> str:
    """Regra de decisão da seção 5: delta = MAE(reference) - MAE(candidate).
    IC 95% inteiro acima de 0 -> candidate é melhor; inteiro abaixo de 0 -> reference
    é melhor; contém 0 -> empate técnico, prefere reference (o termo mais simples).
    """
    if ci_lower > 0:
        return candidate_label
    if ci_upper < 0:
        return reference_label
    return reference_label


def importances(
    model,
    X_test: pd.DataFrame,
    y_test,
    n_repeats: int = 10,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": X_test.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
            "native_importance": model.feature_importances_,
        }
    )


def _position_slices(predictions: pd.DataFrame, position_col: str):
    """Gera (label, subset) para cada posição presente + um 'geral' com tudo."""
    yield "geral", predictions
    for code in sorted(predictions[position_col].dropna().unique()):
        label = F.POSITION_LABELS.get(code, str(code))
        yield label, predictions[predictions[position_col] == code]


def metrics_table(
    predictions: pd.DataFrame,
    y_col: str,
    pred_cols: dict[str, str],
    position_col: str = F.POSITION_COLUMN,
    baseline_model: str = "baseline",
    athlete_baseline_model: str | None = None,
) -> pd.DataFrame:
    """1 linha por (modelo, posição incl. 'geral'): mae, rmse, r2, ganho sobre a
    baseline (mesma posição), n. Com `athlete_baseline_model`, também o ganho
    sobre a baseline do atleta (`gain_over_athlete`)."""
    rows = []
    for position_label, subset in _position_slices(predictions, position_col):
        baseline_mae = mean_absolute_error(subset[y_col], subset[pred_cols[baseline_model]])
        athlete_mae = (
            mean_absolute_error(subset[y_col], subset[pred_cols[athlete_baseline_model]])
            if athlete_baseline_model else None
        )
        for model_name, pred_col in pred_cols.items():
            row = {"model": model_name, "position": position_label}
            row.update(metrics(subset[y_col], subset[pred_col]))
            row["gain_over_baseline"] = 1 - row["mae"] / baseline_mae if baseline_mae else float("nan")
            if athlete_baseline_model:
                row["gain_over_athlete"] = 1 - row["mae"] / athlete_mae if athlete_mae else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def compare_models(
    predictions: pd.DataFrame,
    y_col: str,
    reference_col: str,
    candidate_col: str,
    reference_label: str,
    candidate_label: str,
    comparison_name: str,
    position_col: str = F.POSITION_COLUMN,
    group_col: str = "match_id",
    n_bootstrap: int = config.N_BOOTSTRAP,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """1 linha por posição (incl. 'geral'): delta_mae (ponto, calculado nos dados
    reais), ci_lower, ci_upper, conclusion. Reamostra só os match_id daquela
    posição -- cada posição tem seu próprio bootstrap pareado por partida."""
    rows = []
    for position_label, subset in _position_slices(predictions, position_col):
        mae_reference = mean_absolute_error(subset[y_col], subset[reference_col])
        mae_candidate = mean_absolute_error(subset[y_col], subset[candidate_col])
        delta_mae = mae_reference - mae_candidate
        deltas = bootstrap_mae_diff(
            subset, y_col, reference_col, candidate_col,
            group_col=group_col, n_bootstrap=n_bootstrap, random_state=random_state,
        )
        ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5])
        conclusion = decide_winner(ci_lower, ci_upper, reference_label, candidate_label)
        rows.append(
            {
                "comparison": comparison_name,
                "position": position_label,
                "delta_mae": delta_mae,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)
