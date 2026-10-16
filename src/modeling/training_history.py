"""Training step for the pre-match experiment: cleaned dataset -> baseline /
generic / specialists, using only context (opponent, mandante/visitante) and
player history, validated walk-forward, in data/results_history/ (docs/modelagem.md)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .. import config
from . import evaluation as V
from . import features as F
from . import history as H
from .training import aggregate_importances, build_best_params, build_importances, build_predictions, fit_generic, fit_specialists

log = logging.getLogger("sofascore_scraper")


def _write_outputs(results_dir: Path, predictions, metrics_df, comparison_df, importances_df, best_params_df) -> dict[str, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions.csv": predictions,
        "metrics.csv": metrics_df,
        "comparison.csv": comparison_df,
        "importances.csv": importances_df,
        "best_params.csv": best_params_df,
    }
    for name, table in outputs.items():
        table.to_csv(results_dir / name, index=False, encoding="utf-8-sig")
    return {name: len(table) for name, table in outputs.items()}


def run_history_training(
    source: Path,
    results_dir: Path,
    param_grid: dict | None = None,
    n_splits: int = config.CV_N_SPLITS,
    cv_test_size: int = config.CV_TEST_N_MATCHES,
    n_test_matches: int = config.HISTORY_TEST_N_MATCHES,
    step_matches: int = config.WALK_FORWARD_STEP_MATCHES,
    retune_each_step: bool = False,
) -> dict[str, int]:
    """Walk-forward: treina, prevê um lote de `step_matches` partidas (por
    `kickoff_timestamp`), incorpora esse lote ao treino e prevê o próximo, até
    cobrir as últimas `n_test_matches` partidas (docs/modelagem.md, seção 17).

    Por padrão (`retune_each_step=False`), os hiperparâmetros são escolhidos
    via `GridSearchCV` só no primeiro lote; nos lotes seguintes o modelo só é
    reajustado com esses mesmos hiperparâmetros (`E.refit_with_params`), bem
    mais barato que repetir a busca em grade a cada lote.
    """
    raw = pd.read_csv(source, encoding="utf-8-sig")
    df = H.build_history_features(raw)

    tune_kwargs: dict = {"n_splits": n_splits, "cv_test_size": cv_test_size}
    if param_grid is not None:
        tune_kwargs["param_grid"] = param_grid

    predictions_parts = []
    generic_fixed: dict | None = None
    specialist_fixed: dict[int, dict] | None = None
    generic_cols = generic_result = specialists = test_df = None
    first_generic = first_specialists = None
    importances_parts = []

    for train_df, test_df in H.walk_forward_splits(df, n_test_matches, step_matches):
        baseline_pred = V.baseline_predict(train_df, test_df)
        generic_cols, generic_result, generic_pred = fit_generic(
            train_df, test_df, feature_columns_fn=H.pre_match_feature_columns,
            fixed_params=generic_fixed,
            **({} if generic_fixed is not None else tune_kwargs),
        )
        specialists, specialist_pred = fit_specialists(
            train_df, test_df, feature_columns_fn=H.pre_match_feature_columns,
            fixed_params_by_position=specialist_fixed,
            **({} if specialist_fixed is not None else tune_kwargs),
        )
        predictions_parts.append(build_predictions(test_df, baseline_pred, generic_pred, specialist_pred))
        # importância de cada lote com os modelos daquele lote, sobre o próprio lote
        importances_parts.append(build_importances(generic_cols, generic_result, specialists, test_df))

        if first_generic is None:  # a busca (e o cv_mae) só existe no 1º lote
            first_generic, first_specialists = generic_result, specialists

        if not retune_each_step and generic_fixed is None:
            generic_fixed = generic_result.best_params
            specialist_fixed = {code: result.best_params for code, (_, result) in specialists.items()}

    predictions = pd.concat(predictions_parts, ignore_index=True)
    metrics_df = V.metrics_table(
        predictions, F.TARGET,
        {"baseline": "pred_baseline", "generic": "pred_generic", "specialist": "pred_specialist"},
    )
    comparison_df = V.compare_models(
        predictions, F.TARGET, "pred_generic", "pred_specialist",
        "generic", "specialist", "generic_vs_specialist",
    )
    # importâncias: média dos 8 lotes, ponderada pelas atuações de teste de cada um
    importances_df = aggregate_importances(importances_parts)
    best_params_df = build_best_params(first_generic, first_specialists, generic_result, specialists)

    return _write_outputs(results_dir, predictions, metrics_df, comparison_df, importances_df, best_params_df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare pre-match rating prediction models (context + player history, walk-forward)."
    )
    parser.add_argument("--source", type=Path, default=config.PROCESSED_DIR / config.CLEAN_CSV_NAME)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--step-matches", type=int, default=config.WALK_FORWARD_STEP_MATCHES)
    parser.add_argument("--retune-each-step", action="store_true", help="re-tuna hiperparâmetros a cada lote (mais lento)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results_dir = args.results_dir or config.RESULTS_HISTORY_DIR
    counts = run_history_training(
        args.source, results_dir,
        step_matches=args.step_matches, retune_each_step=args.retune_each_step,
    )
    log.info("done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
