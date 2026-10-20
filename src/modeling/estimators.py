"""Random Forest construction and hyperparameter tuning."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from .. import config

PARAM_GRID: dict[str, list] = {
    "max_features": ["sqrt", 0.33, 1.0],
    "min_samples_leaf": [1, 5, 20, 50, 100],
}


def make_forest(
    random_state: int = config.RANDOM_STATE,
    n_estimators: int = 500,
    **overrides,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        oob_score=True,
        n_jobs=-1,
        **overrides,
    )


@dataclass
class TuneResult:
    model: RandomForestRegressor
    best_params: dict
    cv_mae: float
    oob_mae: float


def tune(
    X,
    y,
    cv_splits,
    param_grid: dict = PARAM_GRID,
    random_state: int = config.RANDOM_STATE,
    n_estimators: int = 500,
) -> TuneResult:
    """Escolhe hiperparâmetros pela validação cruzada em `cv_splits` (as
    partições temporais de `history.temporal_cv_splits`) e reajusta o melhor
    modelo em todo X/y, com oob_score habilitado."""
    search = GridSearchCV(
        estimator=make_forest(random_state=random_state, n_estimators=n_estimators),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    oob_mae = mean_absolute_error(y, best_model.oob_prediction_)
    return TuneResult(
        model=best_model,
        best_params=search.best_params_,
        cv_mae=-search.best_score_,
        oob_mae=oob_mae,
    )


def refit_with_params(
    X,
    y,
    params: dict,
    random_state: int = config.RANDOM_STATE,
    n_estimators: int = 500,
) -> TuneResult:
    """Reajusta um Random Forest com hiperparâmetros JÁ escolhidos, sem rodar
    `GridSearchCV` de novo -- usado no walk-forward
    pra não re-tunar a cada rodada nova incorporada ao treino. `cv_mae`
    fica `NaN` (não há validação cruzada aqui); `oob_mae` continua disponível
    via `oob_score=True`."""
    model = make_forest(random_state=random_state, n_estimators=n_estimators, **params)
    model.fit(X, y)
    oob_mae = mean_absolute_error(y, model.oob_prediction_)
    return TuneResult(model=model, best_params=params, cv_mae=float("nan"), oob_mae=oob_mae)
