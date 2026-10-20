"""Entry point: python terminal_shortcut/grid_edge.py

Diagnóstico da borda da grade: com a grade de `estimators.PARAM_GRID`, os
especialistas de goleiro e defensor e o modelo genérico escolheram
`min_samples_leaf=100`, o maior valor testado. Aqui a mesma busca (validação
cruzada temporal por partida, `history.temporal_cv_splits`, treino do 1º lote
do walk-forward) é refeita só para esses modelos, com `min_samples_leaf` até
400, para ver se o mínimo do erro de validação está dentro da grade.

Grava `data/diagnostics/grid_edge.csv` (MAE de validação por combinação, com a
linha de base na mesma validação) e imprime as tabelas.
"""
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from src import config  # noqa: E402
from src.modeling import estimators as E, features as F, history as H  # noqa: E402

SOURCE = config.DATA_DIR / "processed" / "brasileirao_2025_dataset_modelo.csv"
OUTPUT = config.DATA_DIR / "diagnostics" / "grid_edge.csv"
POSITIONS = {None: "generic", 0: "G", 1: "D"}  # None = modelo genérico (todas as posições)
GRID = {"max_features": ["sqrt", 0.33, 1.0], "min_samples_leaf": [20, 50, 100, 200, 400]}


def main() -> int:
    warnings.filterwarnings("ignore")
    df = H.build_history_features(pd.read_csv(SOURCE, encoding="utf-8-sig"))
    train, _ = next(H.walk_forward_splits(df, config.HISTORY_TEST_N_MATCHES, config.WALK_FORWARD_STEP_MATCHES))

    rows = []
    for code, label in POSITIONS.items():
        tr = train if code is None else train[train[F.POSITION_COLUMN] == code]
        cols = H.pre_match_feature_columns(tr, position=code)
        X, y = tr[cols], tr[F.TARGET]
        cv = H.temporal_cv_splits(tr, n_splits=config.CV_N_SPLITS, test_size=config.CV_TEST_N_MATCHES)

        # linha de base (média da posição no treino) na mesma validação, para comparar
        pos = tr[F.POSITION_COLUMN]
        baseline_mae = np.mean([
            np.abs(y.iloc[va] - pos.iloc[va].map(y.iloc[fit].groupby(pos.iloc[fit]).mean())).mean()
            for fit, va in cv
        ])

        search = GridSearchCV(E.make_forest(), GRID, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
        search.fit(X, y)
        res = pd.DataFrame(search.cv_results_)
        for _, r in res.iterrows():
            rows.append({
                "position": label,
                "n_train": len(tr),
                "max_features": r["param_max_features"],
                "min_samples_leaf": r["param_min_samples_leaf"],
                "cv_mae": -r["mean_test_score"],
                "cv_std": r["std_test_score"],
                "baseline_cv_mae": baseline_mae,
            })

        table = res.assign(cv_mae=-res["mean_test_score"]).pivot(
            index="param_min_samples_leaf", columns="param_max_features", values="cv_mae"
        )
        print(f"\n{label} (n = {len(tr)}), linha de base {baseline_mae:.4f}, melhor {search.best_params_}")
        print(table.round(4).to_string())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\ngravado em {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
