"""Tests for pre-match features (opponent strength + player history): the
core risk is look-ahead leakage, so most tests pit `round` against
`kickoff_timestamp` explicitly (docs/modelagem.md, seção 13)."""
import numpy as np
import pandas as pd
import pytest

from src.modeling import features as F
from src.modeling import history as H
from src.modeling.training_history import run_history_training


def _df(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


# --- team_match_results ---------------------------------------------------


def test_team_match_results_one_row_per_match_and_team():
    df = _df(
        match_id=[1, 1, 2, 2],
        kickoff_timestamp=[100, 100, 200, 200],
        home_team_id=[10, 10, 20, 20],
        away_team_id=[11, 11, 21, 21],
        home_score=[2, 2, 0, 0],
        away_score=[1, 1, 3, 3],
    )
    out = H.team_match_results(df)
    assert len(out) == 4  # 2 matches x 2 teams

    m1_home = out[(out.match_id == 1) & (out.team_id == 10)].iloc[0]
    assert m1_home.goals_for == 2 and m1_home.goals_against == 1
    assert m1_home.opponent_team_id == 11

    m1_away = out[(out.match_id == 1) & (out.team_id == 11)].iloc[0]
    assert m1_away.goals_for == 1 and m1_away.goals_against == 2
    assert m1_away.opponent_team_id == 10


# --- team_prior_form --------------------------------------------------------


def test_team_prior_form_cold_start_and_expanding_mean():
    team_results = _df(
        match_id=[1, 2, 3],
        team_id=[10, 10, 10],
        opponent_team_id=[20, 30, 40],
        kickoff_timestamp=[100, 200, 300],
        goals_for=[2, 0, 1],
        goals_against=[0, 0, 0],
    )
    out = H.team_prior_form(team_results).set_index("match_id")
    assert pd.isna(out.loc[1, "team_goals_for_avg_prior"])
    assert out.loc[1, "team_matches_played_prior"] == 0
    assert out.loc[2, "team_goals_for_avg_prior"] == pytest.approx(2.0)
    assert out.loc[2, "team_matches_played_prior"] == 1
    assert out.loc[3, "team_goals_for_avg_prior"] == pytest.approx(1.0)  # mean(2, 0)
    assert out.loc[3, "team_matches_played_prior"] == 2


def test_team_prior_form_orders_by_kickoff_not_round():
    # Round contradicts kickoff order: match A (round 2) actually happened
    # before match B (round 1) -- a postponed-round scenario.
    team_results = _df(
        match_id=["A", "B"],
        team_id=[10, 10],
        opponent_team_id=[20, 30],
        kickoff_timestamp=[100, 200],
        goals_for=[5, 1],
        goals_against=[0, 0],
    )
    out = H.team_prior_form(team_results).set_index("match_id")
    # B happened after A chronologically, so B's prior average must be A's goals_for.
    assert out.loc["B", "team_goals_for_avg_prior"] == pytest.approx(5.0)
    assert out.loc["B", "team_matches_played_prior"] == 1
    assert pd.isna(out.loc["A", "team_goals_for_avg_prior"])


# --- add_opponent_strength --------------------------------------------------


def test_add_opponent_strength_uses_correct_side():
    df = _df(
        match_id=[1, 1],
        side=[1, 0],  # row 0 = home team, row 1 = away team
        home_team_id=[10, 10],
        away_team_id=[20, 20],
    )
    team_prior = _df(
        match_id=[1, 1],
        team_id=[10, 20],
        team_goals_for_avg_prior=[2.0, 5.0],
        team_goals_against_avg_prior=[1.0, 3.0],
        team_matches_played_prior=[4, 6],
    )
    out = H.add_opponent_strength(df, team_prior)

    home_row = out[out["side"] == 1].iloc[0]
    assert home_row["opponent_team_id"] == 20
    assert home_row["opponent_goals_for_avg_prior"] == pytest.approx(5.0)

    away_row = out[out["side"] == 0].iloc[0]
    assert away_row["opponent_team_id"] == 10
    assert away_row["opponent_goals_for_avg_prior"] == pytest.approx(2.0)


# --- add_player_history ------------------------------------------------------


_TEST_STATS = ("goals", "assists", "key_passes", "duels_won", "tackles_total", "saves")


def _player_df(kickoff, rating, minutes=None, round_=None, match_id=None, player_id=1, **extra):
    n = len(kickoff)
    data = {
        "player_id": [player_id] * n,
        "match_id": match_id if match_id is not None else list(range(1, n + 1)),
        "round": round_ if round_ is not None else list(range(1, n + 1)),
        "kickoff_timestamp": kickoff,
        "rating": rating,
        "minutes_played": minutes if minutes is not None else [90] * n,
    }
    for stat in _TEST_STATS:
        data.setdefault(stat, extra.get(stat, [0] * n))
    data.update(extra)
    return _df(**data)


def test_add_player_history_no_lookahead():
    df = _player_df(kickoff=[100, 200, 300], rating=[6.0, 7.0, 8.0])
    out = H.add_player_history(df).set_index("match_id")
    assert out.loc[3, "player_rating_avg_prior_season"] == pytest.approx(6.5)  # mean(6, 7), not 8
    assert out["player_matches_played_prior"].tolist() == [0, 1, 2]


def test_add_player_history_last5_differs_from_season():
    ratings = [5.0, 6.0, 7.0, 8.0, 9.0, 4.0, 10.0]
    df = _player_df(kickoff=list(range(100, 800, 100)), rating=ratings)
    out = H.add_player_history(df).set_index("match_id")
    season_avg = out.loc[7, "player_rating_avg_prior_season"]  # mean of first 6
    last5_avg = out.loc[7, "player_rating_avg_prior_last5"]  # mean of matches 2-6
    assert season_avg == pytest.approx(sum(ratings[:6]) / 6)
    assert last5_avg == pytest.approx(sum(ratings[1:6]) / 5)
    assert season_avg != pytest.approx(last5_avg)


def test_add_player_history_orders_by_kickoff_not_round():
    # round contradicts kickoff: match "B" has round=1 but happened after "A" (round=2).
    df = _player_df(
        kickoff=[100, 200], rating=[6.0, 9.0], round_=[2, 1], match_id=["A", "B"]
    )
    out = H.add_player_history(df).set_index("match_id")
    assert out.loc["B", "player_rating_avg_prior_season"] == pytest.approx(6.0)
    assert pd.isna(out.loc["A", "player_rating_avg_prior_season"])


def test_add_player_history_debutant_has_nan_history_no_error():
    df = _player_df(kickoff=[100], rating=[7.0])
    out = H.add_player_history(df)
    prior_cols = [c for c in out.columns if c.endswith("_avg_prior_season") or c.endswith("_avg_prior_last5")]
    assert out[prior_cols].isna().all(axis=None)
    assert out["player_matches_played_prior"].iloc[0] == 0


# --- build_history_features --------------------------------------------------


def _full_synthetic_df():
    rng = np.random.default_rng(3)
    rows = []
    match_id = 1
    for round_number in range(1, 6):
        for home, away in [(10, 20), (30, 40)]:
            kickoff = round_number * 1000
            home_score, away_score = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            for team_id, opponent_id, side, score_for, score_against in (
                (home, away, 1, home_score, away_score),
                (away, home, 0, away_score, home_score),
            ):
                rows.append(
                    {
                        "match_id": match_id,
                        "round": round_number,
                        "kickoff_timestamp": kickoff,
                        "home_team_id": home,
                        "away_team_id": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "team_id": team_id,
                        "side": side,
                        "player_id": team_id * 100 + 1,
                        "position": 2,
                        "rating": 6.0 + rng.normal(scale=0.3),
                        "minutes_played": 90,
                        "goals": 0,
                        "assists": 0,
                        "key_passes": 0,
                        "duels_won": 0,
                        "tackles_total": 0,
                        "saves": -1,
                    }
                )
            match_id += 1
    return pd.DataFrame(rows)


def test_build_history_features_preserves_original_columns_and_row_order():
    df = _full_synthetic_df()
    original_columns = set(df.columns)
    out = H.build_history_features(df)
    assert original_columns <= set(out.columns)
    assert len(out) == len(df)
    assert out.index.tolist() == df.index.tolist()


# --- pre_match_feature_columns ------------------------------------------------


RAW_STAT_COLUMNS = ("goals", "touches", "duels_won", "key_passes", "tackles_total", "saves", "assists")


def test_pre_match_feature_columns_excludes_raw_match_stats():
    df = H.build_history_features(_full_synthetic_df())
    generic_cols = H.pre_match_feature_columns(df, position=None)
    assert not set(RAW_STAT_COLUMNS) & set(generic_cols)
    assert "position" in generic_cols

    specialist_cols = H.pre_match_feature_columns(df[df["position"] == 2], position=2)
    assert "position" not in specialist_cols


def test_pre_match_feature_columns_drops_constant_by_position():
    df = _df(
        position=[1, 1, 2, 2],
        side=[1, 0, 1, 0],
        opponent_goals_for_avg_prior=[1.0, 2.0, 1.0, 2.0],
        opponent_goals_against_avg_prior=[1.0, 2.0, 1.0, 2.0],
        opponent_matches_played_prior=[3, 3, 3, 3],
        player_matches_played_prior=[1, 2, 1, 2],
        player_rating_avg_prior_season=[6.0, 6.5, 6.0, 6.5],
        player_rating_avg_prior_last5=[6.0, 6.5, 6.0, 6.5],
        player_minutes_avg_prior_season=[90, 90, 90, 90],
        player_goals_avg_prior_season=[0, 0, 0, 0],
        player_assists_avg_prior_season=[0, 0, 0, 0],
        player_key_passes_avg_prior_season=[0, 0, 0, 0],
        player_duels_won_avg_prior_season=[1, 1, 1, 1],
        player_tackles_total_avg_prior_season=[1, 1, 1, 1],
        player_saves_avg_prior_season=[-1, -1, 3.0, 4.0],  # constant only for position 1 (D)
    )
    defender_only = df[df["position"] == 1]
    cols = H.pre_match_feature_columns(defender_only, position=1)
    assert "player_saves_avg_prior_season" not in cols

    midfielder_only = df[df["position"] == 2]
    cols_mid = H.pre_match_feature_columns(midfielder_only, position=2)
    assert "player_saves_avg_prior_season" in cols_mid


# --- modeling.training_history.run_history_training (integração) -------------


def _integration_dataset() -> pd.DataFrame:
    """20 rodadas, 2 partidas cada (times 10/20/30/40 alternando), 4 jogadores
    por partida (um por posição) -- schema completo exigido por
    build_history_features + build_predictions."""
    rng = np.random.default_rng(11)
    position_codes = {0: "G", 1: "D", 2: "M", 3: "F"}
    rows = []
    match_id = 1
    for round_number in range(1, 21):
        for home, away in [(10, 20), (30, 40)]:
            kickoff = round_number * 1000
            home_score, away_score = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            for team_id, side in ((home, 1), (away, 0)):
                for code, label in position_codes.items():
                    rows.append(
                        {
                            "match_id": match_id,
                            "round": round_number,
                            "match_date": f"2025-01-{round_number:02d}",
                            "kickoff_timestamp": kickoff,
                            "home_team_id": home,
                            "away_team_id": away,
                            "home_score": home_score,
                            "away_score": away_score,
                            "team_id": team_id,
                            "team_name": f"team{team_id}",
                            "side": side,
                            "player_id": team_id * 10 + code,
                            "player_name": f"{label}{team_id}",
                            "position": code,
                            "minutes_played": 90,
                            "rating": 6.8 + 0.1 * code + rng.normal(scale=0.2),
                            "goals": int(rng.integers(0, 3)),
                            "assists": int(rng.integers(0, 2)),
                            "key_passes": int(rng.integers(0, 4)),
                            "duels_won": int(rng.integers(0, 8)),
                            "tackles_total": int(rng.integers(0, 5)),
                            "saves": int(rng.integers(0, 6)) if code == 0 else -1,
                        }
                    )
            match_id += 1
    return pd.DataFrame(rows)


def test_run_history_training_writes_five_csvs(tmp_path):
    source = tmp_path / "clean.csv"
    results_dir = tmp_path / "results_history"
    _integration_dataset().to_csv(source, index=False, encoding="utf-8-sig")

    counts = run_history_training(
        source, results_dir,
        param_grid={"max_features": ["sqrt"], "min_samples_leaf": [1]},
        n_splits=2,
        cv_test_size=4,
        n_test_matches=8,
        step_matches=4,
    )
    assert set(counts) == {
        "predictions.csv", "metrics.csv", "comparison.csv", "importances.csv", "best_params.csv",
    }

    predictions = pd.read_csv(results_dir / "predictions.csv", encoding="utf-8-sig")
    assert {"pred_baseline", "pred_generic", "pred_specialist"} <= set(predictions.columns)
    assert len(predictions) == 8 * 2 * 4  # 8 partidas de teste x 2 times x 4 posições (2 lotes de 4)
    assert predictions["match_id"].is_unique is False  # múltiplas linhas por partida (times/posições)
    assert len(set(predictions["match_id"])) == 8  # mas as 8 partidas de teste, sem repetir

    importances_df = pd.read_csv(results_dir / "importances.csv", encoding="utf-8-sig")
    feature_names = set(importances_df["feature"])
    # importância sobre todos os lotes (2 lotes de 4 partidas), não só o último
    assert (importances_df["n_batches"] == 2).all()
    generic_n = importances_df.loc[importances_df["model"] == "generic", "n"].unique()
    assert list(generic_n) == [len(predictions)]
    assert not set(RAW_STAT_COLUMNS) & feature_names
    assert all(
        f in H.CONTEXT_FEATURE_COLUMNS or f == F.POSITION_COLUMN or H._is_player_form_column(f)
        for f in feature_names
    )

    best_params_df = pd.read_csv(results_dir / "best_params.csv", encoding="utf-8-sig")
    assert len(best_params_df) == 5  # generic + specialist_G/D/M/F
    # a busca só roda no 1º lote: o cv_mae dele precisa sair preenchido, mesmo
    # com o último lote só reajustando a floresta (refit_with_params, sem CV)
    assert best_params_df[["cv_mae_first", "oob_mae_first", "oob_mae_last"]].notna().all().all()


def test_run_history_training_retune_each_step_also_works(tmp_path):
    source = tmp_path / "clean.csv"
    results_dir = tmp_path / "results_history_retune"
    _integration_dataset().to_csv(source, index=False, encoding="utf-8-sig")

    counts = run_history_training(
        source, results_dir,
        param_grid={"max_features": ["sqrt"], "min_samples_leaf": [1]},
        n_splits=2,
        cv_test_size=4,
        n_test_matches=8,
        step_matches=4,
        retune_each_step=True,
    )
    assert set(counts) == {
        "predictions.csv", "metrics.csv", "comparison.csv", "importances.csv", "best_params.csv",
    }


# --- walk_forward_splits -------------------------------------------------------


def _matches_df(n, step_id=100):
    return _df(
        match_id=[step_id + i for i in range(n)],
        round=list(range(1, n + 1)),
        kickoff_timestamp=[1000 + i for i in range(n)],
    )


def test_walk_forward_splits_grows_training_set_each_step():
    df = _matches_df(10)  # matches 0-5 are pre-test, 6-9 are the 4 test matches
    steps = list(H.walk_forward_splits(df, n_test_matches=4, step_matches=2))
    assert len(steps) == 2  # 4 test matches / step_matches=2

    train0, test0 = steps[0]
    assert len(train0) == 6  # matches 0-5
    assert sorted(test0["match_id"]) == [106, 107]

    train1, test1 = steps[1]
    assert len(train1) == 8  # matches 0-7 (test0's batch already incorporated)
    assert sorted(test1["match_id"]) == [108, 109]


def test_walk_forward_splits_covers_every_test_match_exactly_once():
    df = _matches_df(23)
    steps = list(H.walk_forward_splits(df, n_test_matches=13, step_matches=5))
    all_test_ids = [mid for _, test_df in steps for mid in test_df["match_id"]]
    assert sorted(all_test_ids) == sorted(set(all_test_ids))  # no duplicates
    assert len(all_test_ids) == 13


def test_walk_forward_splits_orders_by_kickoff_not_round():
    # round contradicts kickoff: match "B" (round=1) actually happened after "A" (round=2).
    df = _df(
        match_id=["A", "B", "C"],
        round=[2, 1, 3],
        kickoff_timestamp=[100, 200, 300],
    )
    steps = list(H.walk_forward_splits(df, n_test_matches=2, step_matches=1))
    assert steps[0][0]["match_id"].tolist() == ["A"]  # train = chronologically first match
    assert steps[0][1]["match_id"].tolist() == ["B"]  # first test batch = 2nd by kickoff
    assert sorted(steps[1][0]["match_id"]) == ["A", "B"]  # train grew to include B
    assert steps[1][1]["match_id"].tolist() == ["C"]


# --- temporal_cv_splits ----------------------------------------------------------


def test_temporal_cv_splits_validation_always_after_training():
    # 3 linhas por partida, em ordem cronológica inversa no dataframe
    n = 12
    df = _df(
        match_id=[i for i in range(n) for _ in range(3)][::-1],
        round=[1] * (3 * n),
        kickoff_timestamp=[1000 + i for i in range(n) for _ in range(3)][::-1],
    )
    splits = H.temporal_cv_splits(df, n_splits=3, test_size=2)
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        train_kick = df["kickoff_timestamp"].iloc[train_idx]
        val_kick = df["kickoff_timestamp"].iloc[val_idx]
        assert train_kick.max() < val_kick.min()  # validação sempre depois do treino
        train_ids = set(df["match_id"].iloc[train_idx])
        val_ids = set(df["match_id"].iloc[val_idx])
        assert not train_ids & val_ids  # partida inteira de um lado só
        assert len(val_ids) == 2 and len(val_idx) == 6  # 2 partidas x 3 linhas
    # o treino cresce e a última validação são as 2 últimas partidas
    assert [len(t) for t, _ in splits] == [18, 24, 30]
    assert set(df["match_id"].iloc[splits[-1][1]]) == {10, 11}
