import pandas as pd
import yaml

from src.main import main


def test_main_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    dummy_data = {
        "tourney_date": [
            "20210101",
            "20210201",
            "20210301",
            "20210401",
            "20220101",
            "20220201",
            "20220301",
            "20220401",
            "20230101",
            "20230201",
            "20230301",
            "20230401",
        ],
        "surface": ["Hard", "Clay", "Grass", "Hard"] * 3,
        "winner_id": list(range(1, 13)),
        "winner_seed": list(range(1, 13)),
        "winner_rank": [10, 20, 30, 40] * 3,
        "winner_hand": ["R", "L", "R", "L"] * 3,
        "winner_ht": [180, 185, 190, 195] * 3,
        "winner_age": [25.0, 26.0, 27.0, 28.0] * 3,
        "loser_id": list(range(13, 25)),
        "loser_seed": list(range(13, 25)),
        "loser_rank": [70, 80, 90, 100] * 3,
        "loser_hand": ["R", "L", "R", "L"] * 3,
        "loser_ht": [175, 180, 185, 190] * 3,
        "loser_age": [22.0, 23.0, 24.0, 25.0] * 3,
        "tourney_id": [
            "2021-01",
            "2021-02",
            "2021-03",
            "2021-04",
            "2022-01",
            "2022-02",
            "2022-03",
            "2022-04",
            "2023-01",
            "2023-02",
            "2023-03",
            "2023-04",
        ],
        "match_num": [1] * 12,
        "tourney_level": ["G", "A", "M", "G"] * 3,
        "round": ["R32", "R16", "QF", "SF"] * 3,
    }

    df_pd = pd.DataFrame(dummy_data)
    df_21 = df_pd[df_pd["tourney_date"].str.startswith("2021")]
    df_22 = df_pd[df_pd["tourney_date"].str.startswith("2022")]
    df_23 = df_pd[df_pd["tourney_date"].str.startswith("2023")]

    df_21.to_csv(raw_dir / "atp_matches_2021.csv", index=False)
    df_22.to_csv(raw_dir / "atp_matches_2022.csv", index=False)
    df_23.to_csv(raw_dir / "atp_matches_2023.csv", index=False)

    dummy_config = {
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "dataset": {
            "base_url": "http://not-a-real-url/",
            "download_if_missing": False,
            "seasons_train": [2021],
            "seasons_val": [],
            "seasons_test": [2022],
            "seasons_infer": [2023],
        },
        "schema": {
            "required_columns": [
                "tourney_date",
                "surface",
                "winner_id",
                "winner_seed",
                "winner_rank",
                "winner_hand",
                "winner_ht",
                "winner_age",
                "loser_id",
                "loser_seed",
                "loser_rank",
                "loser_hand",
                "loser_ht",
                "loser_age",
            ],
            "allowed_surfaces": ["Hard", "Clay", "Grass"],
            "target": "player_1_win",
            "non_null_columns": [
                "tourney_date",
                "surface",
                "winner_id",
                "loser_id",
                "winner_rank",
                "loser_rank",
            ],
        },
        "split": {"strategy": "seasons"},
        "model": {
            "algorithm": "LogisticRegression",
            "hyperparams": {},
            "random_seed": 42,
        },
        "problem_type": "classification",
        "features": {
            "numeric_pipeline": [
                "p1_rank",
                "p2_rank",
                "rank_diff",
                "age_diff",
                "ht_diff",
            ],
            "categorical_pipeline": [
                "surface",
                "tourney_level",
                "round",
                "p1_hand",
                "p2_hand",
            ],
        },
        "evaluation": {
            "metrics": ["accuracy", "log_loss"],
            "baseline_rule": "dummy rule",
        },
        "wandb": {
            "project": "tennis-atp-test",
            "entity": "",
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(dummy_config, f)

    main(config_path=str(config_path))

    processed_file = tmp_path / "data" / "processed" / "clean.csv"
    assert processed_file.exists()

    model_file = tmp_path / "models" / "model.joblib"
    assert model_file.exists()

    predictions_file = tmp_path / "reports" / "predictions.csv"
    assert predictions_file.exists()

    preds_df = pd.read_csv(predictions_file, index_col=0)
    assert "prediction" in preds_df.columns
    assert len(preds_df) == 4
