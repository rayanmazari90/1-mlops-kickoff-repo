from unittest.mock import patch

import pandas as pd
import pytest

from src.load_data import load_raw_data


def test_load_raw_data_success_local_files(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    df_2018 = pd.DataFrame(
        {
            "tourney_date": ["2018-01-01"],
            "winner_name": ["Player A"],
            "loser_name": ["Player B"],
        }
    )
    df_2019 = pd.DataFrame(
        {
            "tourney_date": ["2019-01-01"],
            "winner_name": ["Player C"],
            "loser_name": ["Player D"],
        }
    )

    df_2018.to_csv(raw_dir / "atp_matches_2018.csv", index=False)
    df_2019.to_csv(raw_dir / "atp_matches_2019.csv", index=False)

    df_combined = load_raw_data(
        raw_dir=raw_dir,
        base_url="http://fake-url.com/",
        seasons=[2018, 2019],
        download_if_missing=False,
    )

    assert len(df_combined) == 2
    assert "Player A" in df_combined["winner_name"].values
    assert "Player C" in df_combined["winner_name"].values


def test_load_raw_data_missing_file_no_download(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="is missing in"):
        load_raw_data(
            raw_dir=raw_dir,
            base_url="http://fake-url.com/",
            seasons=[2020],
            download_if_missing=False,
        )


@patch("src.load_data.urllib.request.urlretrieve")
def test_load_raw_data_download_if_missing(mock_urlretrieve, tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    def mock_download(url, file_path):
        df = pd.DataFrame({"tourney_date": ["2020-01-01"]})
        df.to_csv(file_path, index=False)

    mock_urlretrieve.side_effect = mock_download

    df_combined = load_raw_data(
        raw_dir=raw_dir,
        base_url="http://fake-url.com/",
        seasons=[2020],
        download_if_missing=True,
    )

    assert len(df_combined) == 1
    expected_url = "http://fake-url.com/atp_matches_2020.csv"
    expected_path = raw_dir / "atp_matches_2020.csv"
    mock_urlretrieve.assert_called_once_with(expected_url, expected_path)


@patch("src.load_data.urllib.request.urlretrieve")
def test_load_raw_data_download_failure_cleans_partial(mock_urlretrieve, tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    def fail_download(url, file_path):
        from pathlib import Path

        Path(file_path).write_text("partial")
        raise ConnectionError("Network error")

    mock_urlretrieve.side_effect = fail_download

    with pytest.raises(RuntimeError, match="Failed to download"):
        load_raw_data(
            raw_dir=raw_dir,
            base_url="http://fake-url.com/",
            seasons=[2020],
            download_if_missing=True,
        )

    assert not (raw_dir / "atp_matches_2020.csv").exists()


def test_load_raw_data_empty_seasons(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="No datasets were loaded"):
        load_raw_data(
            raw_dir=raw_dir,
            base_url="http://fake-url.com/",
            seasons=[],
            download_if_missing=False,
        )


@patch("src.load_data.load_csv")
def test_load_raw_data_bad_csv(mock_load_csv, tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "atp_matches_2020.csv").write_text("col1\nval1")

    mock_load_csv.side_effect = Exception("parse error")

    with pytest.raises(RuntimeError, match="Failed to read"):
        load_raw_data(
            raw_dir=raw_dir,
            base_url="http://fake-url.com/",
            seasons=[2020],
            download_if_missing=False,
        )
