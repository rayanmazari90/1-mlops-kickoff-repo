import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.load_data import load_raw_data


def test_load_raw_data_success_local_files(tmp_path):
    """Test that existing local files are loaded and concatenated successfully without downloading."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Create fake CSVs for 2018 and 2019
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

    # Run the function
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
    """Test that missing files raise FileNotFoundError when download_if_missing is False."""
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
    """Test that downloading triggers urllib and correctly loads the new file."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    def mock_download(url, file_path):
        # When urllib is called, create a fake file at the destination
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
    mock_urlretrieve.assert_called_once_with(
        "http://fake-url.com/atp_matches_2020.csv", raw_dir / "atp_matches_2020.csv"
    )
