"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

from pathlib import Path
import urllib.request

import pandas as pd

from src.logger import get_logger
from src.utils import load_csv

logger = get_logger(__name__)


def load_raw_data(
    raw_dir: Path, base_url: str, seasons: list[int], download_if_missing: bool = True
) -> pd.DataFrame:
    """
    Inputs:
    - raw_dir: Path object pointing to the data/raw directory.
    - base_url: The GitHub raw URL where the CSVs are hosted.
    - seasons: List of years (integers) to load.
    - download_if_missing: If True, downloads missing files.
    Outputs:
    - pd.DataFrame containing the raw, unmodified data
      concatenated across all seasons.
    """
    logger.info(
        "Attempting to load raw data for seasons %s from %s ...", seasons, raw_dir
    )

    raw_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for season in seasons:
        file_name = f"atp_matches_{season}.csv"
        file_path = raw_dir / file_name

        if not file_path.exists():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"File {file_name} is missing in {raw_dir} "
                    "and download is disabled."
                )

            url = f"{base_url}{file_name}"
            logger.info("Downloading missing file: %s", url)

            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                if file_path.exists():
                    file_path.unlink()
                raise RuntimeError(f"Failed to download {url}: {e}") from e

        try:
            df_season = load_csv(file_path)
            dfs.append(df_season)
        except Exception as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}") from e

    if not dfs:
        raise ValueError("No datasets were loaded. Please check the seasons list.")

    df_combined = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d records across %d seasons.", len(df_combined), len(seasons))

    return df_combined
