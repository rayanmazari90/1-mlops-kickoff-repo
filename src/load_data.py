"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Ingests raw data from external sources into the pipeline.
- Responsibility (separation of concerns): Handling data fetching, downloading, or reading from the raw zone.
- Pipeline contract (inputs and outputs): Takes a source location, outputs a raw Pandas DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from pathlib import Path
import urllib.request
import logging

from src.utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(
    raw_dir: Path, base_url: str, seasons: list[int], download_if_missing: bool = True
) -> pd.DataFrame:
    """
    Inputs:
    - raw_dir: Path object pointing to the data/raw directory.
    - base_url: The GitHub raw URL where the CSVs are hosted.
    - seasons: List of years (integers) to load.
    - download_if_missing: If True, downloads missing files. If False, raises an error.
    Outputs:
    - pd.DataFrame containing the raw, unmodified data concatenated across all seasons.

    Why this contract matters for reliable ML delivery:
    - Establishes an immutable starting point by caching external data.
    - Makes the ingestion idempotent and reproducible.
    """
    logger.info(f"Attempting to load raw data for seasons {seasons} from {raw_dir}...")

    # Ensure raw_dir exists
    raw_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for season in seasons:
        file_name = f"atp_matches_{season}.csv"
        file_path = raw_dir / file_name

        if not file_path.exists():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"File {file_name} is missing in {raw_dir} and download is disabled."
                )

            # Download the file
            url = f"{base_url}{file_name}"
            # logger.info(f"Downloading missing file: {url}")
            print(f"Downloading missing file: {url}")

            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                # If download fails, remove partially downloaded file to avoid corrupted cache
                if file_path.exists():
                    file_path.unlink()
                raise RuntimeError(f"Failed to download {url}: {e}") from e

        # Load the CSV and append to list
        # Using utils.load_csv for consistency
        try:
            df_season = load_csv(file_path)
            dfs.append(df_season)
        except Exception as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}") from e

    if not dfs:
        raise ValueError("No datasets were loaded. Please check the seasons list.")

    # Concatenate all seasons into a single dataframe
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_combined)} records across {len(seasons)} seasons.")

    return df_combined
