#!/usr/bin/env python3

import logging
from pathlib import Path

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_probe_pairs(probe_path: str) -> pd.DataFrame:
    """
    Parse Netflix Kaggle probe.txt into a DataFrame of (movie_id, user_id).
    Format example:

        1:
        30878
        2647871
        ...

        2:
        12345
        67890
        ...

    Returns:
        DataFrame with columns: movie_id, user_id
    """
    logging.info("Reading probe file: %s", probe_path)
    movie_ids = []
    user_ids = []

    current_movie_id = None

    with open(probe_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith(":"):
                # movie header, e.g. "1234:"
                mid_str = line[:-1]
                try:
                    current_movie_id = int(mid_str)
                except ValueError:
                    logging.warning("Bad movie id line in probe file: %r", line)
                    current_movie_id = None
                continue

            # user_id line
            if current_movie_id is None:
                logging.warning("User line without movie id in probe file: %r", line)
                continue

            try:
                uid = int(line)
            except ValueError:
                logging.warning("Bad user id line in probe file: %r", line)
                continue

            movie_ids.append(current_movie_id)
            user_ids.append(uid)

    probe_df = pd.DataFrame({"movie_id": movie_ids, "user_id": user_ids})
    logging.info("Probe pairs loaded: %d rows", len(probe_df))
    logging.info(
        "Unique movies in probe: %d, unique users in probe: %d",
        probe_df["movie_id"].nunique(),
        probe_df["user_id"].nunique(),
    )
    return probe_df


def remove_probe_rows(ratings_parquet: str, probe_path: str, output_parquet: str):
    logging.info("Loading ratings from: %s", ratings_parquet)
    ratings = pd.read_parquet(ratings_parquet)
    logging.info("Ratings rows (full): %d", len(ratings))

    probe_df = load_probe_pairs(probe_path)

    # Mark probe rows and remove them
    logging.info("Merging ratings with probe pairs to mark probe rows...")
    probe_df["is_probe"] = True

    merged = ratings.merge(
        probe_df,
        on=["movie_id", "user_id"],
        how="left",
    )

    logging.info("Merged shape: %s", merged.shape)

    # Rows where is_probe is True are probe entries; we drop them
    before = len(merged)
    training = merged[merged["is_probe"].isna()].drop(columns=["is_probe"])
    after = len(training)
    removed = before - after

    logging.info("Rows before removing probe: %d", before)
    logging.info("Rows after removing probe: %d", after)
    logging.info("Rows removed as probe: %d", removed)

    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing training (no probe) Parquet to: %s", output_parquet)
    training.to_parquet(out_path, index=False)
    logging.info("Done.")


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parents[2]

    ratings_parquet = project_root / "data" / "processed" / "ratings_full.parquet"
    probe_path = project_root / "data" / "raw" / "probe.txt"
    output_parquet = project_root / "data" / "processed" / "ratings_train_no_probe.parquet"

    logging.info("Project root: %s", project_root)
    logging.info("Ratings (full): %s", ratings_parquet)
    logging.info("Probe file: %s", probe_path)
    logging.info("Output (train no probe): %s", output_parquet)

    remove_probe_rows(str(ratings_parquet), str(probe_path), str(output_parquet))


if __name__ == "__main__":
    main()
