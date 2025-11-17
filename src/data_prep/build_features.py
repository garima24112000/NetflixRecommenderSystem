#!/usr/bin/env python3

import logging
from pathlib import Path

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def build_features(ratings_parquet: str,
                   movie_features_out: str,
                   user_features_out: str):
    logging.info("Loading training ratings (no probe) from: %s", ratings_parquet)

    # Load only needed columns
    df = pd.read_parquet(
        ratings_parquet,
        columns=["movie_id", "user_id", "rating", "date"],
    )
    logging.info("Ratings rows: %d", len(df))

    # Make sure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    logging.info("Rows after dropping bad dates: %d", len(df))

    # --------------------------
    # Movie-level features
    # --------------------------
    logging.info("Computing movie-level features...")

    movie_grp = df.groupby("movie_id")

    movie_features = movie_grp.agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        first_rating_date=("date", "min"),
        last_rating_date=("date", "max"),
    ).reset_index()

    logging.info("Movie features shape: %s", movie_features.shape)

    # --------------------------
    # User-level features
    # --------------------------
    logging.info("Computing user-level features...")

    user_grp = df.groupby("user_id")

    user_features = user_grp.agg(
        n_ratings=("rating", "size"),
        mean_rating_given=("rating", "mean"),
        std_rating_given=("rating", "std"),
        min_rating_given=("rating", "min"),
        max_rating_given=("rating", "max"),
        first_rating_date=("date", "min"),
        last_rating_date=("date", "max"),
    ).reset_index()

    # rating_span_days = last - first in days
    user_features["rating_span_days"] = (
        user_features["last_rating_date"] - user_features["first_rating_date"]
    ).dt.days

    logging.info("User features shape: %s", user_features.shape)

    # --------------------------
    # Write outputs
    # --------------------------
    movie_out_path = Path(movie_features_out)
    user_out_path = Path(user_features_out)
    movie_out_path.parent.mkdir(parents=True, exist_ok=True)
    user_out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing movie features to: %s", movie_out_path)
    movie_features.to_parquet(movie_out_path, index=False)

    logging.info("Writing user features to: %s", user_out_path)
    user_features.to_parquet(user_out_path, index=False)

    logging.info("Done building features.")


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parents[2]

    ratings_parquet = project_root / "data" / "processed" / "ratings_train_no_probe.parquet"
    movie_features_out = project_root / "data" / "processed" / "movie_features.parquet"
    user_features_out = project_root / "data" / "processed" / "user_features.parquet"

    logging.info("Project root: %s", project_root)
    logging.info("Ratings (train no probe): %s", ratings_parquet)
    logging.info("Movie features out: %s", movie_features_out)
    logging.info("User features out: %s", user_features_out)

    build_features(
        str(ratings_parquet),
        str(movie_features_out),
        str(user_features_out),
    )


if __name__ == "__main__":
    main()
