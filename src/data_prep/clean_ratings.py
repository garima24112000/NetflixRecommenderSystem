#!/usr/bin/env python3

import logging
from pathlib import Path

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def clean_and_convert(input_csv: str, output_parquet: str):
    logging.info("Reading CSV: %s", input_csv)
    # Read entire CSV; SeaWulf job will have enough memory.
    df = pd.read_csv(
        input_csv,
        dtype={
            "movie_id": "int32",
            "user_id": "int32",
            "rating": "int8",
        },
        parse_dates=["date"],
        # if parsing fails, you'll get NaT; we'll drop those
        infer_datetime_format=True,
    )

    logging.info("Initial rows: %d", len(df))

    # Drop any rows with nulls in key columns
    df = df.dropna(subset=["movie_id", "user_id", "rating", "date"])
    logging.info("Rows after dropping nulls: %d", len(df))

    # Keep only ratings in [1,5]
    df = df[df["rating"].between(1, 5)]
    logging.info("Rows after rating range filter: %d", len(df))

    # Make sure date is datetime (if parsing failed it would be NaT and got dropped above)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    logging.info("Rows after dropping bad dates: %d", len(df))

    # Optionally drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["movie_id", "user_id", "date"])
    logging.info(
        "Rows after dropping duplicates: %d (removed %d)",
        len(df),
        before - len(df),
    )

    # Some quick summary stats
    n_users = df["user_id"].nunique()
    n_movies = df["movie_id"].nunique()
    logging.info("Unique users: %d", n_users)
    logging.info("Unique movies: %d", n_movies)

    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing Parquet: %s", output_parquet)
    df.to_parquet(out_path, index=False)
    logging.info("Done. Final rows: %d", len(df))


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parents[2]
    input_csv = project_root / "data" / "raw" / "ratings_full.csv"
    output_parquet = project_root / "data" / "processed" / "ratings_full.parquet"

    logging.info("Project root: %s", project_root)
    logging.info("Input CSV: %s", input_csv)
    logging.info("Output Parquet: %s", output_parquet)

    clean_and_convert(str(input_csv), str(output_parquet))


if __name__ == "__main__":
    main()
