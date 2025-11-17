#!/usr/bin/env python3

import os
import glob
import csv
import logging
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_combined_files(raw_dir: str, output_csv: str):
    """
    Parse Netflix Kaggle combined_data_*.txt files into one big CSV.

    Input files format example:
        1:
        30878,3,2005-09-06
        2647871,3,2005-12-26
        ...

    Output CSV columns:
        movie_id,user_id,rating,date
    """
    raw_path = Path(raw_dir)
    pattern = str(raw_path / "combined_data_*.txt")
    input_files = sorted(glob.glob(pattern))

    if not input_files:
        raise FileNotFoundError(f"No files matching {pattern}")

    logging.info("Found %d input files.", len(input_files))
    for f in input_files:
        logging.info("  %s", f)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    current_movie_id = None

    with output_path.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        # header
        writer.writerow(["movie_id", "user_id", "rating", "date"])

        for fname in input_files:
            logging.info("Processing file: %s", fname)
            with open(fname, "r") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue

                    # Movie header line: e.g. "12345:"
                    if line.endswith(":"):
                        movie_id_str = line[:-1]  # drop ':'
                        try:
                            current_movie_id = int(movie_id_str)
                        except ValueError:
                            logging.warning("Bad movie id line: %r in file %s", line, fname)
                            current_movie_id = None
                        continue

                    # Rating line: user_id,rating,date
                    if current_movie_id is None:
                        # Shouldn't happen if file is well-formed
                        logging.warning("Rating line without movie id: %r", line)
                        continue

                    parts = line.split(",")
                    if len(parts) != 3:
                        logging.warning("Bad rating line: %r", line)
                        continue

                    user_id_str, rating_str, date_str = parts
                    try:
                        user_id = int(user_id_str)
                        rating = int(rating_str)
                    except ValueError:
                        logging.warning("Non-integer user_id or rating in line: %r", line)
                        continue

                    writer.writerow([current_movie_id, user_id, rating, date_str])
                    total_rows += 1

                    # Optional: log every 1M rows
                    if total_rows % 1_000_000 == 0:
                        logging.info("Written %d rows so far...", total_rows)

    logging.info("Done. Total rows written: %d", total_rows)
    logging.info("Output CSV: %s", output_path)


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    output_csv = raw_dir / "ratings_full.csv"

    logging.info("Project root: %s", project_root)
    logging.info("Raw dir: %s", raw_dir)
    logging.info("Output CSV: %s", output_csv)

    parse_combined_files(str(raw_dir), str(output_csv))


if __name__ == "__main__":
    main()
