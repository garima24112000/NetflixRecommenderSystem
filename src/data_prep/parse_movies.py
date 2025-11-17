#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]  # go up from src/data_prep
    data_dir = project_root / "data"
    raw_path = data_dir / "raw" / "movie_titles.csv"
    out_path = data_dir / "processed" / "movies.parquet"

    movies = []
    with raw_path.open(encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # MovieID,YearOfRelease,Title
            parts = line.split(",", maxsplit=2)
            movie_id = int(parts[0])
            year_str = parts[1]
            year = int(year_str) if year_str.isdigit() else None
            title = parts[2]
            movies.append((movie_id, year, title))

    df = pd.DataFrame(movies, columns=["movie_id", "year", "title"])

    # Basic validations
    assert df["movie_id"].notna().all(), "Found missing movie_id"
    assert df["year"].isna().sum() < len(df), "All years missing? Check input."
    assert df["title"].notna().all(), "Found missing title"

    df["movie_id"] = df["movie_id"].astype("int32")
    df["year"] = df["year"].astype("float32")  # can be NaN
    # title stays as string

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} movies to {out_path}")

if __name__ == "__main__":
    main()