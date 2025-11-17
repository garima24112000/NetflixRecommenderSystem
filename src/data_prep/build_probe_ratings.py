#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    ratings_path = data_dir / "processed" / "ratings_full.parquet"  # from Person A
    probe_pairs_path = data_dir / "processed" / "probe_pairs.parquet"
    out_path = data_dir / "processed" / "probe_ratings.parquet"

    print(f"Loading {ratings_path}")
    ratings = pd.read_parquet(ratings_path)[["movie_id", "user_id", "rating", "date"]]

    print(f"Loading {probe_pairs_path}")
    probe_pairs = pd.read_parquet(probe_pairs_path)

    merged = probe_pairs.merge(
        ratings,
        on=["movie_id", "user_id"],
        how="left",
        validate="one_to_one"
    )

    missing = merged["rating"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} probe pairs had no matching rating in training data")

    # Type checks
    assert merged["rating"].between(1, 5).all(), "Probe ratings outside 1–5"
    assert merged["date"].notna().all(), "Missing date in probe ratings"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Saved {len(merged)} probe ratings to {out_path}")

if __name__ == "__main__":
    main()
