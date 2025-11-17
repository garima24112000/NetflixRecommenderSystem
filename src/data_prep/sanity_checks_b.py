#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    proc = data_dir / "processed"

    movies = pd.read_parquet(proc / "movies.parquet")
    probe_pairs = pd.read_parquet(proc / "probe_pairs.parquet")
    qualifying = pd.read_parquet(proc / "qualifying_to_predict.parquet")

    print("=== Movies ===")
    print("rows:", len(movies))
    print("min movie_id:", movies["movie_id"].min(), "max:", movies["movie_id"].max())
    print("nulls per column:\n", movies.isna().sum())

    print("\n=== Probe pairs ===")
    print("rows:", len(probe_pairs))
    print("unique movies:", probe_pairs["movie_id"].nunique())
    print("unique users:", probe_pairs["user_id"].nunique())
    print("nulls per column:\n", probe_pairs.isna().sum())

    print("\n=== Qualifying ===")
    print("rows:", len(qualifying))
    print("unique movies:", qualifying["movie_id"].nunique())
    print("unique users:", qualifying["user_id"].nunique())
    print("date range:", qualifying["date"].min(), "→", qualifying["date"].max())
    print("nulls per column:\n", qualifying.isna().sum())

if __name__ == "__main__":
    main()
