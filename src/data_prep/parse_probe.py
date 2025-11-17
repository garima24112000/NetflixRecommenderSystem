#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    raw_path = data_dir / "raw" / "probe.txt"
    out_path = data_dir / "processed" / "probe_pairs.parquet"

    rows = []
    current_movie = None

    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith(":"):
                # MovieID:
                current_movie = int(line[:-1])
            else:
                # CustomerID
                if current_movie is None:
                    raise ValueError("Found user line before any movie id line")
                user_id = int(line)
                rows.append((current_movie, user_id))

    df = pd.DataFrame(rows, columns=["movie_id", "user_id"])

    # Validations
    assert df["movie_id"].notna().all(), "Null movie_id in probe_pairs"
    assert df["user_id"].notna().all(), "Null user_id in probe_pairs"

    df["movie_id"] = df["movie_id"].astype("int32")
    df["user_id"] = df["user_id"].astype("int32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} probe pairs to {out_path}")

if __name__ == "__main__":
    main()
