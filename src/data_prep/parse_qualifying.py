#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    raw_path = data_dir / "raw" / "qualifying.txt"
    out_path = data_dir / "processed" / "qualifying_to_predict.parquet"

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
                # CustomerID,Date
                if current_movie is None:
                    raise ValueError("Found user line before any movie id line")
                user_str, date_str = line.split(",")
                rows.append((current_movie, int(user_str), date_str))

    df = pd.DataFrame(rows, columns=["movie_id", "user_id", "date"])
    df["movie_id"] = df["movie_id"].astype("int32")
    df["user_id"] = df["user_id"].astype("int32")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")

    # Null checks
    assert df["movie_id"].notna().all()
    assert df["user_id"].notna().all()
    assert df["date"].notna().all()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} qualifying rows to {out_path}")

if __name__ == "__main__":
    main()
