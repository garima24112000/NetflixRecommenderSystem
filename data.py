from pathlib import Path
import pandas as pd
import numpy as np
import json


def build_mappings(train_df, out_dir: Path):
    users = train_df['user_id'].unique()
    items = train_df['movie_id'].unique()
    user2idx = {int(u): i for i, u in enumerate(sorted(users))}
    item2idx = {int(i): j for j, i in enumerate(sorted(items))}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'user2idx.json', 'w') as f:
        json.dump(user2idx, f)
    with open(out_dir / 'item2idx.json', 'w') as f:
        json.dump(item2idx, f)
    return user2idx, item2idx


def load_parquet_as_pandas(path: Path, sample_rows: int = 0):
    df = pd.read_parquet(path, engine='pyarrow')
    if sample_rows and sample_rows > 0:
        df = df.sample(n=sample_rows, random_state=42)
    return df


def prepare_data(base_path: str, out_dir: str, sample_rows: int = 0):
    BASE = Path(base_path)
    out_dir = Path(out_dir)
    ratings_path = BASE / 'ratings_train_no_probe.parquet'
    probe_path = BASE / 'probe_ratings.parquet'
    qual_path = BASE / 'qualifying_to_predict.parquet'

    print('Loading ratings...')
    ratings = load_parquet_as_pandas(ratings_path, sample_rows)
    print('Ratings shape:', ratings.shape)

    print('Building mappings...')
    user2idx, item2idx = build_mappings(ratings, out_dir)

    # map indices
    ratings['user_idx'] = ratings['user_id'].map(lambda x: user2idx.get(int(x), -1)).astype(np.int64)
    ratings['item_idx'] = ratings['movie_id'].map(lambda x: item2idx.get(int(x), -1)).astype(np.int64)

    # filter any -1
    ratings = ratings[(ratings['user_idx'] >= 0) & (ratings['item_idx'] >= 0)]

    # save a compact csv for training
    train_csv = out_dir / 'train_sample.csv'
    ratings[['user_idx', 'item_idx', 'rating']].to_csv(train_csv, index=False)

    # also load probe
    print('Loading probe...')
    probe = pd.read_parquet(probe_path, engine='pyarrow')
    probe['user_idx'] = probe['user_id'].map(lambda x: user2idx.get(int(x), -1)).astype(np.int64)
    probe['item_idx'] = probe['movie_id'].map(lambda x: item2idx.get(int(x), -1)).astype(np.int64)
    probe = probe[(probe['user_idx'] >= 0) & (probe['item_idx'] >= 0)]
    probe.to_csv(out_dir / 'probe_sample.csv', index=False)

    # qualifying
    qual = pd.read_parquet(qual_path, engine='pyarrow')
    qual['user_idx'] = qual['user_id'].map(lambda x: user2idx.get(int(x), -1)).astype(np.int64)
    qual['item_idx'] = qual['movie_id'].map(lambda x: item2idx.get(int(x), -1)).astype(np.int64)
    qual = qual[(qual['user_idx'] >= 0) & (qual['item_idx'] >= 0)]
    qual.to_csv(out_dir / 'qual_sample.csv', index=False)

    return {
        'train_csv': str(train_csv),
        'probe_csv': str(out_dir / 'probe_sample.csv'),
        'qual_csv': str(out_dir / 'qual_sample.csv'),
        'n_users': len(user2idx),
        'n_items': len(item2idx)
    }
