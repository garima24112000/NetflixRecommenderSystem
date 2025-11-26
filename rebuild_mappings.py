#!/usr/bin/env python3
"""
Rebuild ID mappings from ratings_full.parquet
"""
import pandas as pd
import pickle
from pathlib import Path

print("Loading ratings_full.parquet...")
ratings_df = pd.read_parquet('data/processed/ratings_full.parquet')

print(f"✓ Loaded {len(ratings_df):,} ratings")
print(f"Columns: {ratings_df.columns.tolist()}")
print(f"\nSample:")
print(ratings_df.head())

# Build mappings (sorted order to match data.py logic)
print("\nBuilding user mappings...")
unique_users = sorted(ratings_df['user_id'].unique())
user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
print(f"  Users: {len(user_to_idx):,} (0 to {len(user_to_idx)-1})")

print("\nBuilding movie mappings...")
unique_movies = sorted(ratings_df['movie_id'].unique())
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
print(f"  Movies: {len(movie_to_idx):,} (0 to {len(movie_to_idx)-1})")

# Verify with training data
print("\nVerifying with training data...")
train_sample = pd.read_csv('final_model_v2/train_sample.csv', nrows=1000)
print(f"  Train user_idx range: {train_sample['user_idx'].min()}-{train_sample['user_idx'].max()}")
print(f"  Train item_idx range: {train_sample['item_idx'].min()}-{train_sample['item_idx'].max()}")
print(f"  Expected: users 0-{len(user_to_idx)-1}, items 0-{len(movie_to_idx)-1}")

if train_sample['user_idx'].max() >= len(user_to_idx):
    print("  ⚠️  Warning: Training indices exceed mapping range!")
if train_sample['item_idx'].max() >= len(movie_to_idx):
    print("  ⚠️  Warning: Item indices exceed mapping range!")

# Save mappings
output_dir = Path('final_model_v2')
output_dir.mkdir(exist_ok=True)

print("\nSaving mappings...")
with open(output_dir / 'user_mapping.pkl', 'wb') as f:
    pickle.dump(user_to_idx, f)
print(f"✓ {output_dir / 'user_mapping.pkl'}")

with open(output_dir / 'item_mapping.pkl', 'wb') as f:
    pickle.dump(movie_to_idx, f)
print(f"✓ {output_dir / 'item_mapping.pkl'}")

# Save CSV for inspection
user_df = pd.DataFrame(list(user_to_idx.items()), columns=['user_id', 'user_idx'])
user_df.to_csv(output_dir / 'user_mapping.csv', index=False)
print(f"✓ {output_dir / 'user_mapping.csv'}")

movie_df = pd.DataFrame(list(movie_to_idx.items()), columns=['movie_id', 'item_idx'])
movie_df.to_csv(output_dir / 'item_mapping.csv', index=False)
print(f"✓ {output_dir / 'item_mapping.csv'}")

print(f"\n{'='*60}")
print("MAPPINGS CREATED")
print(f"{'='*60}")
print(f"Users:  {len(user_to_idx):,} mappings")
print(f"Movies: {len(movie_to_idx):,} mappings")
print(f"\nSample mappings:")
for k, v in list(user_to_idx.items())[:3]:
    print(f"  user_id {k:>7} → user_idx {v}")
for k, v in list(movie_to_idx.items())[:3]:
    print(f"  movie_id {k:>6} → item_idx {v}")
print(f"{'='*60}")
