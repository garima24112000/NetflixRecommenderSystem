#!/usr/bin/env python3
"""
Generate predictions with ID mapping
"""
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from models import NCF


def load_mappings(model_dir):
    """Load ID to index mappings"""
    user_map_path = Path(model_dir) / 'user_mapping.pkl'
    item_map_path = Path(model_dir) / 'item_mapping.pkl'
    
    # Try pickle first
    if user_map_path.exists() and item_map_path.exists():
        print("Loading mappings from pickle files...")
        with open(user_map_path, 'rb') as f:
            user_map = pickle.load(f)
        with open(item_map_path, 'rb') as f:
            item_map = pickle.load(f)
        return user_map, item_map
    
    # Otherwise, extract from training data
    print("Extracting mappings from training data...")
    train_path = Path(model_dir) / 'train_sample.csv'
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Cannot find training data at {train_path}\n"
            "Need it to create ID mappings!"
        )
    
    train_df = pd.read_csv(train_path)
    
    user_map = train_df[['user_id', 'user_idx']].drop_duplicates().set_index('user_id')['user_idx'].to_dict()
    item_map = train_df[['movie_id', 'item_idx']].drop_duplicates().set_index('movie_id')['item_idx'].to_dict()
    
    # Save for future use
    with open(user_map_path, 'wb') as f:
        pickle.dump(user_map, f)
    with open(item_map_path, 'wb') as f:
        pickle.dump(item_map, f)
    
    print(f"✓ Saved mappings to {model_dir}")
    
    return user_map, item_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--ratings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16384)
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    model_dir = checkpoint_path.parent
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['args']
    
    n_users = 480189
    n_items = 17770
    
    print(f"✓ Checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation RMSE: {checkpoint['val_rmse']:.4f}")
    
    # 2. Load ID mappings
    print("\nLoading ID mappings...")
    user_map, item_map = load_mappings(model_dir)
    print(f"✓ User mappings: {len(user_map)}")
    print(f"✓ Item mappings: {len(item_map)}")
    
    # 3. Create model
    print("\nCreating model...")
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        emb_dim=config['embedding_dim'],
        mlp_layers=config['mlp_layers']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(args.device)
    model.eval()
    print("✓ Model ready")
    
    # 4. Load test data
    print(f"\nLoading test data...")
    pairs_df = pd.read_parquet(args.pairs)
    ratings_df = pd.read_parquet(args.ratings)
    
    print(f"  Pairs: {len(pairs_df)} samples")
    print(f"  Ratings: {len(ratings_df)} samples")
    
    # 5. Combine
    if len(pairs_df) == len(ratings_df):
        data_df = pairs_df.copy()
        data_df['rating'] = ratings_df['rating'].values
    else:
        data_df = pd.merge(pairs_df, ratings_df, on=['movie_id', 'user_id'], how='inner')
    
    print(f"✓ Combined: {len(data_df)} samples")
    
    # 6. Map IDs to indices
    print("\nMapping IDs to indices...")
    data_df['user_idx'] = data_df['user_id'].map(user_map)
    data_df['item_idx'] = data_df['movie_id'].map(item_map)
    
    # Check for unmapped IDs
    unmapped_users = data_df['user_idx'].isna().sum()
    unmapped_items = data_df['item_idx'].isna().sum()
    
    print(f"  Unmapped users: {unmapped_users}")
    print(f"  Unmapped items: {unmapped_items}")
    
    if unmapped_users > 0 or unmapped_items > 0:
        print(f"\n⚠️  Warning: Found {unmapped_users + unmapped_items} unmapped IDs")
        print(f"  These are users/items not seen during training")
        print(f"  Filtering them out...")
        
        # Show some examples
        if unmapped_users > 0:
            sample_unmapped = data_df[data_df['user_idx'].isna()]['user_id'].head(5).tolist()
            print(f"  Example unmapped user IDs: {sample_unmapped}")
        if unmapped_items > 0:
            sample_unmapped = data_df[data_df['item_idx'].isna()]['movie_id'].head(5).tolist()
            print(f"  Example unmapped movie IDs: {sample_unmapped}")
        
        # Filter
        before = len(data_df)
        data_df = data_df.dropna(subset=['user_idx', 'item_idx'])
        after = len(data_df)
        print(f"  Kept {after}/{before} samples ({100*after/before:.1f}%)")
    
    # Convert to int
    data_df['user_idx'] = data_df['user_idx'].astype(int)
    data_df['item_idx'] = data_df['item_idx'].astype(int)
    
    print(f"\n✓ Ready for prediction: {len(data_df)} samples")
    print(f"  User indices: {data_df['user_idx'].min()} to {data_df['user_idx'].max()}")
    print(f"  Item indices: {data_df['item_idx'].min()} to {data_df['item_idx'].max()}")
    
    # 7. Generate predictions
    print("\nGenerating predictions...")
    users = data_df['user_idx'].values
    items = data_df['item_idx'].values
    ratings = data_df['rating'].values
    
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(users), args.batch_size), desc="Predicting"):
            batch_users = torch.tensor(
                users[i:i+args.batch_size], 
                dtype=torch.long
            ).to(args.device)
            
            batch_items = torch.tensor(
                items[i:i+args.batch_size], 
                dtype=torch.long
            ).to(args.device)
            
            batch_preds = model(batch_users, batch_items).cpu().numpy()
            predictions.append(batch_preds)
    
    predictions = np.concatenate(predictions)
    
    # 8. Calculate metrics
    rmse = np.sqrt(np.mean((predictions - ratings) ** 2))
    mae = np.mean(np.abs(predictions - ratings))
    
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(predictions):,}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"{'='*60}")
    
    # 9. Save results
    data_df['prediction'] = predictions
    data_df['error'] = np.abs(predictions - ratings)
    data_df['squared_error'] = (predictions - ratings) ** 2
    
    # Select columns to save
    output_cols = ['user_id', 'movie_id', 'rating', 'prediction', 'error']
    data_df[output_cols].to_csv(output_path, index=False)
    
    print(f"\n✓ Saved predictions to: {output_path}")
    
    # 10. Summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"NCF Model Predictions\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
        f.write(f"Validation RMSE: {checkpoint['val_rmse']:.4f}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Samples: {len(predictions):,}\n")
        f.write(f"  RMSE: {rmse:.4f}\n")
        f.write(f"  MAE:  {mae:.4f}\n\n")
        f.write(f"Prediction Stats:\n{pd.Series(predictions).describe()}\n\n")
        f.write(f"Error Stats:\n{data_df['error'].describe()}\n")
    
    print(f"✓ Saved summary to: {summary_path}")
    
    # Show samples
    print(f"\nSample predictions:")
    print(data_df[['user_id', 'movie_id', 'rating', 'prediction', 'error']].head(10))


if __name__ == '__main__':
    main()
