#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import json
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models import NCF
from data import prepare_data

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Install with: pip install wandb")


class RatingsDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.users = df['user_idx'].values.astype(np.int64)
        self.items = df['item_idx'].values.astype(np.int64)
        self.ratings = df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--out-dir', default='deep_recsys/output')
    p.add_argument('--sample-rows', type=int, default=0)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--accumulation-steps', type=int, default=1, 
                   help='Gradient accumulation steps (simulates larger batch)')
    p.add_argument('--embedding-dim', type=int, default=32)
    p.add_argument('--mlp-layers', nargs='+', type=int, default=[64,32])
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    
    # Early stopping
    p.add_argument('--early-stopping', action='store_true', 
                   help='Enable early stopping')
    p.add_argument('--patience', type=int, default=5,
                   help='Early stopping patience (epochs without improvement)')
    p.add_argument('--min-delta', type=float, default=0.0001,
                   help='Minimum improvement to reset patience')
    
    # WandB
    p.add_argument('--use-wandb', action='store_true',
                   help='Log to Weights & Biases')
    p.add_argument('--wandb-project', default='netflix-ncf',
                   help='WandB project name')
    p.add_argument('--wandb-name', default=None,
                   help='WandB run name (auto-generated if not provided)')
    p.add_argument('--wandb-log-freq', type=int, default=500,
                   help='Log to WandB every N batches (default: 500)')
    
    return p.parse_args()


def rmse(preds, labels):
    return np.sqrt(np.mean((preds - labels) ** 2))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize WandB if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_name = args.wandb_name or f"ncf_emb{args.embedding_dim}_bs{args.batch_size}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args)
        )
        print(f"✓ WandB initialized: {wandb_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  WandB requested but not installed. Continuing without WandB.")
        args.use_wandb = False

    print('Preparing data...')
    data_info = prepare_data(args.base, str(out_dir), sample_rows=args.sample_rows)
    print('Data info:', data_info)

    train_ds = RatingsDataset(data_info['train_csv'])
    
    # Verify data integrity
    print(f'\nData integrity check:')
    print(f'  Train samples: {len(train_ds)}')
    print(f'  Max user_idx in train: {train_ds.users.max()} (limit: {data_info["n_users"]-1})')
    print(f'  Max item_idx in train: {train_ds.items.max()} (limit: {data_info["n_items"]-1})')
    
    # Critical validation
    bad_users = (train_ds.users >= data_info['n_users']).sum()
    bad_items = (train_ds.items >= data_info['n_items']).sum()
    if bad_users > 0 or bad_items > 0:
        print(f'ERROR: Found {bad_users} bad user indices and {bad_items} bad item indices!')
        print('Data preparation has a bug - aborting.')
        return
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        pin_memory=True,
        shuffle=True,
        num_workers=4
    )

    probe_df = pd.read_csv(data_info['probe_csv'])
    qual_df = pd.read_csv(data_info['qual_csv'])

    n_users = data_info['n_users']
    n_items = data_info['n_items']

    # Filter evaluation datasets
    print(f'\nFiltering evaluation datasets...')
    probe_before = len(probe_df)
    probe_df = probe_df[
        (probe_df['user_idx'] >= 0) & (probe_df['user_idx'] < n_users) &
        (probe_df['item_idx'] >= 0) & (probe_df['item_idx'] < n_items)
    ]
    probe_after = len(probe_df)
    print(f'  Probe: kept {probe_after}/{probe_before} samples')
    
    qual_before = len(qual_df)
    qual_df = qual_df[
        (qual_df['user_idx'] >= 0) & (qual_df['user_idx'] < n_users) &
        (qual_df['item_idx'] >= 0) & (qual_df['item_idx'] < n_items)
    ]
    qual_after = len(qual_df)
    print(f'  Qual: kept {qual_after}/{qual_before} samples')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    if args.accumulation_steps > 1:
        print(f'Using gradient accumulation:')
        print(f'  Physical batch size: {args.batch_size}')
        print(f'  Accumulation steps: {args.accumulation_steps}')
        print(f'  Effective batch size: {args.batch_size * args.accumulation_steps}')
    
    # Initialize early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience, 
            min_delta=args.min_delta,
            verbose=True
        )
        print(f'✓ Early stopping enabled (patience={args.patience}, min_delta={args.min_delta})')
    
    model = NCF(n_users, n_items, emb_dim=args.embedding_dim, mlp_layers=args.mlp_layers)
    model.to(device)

    # LIGHTWEIGHT WandB watching - only log gradients, very infrequently
    if args.use_wandb:
        wandb.watch(model, log='gradients', log_freq=args.wandb_log_freq)  # Very infrequent

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_rmse = 1e9
    checkpoint_path = out_dir / 'checkpoint.pth'

    print('\nStarting training...')
    global_step = 0  # Track total batches across all epochs
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # TRAINING with gradient accumulation
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (u, i, r) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            u = u.to(device).long()
            i = i.to(device).long()
            r = r.to(device).float()
            
            # Forward pass
            preds = model(u, i)
            loss = criterion(preds, r)
            
            # Normalize loss for accumulation
            loss = loss / args.accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log to WandB OCCASIONALLY (not every step!)
                if args.use_wandb and global_step % args.wandb_log_freq == 0:
                    wandb.log({
                        'train/batch_loss': loss.item() * args.accumulation_steps,
                        'train/step': global_step,
                        'train/epoch': epoch
                    }, step=global_step)
            
            running_loss += loss.item() * r.size(0) * args.accumulation_steps
        
        # Final optimizer step if there are remaining gradients
        if (batch_idx + 1) % args.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = running_loss / len(train_ds)
        epoch_rmse = np.sqrt(epoch_loss)
        
        # EVALUATION
        model.eval()
        val_rmse = None
        with torch.no_grad():
            if len(probe_df) == 0:
                print('  Warning: No valid probe samples!')
            else:
                users = torch.tensor(probe_df['user_idx'].values, dtype=torch.long)
                items = torch.tensor(probe_df['item_idx'].values, dtype=torch.long)
                labels = probe_df['rating'].values.astype(np.float32)
                
                preds = []
                batch = 8192
                for i0 in range(0, len(users), batch):
                    up = users[i0:i0+batch].to(device)
                    ip = items[i0:i0+batch].to(device)
                    p = model(up, ip).cpu().numpy()
                    preds.append(p)
                
                preds = np.concatenate(preds)
                val_rmse = rmse(preds, labels)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f'Epoch {epoch} ({epoch_time:.1f}s):')
        print(f'  Train MSE: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}')
        if val_rmse is not None:
            print(f'  Probe RMSE: {val_rmse:.4f}')
        
        # Log epoch summary to WandB
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/epoch_loss': epoch_loss,
                'train/epoch_rmse': epoch_rmse,
                'train/epoch_time': epoch_time,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }
            if val_rmse is not None:
                log_dict['val/rmse'] = val_rmse
            wandb.log(log_dict, step=global_step)
        
        # Save best model
        if val_rmse is not None and val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_rmse': epoch_rmse,
                'val_rmse': val_rmse,
                'args': vars(args)
            }, checkpoint_path)
            print(f'  ✓ Saved checkpoint (best val RMSE: {best_rmse:.4f})')
        
        # Early stopping check
        if early_stopping is not None and val_rmse is not None:
            if early_stopping(val_rmse):
                print(f'\n✓ Early stopping triggered at epoch {epoch}')
                print(f'  Best val RMSE: {best_rmse:.4f}')
                break

    # Run info
    run_info = {
        'timestamp': time.asctime(),
        'args': vars(args),
        'best_rmse': float(best_rmse),
        'total_epochs': epoch,
        'early_stopped': early_stopping.early_stop if early_stopping else False
    }
    with open(out_dir / 'run_info.json', 'w') as f:
        json.dump(run_info, f, indent=2)

    # Load best model for predictions
    print('\nLoading best model for predictions...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f'  Loaded checkpoint from epoch {checkpoint["epoch"]} (val RMSE: {checkpoint["val_rmse"]:.4f})')

    # QUAL PREDICTIONS
    print('\nPredicting qualifying set...')
    if len(qual_df) == 0:
        print('  Warning: No valid qual samples!')
    else:
        model.eval()
        pred_rows = []
        with torch.no_grad():
            batch = 8192
            for i0 in range(0, len(qual_df), batch):
                chunk = qual_df.iloc[i0:i0+batch]
                users = torch.tensor(chunk['user_idx'].values, dtype=torch.long).to(device)
                items = torch.tensor(chunk['item_idx'].values, dtype=torch.long).to(device)
                p = model(users, items).cpu().numpy()
                for mv, us, pv in zip(chunk['movie_id'].values, chunk['user_id'].values, p):
                    pred_rows.append((int(mv), int(us), float(pv)))

        out_csv = out_dir / 'predictions.csv'
        with open(out_csv, 'w') as f:
            f.write('movie_id,user_id,pred_rating\n')
            for mv, us, pv in pred_rows:
                f.write(f'{mv},{us},{pv}\n')

        print(f'Predictions saved to {out_csv}')
    
    # Finish WandB
    if args.use_wandb:
        wandb.finish()
        print('✓ WandB run finished')


if __name__ == '__main__':
    main()
