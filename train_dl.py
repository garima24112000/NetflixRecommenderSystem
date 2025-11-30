#!/usr/bin/env python3
"""
train.py - Universal trainer for AutoRec, MultVAE, NeuMF with comparison mode.

Usage:
    python train.py --base /path/to/data --model-name neumf
    python train.py --base /path/to/data --model-name all

Notes:
 - AutoRec/MultVAE expect dense user-item vectors. The script will attempt to build
   a full dense matrix if it fits in memory (heuristic). Otherwise it will
   construct dense vectors per-batch from triplets (slower but memory-safe).
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on sys.path so deep_recsys imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from deep_recsys.models_autorec import AutoRec
from deep_recsys.models_multivae import MultVAE
from deep_recsys.models_neumf import NeuMF
from deep_recsys.data import prepare_data


MODEL_REGISTRY = {
    "autorec": AutoRec,
    "multivae": MultVAE,
    "neumf": NeuMF,
}


# ----------------------------
# Small utilities / datasets
# ----------------------------
class TripletRatingsDataset(Dataset):
    """Standard triplet dataset (user_idx, item_idx, rating)."""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.users = df['user_idx'].values.astype(np.int64)
        self.items = df['item_idx'].values.astype(np.int64)
        self.ratings = df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def build_user_item_sparse_map(csv_path: str, n_users: int, n_items: int):
    """Return dictionary mapping user -> list of (item_idx, rating).
    Memory: O(#ratings) for the mapping.
    """
    df = pd.read_csv(csv_path)
    groups = df.groupby('user_idx')
    user_map = {}
    for uid, g in groups:
        user_map[int(uid)] = (g['item_idx'].astype(np.int64).values,
                              g['rating'].astype(np.float32).values)
    return user_map


def dense_vector_from_user_map(user_map_entry: Tuple[np.ndarray, np.ndarray], n_items: int):
    """Given (items, ratings) arrays for one user, produce dense vector."""
    vec = np.zeros(n_items, dtype=np.float32)
    items, ratings = user_map_entry
    vec[items] = ratings
    return vec


def can_build_dense_matrix(n_users: int, n_items: int, max_bytes: int = 2_000_000_000):
    """
    Heuristic: if n_users * n_items * 4 bytes (float32) < max_bytes, allow dense matrix.
    Default max_bytes = 2GB. Adjust as you see fit.
    """
    needed = n_users * n_items * 4
    return needed <= max_bytes


# ----------------------------
# Training/Eval helpers
# ----------------------------
def rmse_numpy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def prepare_auto_dense_matrix(train_csv: str, n_users: int, n_items: int, allow_build_dense: bool):
    """
    Try to build full dense user-item matrix (shape n_users x n_items).
    If not allowed or too large, returns None to indicate per-batch construction should be used.
    """
    if not allow_build_dense:
        return None

    print("Building dense user-item matrix in memory (may be large)...")
    # attempt to build; if MemoryError, return None
    try:
        mat = np.zeros((n_users, n_items), dtype=np.float32)
        df = pd.read_csv(train_csv)
        # assume user_idx and item_idx are zero-based
        mat[df['user_idx'].values.astype(np.int64),
            df['item_idx'].values.astype(np.int64)] = df['rating'].values.astype(np.float32)
        return mat
    except Exception as e:
        print("Could not build dense matrix:", e)
        return None


# ----------------------------
# Train / evaluate a single model
# ----------------------------
def train_single_model(
        model_name: str,
        data_info: Dict,
        out_dir: Path,
        args
) -> Tuple[float, Path]:
    """
    Train one model and return (best_probe_rmse, checkpoint_path)
    """
    assert model_name in MODEL_REGISTRY, f"Unknown model {model_name}"
    n_users = data_info['n_users']
    n_items = data_info['n_items']
    train_csv = data_info['train_csv']
    probe_csv = data_info['probe_csv']
    qual_csv = data_info['qual_csv']

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else
                          ('mps' if (args.device == 'mps' and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()) else 'cpu'))
    print(f"[{model_name}] Using device: {device}")

    # instantiate model with sensible defaults depending on model type
    if model_name == 'autorec':
        model = AutoRec(n_items=n_items, hidden_dim=args.hidden_dim)
        # AutoRec expects a dense vector per user -> training logic below handles it
    elif model_name == 'multivae':
        model = MultVAE(n_items=n_items, hidden_dim=args.mvae_hidden, latent_dim=args.mvae_latent)
    elif model_name == 'neumf':
        model = NeuMF(n_users=n_users, n_items=n_items, emb_dim=args.embedding_dim, mlp_layers=args.mlp_layers)
    else:
        raise RuntimeError("unreachable")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # checkpoint paths per-model
    ckpt_path = out_dir / f'checkpoint_{model_name}.pth'
    best_rmse = float('inf')
    start_epoch = 1

    # termination flag
    terminate_now = {'flag': False}
    def _handler(signum, frame):
        terminate_now['flag'] = True
    signal.signal(signal.SIGTERM, _handler)

    # attempt to build dense matrix for AutoRec/MultVAE if feasible
    dense_user_item = None
    if model_name in ('autorec', 'multivae'):
        allow_dense = can_build_dense_matrix(n_users, n_items, max_bytes=args.max_dense_bytes)
        dense_user_item = prepare_auto_dense_matrix(train_csv, n_users, n_items, allow_dense)

        # if dense matrix wasn't built, build a sparse map for on-the-fly densification
        if dense_user_item is None:
            user_map = build_user_item_sparse_map(train_csv, n_users, n_items)
        else:
            user_map = None
    else:
        user_map = None

    # prepare triplet dataset for NeuMF (and used to generate train batches for all)
    triplet_ds = TripletRatingsDataset(train_csv)
    triplet_loader = DataLoader(triplet_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    probe_df = pd.read_csv(probe_csv)
    qual_df = pd.read_csv(qual_csv)

    epochs_since_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        it = triplet_loader
        if args.verbose:
            it = tqdm(triplet_loader, desc=f'[{model_name}] Epoch {epoch}')

        try:
            # MODEL-SPECIFIC TRAINING
            if model_name == 'neumf':
                # standard triplet training
                for u, i, r in it:
                    u = u.to(device).long()
                    i = i.to(device).long()
                    r = r.to(device).float()
                    optimizer.zero_grad()
                    preds = model(u, i).squeeze() if preds := None else model(u, i)
                    loss = criterion(preds, r)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * r.size(0)
                    if terminate_now['flag']:
                        break

            elif model_name == 'autorec':
                # train AutoRec by sampling a batch of users and reconstructing their full item vector
                # we'll create batches using either the dense matrix or user_map
                user_indices = np.arange(n_users)
                if args.shuffle_users:
                    np.random.shuffle(user_indices)
                # create batches of users
                for st in range(0, n_users, args.user_batch_size):
                    batch_users = user_indices[st:st + args.user_batch_size]
                    # build dense batch [B, n_items]
                    if dense_user_item is not None:
                        batch_X = torch.tensor(dense_user_item[batch_users], dtype=torch.float32, device=device)
                    else:
                        # on-the-fly build
                        batch_list = []
                        for uid in batch_users:
                            entry = user_map.get(int(uid), (np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
                            vec = dense_vector_from_user_map(entry, n_items)
                            batch_list.append(vec)
                        batch_X = torch.tensor(np.stack(batch_list, axis=0), dtype=torch.float32, device=device)
                    optimizer.zero_grad()
                    recon = model(batch_X)  # [B, n_items]
                    # compute loss only on observed entries to mimic AutoRec setting
                    mask = (batch_X != 0).float()
                    # avoid zero-mask rows
                    denom = mask.sum(dim=1)
                    # if denom == 0, skip
                    valid = denom > 0
                    if valid.sum() == 0:
                        continue
                    obs_loss = ((recon - batch_X) ** 2) * mask
                    per_user_loss = obs_loss.sum(dim=1) / torch.clamp(denom, min=1.0)
                    loss = per_user_loss[valid].mean()
                    loss.backward()
                    optimizer.step()
                    running_loss += float(per_user_loss.mean().item()) * batch_users.shape[0]
                    if terminate_now['flag']:
                        break

            elif model_name == 'multivae':
                # MultVAE expects input vectors (e.g., counts or normalized) per user
                user_indices = np.arange(n_users)
                if args.shuffle_users:
                    np.random.shuffle(user_indices)
                for st in range(0, n_users, args.user_batch_size):
                    batch_users = user_indices[st:st + args.user_batch_size]
                    if dense_user_item is not None:
                        batch_X = torch.tensor(dense_user_item[batch_users], dtype=torch.float32, device=device)
                    else:
                        batch_list = []
                        for uid in batch_users:
                            entry = user_map.get(int(uid), (np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
                            vec = dense_vector_from_user_map(entry, n_items)
                            batch_list.append(vec)
                        batch_X = torch.tensor(np.stack(batch_list, axis=0), dtype=torch.float32, device=device)

                    # optional normalization for MultVAE (common practice)
                    if args.mvae_normalize:
                        row_sums = batch_X.sum(dim=1, keepdim=True)
                        row_sums[row_sums == 0] = 1.0
                        batch_in = batch_X / row_sums
                    else:
                        batch_in = batch_X

                    optimizer.zero_grad()
                    logits, mu, logvar = model(batch_in)
                    # reconstruction loss: -sum(x * log_softmax(logits))
                    recon_loss = -(batch_in * nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
                    # KL
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    loss = recon_loss + args.mvae_beta * kl
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * batch_users.shape[0]
                    if terminate_now['flag']:
                        break

            else:
                raise RuntimeError("Unknown model during training")

        except KeyboardInterrupt:
            print(f"\n[{model_name}] Training interrupted by user. Saving checkpoint.")
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_rmse': best_rmse,
                        'epoch': epoch}, ckpt_path)
            return best_rmse, ckpt_path

        # epoch stats
        epoch_loss = running_loss / max(1, (n_users if model_name in ('autorec', 'multivae') else len(triplet_ds)))
        if args.verbose:
            print(f"[{model_name}] Epoch {epoch} avg loss: {epoch_loss:.6f}")

        # evaluate on probe
        cur_rmse = evaluate_model_on_probe(model, model_name, probe_df, device, n_items, user_map, dense_user_item, args)
        if args.verbose:
            print(f"[{model_name}] Probe RMSE after epoch {epoch}: {cur_rmse:.6f}")

        # early stopping and checkpointing
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            epochs_since_improve = 0
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_rmse': best_rmse,
                        'epoch': epoch}, ckpt_path)
            if args.verbose:
                print(f"[{model_name}] Saved checkpoint to {ckpt_path}")
        else:
            epochs_since_improve += 1
            if args.verbose:
                print(f"[{model_name}] No improvement for {epochs_since_improve} epoch(s)")
            if epochs_since_improve > args.patience:
                if args.verbose:
                    print(f"[{model_name}] Early stopping triggered (patience={args.patience})")
                break

        if terminate_now['flag']:
            if args.verbose:
                print(f"[{model_name}] External termination requested; saving checkpoint.")
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_rmse': best_rmse,
                        'epoch': epoch}, ckpt_path)
            break

    return best_rmse, ckpt_path


def evaluate_model_on_probe(model, model_name: str, probe_df: pd.DataFrame, device: torch.device,
                            n_items: int, user_map, dense_user_item, args) -> float:
    """Evaluate the given model on the probe dataframe. Returns RMSE (float)."""
    model.eval()
    preds = []
    labels = probe_df['rating'].values.astype(np.float32)

    if model_name == 'neumf':
        users = torch.tensor(probe_df['user_idx'].values, dtype=torch.long, device=device)
        items = torch.tensor(probe_df['item_idx'].values, dtype=torch.long, device=device)
        batch = args.eval_batch
        for i0 in range(0, len(users), batch):
            up = users[i0:i0 + batch]
            ip = items[i0:i0 + batch]
            with torch.no_grad():
                p = model(up, ip).cpu().numpy()
            preds.append(p)
    elif model_name == 'autorec':
        # We need per-user reconstructions for probe users
        batch = args.eval_batch
        for i0 in range(0, len(probe_df), batch):
            chunk = probe_df.iloc[i0:i0 + batch]
            out_ps = []
            for uid, iid in zip(chunk['user_idx'].values.astype(np.int64), chunk['item_idx'].values.astype(np.int64)):
                # build dense vector for this user
                if dense_user_item is not None:
                    user_vec = torch.tensor(dense_user_item[uid:uid + 1], dtype=torch.float32, device=device)
                else:
                    entry = user_map.get(int(uid), (np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
                    vec = dense_vector_from_user_map(entry, n_items)
                    user_vec = torch.tensor(vec[None, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    recon = model(user_vec).cpu().numpy()[0]
                out_ps.append(float(recon[iid]))
            preds.append(np.array(out_ps))
    elif model_name == 'multivae':
        # MultVAE needs user input; compute reconstruction scores per user encountered in probe (cache per-uid)
        cache = {}
        batch = args.eval_batch
        for i0 in range(0, len(probe_df), batch):
            chunk = probe_df.iloc[i0:i0 + batch]
            out_ps = []
            for uid, iid in zip(chunk['user_idx'].values.astype(np.int64), chunk['item_idx'].values.astype(np.int64)):
                if uid not in cache:
                    if dense_user_item is not None:
                        user_vec = torch.tensor(dense_user_item[uid:uid + 1], dtype=torch.float32, device=device)
                    else:
                        entry = user_map.get(int(uid), (np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
                        vec = dense_vector_from_user_map(entry, n_items)
                        user_vec = torch.tensor(vec[None, :], dtype=torch.float32, device=device)
                    if args.mvae_normalize:
                        s = user_vec.sum(dim=1, keepdim=True)
                        s[s == 0] = 1.0
                        user_in = user_vec / s
                    else:
                        user_in = user_vec
                    with torch.no_grad():
                        logits, mu, logvar = model(user_in)
                        probs = nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                    cache[uid] = probs
                out_ps.append(float(cache[uid][iid]))
            preds.append(np.array(out_ps))
    else:
        raise RuntimeError("Unknown model for evaluation")

    preds = np.concatenate(preds)
    return rmse_numpy(preds, labels)


def generate_predictions_from_checkpoint(model_name: str, ckpt_path: Path, data_info: Dict, args):
    """Load checkpoint and generate qualifying predictions CSV."""
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    n_items = data_info['n_items']
    qual_df = pd.read_csv(data_info['qual_csv'])

    # re-instantiating model to load weights
    if model_name == 'autorec':
        model = AutoRec(n_items=n_items, hidden_dim=args.hidden_dim)
    elif model_name == 'multivae':
        model = MultVAE(n_items=n_items, hidden_dim=args.mvae_hidden, latent_dim=args.mvae_latent)
    elif model_name == 'neumf':
        model = NeuMF(n_users=data_info['n_users'], n_items=n_items, emb_dim=args.embedding_dim, mlp_layers=args.mlp_layers)
    else:
        raise RuntimeError("Unknown model")

    ck = torch.load(str(ckpt_path), map_location='cpu')
    model.load_state_dict(ck['model_state'])
    model.to(device)
    model.eval()

    # for AutoRec/MultVAE we need user input; for NeuMF we use (u,i)
    pred_rows = []
    # For AutoRec/MultVAE we need per-user dense input. Attempt to build a dense matrix
    # or a sparse user_map from the training CSV so we can create realistic predictions.
    train_csv = data_info.get('train_csv')
    dense_user_item = None
    user_map = None
    if model_name in ('autorec', 'multivae') and train_csv:
        allow_dense = can_build_dense_matrix(data_info['n_users'], data_info['n_items'], max_bytes=args.max_dense_bytes)
        dense_user_item = prepare_auto_dense_matrix(train_csv, data_info['n_users'], data_info['n_items'], allow_dense)
        if dense_user_item is None:
            # fall back to streaming sparse map
            user_map = build_user_item_sparse_map(train_csv, data_info['n_users'], data_info['n_items'])

    with torch.no_grad():
        batch = args.eval_batch
        # cache reconstructions for AutoRec/MultVAE to avoid repeated forward passes per user
        recon_cache = {}
        for i0 in range(0, len(qual_df), batch):
            chunk = qual_df.iloc[i0:i0+batch]
            if model_name == 'neumf':
                users = torch.tensor(chunk['user_idx'].values, dtype=torch.long, device=device)
                items = torch.tensor(chunk['item_idx'].values, dtype=torch.long, device=device)
                p = model(users, items).cpu().numpy()
                for mv, us, pv in zip(chunk['movie_id'].values, chunk['user_id'].values, p):
                    pred_rows.append((int(mv), int(us), float(pv)))
            else:
                # AutoRec / MultVAE: for each unique user in chunk, compute reconstruction once
                uids = chunk['user_idx'].values.astype(np.int64)
                iids = chunk['item_idx'].values.astype(np.int64)
                for uid, iid, mv, us in zip(uids, iids, chunk['movie_id'].values, chunk['user_id'].values):
                    if int(uid) in recon_cache:
                        recon = recon_cache[int(uid)]
                    else:
                        # build user input vector
                        if dense_user_item is not None:
                            user_vec = torch.tensor(dense_user_item[int(uid):int(uid)+1], dtype=torch.float32, device=device)
                        else:
                            entry = user_map.get(int(uid), (np.array([], dtype=np.int64), np.array([], dtype=np.float32))) if user_map is not None else (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
                            vec = dense_vector_from_user_map(entry, n_items)
                            user_vec = torch.tensor(vec[None, :], dtype=torch.float32, device=device)
                        if model_name == 'multivae' and args.mvae_normalize:
                            s = user_vec.sum(dim=1, keepdim=True)
                            s[s == 0] = 1.0
                            user_in = user_vec / s
                        else:
                            user_in = user_vec
                        # forward pass
                        out = model(user_in)
                        if model_name == 'multivae':
                            logits = out[0] if isinstance(out, tuple) else out
                            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                            recon = probs
                        else:
                            # AutoRec returns reconstruction directly
                            recon = out.cpu().numpy()[0]
                        recon_cache[int(uid)] = recon
                    # get prediction for item index
                    pred_val = float(recon[int(iid)]) if 0 <= int(iid) < len(recon) else float(np.mean(recon))
                    pred_rows.append((int(mv), int(us), pred_val))
    out_csv = Path(args.out_dir) / f'predictions_{model_name}.csv'
    with open(out_csv, 'w') as f:
        f.write('movie_id,user_id,pred_rating\n')
        for mv, us, pv in pred_rows:
            f.write(f'{mv},{us},{pv}\n')
    print(f"[{model_name}] Predictions saved to {out_csv}")


# ----------------------------
# CLI and main orchestration
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='Path to raw data directory (input to prepare_data)')
    p.add_argument('--out-dir', default='deep_recsys/output')
    p.add_argument('--model-name', choices=['autorec', 'multivae', 'neumf', 'all'], default='neumf')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--user-batch-size', type=int, default=512, dest='user_batch_size',
                   help='Batch size in users when training AutoRec/MultVAE (number of users per step)')
    p.add_argument('--embedding-dim', type=int, default=16)
    p.add_argument('--mlp-layers', nargs='+', type=int, default=[64, 32])
    p.add_argument('--hidden-dim', type=int, default=512, help='AutoRec hidden dim')
    p.add_argument('--mvae-hidden', type=int, default=600)
    p.add_argument('--mvae-latent', type=int, default=200, dest='mvae_latent')
    p.add_argument('--mvae-beta', type=float, default=0.2, dest='mvae_beta')
    p.add_argument('--mvae-normalize', action='store_true', dest='mvae_normalize',
                   help='Normalize rows for MultVAE input (common practice)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default='cuda')
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample-rows', type=int, default=0,
                   help='If >0, load only this many rows from the training parquet for quick tests')
    p.add_argument('--shuffle-users', action='store_true', help='Shuffle user order for AutoRec/MultVAE epochs')
    p.add_argument('--eval-batch', type=int, default=10000)
    p.add_argument('--max-dense-bytes', type=int, default=2_000_000_000,
                   dest='max_dense_bytes', help='Max bytes allowed to attempt building dense user-item matrix (float32 bytes)')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Preparing data...")
    data_info = prepare_data(args.base, str(out_dir), sample_rows=getattr(args, 'sample_rows', 0))
    print("Data info:", data_info)

    models_to_run = [args.model_name] if args.model_name != 'all' else ['autorec', 'multivae', 'neumf']
    results = {}
    ckpts = {}

    for m in models_to_run:
        print(f"=== Training model: {m} ===")
        best_rmse, ckpt = train_single_model(m, data_info, out_dir, args)
        results[m] = best_rmse
        ckpts[m] = ckpt
        print(f"=== {m} done: best_probe_rmse={best_rmse:.6f}, checkpoint={ckpt} ===")

    # Comparison reporting
    if len(models_to_run) > 1:
        print("=== Model comparison (probe RMSE) ===")
        for k, v in results.items():
            print(f"{k}: {v:.6f}")
        best_model = min(results, key=results.get)
        print("Best model:", best_model)
    else:
        best_model = models_to_run[0]

    # generate predictions using the best model's checkpoint
    print(f"Generating predictions for best model: {best_model}")
    generate_predictions_from_checkpoint(best_model, ckpts[best_model], data_info, args)

    # write run info
    run_info = {
        'timestamp': time.asctime(),
        'args': vars(args),
        'results': results
    }
    with open(out_dir / 'run_info.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    print("Finished.")

if __name__ == '__main__':
    main()
