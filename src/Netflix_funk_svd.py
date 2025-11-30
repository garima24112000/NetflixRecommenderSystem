"""
SVD via Stochastic Gradient Descent (FunkSVD) - Complete Pipeline

Features:
1. Numba Acceleration for high performance.
2. Early Stopping (prevents overfitting).
3. Best Model Checkpointing (saves the best epoch, not the last).
4. History Logging (saves RMSE per epoch to CSV).
"""
import numpy as np
import pandas as pd
from numba import jit
from pathlib import Path
from datetime import datetime
import time
import sys

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Hyperparameters
RANK = 100
INITIAL_LR = 0.007
REGULARIZATION = 0.03   # Increased to fight overfitting
EPOCHS = 50
SEED = 42
PATIENCE = 5            # Stop if no improvement for 5 epochs

# Create Dynamic Output Directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"sgd_rank{RANK}_earlystop_{timestamp}"
OUTPUT_DIR = Path(__file__).parent / "Output" / folder_name

def load_data():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_file = DATA_DIR / "ratings_train_no_probe.parquet"
    train_df = pd.read_parquet(train_file)
    probe_file = DATA_DIR / "probe_ratings.parquet"
    probe_df = pd.read_parquet(probe_file)
    qual_file = DATA_DIR / "qualifying_to_predict.parquet"
    qual_df = pd.read_parquet(qual_file)
    return train_df, probe_df, qual_df

def preprocess_ids(train_df, probe_df, qual_df):
    print("\n" + "=" * 60)
    print("PREPROCESSING IDs")
    print("=" * 60)
    unique_users = train_df['user_id'].unique()
    unique_movies = train_df['movie_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
    
    def map_ids(df, col, mapping):
        return df[col].map(mapping).fillna(-1).astype(np.int32)

    train_df['user_idx'] = map_ids(train_df, 'user_id', user_to_idx)
    train_df['movie_idx'] = map_ids(train_df, 'movie_id', movie_to_idx)
    probe_df['user_idx'] = map_ids(probe_df, 'user_id', user_to_idx)
    probe_df['movie_idx'] = map_ids(probe_df, 'movie_id', movie_to_idx)
    qual_df['user_idx'] = map_ids(qual_df, 'user_id', user_to_idx)
    qual_df['movie_idx'] = map_ids(qual_df, 'movie_id', movie_to_idx)

    return train_df, probe_df, qual_df, len(unique_users), len(unique_movies), user_to_idx, movie_to_idx

# ==========================================
# NUMBA FUNCTIONS
# ==========================================
@jit(nopython=True)
def sgd_epoch(user_ids, movie_ids, ratings, P, Q, bu, bi, global_mean, lr, reg):
    for i in range(len(ratings)):
        u = user_ids[i]
        m = movie_ids[i]
        r = ratings[i]
        
        dot_prod = 0.0
        for f in range(P.shape[1]):
            dot_prod += P[u, f] * Q[m, f]
        pred = global_mean + bu[u] + bi[m] + dot_prod
        err = r - pred
        
        bu[u] += lr * (err - reg * bu[u])
        bi[m] += lr * (err - reg * bi[m])
        
        for f in range(P.shape[1]):
            p_uf = P[u, f]
            q_mf = Q[m, f]
            P[u, f] += lr * (err * q_mf - reg * p_uf)
            Q[m, f] += lr * (err * p_uf - reg * q_mf)

@jit(nopython=True)
def compute_rmse(user_ids, movie_ids, ratings, P, Q, bu, bi, global_mean):
    squared_error = 0.0
    n = 0
    for i in range(len(ratings)):
        u = user_ids[i]
        m = movie_ids[i]
        if u == -1 or m == -1: continue
        r = ratings[i]
        dot_prod = 0.0
        for f in range(P.shape[1]):
            dot_prod += P[u, f] * Q[m, f]
        pred = global_mean + bu[u] + bi[m] + dot_prod
        if pred > 5.0: pred = 5.0
        if pred < 1.0: pred = 1.0
        squared_error += (r - pred) ** 2
        n += 1
    if n == 0: return 0.0
    return np.sqrt(squared_error / n)

@jit(nopython=True)
def predict_batch(user_ids, movie_ids, P, Q, bu, bi, global_mean):
    preds = np.zeros(len(user_ids))
    for i in range(len(user_ids)):
        u = user_ids[i]
        m = movie_ids[i]
        if u == -1 or m == -1:
            preds[i] = global_mean
            continue
        dot_prod = 0.0
        for f in range(P.shape[1]):
            dot_prod += P[u, f] * Q[m, f]
        pred = global_mean + bu[u] + bi[m] + dot_prod
        if pred > 5.0: pred = 5.0
        if pred < 1.0: pred = 1.0
        preds[i] = pred
    return preds

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "=" * 60)
    print(f"SGD FUNKSVD | Rank {RANK} | Reg {REGULARIZATION} | Early Stop {PATIENCE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    start_total = time.time()
    train_df, probe_df, qual_df = load_data()
    train_df, probe_df, qual_df, n_users, n_movies, user_map, movie_map = \
        preprocess_ids(train_df, probe_df, qual_df)
    
    print("\n⚙️  Initializing Model Parameters...")
    global_mean = train_df['rating'].mean()
    np.random.seed(SEED)
    
    # Initialize Weights
    P = np.random.normal(0, 0.1, (n_users, RANK))
    Q = np.random.normal(0, 0.1, (n_movies, RANK))
    bu = np.zeros(n_users)
    bi = np.zeros(n_movies)
    
    # Prepare arrays
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_users = train_df['user_idx'].values
    train_movies = train_df['movie_idx'].values
    train_ratings = train_df['rating'].values
    probe_users = probe_df['user_idx'].values
    probe_movies = probe_df['movie_idx'].values
    probe_ratings = probe_df['rating'].values
    
    lr = INITIAL_LR
    
    # --- VARIABLES FOR BEST MODEL SAVING ---
    best_rmse = float('inf')
    best_epoch = -1
    patience_counter = 0
    
    best_P = None
    best_Q = None
    best_bu = None
    best_bi = None
    
    # --- VARIABLE FOR HISTORY LOGGING ---
    history = []
    
    print(f"\n🚀 Starting Training...")
    print("-" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        start_epoch = time.time()
        
        # Train
        sgd_epoch(train_users, train_movies, train_ratings, 
                  P, Q, bu, bi, global_mean, lr, REGULARIZATION)
        
        # Evaluate
        rmse = compute_rmse(probe_users, probe_movies, probe_ratings, 
                            P, Q, bu, bi, global_mean)
        
        elapsed = time.time() - start_epoch
        
        # 1. Log History
        history.append({
            'epoch': epoch + 1,
            'rmse': rmse,
            'elapsed_seconds': elapsed,
            'learning_rate': lr
        })
        
        # 2. Check for Best Model
        if rmse < best_rmse:
            print(f"✅ Epoch {epoch+1:02d}: RMSE = {rmse:.4f} [New Best!] [⏱️ {elapsed:.1f}s]")
            best_rmse = rmse
            best_epoch = epoch + 1
            patience_counter = 0
            # Deep copy the best weights
            best_P = P.copy()
            best_Q = Q.copy()
            best_bu = bu.copy()
            best_bi = bi.copy()
        else:
            print(f"⚠️ Epoch {epoch+1:02d}: RMSE = {rmse:.4f} [No Improve] [⏱️ {elapsed:.1f}s]")
            patience_counter += 1
            
        # 3. Early Stopping Check
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered! No improvement for {PATIENCE} epochs.")
            print(f"   Restoring best model from Epoch {best_epoch} (RMSE: {best_rmse:.4f})")
            break
        
        if epoch > 5:
            lr *= 0.98

    # SAVE HISTORY TO CSV
    history_df = pd.DataFrame(history)
    history_file = OUTPUT_DIR / "training_history.csv"
    history_df.to_csv(history_file, index=False)
    print(f"\n📊 Saved Training History: {history_file}")

    # RESTORE BEST WEIGHTS
    if best_P is not None:
        P, Q, bu, bi = best_P, best_Q, best_bu, best_bi
    
    print("\n" + "=" * 60)
    print(f"GENERATING PREDICTIONS (Using Best Epoch {best_epoch})")
    print("=" * 60)
    
    qual_users = qual_df['user_idx'].values
    qual_movies = qual_df['movie_idx'].values
    preds = predict_batch(qual_users, qual_movies, P, Q, bu, bi, global_mean)
    qual_df['pred_rating'] = preds
    
    pred_file = OUTPUT_DIR / "sgd_predictions_best.csv"
    qual_df[['movie_id', 'user_id', 'pred_rating']].to_csv(pred_file, index=False)
    print(f"💾 Saved Predictions: {pred_file}")
    
    print("\n💾 Saving Best Embeddings...")
    user_emb = pd.DataFrame(P, columns=[f'dim_{i}' for i in range(RANK)])
    idx_to_user = {v: k for k, v in user_map.items()}
    user_emb['user_id'] = [idx_to_user[i] for i in range(n_users)]
    user_emb['bias'] = bu
    user_emb.to_parquet(OUTPUT_DIR / "user_embeddings_sgd.parquet")
    
    movie_emb = pd.DataFrame(Q, columns=[f'dim_{i}' for i in range(RANK)])
    idx_to_movie = {v: k for k, v in movie_map.items()}
    movie_emb['movie_id'] = [idx_to_movie[i] for i in range(n_movies)]
    movie_emb['bias'] = bi
    movie_emb.to_parquet(OUTPUT_DIR / "movie_embeddings_sgd.parquet")
    
    with open(OUTPUT_DIR / "model_metadata.txt", "w") as f:
        f.write(f"Rank: {RANK}\nBest Epoch: {best_epoch}\nBest RMSE: {best_rmse}\n")
        f.write(f"Reg: {REGULARIZATION}\n")

    print(f"✅ Completed in {(time.time()-start_total)/60:.1f} minutes")

if __name__ == "__main__":
    main()
