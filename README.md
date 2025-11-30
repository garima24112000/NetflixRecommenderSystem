# 🎬 Netflix Recommender System
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg" width="250"/>
</p>


End-to-end **Netflix Recommendation System** built on the **Netflix Prize** dataset — covering large-scale data engineering, distributed training (Spark + MPI + SeaWulf HPC), classical collaborative filtering, matrix factorization, and deep learning–based recommenders.

This repository accompanies the **AMS 598 – Netflix Recommendation Systems** group project (Fall 2025, Stony Brook University).

---

## 📌 Project Highlights

- **100M+ ratings** from the Netflix Prize dataset, converted into efficient columnar format.
- **Distributed preprocessing** with **PySpark** (parquet conversion, joins, filtering, feature creation).:contentReference[oaicite:1]{index=1}  
- **Collaborative Filtering with ALS** using Spark MLlib for scalable matrix factorization.:contentReference[oaicite:2]{index=2}  
- **Biased Matrix Factorization (FunkSVD)** with SGD + Numba JIT acceleration for fast training.:contentReference[oaicite:3]{index=3}  
- **Deep Learning (NeuMF)** recommender implemented in PyTorch (GMF + MLP hybrid) with:
  - Gradient accumulation
  - Mixed precision (AMP)
  - GPU-optimized data loading
  - Checkpointing + early stopping:contentReference[oaicite:4]{index=4}  
- Designed to run on **SeaWulf HPC**: Slurm job scripts, MPI-friendly I/O, and large-scale training.

---

## 🗂 Repository Structure

```text
NetflixRecommenderSystem/
│
├── data/                  # Raw / intermediate / processed data (Netflix Prize splits, features, etc.)
│
├── results/               # Saved metrics, plots, and model outputs (RMSE curves, predictions, logs)
│
├── slurm/                 # Slurm job scripts for SeaWulf / HPC runs (ALS, FunkSVD, NeuMF, etc.)
│
├── src/                   # Core library code (utilities, data loaders, helpers)
│
├── collaborativefiltering.py  # Spark ALS pipeline: training + evaluation on large-scale ratings
├── data.py                    # Dataset / preprocessing utilities (parsing, feature creation, splits)
├── memmap_dataset.py          # Memory-mapped dataset for efficient GPU training
├── models_neumf.py            # NeuMF (GMF + MLP hybrid) model definition
├── train_dl.py                # Training loop for NeuMF (AMP, grad accumulation, checkpoints)
└── README.md                  # You are here 🎯
```
## 📁 Project Paths (SeaWulf / GPFS Directories)
Below are the directories contributed by each team member for different components of the Netflix Recommender System project:

### **Preprocessing**
```
/gpfs/projects/AMS598/class2025/Kumari_Manasa/NetflixRecommenderSystemAMS598
```
###  **Collaborative Filtering**
```
/gpfs/projects/AMS598/class2025/Shaikh_Tasfia/ams598_netflixrecsys/collaborative_filtering
```
### **Matrix Factorization (SVD)**
```
/gpfs/projects/AMS598/class2025/Kapoor_Moraish/netflix/matrix_factorization/SVD/
```
### **SVD Outputs**
```
/gpfs/projects/AMS598/class2025/Kapoor_Moraish/netflix/matrix_factorization/SVD/Output
```
### **Deep Learning**
```
/gpfs/projects/AMS598/class2025/Potle_IshaanSantosh/AMS598_FinalProject/deep_recsys
```
### **Deep Learning (NeuMF)**
```
/gpfs/projects/AMS598/class2025/Bandham_Manikanta/final_project
```
