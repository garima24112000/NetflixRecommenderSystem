#!/bin/bash
#SBATCH --job-name=build_features
#SBATCH --partition=extended-40core-shared
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/build_features.out

module purge
module load anaconda/3-new
source activate netflix_env

cd /gpfs/projects/AMS598/class2025/Kumari_Manasa/NetflixRecommenderSystemAMS598

python src/data_prep/build_features.py
