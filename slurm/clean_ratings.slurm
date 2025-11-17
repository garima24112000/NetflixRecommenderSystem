#!/bin/bash
#SBATCH --job-name=clean_ratings
#SBATCH --partition=extended-40core-shared
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/clean_ratings.out

module purge
module load anaconda/3-new

# Activate your Conda env
source activate netflix_env

cd /gpfs/projects/AMS598/class2025/Kumari_Manasa/NetflixRecommenderSystemAMS598

python src/data_prep/clean_ratings.py
