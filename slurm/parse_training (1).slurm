#!/bin/bash
#SBATCH --job-name=parse_ratings
#SBATCH --partition=extended-40core-shared
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/parse_ratings.out

module load python/3.11.2

cd /gpfs/projects/AMS598/class2025/Kumari_Manasa/NetflixRecommenderSystemAMS598

python src/data_prep/build_ratings_csv.py
