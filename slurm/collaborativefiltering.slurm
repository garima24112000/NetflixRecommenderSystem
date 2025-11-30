#!/usr/bin/env bash
#SBATCH --job-name=netflix_cf
#SBATCH --output=netflix.out
#SBATCH --error=netflix.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=short-28core
#SBATCH --mem-per-cpu=8G   

module purge
module load anaconda/3-new
conda activate netflix_env

# Use Java 17 from the conda env
export JAVA_HOME=$(dirname "$(dirname "$(which java)")")
export PATH="$JAVA_HOME/bin:$PATH"

# Spark / Java heap settings (stay well under 64GB node limit)
export SPARK_DRIVER_MEMORY=40g
export SPARK_EXECUTOR_MEMORY=40g
export _JAVA_OPTIONS="-Xms40G -Xmx40G"

cd /gpfs/projects/AMS598/class2025/Shaikh_Tasfia/ams598_netflixrecsys/collaborative_filtering

python netflix.py
