#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G 
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_db_1
#SBATCH --partition=students-prod
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=325524@studenti.unimore.it
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Attiva ambiente Conda "base"
conda activate base

# Lancia il programma Python
python eval_db_1.py
