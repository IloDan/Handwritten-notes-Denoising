#!/bin/bash
#SBATCH --job-name=Visione-dei-calcolatori-Distribuita
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=30G
#SBATCH --partition=students-prod
#SBATCH --exclude=aimagelab-srv-00,ajeje,germano,helmut,tafazzi,vegeta,carabbaggio,lurcanio
#SBATCH --array=0-2%1
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR="${nodelist[0]}"
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE=2
export LOCAL_RANK=0
export RANK=0
export CUDA_HOME=/usr/local/cuda
python -m torch.distributed.launch --nproc_per_node=${WORLD_SIZE} train_ddp.py