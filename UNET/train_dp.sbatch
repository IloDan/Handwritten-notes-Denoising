#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G # Inserisci la quantità di memoria richiesta
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=Visione-dei-calcolatori-UNET
#SBATCH --partition=students-prod
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=246019@studenti.unimore.it
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Attiva l'ambiente conda in cui hai installato PyTorch
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/bin/:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate
export PYTHONPATH='/usr/local/anaconda3/bin/python' # Modifica il percorso Python se necessario

python mean_std.py