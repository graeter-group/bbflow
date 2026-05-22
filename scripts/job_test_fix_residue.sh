#!/bin/bash

#SBATCH --time=0-01:00:00
#SBATCH --partition=genoa-deep.p
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%j
#SBATCH --error=slurm_logs/%j.err
#SBATCH --job-name=bbflow_fix_residue

source ~/.bashrc
eval "$(/hits/sw/mli/seutelf/anaconda3/bin/conda shell.bash hook)"

set -e

conda activate bbflow

srun bash /hits/basement/mli/seutelf/bbflow/scripts/test_fix_residue_cli.sh
