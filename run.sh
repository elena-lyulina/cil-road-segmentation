#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --account=cil
#SBATCH --job-name="masked-training"

. /etc/profile.d/modules.sh
module add cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roadseg

nvidia-smi

python src/experiments/deepLabv3Plus/main_masked.py
