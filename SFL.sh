#!/bin/bash
#SBATCH --job-name=SFL
#SBATCH --output=sfl.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate


python main_oneDS.py --Federated_mode SFL