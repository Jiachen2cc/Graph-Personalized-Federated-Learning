#!/bin/bash
#SBATCH --job-name=sround_test6
#SBATCH --output=sround6.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate

python main_oneDS.py --toy_rate 0.6
