#!/bin/bash
#SBATCH --job-name=test7
#SBATCH --output=sround7.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate

python main_oneDS.py --toy_rate 0.7