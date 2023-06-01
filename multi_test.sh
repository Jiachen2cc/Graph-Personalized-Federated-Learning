#!/bin/bash
#SBATCH --job-name=seg_test
#SBATCH --output=res_v0
#SBATCH --gres=gpu:1


cd /local/scratch/jzhou50/GCFL-reproduction
source /local/scratch/jzhou50/sfn/bin/activate

python communication.py --Fed_mode self_training --multi_test molecules
python communication.py --Fed_mode Prox --multi_test molecules
python communication.py --Fed_mode Avg --multi_test molecules

python communication.py --Fed_mode SFGL --multi_test molecules

python communication.py --Fed_mode self_training --multi_test small
python communication.py --Fed_mode Prox --multi_test small
python communication.py --Fed_mode Avg --multi_test small

python communication.py --Fed_mode SFGL --multi_test small

python communication.py --Fed_mode self_training --multi_test mix
python communication.py --Fed_mode Prox --multi_test mix
python communication.py --Fed_mode Avg --multi_test mix

python communication.py --Fed_mode SFGL --multi_test mix

python communication.py --Fed_mode self_training --multi_test biochem
python communication.py --Fed_mode Prox --multi_test biochem
python communication.py --Fed_mode Avg --multi_test biochem

python communication.py --Fed_mode SFGL --multi_test biochem

