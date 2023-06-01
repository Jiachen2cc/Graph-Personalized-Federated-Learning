#!/bin/bash
#SBATCH --job-name=avg_test
#SBATCH --output=avg_test.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate

python main_oneDS.py --initial_graph uniform --num_splits 1 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 2 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 3 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 4 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 5 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 6 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 7 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 8 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 9 --toy_rate 0.5
python main_oneDS.py --initial_graph uniform --num_splits 10 --toy_rate 0.5

python main_oneDS.py --para_choice avg --num_splits 1 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 2 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 3 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 4 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 5 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 6 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 7 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 8 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 9 --toy_rate 0.5
python main_oneDS.py --para_choice avg --num_splits 10 --toy_rate 0.5


