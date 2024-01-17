#!/bin/bash
#SBATCH --job-name=toy06
#SBATCH --output=supexp/GCFL/toy06.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate

python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 1 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 2 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 3 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 4 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 5 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 6 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 7 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 8 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 9 --test_fold 5 --Federated_mode GCFL
python main.py --data_group NCI1 --split_way toy --toy_rate 0.6  --num_clients 2 --num_splits 10 --test_fold 5 --Federated_mode GCFL