#!/bin/bash
#SBATCH --job-name=fedampp
#SBATCH --output=fedamp/fedampp.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/pFedGraph
source /local/scratch/jzhou50/zjc/bin/activate


python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 15 --split_way blabel_skew --skew_rate 1 --test_fold 5 --repeat_num 3
python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 20 --split_way blabel_skew --skew_rate 1 --test_fold 5 --repeat_num 3
python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 25 --split_way blabel_skew --skew_rate 1 --test_fold 5 --repeat_num 3

python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 15 --split_way blabel_skew --skew_rate 5 --test_fold 5 --repeat_num 3
python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 20 --split_way blabel_skew --skew_rate 5 --test_fold 5 --repeat_num 3
python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 25 --split_way blabel_skew --skew_rate 5 --test_fold 5 --repeat_num 3

python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 15 --split_way blabel_skew --skew_rate 10 --test_fold 5 --repeat_num 3
python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 20 --split_way blabel_skew --skew_rate 10 --test_fold 5 --repeat_num 3
python fedamp.py --data_group PROTEINS --Federated_mode fedamp --num_clients 1 --num_splits 25 --split_way blabel_skew --skew_rate 10 --test_fold 5 --repeat_num 3