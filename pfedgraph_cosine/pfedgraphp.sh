#!/bin/bash
#SBATCH --job-name=pfedgraphp
#SBATCH --output=pfedgraph_cosine/pfedgraphp.txt
#SBATCH --gres=gpu:1



python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 15 --split_way blabel_skew --skew_rate 1 --test_fold 5 --repeat_num 3
python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 20 --split_way blabel_skew --skew_rate 1 --test_fold 5 --repeat_num 3
python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 25 --split_way blabel_skew --skew_rate 1 --test_fold 5 --repeat_num 3

python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 15 --split_way blabel_skew --skew_rate 5 --test_fold 5 --repeat_num 3
python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 20 --split_way blabel_skew --skew_rate 5 --test_fold 5 --repeat_num 3
python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 25 --split_way blabel_skew --skew_rate 5 --test_fold 5 --repeat_num 3

python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 15 --split_way blabel_skew --skew_rate 10 --test_fold 5 --repeat_num 3
python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 20 --split_way blabel_skew --skew_rate 10 --test_fold 5 --repeat_num 3
python pfedgraph_cosine.py --data_group PROTEINS --Federated_mode pfedgraph --num_clients 1 --num_splits 25 --split_way blabel_skew --skew_rate 10 --test_fold 5 --repeat_num 3