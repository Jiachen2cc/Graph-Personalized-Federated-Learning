#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=train_plot/plot_sh/exp.txt
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1

source ../zjc/bin/activate

python main.py --data_group NCI1 --Federated_mode Selftraining --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
python main.py --data_group NCI1 --Federated_mode FedAvg --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
python main.py --data_group NCI1 --Federated_mode FedProx --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
python main.py --data_group NCI1 --Federated_mode Scaffold --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
python main.py --data_group NCI1 --Federated_mode GCFL --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
python main.py --data_group NCI1 --Federated_mode fedstar --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
python main.py --data_group NCI1 --Federated_mode GPFL --skew_rate 1 --num_clients 1 --num_splits 25 --split_way blabel_skew
