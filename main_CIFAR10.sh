#!/bin/bash
#SBATCH --job-name=CIFAR101-main
#SBATCH --output=CIFAR1.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate

python main_oneDS.py --data_group CIFAR10 --Federated_mode Selftraining --num_splits 5 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode FedAvg --num_splits 5 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode FedProx --num_splits 5 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode SFL --num_splits 5 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3

python main_oneDS.py --data_group CIFAR10 --Federated_mode Selftraining --num_splits 10 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode FedAvg --num_splits 10 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode FedProx --num_splits 10 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode SFL --num_splits 10 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3

python main_oneDS.py --data_group CIFAR10 --Federated_mode Selftraining --num_splits 15 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode FedAvg --num_splits 15 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode FedProx --num_splits 15 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3
python main_oneDS.py --data_group CIFAR10 --Federated_mode SFL --num_splits 15 --split_way blabel_skew --skew_rate 4 --test_fold 5 --repeat_num 3