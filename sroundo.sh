#!/bin/bash
#SBATCH --job-name=sround_test5
#SBATCH --output=sround5.txt
#SBATCH --gres=gpu:1

cd /local/scratch/jzhou50/sfl
source /local/scratch/jzhou50/zjc/bin/activate

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 1
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 1
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 1
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 1
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 1

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 2
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 2
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 2
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 2
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 2

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 3
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 3
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 3
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 3
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 3

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 4
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 4
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 4
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 4
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 4

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 5
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 5
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 5
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 5
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 5

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 6
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 6
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 6
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 6
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 6

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 7
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 7
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 7
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 7
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 7

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 8
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 8
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 8
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 8
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 8

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 9
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 9
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 9
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 9
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 9

python main_oneDS.py --Federated_mode Selftraining --toy_rate 0.5 --num_splits 10
python main_oneDS.py --Federated_mode FedAvg --toy_rate 0.5 --num_splits 10
python main_oneDS.py --Federated_mode SFL --toy_rate 0.5 --num_splits 10
python main_oneDS.py --Federated_mode FedProx --toy_rate 0.5 --num_splits 10
python main_oneDS.py --Federated_mode Scaffold --toy_rate 0.5 --num_splits 10