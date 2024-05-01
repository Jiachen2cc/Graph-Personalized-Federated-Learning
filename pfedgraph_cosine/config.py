import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default='gin', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='NCI1', help='dataset used for training')
    parser.add_argument('--data_group', type = str,default = 'NCI1')
    parser.add_argument('--datapath', type = str,default= './data')
    parser.add_argument('--convert_x', type = bool,default = False)
    
    parser.add_argument('--Federated_mode', type = str, default = 'pfedgraph')
    parser.add_argument('--partition', type=str, default='noniid-skew', help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    
    # parameters for local training
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--hidden',type = int, default = 64)
    parser.add_argument('--nlayer',type = int, default = 3)
    parser.add_argument('--dropout',type = float, default = 0.5)
    parser.add_argument('--weight_decay',type = float, default = 5e-4)
    
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--split_way', type = str, default = 'toy',
                    help = ' the split methods for global datasets',choices = ['toy','label_skew','blabel_skew','random'])
    parser.add_argument('--fold_num', type = int,
                        default = 5)
    parser.add_argument('--test_fold', type = int,
                        default = 1)
    parser.add_argument('--repeat_num', type = int,
                        default = 1)
    # toy split
    parser.add_argument('--toy_rate',type = float, default = 0.7,
                    help = 'the rate for label split')
    parser.add_argument('--num_clients',type = int, default = 2,
                    help = 'the number of client')
    parser.add_argument('--num_splits',type = int, default = 5,
                    help = 'the split number of each client dataset')
    # label skew & skew balance
    parser.add_argument('--skew_rate',type = float, default = 1,
                    help = 'the rate for parameterize the Dirichlet distribution')
    parser.add_argument('--comm_round', type=int, default=200, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 2, help='The parameter for the noniid-skew for data partitioning')
    #parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--reg', type=float, default=5e-4, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    # parser.add_argument('--optimizer', type = str, default = 'sgd), help = 'the optimizer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--concen_loss', type=str, default='uniform_norm', choices=['norm', 'uniform_norm'], help='How to measure the modle difference')
    parser.add_argument('--weight_norm', type=str, default='relu', choices=['sum', 'softmax', 'abs', 'relu', 'sigmoid'], help='How to measure the modle difference')
    parser.add_argument('--difference_measure', type=str, default='all', help='How to measure the model difference')
    
    parser.add_argument('--alpha', type=float, default=0.8, help='Hyper-parameter to avoid concentration')
    parser.add_argument('--lam', type=float, default=0.01, help="Hyper-parameter in the objective")

    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)
    
    
    
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #args.num_clients = args.num_clients * args.num_splits
    
    cfg = dict()
    cfg["comm_round"] = args.comm_round
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    cfg["epochs"] = args.epochs
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'yahoo_answers'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    cfg['client_num'] = args.num_clients
    cfg['model_name'] = args.model
    cfg['self_wight'] = 'loss'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args , cfg