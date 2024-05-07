import argparse
import os
import torch

# hyper parameters for fedamp training
cfg = {
    'lambda_1': 1,
    
}
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")    
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--data_group', type = str,default = 'NCI1')
    parser.add_argument('--datapath', type = str,default= './data')
    parser.add_argument('--convert_x', type = bool,default = False)
    
    parser.add_argument('--Federated_mode', type = str, default = 'fedamp')
    
    parser.add_argument('--partition', type=str, default='noniid-skew', help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    
    parser.add_argument('--batch_size', type=int, default = 128, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default= 1e-3, help='learning rate (default: 0.1)')
    
    parser.add_argument('--hidden',type = int, default = 64)
    parser.add_argument('--nlayer',type = int, default = 3)
    parser.add_argument('--dropout',type = float, default = 0.5)
    parser.add_argument('--weight_decay',type = float, default = 5e-4)
    
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--split_way', type = str, default = 'blabel_skew',
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
    parser.add_argument('--num_clients',type = int, default = 1,
                    help = 'the number of client')
    parser.add_argument('--num_splits',type = int, default = 25,
                    help = 'the split number of each client dataset')
    # label skew & skew balance
    parser.add_argument('--skew_rate',type = float, default = 1,
                    help = 'the rate for parameterize the Dirichlet distribution')
    
    
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default= 200, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 2, help='The parameter for the noniid-skew for data partitioning')   
    parser.add_argument('--reg', type=float, default= 5e-4, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--lambda_1', type=float, default=1.0, help='hyper param used in local training')
    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    cfg = dict()
    cfg["comm_round"] = args.comm_round
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'yahoo_answers'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    cfg['client_num'] = args.n_parties
    cfg['model_name'] = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args , cfg