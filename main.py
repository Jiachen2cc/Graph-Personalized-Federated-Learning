import os
import argparse
import random
import copy

import torch
import numpy as np
from pathlib import Path
import pandas as pd

import setupGC
from training import *
from analyze_dataset import structure_sim,pg_analysis,label_dis
from client import Client_GC
from utils import cross_res_analyze
from graph_utils import normalize
from data_utils import gcfl_param,device
import time
from dataprocess.setup import SetUp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True

def process_gpfl(clients, server,args):
    # structure federated learning based on the gradients of the model
    allAccs = run_gpfl(clients, server, args.num_rounds, args.local_epoch, args)
    return allAccs

def process_selftrain(clients, server,args):
    #print("Self-training ...")
    allAccs = run_selftrain_GC(clients, server,args)
    
    return allAccs

def process_fedavg(clients, server,args = None):
    #print("\nDone setting up FedAvg devices.")

    #print("Running FedAvg ...")
    #frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    all_Accs = run_fedavg(clients, server, args.num_rounds, args.local_epoch, args, samp=None)
    #all_Accs = run_fedavg(clients, server, args.num_rounds, 5, samp=None)
    return all_Accs
    '''
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")
    '''

def process_fedprox(clients, server, args):
    allAccs = run_fedprox(clients, server, args.num_rounds, args.local_epoch, args, samp=None)

    return allAccs
    '''
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")
    '''

def process_scaffold(clients,server,args):
    allAccs = run_scaffold(clients, server, args.num_rounds,args.local_epoch,args)
    return allAccs

def process_gcfl(clients, server, args):
    allAccs = run_gcfl(clients, server, args.num_rounds, args.local_epoch, args.epsilon1, args.epsilon2,args)
    return allAccs

def process_fedstar(clients,server,args):
    allAccs = run_fedstar(clients, server, args.num_rounds,args.local_epoch,args)
    return allAccs

def process_pfedgraph(clients,server,args):
    allAccs = run_pFedGraph(clients, server, args.num_rounds,args.local_epoch,args)
    return allAccs


def process_gcflplusdWs(clients, server,args):

    allAccs = run_gcflplus_dWs(clients, server, args.num_rounds, args.local_epoch, args.epsilon1, args.epsilon2, args.seq_length, args.standardize,args)
    return allAccs

parser = argparse.ArgumentParser()

parser.add_argument('--setting',type = str, default = 'single',
                        choices = ['single','multi'])
parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
# parameter for local model training
parser.add_argument('--num_repeat', type=int, default= 5,
                        help='number of repeating rounds to simulate;')
parser.add_argument('--num_rounds', type=int, default= 200,
                        help='number of rounds to simulate;')
parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default = 0)
parser.add_argument('--multiseed', help = 'enable multiseed experiment',
                        type = int, default = 0)
# input and output
parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
parser.add_argument('--outpath', type=str, default='./output')
parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
parser.add_argument('--plotpath', type = str, default = './train_plot')
# data process
parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default = 'NCI1')
parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
parser.add_argument('--plot_train',type = int, default = 0,
                    help = 'whether plot train')

# the repeat time for the experiment
parser.add_argument('--fold_num', type = int, default = 5,
                    help = 'the k for k-fold cross-validation')
parser.add_argument('--test_fold', type = int, default = 0,
                    help = 'the number of used fold')
parser.add_argument('--repeat_num',type = int, default = 1,
                    help = 'the repeat times for main experiment')

# server & local model sharing options
parser.add_argument('--server_sharing',type = str, default = 'full',
                        choices = ['center','full'])
# GCFL parameters
parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.01)
parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)

# GPFL parameters
parser.add_argument('--gen_mode', type = str, default = 'GAEAT',
                    help = 'the type of graph generator')
parser.add_argument('--glr', type = float, default = 1e-3,
                    help = 'the learning rate of graph generator')
parser.add_argument('--gweight_decay', type = float, default = 5e-4,
                    help = 'the weight decay of graph generator')
parser.add_argument('--mask_ratio', type = float, default = 0.)
parser.add_argument('--loss_gama', type = float, default = 0.5)
parser.add_argument('--compress_mode', type = str, default = 'shape',
                    help = 'the parameter compression model',
                    choices = ['continous','discrete','shape'])
parser.add_argument('--compress_dim', type = int, default = 100)
parser.add_argument('--gc_epoch', type = int, default = 100,
                    help = 'the epoch for training ')
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--serveralpha', type = float, default = 0.95,
                    help = 'server prop alpha')
parser.add_argument('--serverbeta',type = float, default = 0.1,
                    help = 'parameter replace rate')
parser.add_argument('--interval', type = int, default = 1,
                    help = 'the client graph update interval')
parser.add_argument('--mix_rate',type = float,default = 0)

#FedStar Parameters
parser.add_argument('--n_rw', type=int, default=16,
                        help='Size of position encoding (random walk).')
parser.add_argument('--n_dg', type=int, default=16,
                        help='Size of position encoding (max degree).')
parser.add_argument('--n_ones', type=int, default=16,
                        help='Size of position encoding (ones).')
parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])
# pFedGraph parameters
parser.add_argument('--lam', type = float, default = 0.01,
                    help = 'hyper parameters in local objective')
parser.add_argument('--fedgraphalpha', type = float, default = 0.8)

# dataset difficulty modifications
parser.add_argument('--difficulty', type = int, default = 0,
                    help = 'decide whether to do the difficulty test')
parser.add_argument('--noise_type',type = str, default = 'feature',
                    choices = ['feature','structure','node'])

# feature perturbation configuration
parser.add_argument('--noise_rate',type = float, default = 1,
                    help = 'the perturbated rate for each nodes/edges in a single graph')
parser.add_argument('--fptype',type = str, default = 'add',
                    help = 'the feature perturbation type',
                    choices = ['add','replace'])
parser.add_argument('--SNR',type = float,default = 10,
                    choices = [20,15,10,5,0,-5,-10])
parser.add_argument('--per_edge',type = int, default = 0,
                help = 'decide whether to perturbate edges when performing node feature perturbation')
parser.add_argument('--ex_rate',type = float, default = 0.4,
                help = 'the rates for changing original edges into noise similarity edges')


parser.add_argument('--feature_pertur',type = int, default = 0)
parser.add_argument('--structure_pertur',type = int, default = 0)
#edge perturbation configuration
parser.add_argument('--prate',type = float, default = 1,
                    help = 'the probability for dropping positive edges')
parser.add_argument('--nrate',type = float, default = 0,
                    help = 'the probability for adding negative edges')
parser.add_argument('--fmask',type = str, default = 'None',
                    help = 'the type of feature mask when performing edge perturbation',
                    choices = ['None','zero','one','Gaussian'])

# node downsample configuration
parser.add_argument('--downsample_rate', type = float, default = 1,
                    help = 'the downsample rate for node number')

# data downsample configuration
parser.add_argument('--data_down', type = float, default = 1,
                    help = 'the downsample rate fot the whole dataset')

# dataset heterogeneity modifications
parser.add_argument('--hetero',type = int, default = 0,
                    help = 'choose whether to strengthen the hetergeneity between datasets')
parser.add_argument('--target_dataset', type = str, default = 'IMDB-BINARY')

parser.add_argument('--split_way', type = str, default = 'blabel_skew',
                    help = ' the split methods for global datasets',choices = ['toy','label_skew','blabel_skew',
                                                                               'random','fix_num'])
# toy split
parser.add_argument('--toy_rate',type = float, default = 0.5,
                    help = 'the rate for label split')
parser.add_argument('--num_clients',type = int, default = 1,
                    help = 'the number of client')
parser.add_argument('--num_splits',type = int, default = 25,
                    help = 'the split number of each client dataset')
# label skew & skew balance
parser.add_argument('--skew_rate',type = float, default = 0.5,
                    help = 'the rate for parameterize the Dirichlet distribution')

# choose federated parameters
parser.add_argument('--Federated_mode', type = str, default ='GPFL',
                        choices = ['Selftraining','FedAvg','FedProx','GPFL','GCFL','Scaffold','fedstar','pfedgraph'])
parser.add_argument('--initial_graph', type = str, default = 'property',
                        choices = ['uniform','sim','ans','property','randomc'])
parser.add_argument('--graph_eps', type = float, default = 0.3,
                        help = 'the eps term for initial client graph normalization')
parser.add_argument('--graph_rate', type = float, default = 0.05,
                    help = 'the update rate of the initial graph')
parser.add_argument('--para_choice', type = str, default = 'param',
                        choices = ['param','embed','ans','self','avg','label'])
parser.add_argument('--graph_choice', type = str,default = 'embed',
                    help = 'the choice for parameterize the initial graph')
parser.add_argument('--input_choice', type = str, default = 'diff',
                        choices = ['whole','gradient','seq','diff','ans','normalize'])
parser.add_argument('--diff_rate',type = float, default = 1,
                    help = 'the remove rate of mean value')
parser.add_argument('--timelen', type = int, default = 20)


# update model sharing mechanism
parser.add_argument('--sharing_mode', type = str, default = 'gradient',
                        choices = ['gradient','total','difference','ALA'])

# feature normalization method
parser.add_argument('--norm_way', type = str, default = 'F_norm',
                        choices = ['F_norm','minmax_norm',''])
parser.add_argument('--global_model', type = int, default = 0,
                    help = 'we do not need a global model in cross-dataset setting')
parser.add_argument('--sround', type = int, default = 0,
                    help = 'the start round of normal sharing')
parser.add_argument('--pshare', type = str, default = 'uniform',
                    choices = ['null','uniform','init'],help = 'the sharing method before normal sharing start')

# the mu coefficient for fedprox
parser.add_argument('--mu',type = float, default = 0.01,
                    help = 'the mu coefficient for fedprox')
parser.add_argument('--sigmoid',type = int, default = 1,
                    help = 'the activation function for generated graph')
parser.add_argument('--discrete',type = int, default = 0,
                    help = 'whether to train a discrete or continual client graph')

# control parameter for ablation study
# lu : graph learner      | initialization graph update
# l  : only graph learner | no graph update
# n  : no graph learner   | no update  

parser.add_argument('--ablation',type = str,default = 'lu',
                    choices = ['lu','l','u','n'])
# ALA train
parser.add_argument('--ala_ratio', type = float, default = 0.2,
                    help = 'the ratio for the subdataset')
parser.add_argument('--ala_lr', type = float, default = 1,
                    help = 'the learning rate for ala tuning')
parser.add_argument('--ala_threshold', type = float, default = 1e-2,
                    help = 'the threshold for stopping ala tuning')
parser.add_argument('--ala_round', type = int, default = 5,
                    help = 'the max training round for ALA step')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

seed_dataSplit = 123


#args.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
args.device = device
args.n_se = args.n_rw + args.n_dg

EPS_1 = args.epsilon1
EPS_2 = args.epsilon2

# TODO: change the data input path and output path
outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')

if args.overlap and args.standardize:
    outpath = os.path.join(outbase, f"standardizedDTW/multiDS-overlap")
elif args.overlap:
    outpath = os.path.join(outbase, f"multiDS-overlap")
elif args.standardize:
    outpath = os.path.join(outbase, f"standardizedDTW/multiDS-nonOverlap")
else:
    outpath = os.path.join(outbase, f"multiDS-nonOverlap")
outpath = os.path.join(outpath, args.data_group, f'eps_{EPS_1}_{EPS_2}')
Path(outpath).mkdir(parents=True, exist_ok=True)
print(f"Output Path: {outpath}")

# preparing data
if not args.convert_x:
    """ using original features """
    suffix = ""
    print("Preparing data (original features) ...")
else:
    """ using node degree features """
    suffix = "_degrs"
    print("Preparing data (one-hot degree features) ...")

if args.repeat is not None:
    Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

def preparation(args):

    #splitedData = setupGC.prepareData_oneDS(args.num_clients,args,seed=args.seed)
    splitedData = SetUp(args).splited_graphs
    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)

    return init_clients, init_server
    
def training_round(init_clients,init_server,args):
    
    
    idx_clients = copy.deepcopy(init_clients)
    cross_results,cross_A = [],[]
    tarfold = range(args.fold_num) if args.test_fold == args.fold_num else [args.test_fold] 
    for idx in tarfold:

        # split the train and test dataset
        for client in idx_clients:
            assert isinstance(client,Client_GC)
            struc_feature = client.split_traintest(idx,args.batch_size,args)
        
        #analyze the propeties of all the datasets
        #pg_analysis(idx_clients)
        #exit(0)

        if args.Federated_mode == 'GPFL':
            res,avgA = process_gpfl(copy.deepcopy(idx_clients), copy.deepcopy(init_server), args)
            cross_A.append(avgA)
        elif args.Federated_mode == 'Selftraining':
            res = process_selftrain(clients=copy.deepcopy(idx_clients), server=copy.deepcopy(init_server), args = args)
        elif args.Federated_mode == 'FedAvg':
            res = process_fedavg(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
        elif args.Federated_mode == 'FedProx':
            res = process_fedprox(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server), args = args)
        elif args.Federated_mode == 'Scaffold':
            res = process_scaffold(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
        elif args.Federated_mode == 'GCFL':
            res = process_gcflplusdWs(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
        elif args.Federated_mode == 'fedstar':
            res = process_fedstar(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
        elif args.Federated_mode == 'pfedgraph':
            res = process_pfedgraph(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
        cross_results.append(res)
        #break
        
    return cross_results,cross_A

def report_results(cross_results,args):
    report = cross_res_analyze(cross_results)
    print(args.data_group+'_'+args.Federated_mode+'_'+args.initial_graph+'_'+str(args.num_rounds))
    print(report)
    if args.Federated_mode != 'SFL':
        filename = os.path.join(args.outpath,args.data_group,str(args.num_rounds),args.Federated_mode+'.csv')
    else:
        filename = os.path.join(args.outpath,args.data_group,str(args.num_rounds),args.Federated_mode + '_' + args.initial_graph + '.csv')
    #filename = os.path.join(args.outpath,str(args.num_rounds),args.data_group,args.Federated_mode +'_' +args.initial_graph+'.csv')
    #report.to_csv(filename)

# to ensure the result re-run the algorithm under same arguments and return report
def multi_rounds_traing(rounds,args):
    test_seeds = [0,10,42,111,123] if args.multiseed else [args.seed]
    multi_results,multi_A = [],[]
    
    for seed in test_seeds:
        args.seed = seed
        set_seed(args.seed)
        ics,iss = preparation(args)
        for _ in range(rounds):
            res,resA = training_round(ics,iss,args)
            multi_results.extend(res)
            multi_A.extend(resA)

    if len(multi_A) != 0:
        res = torch.concat([a.unsqueeze(0) for a in multi_A],dim = 0)
        print('average client graph')
        print(torch.mean(res,dim = 0))

    report_results(multi_results,args)

def gcfl_args(args):
    args.Federated_mode = 'GCFL'
    if args.data_group not in gcfl_param.keys():
        print(" have not get parameters setting for this data group yet")
        exit(0)
    args.epsilon1,args.epsilon2 = gcfl_param[args.data_group]
    return args

if __name__ == '__main__':
    
    if args.Federated_mode == 'GCFL':
        args = gcfl_args(args)
    '''
    for sr in [0.5,1,2,4]:
        for ns in [10,15,20]:
            args.split_way = 'blabel_skew'
            args.skew_rate = sr
            args.num_splits = ns
            multi_rounds_traing(5,args)
    '''
    start = time.time()
    multi_rounds_traing(args.repeat_num,args)
    print('training time:{:.4f}'.format(time.time()-start))
    '''
    for s in range(1,11):
        args.num_splits = s
        args.Federated_mode = 'SFL'
        multi_rounds_traing(5,args)
        args.Federated_mode = 'Selftraining'
        multi_rounds_traing(5,args)
        args.Federated_mode = 'FedAvg'
        multi_rounds_traing(5,args)
        args.Federated_mode = 'FedProx'
        multi_rounds_traing(5,args)
    '''
