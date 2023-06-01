import os
import argparse
import random
import copy

import torch
from pathlib import Path
import pandas as pd

import setupGC
from training import *
from analyze_dataset import structure_sim
from client import Client_GC
from utils import cross_res_analyze
from graph_utils import normalize
from data_utils import gcfl_param

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
set_seed(0)

def process_sfl(clients, server,args):
    # structure federated learning based on the gradients of the model
    allAccs = run_sfl(clients, server, args.num_rounds, args.local_epoch, args)
    #allAccs = run_tosfl(clients, server, args.num_rounds, args.local_epoch, args)
    #allAccs = run_sfl(clients, server, args.num_rounds, 5, args)

    return allAccs

def process_tosfl(clients, server, args):
    # strcuture federated learning based on the parameters of the model
    allAccs = run_tosfl(clients, server, args.num_rounds, args.local_epoch, args)
    
    return allAccs

def process_bisfl(clients, server, args):
    allAccs = run_bisfl(clients, server, args.num_rounds, args.local_epoch, args)

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

def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    allAccs = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)

    return allAccs
    '''
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")
    '''

def process_gcfl(clients, server, args):

    allAccs = run_gcfl(clients, server, args.num_rounds, args.local_epoch, args.epsilon1, args.epsilon2)
    
    return allAccs

def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}.csv')

    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcflplusdWs(clients, server,args):

    allAccs = run_gcflplus_dWs(clients, server, args.num_rounds, args.local_epoch, args.epsilon1, args.epsilon2, args.seq_length, args.standardize)
    return allAccs

parser = argparse.ArgumentParser()

parser.add_argument('--setting',type = str, default = 'multi',
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
                        type=int, default=0)
# input and output
parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
parser.add_argument('--outpath', type=str, default='./output')
parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
# data process
parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default = 'molecules')
parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
parser.add_argument('--fold_num', type = int, default = 5,
                    help = 'the k for k-fold cross-validation')
# server & local model sharing options
parser.add_argument('--server_sharing',type = str, default = 'center',
                        choices = ['center','full'])
# GCFL parameters
parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.01)
parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)

# SFL parameters
parser.add_argument('--gen_mode', type = str, default = 'GAE',
                    help = 'the type of graph generator')
parser.add_argument('--glr', type = float, default = 0.001,
                    help = 'the learning rate of graph generator')
parser.add_argument('--gweight_decay', type = float, default = 5e-4,
                    help = 'the weight decay of graph generator')
parser.add_argument('--mask_ratio', type = float, default = 0.1)
parser.add_argument('--compress_mode', type = str, default = 'shape',
                    help = 'the parameter compression model',
                    choices = ['continous','discrete','shape'])
parser.add_argument('--compress_dim', type = int, default = 1000)
parser.add_argument('--gc_epoch', type = int, default = 200,
                    help = 'the epoch for training ')
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--serveralpha', type = float, default = 1,
                    help = 'server prop alpha')
parser.add_argument('--serverbeta',type = float, default = 0.1,
                    help = 'parameter replace rate')
parser.add_argument('--interval', type = int, default = 1,
                    help = 'the client graph update interval')
parser.add_argument('--mix_rate',type = float,default = 0)

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

# choose federated parameters
parser.add_argument('--Federated_mode', type = str, default ='SFL',
                        choices = ['Selftraining','FedAvg','FedProx','SFL','biSFL','toSFL','GCFL'])
parser.add_argument('--initial_graph', type = str, default = 'sim',
                        choices = ['degree_disb','triangle_disb','distance','hop2_disb','uniform','sim'])
parser.add_argument('--graph_eps', type = float, default = 0.3,
                        help = 'the eps term for initial client graph normalization')
parser.add_argument('--para_choice', type = str, default = 'embed',
                        choices = ['param','embed','ans'])
parser.add_argument('--input_choice', type = str, default = 'diff',
                        choices = ['whole','gradient','seq','diff'])
parser.add_argument('--diff_rate',type = float, default = 0.95,
                    help = 'the remove rate of mean value')
parser.add_argument('--timelen', type = int, default = 20)
parser.add_argument('--num_splits', type = int, default = 1)

# update model sharing mechanism
parser.add_argument('--sharing_mode', type = str, default = 'gradient',
                        choices = ['gradient','total'])

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

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

seed_dataSplit = 123

# set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def training_round(args):

    #splitedData, df_stats = setupGC.prepareData_multiDS(args.datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    splitedData = setupGC.prepareData_multiDS(args,seed=seed_dataSplit)
    #splitedData = setupGC.prepareData_fakeDS(args,seed = seed_dataSplit)
    #property_report = setupGC.property_counts(args,seed=None)
    print("Done")
    
    
    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")
    

    idx_clients = copy.deepcopy(init_clients)
    cross_results,cross_A = [],[]
    for idx in range(5):

        # split the train and test dataset
        for client in idx_clients:
            assert isinstance(client,Client_GC)
            struc_feature = client.split_traintest(idx,args.batch_size,args)
        
        if args.Federated_mode == 'SFL':
            res,avgA = process_sfl(copy.deepcopy(idx_clients), copy.deepcopy(init_server), args)
            cross_A.append(avgA)
        elif args.Federated_mode == 'biSFL':
            res = process_bisfl(copy.deepcopy(idx_clients), copy.deepcopy(init_server), args)
        elif args.Federated_mode == 'toSFL':
            res = process_tosfl(copy.deepcopy(idx_clients),copy.deepcopy(init_server), args)
        elif args.Federated_mode == 'Selftraining':
            res = process_selftrain(clients=copy.deepcopy(idx_clients), server=copy.deepcopy(init_server), args = args)
        elif args.Federated_mode == 'FedAvg':
            res = process_fedavg(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
        elif args.Federated_mode == 'FedProx':
            res = process_fedprox(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server), mu = 0.01)
        elif args.Federated_mode == 'GCFL':
            res = process_gcflplusdWs(clients = copy.deepcopy(idx_clients), server = copy.deepcopy(init_server),args = args)
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

    multi_results,multi_A = [],[]

    for r in range(rounds):
        res,resA = training_round(args)
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
    multi_rounds_traing(1,args)
    #args.Federated_mode = 'Selftraining'
    #multi_rounds_traing(5,args)
    #args.Federated_mode = 'FedAvg'
    #multi_rounds_traing(5,args)