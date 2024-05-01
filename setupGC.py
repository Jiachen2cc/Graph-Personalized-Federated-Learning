import random
from random import choices
import numpy as np
import pandas as pd

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from models import GIN, serverGIN,GINclassifier,classiGIN,featureGIN,GINextractor,GIN_dc,serverGIN_dc
from server import Server
from client import Client_GC
from utils import get_stats, split_data, get_numGraphLabels,convert_to_nodeDegreeFeatures,init_structure_encoding
#from argument_setting import args
from analyze_dataset import *

from sklearn.model_selection import StratifiedKFold
from perturbations import *
from functest import arti_datasets,toy_datasets,feature_padding
from data_utils import nofeature_datasets,graph_process,toy_split,load_attr,subchunk_split,show_label_distribution,fix_size_split





import copy

from data_utils import group2datas, data_process, easy_datasets, kfold_split

def cal_num(total,n_split):

    lb = total // n_split

    deta = total - lb * n_split

    prior = [lb for i in range(n_split)]
    if deta == 0:
        return prior
    else:
        for i in range(deta):
            prior[i] = prior[i] + 1
    
    return prior

def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def label_skew(graphs, num_client, seed, alpha=4):

    random.seed(seed)
    np.random.seed(seed)

    labels = np.array([graph.y.item() for graph in graphs])
    # step 1: divide the dataset according to their labels
    num_classes = labels.max() + 1
    class_idx = [np.argwhere(labels == y).flatten()
                 for y in range(num_classes)]

    # step 2: create label distribution for each client
    label_distribution = np.random.dirichlet(
        [alpha]*num_client, num_classes)

    eps = 0.2/(num_client)
    label_distribution[label_distribution < eps] = eps
    for i, frac in enumerate(label_distribution):
        label_distribution[i] = copy.deepcopy(frac / np.sum(frac))

    # step 3: split the dataset to all classes
    client_idx = [[] for _ in range(num_client)]

    for c, fracs in zip(class_idx, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):

            client_idx[i] += [idcs]

    client_idx = [np.concatenate(idcs) for idcs in client_idx]
    # step 4: split graph_chunks
    graph_chunks = []
    for i in range(num_client):
        graph_chunks.append([graphs[idx] for idx in client_idx[i]])

    # step 5: report the split result
    for idcs in client_idx:
        #print(len(idcs))
        each_num = []
        for i in range(num_classes):
            sum = 0
            for idx in idcs:
                if labels[idx] == i:
                    sum += 1
            each_num.append(sum)
        print(each_num)

    return client_idx


def label_skew_balance(graphs, num_client, seed, alpha=4):

    random.seed(seed)
    np.random.seed(seed)

    labels = np.array([graph.y.item() for graph in graphs])
    num_classes = labels.max() + 1

    class_priors = np.random.dirichlet(
        alpha=[alpha] * num_classes, size=num_client)
    
    for i, frac in enumerate(class_priors):

        # print(class_priors)
        prior_cusum = np.cumsum(class_priors, axis=1)
        idx_list = [np.where(labels == i)[0] for i in range(num_classes)]

        # compute the size of each class
        class_amount = [len(idx_list[i]) for i in range(num_classes)]
        
        # compute the size of each local dataset
        client_sample_nums = np.array(cal_num(len(labels), num_client))
        # print(client_sample_nums)

        client_indices = [np.zeros(client_sample_nums[cid]).astype(
            np.int64) for cid in range(num_client)]

        while np.sum(client_sample_nums) != 0:
            curr_cid = np.random.randint(num_client)
            if client_sample_nums[curr_cid] <= 0:
                continue
            client_sample_nums[curr_cid] -= 1
            curr_prior = prior_cusum[curr_cid]
            while True:
                curr_class = np.argmax(np.random.uniform() <= curr_prior)
                if class_amount[curr_class] <= 0:
                    continue
                class_amount[curr_class] -= 1
                client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                    idx_list[curr_class][class_amount[curr_class]]
                break
            
        graph_chunks = []

        for i in range(num_client):
            graph_chunks.append([graphs[idx] for idx in client_indices[i]])
        for client_idx in client_indices:
            label_res = np.zeros(np.max(labels)+1).astype(int)
            for idx in client_idx:
                label_res[int(labels[idx])] += 1
            #print(label_res)
        return [client_indices[i] for i in range(num_client)]

'''
def select_pertur(graphs,num_client,seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    # assign status flag to each client
    # 1 normal
    # 2 node feature Gaussian
    # 3 node feature perm 
    # 4 structure triangle enclose
    fstat = ['Gaussian','perm']
    sstat = ['tri_en']
    status = ['normal','Gaussian','perm','tri_en']
    cstatus = [random.choice(status) for i in range(num_client)]
    
    data_list = []
    for data,state in zip(graphs,cstatus):
        if state in fstat:
            pdata = node_feature_perturbation(data,state,seed)
        elif state in sstat:
            pdata = structure_perturbation(data,state)
        else:
            pdata = data
        data_list.append(pdata)
    
    return data_list,cstatus
    

def feature_pertur_oneDS(data,chunks,chunks_idx,idx,args):

    if args.mix_type == 'attr':
        attrdata = load_attr(args.datapath,data)
        num_node_features = max(num_node_features,attrdata[0].num_node_features)
        if idx == 1:
            attrchunk = [attrdata[idx] for idx in chunks_idx]
            chunks = feature_mix(chunks,args.mix_type,args.fmix_rate,attrchunk)
        else:
            chunks = feature_padding(chunks,num_node_features)
    elif (args.mix_type in ['ones','Gauss']) and idx == 1:
        print('feature mix!')
        chunks = feature_mix(chunks,args.mix_type,args.fmix_rate)

    return chunks
'''    
def prepareData_oneDS(num_client, args, seed=None):
    
    data = args.data_group
    tudataset = data_process(args.datapath, data, args.convert_x)
    graphs = [x for x in tudataset]
    #show_label_distribution(graphs)
    #exit(0)
    #print(graphs[0].x.shape)
    #print("  **", data, len(graphs))

    #graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    #graphs_chunks = label_skew(graphs, num_client, seed, 1)
    #graphs_chunks = label_skew_balance(graphs, num_client, seed,args.label_skew)

    # only split two clients
    num_client *= args.num_splits
    args.num_clients *= args.num_splits
    if args.split_way == 'toy':
        graphs_chunks_idx = toy_split(graphs,rate = args.toy_rate)

        if args.num_splits > 1:
            graphs_chunks_idx = subchunk_split(graphs,graphs_chunks_idx,args.num_splits if num_client > 1 else args.num_splits//2)
            #num_client *= args.num_splits
            #args.num_clients *= args.num_splits

    elif args.split_way == 'label_skew':
        #num_client *= args.num_splits
        graphs_chunks_idx = label_skew(graphs,num_client,seed,args.skew_rate)
        #exit(0)
    
    elif args.split_way == 'blabel_skew':
        #num_client *= args.num_splits
        graphs_chunks_idx = label_skew_balance(graphs,num_client,seed,args.skew_rate)
    
    elif args.split_way == 'fix_num':
        # toy cases for motivation: assume we use 20 client as most
        # first split data into 2 groups
        # Then split group into clients
        max_number = 5
        sample_per_client = len(graphs) // (2*max_number)
        group_size = sample_per_client * max_number
        graphs_chunks_idx = fix_size_split(graphs,group_size,args.toy_rate)
        graphs_chunks_idx = subchunk_split(graphs,graphs_chunks_idx,max_number)
        graphs_chunks_idx = graphs_chunks_idx[0:args.num_splits] + graphs_chunks_idx[max_number:max_number+args.num_splits]
    
        
    
    splitedData = {}
    num_node_features = graphs[0].num_node_features
    '''
    if args.per_type == 'mix':
        per_data,cs = select_pertur(graphs_chunks,num_client,seed)
    graphs_chunks = per_data
    '''
    cs = ['normal' for i in range(num_client)]
    ##----------------------------------------------
    for idx, chunks_idx,status in zip(list(range(num_client)),graphs_chunks_idx,cs):
        #print(len(chunks))
        ds = f'{idx}-{data}-{status}'
        #print(len(chunks_idx))
        chunks = [graphs[idx] for idx in chunks_idx]
        show_label_distribution(chunks)
        #print(idx,len(chunks))
        
        '''
        if args.feature_pertur:
            chunks = feature_pertur_oneDS(data,chunks,chunks_idx,idx,args)
        
        if idx == 1 and args.structure_pertur:
            chunks = edge_noise(chunks,args.prate,args.nrate)
        '''
        #show_label_distribution(chunks)
        
        train_idx_list, test_idx_list = kfold_split(chunks,args.fold_num,seed)
        
        num_graph_labels = get_numGraphLabels(chunks)
        splitedData[ds] = (chunks,{'train':train_idx_list,'test':test_idx_list},
                        num_node_features, num_graph_labels, data)
        
    if args.global_model:
        ds = f'{data}-global_model'
        train_idx_list, test_idx_list = kfold_split(graphs,args.fold_num,seed)
        num_graph_labels = get_numGraphLabels(graphs)
        splitedData[ds] = (graphs,{'train':train_idx_list,'test':test_idx_list},
                        num_node_features, num_graph_labels, data)
        
    return splitedData


def prepareData_multiDS(args,seed=None):
    #assert group in ['molecules', 'molecules_tiny', 'small',
    #                 'mix', "mix_tiny", "biochem", "biochem_tiny"]

    datasets = group2datas[args.data_group]
    splitedData = {}
    for data in datasets:
        tudataset = data_process(args.datapath,data,args.convert_x)
        graphs = [x for x in tudataset]
        #homo_analyze(graphs)
        graphs = graph_process(data,graphs,args)
        #print(len(graphs))

        # cross validation dataset split
        train_idx_list, test_idx_list = kfold_split(graphs,args.fold_num,seed)

        '''
        graphs_val, graphs_test = split_data(
            graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        
        if group.endswith('tiny'):
            graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
            graphs_train, graphs_test = split_data(
                graphs, test=0.2, shuffle=True, seed=seed)
            
            graphs_val, graphs_test = split_data(
                graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        '''
        #graphs_val = copy.deepcopy(graphs_test)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs)

        #dataloader_train = DataLoader(
        #    graphs_train, batch_size=batchSize, shuffle=True)
        #dataloader_val = DataLoader(
        #    graphs_val, batch_size=batchSize, shuffle=True)
        #dataloader_test = DataLoader(
        #    graphs_test, batch_size=batchSize, shuffle=True)

        #splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
        #                     num_node_features, num_graph_labels, len(graphs_train))

        #splitedData[data] = ({'dataset':graphs,'train': train_idx_list, 'test': test_idx_list, 'analysis': graphs_train},
        #                     num_node_features, num_graph_labels, len(graphs_train))
        splitedData[data] = (graphs,{'train': train_idx_list, 'test': test_idx_list},
                             num_node_features, num_graph_labels, data)
        #df = get_stats(df, data, graphs_train,
        #               graphs_val=None, graphs_test=graphs_test)
    return splitedData#, df

def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    sclassifiers,sextracts = [],[]
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        data, split_idx, num_node_features, num_graph_labels, dataset_name = splitedData[ds]
        if args.Federated_mode == 'fedstar':
            data = init_structure_encoding(args,data,args.type_init)
            cmodel_gc = GIN_dc(num_node_features, args.n_se, args.hidden,
                        num_graph_labels, args.nlayer, args.dropout)
        else:
            cmodel_gc = GIN(num_node_features, args.hidden,
                        num_graph_labels, args.nlayer, args.dropout)
        
        #cmodel_gc = GraphCNN(args.num_layers,args.num_mlp_layers,num_node_features,args.hidden_dim,num_graph_labels
        #,args.final_dropout,args.graph_pooling_type,args.neighbor_pooling_type,args.device)
        # optimizer = torch.optim.Adam(cmodel_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters(
        )), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, dataset_name,
                       data, split_idx, optimizer, args))
        sclassifier = GINclassifier(nhid = args.hidden,nclass=num_graph_labels)
        sextract = GINextractor(nfeat = num_node_features, nhid = args.hidden)

        sclassifiers.append(sclassifier)
        sextracts.append(sextract)

    if args.server_sharing == 'center':
        smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    elif args.server_sharing == 'full':
        if args.Federated_mode == 'fedstar':
            smodel = GIN_dc(num_node_features, args.n_se, args.hidden,
                    num_graph_labels, args.nlayer, args.dropout)
        else:
            smodel = GIN(num_node_features, args.hidden,
                    num_graph_labels, args.nlayer, args.dropout)
    elif args.server_sharing == 'center_class':
        smodel = classiGIN(num_node_features, args.hidden,
                    num_graph_labels, args.nlayer, args.dropout)
    elif args.server_sharing == 'feature':
        smodel = featureGIN(num_node_features, args.hidden,
                    num_graph_labels, args.nlayer, args.dropout)

    #smodel = serverGraphCNN(args.num_layers,args.num_mlp_layers,args.hidden_dim,args.learn_eps,args.device)
    server = Server(smodel, sclassifiers, sextracts, args.device)
    return clients, server, idx_clients

def property_counts(args,seed = None):

    p_dict = {}
    datasets = group2datas[args.data_group]

    for data in datasets:
        tudataset = data_process(args.datapath,data,args.convert_x)
        graphs = [x for x in tudataset] 

        #avg_tri = avg_tri_num(graphs)
        avg_node = avg_nodenum(graphs)
        
        #per_graphs = structure_perturbation(graphs,noise_rate = 1)
        '''
        per_avg_tri = avg_tri_num(per_graphs)

        p_dict[data] = {'average triangle num':avg_tri,'average triangle num after triangle enclose':per_avg_tri}
    
        print(data)
        print('average triangle num:{:.4f}'.format(avg_tri))
        print('average triangle num:{:.4f}'.format(per_avg_tri))
        '''
        
        print(data)
        print('average node num:{:.4f}'.format(avg_node))
    
    return p_dict
