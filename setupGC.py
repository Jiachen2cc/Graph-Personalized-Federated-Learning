import random
from random import choices
import numpy as np
import pandas as pd

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from models import GIN, GINclassifier,GINextractor,GIN_dc
from server import Server
from client import Client_GC
from utils import get_stats, split_data, get_numGraphLabels,convert_to_nodeDegreeFeatures,init_structure_encoding
#from argument_setting import args
from analyze_dataset import *

from sklearn.model_selection import StratifiedKFold
#from perturbations import *
#from functest import arti_datasets,toy_datasets,feature_padding

import copy


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    sclassifiers,sextracts = [],[]
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        data, split_idx, num_node_features, num_graph_labels, dataset_name, property = splitedData[ds]
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
                       data, property, split_idx, optimizer, args))
        sclassifier = GINclassifier(nhid = args.hidden,nclass=num_graph_labels)
        sextract = GINextractor(nfeat = num_node_features, nhid = args.hidden)

        sclassifiers.append(sclassifier)
        sextracts.append(sextract)

   
    if args.Federated_mode == 'fedstar':
        smodel = GIN_dc(num_node_features, args.n_se, args.hidden,
                num_graph_labels, args.nlayer, args.dropout)
    else:
        smodel = GIN(num_node_features, args.hidden,
                num_graph_labels, args.nlayer, args.dropout)
    #smodel = serverGraphCNN(args.num_layers,args.num_mlp_layers,args.hidden_dim,args.learn_eps,args.device)
    server = Server(smodel, sclassifiers, sextracts, args.device)
    return clients, server, idx_clients
