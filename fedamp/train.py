import copy
import math
import random
import time
#from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from fedamp.config import get_args,cfg
from fedamp.utils import aggregation_by_graph, update_graph_matrix_neighbor
from setupGC import prepareData_oneDS,setup_devices
from training import analyze_train
#from model import simplecnn, textcnn
#from prepare_data import get_dataloader


def local_train_fedavg(args, round, nets_this_round, cluster_models, train_local_dls, test_dl, best_val_acc_list, best_test_acc_list):
    
    for net_id,net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        #data_distribution = data_distributions[net_id]
        cluster_model = cluster_models[net_id]

        # Set Optimizer
        #if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        '''
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        '''
        criterion = torch.nn.CrossEntropyLoss()
        
        cluster_model.cuda()
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for _,batch in enumerate(iterator):
            batch.cuda()
            optimizer.zero_grad()
            target = batch.y
            out = net(batch)
            loss = net.loss(out,target)
            
            if round > 0:
                for param_p, param in zip(cluster_model.parameters(), net.parameters()):
                    loss += ((cfg['lambda_1'] / 2) * torch.norm((param - param_p)) ** 2)
                
            loss.backward()
            optimizer.step()
        
    #return ans


def process_fedamp(clients, server, args):
    
    #for c in clients:
    #    c.split_traintest(fold_id,args.batch_size,args)
    
    #n_party_per_round = int(args.num_clients * args.sample_fraction)
    #party_list = [i for i in range(args.num_clients)]
    #party_list_rounds = []
    #if n_party_per_round != num_clients:
    #    for i in range(args.num_rounds):
    #        party_list_rounds.append(random.sample(party_list, n_party_per_round))
    #for i in range(args.num_rounds):
    #    party_list_rounds.append(party_list)
    
    train_local_dls = [c.dataLoader['train'] for c in clients]
    test_dl = [c.dataLoader['test'] for c in clients]
    
    global_p = server.model.to('cpu').named_parameters()
    global_p = {k:v for k,v in global_p}
    global_parameters = server.model.to('cpu').state_dict()
    local_models = []
    cluster_models = []
    best_val_acc_list, best_test_acc_list = [[] for i in range(len(clients))],[[] for i in range(len(clients))]
    dw = []
    for i in range(args.num_clients):
        local_models.append(clients[i].model)
        cluster_models.append(copy.deepcopy(clients[i]).model)
        dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    
    graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
    graph_matrix[range(len(local_models)), range(len(local_models))] = 0

    for net in local_models:
        net.load_state_dict(global_parameters)
    
    for net in cluster_models:
        net.load_state_dict(global_parameters)
    local_models = {i:local_models[i] for i in range(len(local_models))}
    for round in range(args.num_rounds):
        
        #party_list_this_round = party_list_rounds[round]
        
        #nets_this_round = {k: local_models[k] for k in party_list_this_round}
        #nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
        
        local_train_fedavg(args,round,local_models,cluster_models,train_local_dls,test_dl,best_val_acc_list,best_test_acc_list)
        for c in clients:
            c.evaluate()
        #total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
        #fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}
        graph_matrix = update_graph_matrix_neighbor(local_models, global_parameters, dw)   # Graph Matrix is not normalized yet
        aggregation_by_graph(graph_matrix, local_models, global_parameters, cluster_models)   # Aggregation weight is normalized here
        
    allAccs = analyze_train(clients,args)
    return allAccs


  
