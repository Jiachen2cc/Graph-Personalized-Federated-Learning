import copy
import math
import random
import time
#from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from fedamp.config import cfg
from fedamp.utils import aggregation_by_graph, update_graph_matrix_neighbor
from training import analyze_train
#from model import simplecnn, textcnn
#from prepare_data import get_dataloader


def local_train_fedavg(args, round, nets_this_round, cluster_models, train_local_dls):
    
    for net_id,net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        #data_distribution = data_distributions[net_id]
        cluster_model = cluster_models[net_id]
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        
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
    
    train_local_dls = [c.dataLoader['train'] for c in clients]
    
    global_p = server.model.to('cpu').named_parameters()
    global_p = {k:v for k,v in global_p}
    global_parameters = server.model.to('cpu').state_dict()
    local_models = []
    cluster_models = []
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
        
        local_train_fedavg(args,round,local_models,cluster_models,train_local_dls)
        for c in clients:
            c.evaluate()
        #total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
        #fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}
        graph_matrix = update_graph_matrix_neighbor(local_models, global_parameters, dw)   # Graph Matrix is not normalized yet
        aggregation_by_graph(graph_matrix, local_models, global_parameters, cluster_models)   # Aggregation weight is normalized here
        
    allAccs = analyze_train(clients,args)
    return allAccs


  
