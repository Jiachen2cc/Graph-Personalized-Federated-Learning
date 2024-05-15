import torch
import numpy as np
from gpfl.initial_graph import *
from gpfl.client_feature import prepare_features
from gpfl.utils import normalize
from clientgraph.graph_cons import graph_constructor
from training import analyze_train
import copy

def prepare_global_initial(clients,args):
    if args.initial_graph == "property":
        return property_graph([c.train_data_property for c in clients])
    elif args.initial_graph == "randomc":
        return random_graph(args.num_clients)
    return None

def round_update(lastA, embedsim, graph_rate):
    
    if lastA is None:
        A = embedsim
    else:
        A = lastA*(1-graph_rate) + embedsim.to(lastA.device)*graph_rate
    
    mask = (A >= 0).float().to(A.device)
    return mask*A

def prepare_fixed_graph(clients,args):
    label_dis = torch.stack([torch.tensor(c.get_labeldis()) for c in clients], dim = 0).float()
    disg = distribution_graph(label_dis)
    return disg
    

def process_gpfl(clients,server,args):
    
    #1 initialization
    [c.download_from_server(args,server) for c in clients]
    init_A, average_A = prepare_global_initial(clients,args), torch.zeros((args.num_clients,args.num_clients))
    graph_batch = random_graphbatch(20,clients[0].data[0].x.shape[1],seed = 0)
    
    # loop over each communication round
    for i in range(1,args.num_rounds+1):
        
        # 1 local train
        for c in clients:
            c.compute_weight_update(args.local_epoch)
        
        embed = server.graph_modelembedding(clients,graph_batch.to(args.device),'sum')
        parameter = [
            {k:copy.deepcopy(c.W[k]) for k in server.W.keys()}
            for c in clients
        ]  
        # 2 prepare client feature initial graph
        rawfeature,feature,sim = prepare_features(embed,parameter,args)
        
        # 2.1 rule-based property selector | round update
        if i == 1:
            init_A = rule_selector(torch.stack([c.train_data_property for c in clients], dim = 0),
                                   sim.detach().cpu())
        else:
            init_A = round_update(A,sim,args.graph_rate)
        init_A = normalize(init_A,'sym').to(args.device)
        # 3 construct client graph
        client_graph_cons = graph_constructor(feature.shape[1],args)
        if args.initial_graph == 'distri':
            A = normalize(prepare_fixed_graph(clients,args).to(args.device),'row')
        else:
            A = client_graph_cons.graph_based_aggregation(feature, init_A)
        
        # 4 graph-guided model aggregation
        [client.reset() for client in clients]
        server.graph_update(clients,
            [{k:copy.deepcopy(client.dW[k]) for k in server.W.keys()}
            for client in clients], 
            A, args)
        # evaluate and record important info
        [client.evaluate() for client in clients]
        average_A += A.cpu()
    # evaluate train stage
    allAccs = analyze_train(clients,args) 
    average_A /= args.num_rounds
    return allAccs, average_A  
    
        
        