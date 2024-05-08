import copy
import math
import random
import time
#from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from pfedgraph_cosine.config import get_args,cfg
from pfedgraph_cosine.utils import aggregation_by_graph, update_graph_matrix_neighbor,weight_flatten_all
from training import analyze_train
#from model import simplecnn, textcnn
#from prepare_data import get_dataloader
#from attack import *

'''
def compute_acc(net, test_data_loader):
    net.cuda()
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            x, target = batch.cuda(), batch.y.to(dtype=torch.int64).cuda()
            pred = net(x)
            total += batch.num_graphs
            correct += pred.max(dim=1)[1].eq(target).sum().item()
    net.to('cpu')
    return correct / float(total)
'''
def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls):
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        #data_distribution = data_distributions[net_id]
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        if round > 0:
            cluster_model = cluster_models[net_id].cuda()
        
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
                flatten_model = []
                models = net.named_parameters()
                for k,v in models:
                    flatten_model.append(v.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                
                
                loss2 = cfg['lam'] * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        

    

def process_pfedgraph(clients,server,args):
    
    '''
    n_party_per_round = int(num_clients * args.sample_fraction)
    party_list = [i for i in range(num_clients)]
    party_list_rounds = []
    if n_party_per_round != args.num_clients:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
    '''

    #benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
    #benign_client_list = [i for i in range(args.num_clients)]
    #benign_client_list.sort()
    #print(f'>> -------- Benign clients: {benign_client_list} --------')

    #train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)
    train_local_dls = [c.dataLoader['train'] for c in clients]



    #global_model = model(cfg['classes_size'])
    global_p = server.model.to('cpu').named_parameters()
    global_p = {k:v for k,v in global_p}
    global_parameters = server.model.to('cpu').state_dict()
    local_models = []
    dw = []
    for i in range(args.num_clients):
        local_models.append(clients[i].model)
        dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
        #best_val_acc_list.append(0)
        #best_test_acc_list.append(0)

    graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
    graph_matrix[range(len(local_models)), range(len(local_models))] = 0

    for net in local_models:
        net.load_state_dict(global_parameters)


        
    cluster_model_vectors = {}
    for round in range(args.num_rounds):
        #print(weight_flatten_all(global_parameters).shape)
        #print(weight_flatten_all(local_models[0].state_dict()).shape)
        nets_this_round = {k: local_models[k] for k in range(args.num_clients)}
        #nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}

        local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors, train_local_dls)
        for c in clients:
            c.evaluate()
        total_data_points = sum([c.train_size for c in clients])
        fed_avg_freqs = {k: clients[k].train_size/ total_data_points for k in range(args.num_clients)}

        #manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)

        graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, dw, fed_avg_freqs, cfg['alpha'], cfg['difference_measure'])   # Graph Matrix is not normalized yet
        cluster_model_vectors = aggregation_by_graph(graph_matrix, nets_this_round, global_parameters,global_p)                                                    # Aggregation weight is normalized here
    
    allAccs = analyze_train(clients,args)
    return allAccs

'''
if __name__ == '__main__':
    args, cfg = get_args()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    num_clients = args.num_clients*args.num_splits

    splitedData = prepareData_oneDS(args.num_clients,args,seed = args.init_seed)
    clients,server,_ = setup_devices(splitedData, args)
    
    meanacc,stdacc = 0,0
    for i in range(args.repeat_num):
        m,s = train_multi_fold(clients,server,args)
        meanacc,stdacc = meanacc+m,stdacc+s
    meanacc /= args.repeat_num
    stdacc /= args.repeat_num
    print('epxeriment setting')
    print('dataset {}, client number {}, skew rate {}'.format(args.data_group,num_clients,args.skew_rate))
    print('mean client accuracy: {:.4f}'.format(meanacc),
          'accuracy standard deviation: {:.4f}'.format(stdacc))
'''   
    
    
    