import copy
import math
import random
import time
#from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from fedamp.config import get_args
from fedamp.utils import aggregation_by_graph, update_graph_matrix_neighbor
from fedamp.setupGC import prepareData_oneDS,setup_devices

#from model import simplecnn, textcnn
#from prepare_data import get_dataloader


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

def local_train_fedavg(args, round, nets_this_round, cluster_models, train_local_dls, test_dl, best_val_acc_list, best_test_acc_list):
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        #data_distribution = data_distributions[net_id]
        cluster_model = cluster_models[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
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
                    loss += ((args.lambda_1 / 2) * torch.norm((param - param_p)) ** 2)
                
            loss.backward()
            optimizer.step()
        
        '''
        val_acc = compute_acc(net, val_local_dls[net_id])
        
        personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

        if val_acc > best_val_acc_list[net_id]:
            best_val_acc_list[net_id] = val_acc
            best_test_acc_list[net_id] = personalized_test_acc
        print('>> Client {} test 2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        '''
        test_acc = compute_acc(net, test_dl[net_id])
        best_test_acc_list[net_id].append(test_acc)
        net.to('cpu')
        cluster_model.to('cpu')
        
    tar = np.array(best_test_acc_list)
    if round > 10:
        tar = tar[:,round-10:round]
    ans = np.max(np.mean(tar,axis = 0))
    
    return ans


def train_single_fold(clients, server, fold_id, args):
    
    for c in clients:
        c.split_traintest(fold_id,args.batch_size,args)
    
    n_party_per_round = int(num_clients * args.sample_fraction)
    party_list = [i for i in range(num_clients)]
    party_list_rounds = []
    if n_party_per_round != num_clients:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
    
    train_local_dls = [c.dataLoader['train'] for c in clients]
    test_dl = [c.dataLoader['test'] for c in clients]
    
    global_p = server.model.to('cpu').named_parameters()
    global_p = {k:v for k,v in global_p}
    global_parameters = server.model.to('cpu').state_dict()
    local_models = []
    cluster_models = []
    best_val_acc_list, best_test_acc_list = [[] for i in range(len(clients))],[[] for i in range(len(clients))]
    dw = []
    for i in range(num_clients):
        local_models.append(clients[i].model)
        cluster_models.append(copy.deepcopy(clients[i]).model)
        dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    
    graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
    graph_matrix[range(len(local_models)), range(len(local_models))] = 0

    for net in local_models:
        net.load_state_dict(global_parameters)
    
    for net in cluster_models:
        net.load_state_dict(global_parameters)
    
    for round in range(cfg["comm_round"]):
        
        party_list_this_round = party_list_rounds[round]
        
        nets_this_round = {k: local_models[k] for k in party_list_this_round}
        #nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
        
        mean_personalized_acc = local_train_fedavg(args,round,nets_this_round,cluster_models,train_local_dls,test_dl,best_val_acc_list,best_test_acc_list)
        
        #total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
        #fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

        graph_matrix = update_graph_matrix_neighbor(nets_this_round, global_parameters, dw)   # Graph Matrix is not normalized yet
        aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters, cluster_models)   # Aggregation weight is normalized here
        
       
    return mean_personalized_acc
        
def train_multi_fold(clients,server,args):
    records = []
    for i in range(args.test_fold):
        records.append(train_single_fold(copy.deepcopy(clients),
                    copy.deepcopy(server),i,args))
    records = np.array(records)
    
    return np.mean(records),np.std(records)
    
if __name__ == '__main__':
    args, cfg = get_args()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    num_clients = args.num_clients * args.num_splits
    
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
    