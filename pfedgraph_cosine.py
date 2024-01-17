import copy
import math
import random
import time
#from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from pfedgraph_cosine.config import get_args
from pfedgraph_cosine.utils import aggregation_by_graph, update_graph_matrix_neighbor,weight_flatten_all
from pfedgraph_cosine.setupGC import prepareData_oneDS,setup_devices
#from model import simplecnn, textcnn
#from prepare_data import get_dataloader
#from attack import *

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

def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        #data_distribution = data_distributions[net_id]
        if net_id in benign_client_list:
            pass
            #test_acc = compute_acc(net, test_dl[net_id])
            #val_acc = compute_acc(net, val_local_dls[net_id])
            #personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)
            
            #if val_acc > best_val_acc_list[net_id]:
                #best_val_acc_list[net_id] = val_acc
                #best_test_acc_list[net_id] = personalized_test_acc
            #if test_acc > best_test_acc_list[net_id]:
            #best_test_acc_list[net_id].append(test_acc)
            #print('>> Client {} test1 | (Pre) Test Acc: ({:.5f})'.format(net_id, test_acc))

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if round > 0:
            cluster_model = cluster_models[net_id].cuda()
        
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for _,batch in enumerate(iterator):
            '''
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            '''
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
                
                
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:
            
            test_acc = compute_acc(net, test_dl[net_id])
            best_test_acc_list[net_id].append(test_acc)
            #print('>> Client {} test1 | (Pre) Test Acc: ({:.5f})'.format(net_id, test_acc))
        '''
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        '''
        net.to('cpu')
    tar = np.array(best_test_acc_list)[np.array(benign_client_list)]
    if round > 10:
        tar = tar[:,round-10:round]
    ans = np.max(np.mean(tar,axis = 0))
    return ans


def train_multi_fold(clients,server,args):
    records = []
    if args.test_fold == args.fold_num:
        for i in range(args.test_fold):
            records.append(train_single_fold(copy.deepcopy(clients),
                        copy.deepcopy(server),i,args))
    else:
        records.append(train_single_fold(copy.deepcopy(clients),
                        copy.deepcopy(server),args.test_fold,args))
    records = np.array(records)
    
    return np.mean(records),np.std(records)
    print('mean client accuracy: {:.4f}'.format(np.mean(records)),
          'accuracy standard deviation: {:.4f}'.format(np.std(records)))
    

def train_single_fold(clients,server,fold_id,args):
    
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

    #benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
    benign_client_list = [i for i in range(num_clients)]
    benign_client_list.sort()
    #print(f'>> -------- Benign clients: {benign_client_list} --------')

    #train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)
    train_local_dls = [c.dataLoader['train'] for c in clients]
    test_dl = [c.dataLoader['test'] for c in clients]



    #global_model = model(cfg['classes_size'])
    global_p = server.model.to('cpu').named_parameters()
    global_p = {k:v for k,v in global_p}
    global_parameters = server.model.to('cpu').state_dict()
    local_models = []
    best_val_acc_list, best_test_acc_list = [[] for i in range(len(clients))],[[] for i in range(len(clients))]
    dw = []
    for i in range(num_clients):
        local_models.append(clients[i].model)
        dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
        #best_val_acc_list.append(0)
        #best_test_acc_list.append(0)

    graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
    graph_matrix[range(len(local_models)), range(len(local_models))] = 0

    for net in local_models:
        net.load_state_dict(global_parameters)


        
    cluster_model_vectors = {}
    for round in range(cfg["comm_round"]):
        #print(weight_flatten_all(global_parameters).shape)
        #print(weight_flatten_all(local_models[0].state_dict()).shape)
        party_list_this_round = party_list_rounds[round]
        if args.sample_fraction < 1.0:
            print(f'>> Clients in this round : {party_list_this_round}')
        nets_this_round = {k: local_models[k] for k in party_list_this_round}
        nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}

        mean_personalized_acc = local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors, train_local_dls, None, test_dl, None, best_val_acc_list, best_test_acc_list, benign_client_list)
    
        total_data_points = sum([clients[k].train_size for k in party_list_this_round])
        fed_avg_freqs = {k: clients[k].train_size/ total_data_points for k in party_list_this_round}

        manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)

        graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, dw, fed_avg_freqs, args.alpha, args.difference_measure)   # Graph Matrix is not normalized yet
        #print(graph_matrix)
        cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters,global_p)                                                    # Aggregation weight is normalized here

        #print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
        #print('-'*80)
    
    return mean_personalized_acc


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
    
    
    
    