import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from server import Server
import torch
from queue import Queue
import copy
from analyze_client import simi_ana,clf_ana,cos_sim
from analyze_dataset import structure_sim,pg_analysis,label_dis,get_meanfeature
from analyze_graphs import fake_graph
from server import group_sub
from init_client_graph import dist_simi_metrix
from graph_utils import para2metrix,normalize
from functest import random_graphbatch,feature_enlarge
from federated import generate_adj
from torch.utils.tensorboard import SummaryWriter
from analyze_client import graph_diff
from recover_model import GCN_DAE
from client import Client_GC
from utils import graph_truncate,mean_diff,cluster_uniform_graph,random_con_graph
import math
from graph_utils import normalize,update_graph_matrix,graph_aggregate
import torch.nn.functional as F
from tqdm import tqdm
from utils import rule_selector
from clientgraph.graph_cons import graph_constructor


# the length for considering the best accuracy among the last serveral rounds
COUNT_LEN = 10
RECORD_ROUND = [50,100,200]

def interval_print(x,cur_round,interval,desc = None,out = False):

    if not out:
        return
    if (cur_round+1)%interval == 0:
        if desc is not None:
            print(desc)
        print(x)
    
    
def best_final_acc(client_final_accs,length):

    final_avg = np.zeros(length)
    for k in client_final_accs.keys():
        final_avg += np.array(client_final_accs[k])
    final_avg /= len(client_final_accs.keys())

    index = np.argmax(final_avg)

    best_final_acc = {
        k:client_final_accs[k][index] for k in client_final_accs.keys()
    }

    return best_final_acc

def analyze_train(clients,args):

    allAccs,final_accs,final_rocs = {},{},{}
    avg_stats = np.zeros(len(clients[0].eval_stats['testAccs']))
    for client in clients:
        # initialize
        allAccs[client.name] = {}
        stats = np.array(client.eval_stats['testAccs'])
        avg_stats = avg_stats + stats
        #rstats = np.array(client.eval_stats['testRocs'])
        allAccs[client.name]['best_test_acc'] = np.max(stats)
        allAccs[client.name]['final_test_acc'] = stats[-1]
        final_accs[client.name] = stats[-COUNT_LEN:]
        #final_rocs[client.name] = rstats[-(COUNT_LEN+1):-1]
    avg_stats = avg_stats/len(clients)
    if args.plot_train:
        plot_acc(avg_stats,args)
    best_final = best_final_acc(final_accs,COUNT_LEN)
    #best_final_roc = best_final_acc(final_rocs,COUNT_LEN)

    for k in best_final.keys():
        allAccs[k]['final_best_test_acc'] = best_final[k]
        #allAccs[k]['final_best_test_roc'] = best_final_roc[k]

    return allAccs
  
import os
def plot_acc(avg_acc,args): 
     
    file_name = '{}_{}_{}_{}'.format(args.data_group,args.Federated_mode,args.num_clients,args.skew_rate)
    file_name = os.path.join(args.plotpath,file_name)
    np_file = file_name + '.npy'
    plot_file = file_name + '.png'
    
    plt.plot(range(len(avg_acc)),avg_acc)
    plt.xlabel('communication round')
    plt.ylabel('average accuracy')
    plt.title(file_name)
    plt.savefig(plot_file,dpi = 300)
    
    np.save(np_file,avg_acc)
    

def run_selftrain_GC(clients, server, args):
    assert isinstance(server,Server)

    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(args,server)
    
    feature_dim = clients[0].data[0].x.shape[1] if args.setting == 'single' else args.hidden
    '''
    graph_batch = random_graphbatch(20,20,40,feature_dim,seed = 0)
    
    client_similarity = torch.zeros((len(clients),len(clients)))
    embed_similarity = torch.zeros((len(clients),len(clients)))

    oparamsim = torch.zeros((args.num_clients,args.num_clients))
    paramsim = torch.zeros((args.num_clients,args.num_clients))
    '''
    for cround in range(1,args.num_rounds + 1):
        # training
        for client in clients:
            client.local_train(1)
            client.evaluate()

        #analyze the similarity 
        '''
        embed = server.graph_modelembedding(clients,graph_batch.to(args.device))
        params = [{k:copy.deepcopy(client.W[k]) for k in server.W.keys()} for client in clients]
        ofeature,feature,sim = prepare_features(embed,params,cround,args)

        oparamsim += cos_sim(ofeature.detach().cpu())
        paramsim += cos_sim(feature.detach().cpu())
        client_simi = simi_ana(clients,None,server.W.keys())
        embed_simi = cos_sim(embed)
        
        client_similarity += client_simi
        embed_similarity += embed_simi.cpu()
        '''

    
    #print('average similarity')
    #print('average client similarity')
    #print(client_similarity/args.num_rounds)
    #print('average embedding similarity')
    #print(embed_similarity/args.num_rounds)
    allAccs = analyze_train(clients,args)
    '''
    clients[0].download_from_server(clients[1])
    _,acc0 = clients[0].evaluate()
    _,acc1 = clients[1].evaluate()
    print('client_0:{:.4f},client_1:{:.4f}'.format(acc0,acc1))
    '''

    #oparamsim /= args.num_rounds
    #paramsim /= args.num_rounds
    #print(oparamsim-paramsim)

    #np.save('client_feature/vanilla_param.npy',oparamsim.numpy())
    #np.save('client_feature/differential_param.pt',paramsim.numpy())

    return allAccs

def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, args, samp=None, frac=1.0):

    for client in clients:
        client.download_from_server(args,server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        
        for client in clients:
            client.local_train(local_epoch)

        # do not select global model
        if args.global_model:
            selected_clients = clients[:-1]
        else:
            selected_clients = clients
    
        # analyze client.W[k]
        clients_W = []
        for client in clients:
            W = {k: copy.deepcopy(client.W[k]) for k in server.W.keys()}
            clients_W.append(W)
        
        server.aggregate_weights(selected_clients)

        # analyze server.W[k]
        server_W = {k: copy.deepcopy(server.W[k]) for k in server.W.keys()}
    
        #print(simi_ana(clients,server_W,server.W.keys()))
        if args.data_group == 'clf_test':
            print('the similarity of the classifiers')
            clf_ana(clients,server)
        
        for client in selected_clients:
            client.download_from_server(args,server)
        
        for client in clients:
            client.evaluate()



    allAccs = analyze_train(clients,args)
    '''
    clients[0].download_from_server(clients[1])
    _,acc0 = clients[0].evaluate()
    _,acc1 = clients[1].evaluate()
    print('client_0:{:.4f},client_1:{:.4f}'.format(acc0,acc1))
    '''
    return allAccs

def run_fedstar(clients, server: Server, COMMUNICARION_ROUNDS, local_epoch, args, samp = None, frac=1.0):
    
    for client in clients:
        client.download_from_server(args,server)
    
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    
    for c_round in range(1, COMMUNICARION_ROUNDS + 1):
        
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
            
        for client in clients:
            client.local_train(local_epoch)
            
        server.aggregate_weights_se(selected_clients)
        
        for client in selected_clients:
            client.download_from_server(args,server)
        
        for client in clients:
            client.evaluate()
    
    allAccs = analyze_train(clients,args)
    
    return allAccs

def run_pFedGraph(clients, server: Server, COMMUNICATION_ROUNDS, local_epoch, args, samp = None, frac = 1.0):
    for client in clients:
        client.download_from_server(args,server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    if samp == 'random':
        sampling_fn = server.randomSample_clients

    #initialize
    cluster_models = {}
    global_parameters = {k:v for k,v in server.W.items()}
    total_size = sum([client.train_size for client in clients])
    size_freqs = {k:clients[k].train_size/total_size for k in range(len(clients))}
    
    graph_matrix = torch.ones((len(clients),len(clients)))/(len(clients)-1)
    graph_matrix[range(len(clients)),range(len(clients))] = 0
    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        #if (c_round) % 50 == 0:
        #    print(f"  > round {c_round}")
        
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
        for idx,client in enumerate(selected_clients):
            if idx in cluster_models.keys():
                client.local_train_fedgraph(local_epoch, cluster_models[idx],args.lam)
            else:
                client.local_train_fedgraph(local_epoch,lam = args.lam)
            #client.local_train(local_epoch)
                
        graph_matrix = update_graph_matrix(graph_matrix,clients,global_parameters,size_freqs,args.fedgraphalpha)
        cluster_models = graph_aggregate(graph_matrix,clients,args.device)
        
        #server.aggregate_weights(clients)
        # compute client graph and perform graph aggregation
        
        '''
        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(args,server)

            # cache the aggregated weights for next round
            client.cache_weights()
        '''
        
        for client in clients:
            client.evaluate()
            

    allAccs = analyze_train(clients,args)
    
    return allAccs
    

def run_scaffold(clients, server: Server, COMMUNICARION_ROUNDS, local_epoch, args, samp = None, frac = 1.0):

    for client in clients:
        client.download_from_server(args,server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    
    for c_round in range(1, COMMUNICARION_ROUNDS + 1):
        
        for _ in range(local_epoch):
            for client in clients:
                client.local_train(1)
                for k in server.W.keys():
                    client.W[k].data = client.W[k].data - args.lr*(server.control[k].data - client.control[k].data)
                client.evaluate()
                
        #update the client model with control parameters
        server.scaffold_update(clients,local_epoch,args)
    
    allAccs = analyze_train(clients,args)

    return allAccs


def run_fedprox(clients, server, COMMUNICATION_ROUNDS, local_epoch, args, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(args,server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    if samp == 'random':
        sampling_fn = server.randomSample_clients

    #initialize
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        #if (c_round) % 50 == 0:
        #    print(f"  > round {c_round}")
        
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
        for client in selected_clients:
            client.local_train_prox(local_epoch, args.mu)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(args,server)

            # cache the aggregated weights for next round
            client.cache_weights()

        for client in clients:
            client.evaluate()

    allAccs = analyze_train(clients,args)
    
    return allAccs


def pre_finigraph(tag,clients,eps,args):
    # prepare all the initial client graph except for 
    # distance & similarity(these two needed to be computed in each round)

    if tag in ['degree_disb','triangle_disb','hop2_disb']:
        distributions = [client.structure_feature_analysis(tag) for client in clients]
        init_A = structure_sim(distributions,eps)
    elif tag == 'uniform':
        num = len(clients)
        init_A = torch.ones(num,num)/num
    elif tag == 'property':
        init_A = pg_analysis(clients)
        #pass
    elif tag == 'ans':
        init_A = cluster_uniform_graph(args.num_clients*args.num_splits,args.num_splits)
    elif tag == 'randomc':
        init_A = random_con_graph(args.num_clients*args.num_splits)
    else:
        init_A = None
    
    return init_A

def pre_vinigraph(init_A,tag,param,sim,eps,cround,lastA,args):
    
    flex = ['distance']
    # filter last graph to avoid introducing wrong information
    
    if init_A is not None and tag not in flex:
        # adjust the graph update rate according to the current round
        #gr = args.graph_rate*(1-cround/args.num_rounds)
        if 'u' in args.ablation:
            gr = args.graph_rate
            #A = init_A*(1 - args.graph_rate) + lastA.to(init_A.device)*args.graph_rate if lastA is not None else init_A
            A = init_A.to(init_A.device)*(1 - gr) + sim.to(init_A.device)*gr if lastA is not None else init_A
        else:
            A = init_A
        #print(cround)
        #print(torch.sum((A-init_A)**2))
        #A = init_A
    elif tag == 'distance':
        A = dist_simi_metrix(param,eps)
    elif tag == 'sim':
        #print('correct logic!')
        A = sim
    #A = normalize(graph_truncate(A.cpu(),math.ceil((A.shape[1]+1)/2)),'sym')
    #A = graph_truncate(A.cpu(),math.ceil((A.shape[1]+1)/2))
    mask = (A >= 0).float().to(A.device)
    A = mask*A

    #interval_print(A,cround,40,'initial client graph')

    return A

def get_finalgraph(args,feature,init_A,cround,clients):
    nclient = init_A.shape[0]
    if args.para_choice not in ['ans','self','avg']:
        if 'l' in args.ablation:
            _,A,_ = generate_adj(clients,feature,init_A,args,None)
        else:
            A = init_A
    elif args.para_choice == 'ans':
        A = cluster_uniform_graph(nclient,args.num_splits)
    elif args.para_choice == 'self':
        A = torch.eye(nclient)
    elif args.para_choice == 'avg':
        A = torch.ones((nclient,nclient))/nclient
    elif args.para_choice == 'label':
        A = label_dis(clients,args.graph_eps)
    #interval_print(A,cround,40,'result client graph')
    return A

def prepare_features(embed,param,cround,args):
    # param to metrix
    cparam = (para2metrix(param,None,args.compress_dim)).to(args.device)
    ofeature = {'embed':embed,'param':cparam}
    choices = {'embed':embed,'param':cparam}

    # apply transforms to parameters
    if args.input_choice == 'diff':
        choices = {k: F.normalize(mean_diff(f,args.diff_rate),p=2,dim=1) for k,f in choices.items()}
    
    # compute similarity
    sims = {k:cos_sim(f) for k,f in choices.items()}
    for k,s in sims.items():
        interval_print(s,cround,1,'the similarity of '+ k)
    k = args.para_choice if args.para_choice in choices.keys() else 'embed'
    return ofeature[k],choices[k],sims[args.graph_choice]

# try different sharing ways before paramter converge to stable value
def pre_sharing(server,clients,agg_dWs,method,init_A,args):

    if method == 'null':
        return
    [client.reset() for client in clients]
    nc = init_A.shape[0]
    choices = {'uniform':torch.ones(nc,nc)/nc,'init':init_A.cpu()}
    server.graph_update(clients,agg_dWs,choices[method],args)

def collect_info(info,ofeature,feature,initA,resA):
    
    info['ofeature'].append(ofeature)
    info['feature'].append(feature)
    info['initA'].append(initA)
    info['resA'].append(resA)

    return info

def run_gpfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, args):

    assert isinstance(server,Server)
    
    #seq_grads = {c.id:[] for c in clients}
    [client.download_from_server(args,server) for client in clients]
    
    # generate the input feature based on the gradients sequence
    
    #q = Queue(maxsize = 5)

    # compute the initial graph 
    init_A = pre_finigraph(args.initial_graph,clients,args.graph_eps,args)
    property_feature = get_meanfeature(clients)
    nclient = len(clients)

    #interval update
    last_client_W = None
    A,average_A = None,torch.zeros(nclient,nclient)
    graph_batch = random_graphbatch(20,20,30,min(c.data[0].x.shape[1] for c in clients),'structure',seed = 0)

    sharing_start = 0
    if args.setting == 'multi':
        graph_batch = [feature_enlarge(graph_batch,c.data[0].x.shape[1]) for c in clients]
    
    infos = {'ofeature':[],'feature':[],'initA':[],'resA':[]}

    # record the performance gain of each client after knowledge sharing
    #performance_gain = torch.zeros(len(clients))
    for cround in range(1, COMMUNICATION_ROUNDS + 1):
        
        average = 0
        for i,client in enumerate(clients):
            client.compute_weight_update(local_epoch)
            #[client.evaluate() for client in clients]
        
        if args.setting == 'single':
            embed = server.graph_modelembedding(clients,graph_batch.to(args.device),'sum')
        else:
            embed = server.multi_embedding(clients,[cbatch.to(args.device) for cbatch in graph_batch],'sum')
        agg_dWs,graph_dWs = get_interval_dW(clients,last_client_W,server.W.keys())

        if (graph_dWs is not None) and cround > args.sround:
            # prepare input_features
            ofeature,feature,sim = prepare_features(embed,graph_dWs,cround,args)
            # select property features
            if args.initial_graph == 'property' and cround == 1:
                #print(property_feature.shape)
                init_A = rule_selector(property_feature,sim.detach().cpu())
            init_A = pre_vinigraph(init_A, args.initial_graph,graph_dWs,sim,args.graph_eps,cround,A,args).to(args.device)
            init_A = normalize(init_A,'sym')
            
            #print('initial graph')
            #print(init_A)
            client_graph_cons = graph_constructor(feature.shape[1],args)
            #A = get_finalgraph(args,feature,init_A,cround,clients)
            A = client_graph_cons.graph_based_aggregation(feature, init_A)
            #if cround > 5:
            #    exit(0)
            
            
            #print('out')
            #print(A)
            
            
            #A = get_finalgraph(args,feature,init_A,cround,clients)
            
            #num = torch.tensor([client.train_size]).to(A.device)
            #A = normalize(A*num[None,:],'row')
            
            average_A += A.cpu()
            
            collect_info(infos,ofeature.cpu().numpy(),feature.cpu().numpy(),init_A.cpu().numpy(),A.cpu().numpy())
            # update initial A per rounds
            #init_A = (1 - args.serverbeta) * init_A + args.serverbeta * A.to(args.device)

        # 2 update the local models
        if cround >= args.sround:
            [client.reset() for client in clients]
            server.graph_update(clients,agg_dWs,A,args)
        else:
            pre_sharing(server,clients,agg_dWs,args.pshare,init_A,args)
        #q = server.inter_graph_update(clients,q,A,5)

        # evaluate the performance
        [client.evaluate() for client in clients]
        #print('accuracy after knowledge sharing {:.4f}'.format(average/len(clients)))
    
    allAccs = analyze_train(clients,args)

    average_A /= COMMUNICATION_ROUNDS - sharing_start
    # save info
    np.save('ee/info_record/info.npy',infos)
    #print('average sharing client graph')
    #print(average_A)
    return allAccs,average_A
               
def run_gcfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2,args):
    assert isinstance(server,Server)

    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            pass
            #print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(args,server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset() 

        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)
    
    '''
    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)
    '''
    all_Accs = analyze_train(clients,args)

    return all_Accs

def run_gcflplus(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize, args):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(args,server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(args,server)
        
        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convGradsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])

    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame

def run_gcflplus_dWs(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize, args):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(args,server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            pass
            #print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(args,server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convDWsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    allAccs = analyze_train(clients,args)

    return allAccs

def get_interval_dW(clients,last_client_W,server_key):

    agg_dWs = [
            {k:copy.deepcopy(client.dW[k]) for k in server_key}
            for client in clients
    ]
    graph_dWs = [
            {k:copy.deepcopy(client.W[k]) for k in server_key}
            for client in clients
    ]
            
    return agg_dWs,graph_dWs

   


