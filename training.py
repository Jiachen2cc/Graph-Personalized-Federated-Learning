import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from server import Server
import torch
from queue import Queue
import copy
from analyze_client import simi_ana,clf_ana,cos_sim
from analyze_dataset import structure_sim
from analyze_graphs import fake_graph
from server import group_sub
from init_client_graph import dist_simi_metrix
from graph_utils import para2metrix,normalize
from functest import random_graphbatch,struc_graphs
from federated import generate_adj
from torch.utils.tensorboard import SummaryWriter
from analyze_client import graph_diff
from recover_model import GCN_DAE
from client import Client_GC
from utils import graph_truncate,mean_diff,cluster_uniform_graph
import math

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
    
def mean_performance(clients):

    mean_loss,mean_acc = 0,0
    for client in clients:
        loss,acc = client.evaluate()
        mean_loss += loss
        mean_acc += acc
    mean_loss /= len(clients)
    mean_acc /= len(clients)
    return mean_loss,mean_acc

def monitor_performance(writer,clients):

    assert isinstance(writer,SummaryWriter)
    #assert isinstance(client,Client_GC)
    # monitor loss
    for client in clients:
        trainloss,trainacc,testloss,testacc = client.train_stats.values()
        #print(trainloss)
        for idx in range(len(np.array(trainloss))):
            writer.add_scalar('Loss/train',trainloss[idx],idx)
        for idx in range(len(testloss)):
            writer.add_scalar('Loss/eval',testloss[idx],idx)
        #monitor acc
        for idx in range(len(trainacc)):
            writer.add_scalar('Acc/train',trainacc[idx],idx)
        for idx in range(len(testacc)):
            writer.add_scalar('Acc/test',testacc[idx],idx)
    writer.close()
    
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

def analyze_train(clients):

    allAccs,final_accs = {},{}
    for client in clients:
        # initialize
        allAccs[client.name] = {}
        stats = np.array(client.eval_stats['testAccs'])
        allAccs[client.name]['best_test_acc'] = np.max(stats)
        allAccs[client.name]['final_test_acc'] = stats[-1]
        final_accs[client.name] = stats[-(COUNT_LEN+1):-1]

    best_final = best_final_acc(final_accs,COUNT_LEN)

    for k in best_final.keys():
        allAccs[k]['final_best_test_acc'] = best_final[k]

    return allAccs
    

def run_selftrain_GC(clients, server, args):
    assert isinstance(server,Server)

    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)
    
    #writer = SummaryWriter('run/exp1')
    feature_dim = clients[0].data[0].x.shape[1] if args.setting == 'single' else args.hidden
    graph_batch = struc_graphs(20,30,feature_dim,0.1,seed = 0)
    
    client_similarity = torch.zeros((len(clients),len(clients)))
    embed_similarity = torch.zeros((len(clients),len(clients)))

    for cround in range(1,args.num_rounds + 1):
        # training
        for client in clients:
            client.local_train(1)
            client.evaluate()

        #analyze the similarity 
        embed = server.graph_modelembedding(clients,graph_batch.to(args.device))
        #print('the similarity of the model parameters')
        client_simi = simi_ana(clients,None,server.W.keys())
        #print('the similarity of the inference result')
        embed_simi = cos_sim(embed)
        
        client_similarity += client_simi
        embed_similarity += embed_simi.cpu()


        if (cround+1)%10 == 0:
            mean_loss,mean_acc = mean_performance(clients)
            #print('average accuracy:{:.4f}, average loss:{:.4f}'.format(mean_acc, mean_loss))
        if args.data_group == 'clf_test':
            print('the similarity of the classifiers')
            clf_ana(clients,server)
    
    #print('average similarity')
    print('average client similarity')
    print(client_similarity/args.num_rounds)
    print('average embedding similarity')
    print(embed_similarity/args.num_rounds)
    allAccs = analyze_train(clients)
    '''
    clients[0].download_from_server(clients[1])
    _,acc0 = clients[0].evaluate()
    _,acc1 = clients[1].evaluate()
    print('client_0:{:.4f},client_1:{:.4f}'.format(acc0,acc1))
    '''
    #if args.global_model:
        #monitor_performance(writer,clients[-1])
    #monitor_performance(writer,clients)

    return allAccs

def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, args, samp=None, frac=1.0):

    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            pass
            #print(f"  > round {c_round}")
        
        '''
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
        '''
        # do not select global model
        #if args.global_model:
        #    selected_clients = clients[:-1]
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
            client.download_from_server(server)
        
        for client in clients:
            client.evaluate()



    allAccs = analyze_train(clients)
    '''
    clients[0].download_from_server(clients[1])
    _,acc0 = clients[0].evaluate()
    _,acc1 = clients[1].evaluate()
    print('client_0:{:.4f},client_1:{:.4f}'.format(acc0,acc1))
    '''
    # summary writer
    #monitor_performance(writer,allloss,allacc)
    return allAccs


def run_fedprox(clients, server, COMMUNICATION_ROUNDS, local_epoch, mu, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    if samp == 'random':
        sampling_fn = server.randomSample_clients

    allAccs,final_accs = {},{}
    #initialize
    for client in clients:
        allAccs[client.name] = {'best_test_acc':[],'final_test_acc':None}
        final_accs[client.name] = []

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train_prox(local_epoch, mu)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

            # cache the aggregated weights for next round
            client.cache_weights()

        for client in clients:
            loss, acc = client.evaluate()
            allAccs[client.name]['best_test_acc'].append(acc)
            allAccs[client.name]['final_test_acc'] = acc
            if COMMUNICATION_ROUNDS - c_round <= COUNT_LEN - 1:
                final_accs[client.name].append(acc)

    best_final = best_final_acc(final_accs,COUNT_LEN)
    for key in allAccs.keys():
        allAccs[key]['best_test_acc'] = np.max(np.array(allAccs[key]['best_test_acc']))
        allAccs[key]['final_best_test_acc'] = best_final[key]
    
    return allAccs

    

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame

def run_bisfl(clients,server,COMMUNICATION_ROUNDS, local_epoch, args):
    assert isinstance(server,Server)

    for client in clients:
        client.download_from_server(server)
    

    # compute the initial graph 
    init_A = None
    if args.initial_graph != 'distance':
        distributions = [client.structure_feature_analysis(args.initial_graph) for client in clients]
        init_A = structure_sim(distributions).to(args.device)
    A = None

    for cround in range(1, COMMUNICATION_ROUNDS + 1):

        for client in clients:
            client.compute_weight_update(local_epoch)
            #client.reset()

        # get local weights
        client_Ws = get_dW(clients,server.W.keys())

        simi_ana(clients,None,server.W.keys())
        # 1 build client graph
        # interval update
        if init_A is None:
            init_A = dist_simi_metrix(client_Ws,args.graph_eps).to(args.device)

        interval_print(init_A,cround,20,'initial client graph')
        #A = server.graph_build(client_Ws,init_A,args)
        # try fixed graphs 
        #A = init_A
        #print(A)
        # 2 update the local models
        server.bigraph_update(clients,client_Ws,init_A,args)
        
        # 3 evaluate new aggregated model
        for client in clients:
            client.evaluate()
    

    allAccs = analyze_train(clients)
    return allAccs

def run_tosfl(clients,server,COMMUNICATION_ROUNDS, local_epoch, args):
    
    '''
    writer1 = SummaryWriter(log_dir = 'runs',comment = 'structure federated learning(01)')
    writer3 = SummaryWriter(log_dir = 'runs',comment = 'structure federated learning(03)')
    writer6 = SummaryWriter(log_dir = 'runs',comment = 'structure federated learning(06)')
    '''
    
    assert isinstance(server,Server)
    
    
    for client in clients:
        client.download_from_server(server)
    
    best_acc = []

    allAccs,final_accs = {},{}
    #initialize
    for client in clients:
        allAccs[client.name] = {'best_test_acc':[],'final_test_acc':None}
        final_accs[client.name] = []

    # compute the initial graph 
    init_A = None
    if args.initial_graph != 'distance':
        distributions = [client.structure_feature_analysis(args.initial_graph) for client in clients]
        init_A = structure_sim(distributions).to(args.device)
    #init_A = 
    #interval update
    A = None
    graph_batch = random_graphbatch(10,30,args.hidden,seed = args.seed)
    records_A = []
    for cround in range(1, COMMUNICATION_ROUNDS + 1):

        #A = torch.from_numpy(server.compute_pairwise_similarities(clients)).to(args.device)
        
        for client in clients:

            client.compute_weight_update(local_epoch)
            #client.reset()
        # generate the graph based on the initial graph
        # compute client_dWS
        # interval compute 
        client_Ws = get_dW(clients,server.W.keys())
        embed = server.graph_modelembedding(clients,graph_batch.to(args.device))
        #compress_param = (para2metrix(client_Ws,args.compress_mode,args.compress_dim)).to(args.device)
        #embed = compress_param
        
        #print('the norm of input features')
        #print(torch.norm(embed,dim = 1))
        
        
        print('client parameters similarity')
        cs = simi_ana(clients,None,server.W.keys())
        '''
        writer1.add_scalar('para_sim',cs[0,1],cround)
        writer3.add_scalar('para_sim',cs[0,3],cround)
        writer6.add_scalar('para_sim',cs[0,6],cround)
        '''
        print('embedding similarity')
        ecs = cos_sim(embed)
        print(ecs)
        '''
        writer1.add_scalar('embed_sim',ecs[0,1],cround)
        writer3.add_scalar('embed_sim',ecs[0,3],cround)
        writer6.add_scalar('embed_sim',ecs[0,6],cround)
        '''
        # 5 rounds interval for knowledge sharing

        if cround >= 25 and cround % 2 == 0:
            # 1 build client graph
            # interval update
            if init_A is None:
                dis_A = dist_simi_metrix(client_Ws)
                init_A = dis_A
            init_A = normalize(init_A,'sym').to(args.device)
            
            print('init client graph')
            print(init_A)
            
            _,A = generate_adj(embed,init_A,args)
            
            #A = init_A
            #A = fake_graph(len(clients),'test1')
            print('result client graph')
            print(A)
            
            records_A.append(A)

            # try fixed graphs 
            #A = init_A
            #print(A)
            # 2 update the local models

            # fix the client graph to test the relationship between client graph model and client similarity
            # since simply sharing the whole model is not good enough we try to only sharing the gradient
            server.tograph_update(clients,client_Ws,A.to('cpu'),args)
        
        
        
        
        all_loss,all_acc = [],[]
        for client in clients:
            loss, acc = client.evaluate()
            allAccs[client.name]['best_test_acc'].append(acc)
            allAccs[client.name]['final_test_acc'] = acc
            if COMMUNICATION_ROUNDS - cround <= COUNT_LEN - 1:
                final_accs[client.name].append(acc)

            all_loss.append(loss)
            all_acc.append(acc)

        mean_loss = np.array(all_loss).mean()
        mean_acc = np.array(all_acc).mean()

        #print('loss:{:.4f},acc:{:.4f}'.format(mean_loss,mean_acc))
        best_acc.append(mean_acc)
    
    print(len(records_A))
    graph_diff(records_A)

    best_acc = np.array(best_acc).max()
    #print('best_acc:{:.4f}'.format(best_acc))

    best_final = best_final_acc(final_accs,COUNT_LEN)
    for key in allAccs.keys():
        allAccs[key]['best_test_acc'] = np.max(np.array(allAccs[key]['best_test_acc']))
        allAccs[key]['final_best_test_acc'] = best_final[key]
    '''
    writer1.close()
    writer3.close()
    writer6.close()
    '''
    return allAccs

def run_sfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, args):

    assert isinstance(server,Server)
    
    seq_grads = {c.id:[] for c in clients}
    [client.download_from_server(server) for client in clients]
    
    # generate the input feature based on the gradients sequence
    
    #q = Queue(maxsize = 5)

    # compute the initial graph 
    init_A = None

    if args.initial_graph not in ['distance','uniform','similarity']:
        distributions = [client.structure_feature_analysis(args.initial_graph) for client in clients]
        init_A = structure_sim(distributions,args.graph_eps)

    #init_A = 
    #interval update
    last_client_W = None
    nclient = len(clients)
    A,average_A = None,torch.zeros(nclient,nclient)
    feature_dim = clients[0].data[0].x.shape[1] if args.setting == 'single' else args.hidden
    graph_batch = struc_graphs(20,30,feature_dim,0.1,seed = 0)
    #graph_model = GCN_DAE(2,40,128,40,0.5,0,args.gen_mode,64,32,2).to(args.device)
    sharing_start = 10

    for cround in range(1, COMMUNICATION_ROUNDS + 1):

        init_embed = server.graph_modelembedding(clients,graph_batch.to(args.device),'sum')

        for client in clients:
            client.compute_weight_update(local_epoch)
            seq_grads[client.id].append({k:client.dW[k] for k in client.gconvNames})

        embed = server.graph_modelembedding(clients,graph_batch.to(args.device),'sum')
        # calculate the embed difference
        if args.input_choice == 'gradient':
            embed = embed - init_embed
        embed_sim = cos_sim(embed)
        interval_print(embed_sim,cround,40,'embed similarity')

        agg_dWs,graph_dWs,last_client_W = get_interval_dW(seq_grads,clients,last_client_W,server.W.keys(),cround,args)
        
        # 1 build client graph
        interval_print(simi_ana(clients,None,server.W.keys()),cround,40,'client similarity')

        #print('embed similarity')
        #print(cos_sim(embed))

        if (graph_dWs is not None) and cround > sharing_start:
            if args.initial_graph == 'distance':
                init_A = dist_simi_metrix(graph_dWs,args.graph_eps)
            elif args.initial_graph == 'uniform':
                init_A = torch.ones(nclient,nclient)/nclient

            # graph build
            #compress_param = (para2metrix(graph_dWs,args.compress_mode,args.compress_dim)).to(args.device)
            compress_param = (para2metrix(graph_dWs,None,args.compress_dim)).to(args.device)
            if args.input_choice == 'diff':
                compress_param = mean_diff(compress_param,args.diff_rate)
                embed = mean_diff(embed,args.diff_rate)
            param_sim = cos_sim(compress_param)
            interval_print(param_sim,cround,40,'client model similarity')

            if args.initial_graph == 'similarity':
                choices = {'param':param_sim,'embed':embed_sim}
                init_A = choices[args.para_choice]
            # choose whether to use graph truncate
            #init_A = normalize(graph_truncate(init_A.cpu(),math.ceil((init_A.shape[1]+1)/2)),'sym').to(args.device)
            init_A = normalize(init_A,'sym').to(args.device)

            #compress_param = (para2metrix(graph_dWs,args.compress_mode,args.compress_dim)).to(args.device)
            #interval_print(cos_sim(compress_param),cround,20,'compressed client model similarity')
            #init_A = (normalize(cos_sim(embed),'sym').to(args.device)+normalize(cos_sim(compress_param),'sym').to(args.device))/2
            interval_print(init_A,cround,40,'initial client graph')

            #_,A = generate_adj(compress_param,init_A,args)
            choices = {'param':compress_param,'embed':embed}
            if args.para_choice in choices.keys():
                _,A,graph_model = generate_adj(choices[args.para_choice],init_A,args,None)
            elif args.para_choice == 'ans':
                A = cluster_uniform_graph(nclient,args.num_splits)

            
            #A = 0.95 * A.to(args.device) + 0.05 * init_A
            #A = torch.tensor([[0.5,0.5],[0.5,0.5]]).to(args.device)
            interval_print(A,cround,40,'result client graph')
            average_A += A.cpu()

            # update initial A per rounds
            #init_A = (1 - args.serverbeta) * init_A + args.serverbeta * A.to(args.device)

        # try fixed graphs 
        #A = init_A
        #print(A)
        # 2 update the local models
        if cround > sharing_start:
            [client.reset() for client in clients]
            server.graph_update(clients,agg_dWs,A.to('cpu'),args)
        #q = server.inter_graph_update(clients,q,A,5)

        # evaluate the performance
        [client.evaluate() for client in clients]
    
    allAccs = analyze_train(clients)

    average_A /= COMMUNICATION_ROUNDS - sharing_start

    #print('average sharing client graph')
    #print(average_A)
    return allAccs,average_A
               
def run_gcfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2):
    assert isinstance(server,Server)

    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            pass
            #print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset() # ???  reset the model weight with the weight before training

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
    all_Accs = analyze_train(clients)

    return all_Accs

def run_gcflplus(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)
        
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

def run_gcflplus_dWs(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            pass
            #print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

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

    allAccs = analyze_train(clients)

    return allAccs

def get_interval_dW(seq_grad,clients,last_client_W,server_key,cround,args):

    agg_dWs = [
            {k:copy.deepcopy(client.dW[k]) for k in server_key}
            for client in clients
    ]
    if args.input_choice == 'gradient':
        graph_dWs = [
            {k:copy.deepcopy(client.dW[k]) for k in server_key}
            for client in clients
        ]
    elif args.input_choice == 'seq':
        if len(seq_grad[clients[0].id]) < args.timelen:
            graph_dWs = [
            {k:copy.deepcopy(client.W[k]) for k in server_key}
            for client in clients
            ]
        else:
            graph_dWs = [
                time_mean(seq_grad[client.id][-args.timelen:])
                for client in clients
            ]
    else:
        graph_dWs = [
            {k:copy.deepcopy(client.W[k]) for k in server_key}
            for client in clients
        ]
            
    last_client_W = None
    return agg_dWs,graph_dWs,last_client_W
    # update every round
    if interval == 1 or ((last_client_W is None) and cround == 1):
        graph_dWs = [
            {k:copy.deepcopy(client.W[k]) for k in server_key}
            for client in clients
        ]
        if last_client_W == None:
            last_client_W = [
                {k:copy.deepcopy(client.W[k]) for k in server_key}
                for client in clients
            ]
    
    elif cround % interval == 1:
        now_client_W = [
            {k:copy.deepcopy(client.W[k]) for k in server_key}
            for client in clients
        ]
        graph_dWs = group_sub(now_client_W,last_client_W)

        last_client_W = copy.deepcopy(now_client_W)
    
    else:
        graph_dWs = None

    #return agg_dWs,graph_dWs,last_client_W

def get_dW(clients,server_key):

    client_Ws = [
        {k:copy.deepcopy(client.W[k]) for k in server_key}
        for client in clients
    ]

    return client_Ws

def time_mean(seq):
    mean = {}
    for term in seq:
        for k in term.keys():
            if k not in mean.keys():
                mean[k] = term[k].clone()
            else:
                mean[k] += term[k].clone()
        mean[k] /= len(seq)
    return mean    


