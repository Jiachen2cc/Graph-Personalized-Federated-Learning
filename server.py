import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
#from argument_setting import args
from init_client_graph import dist_simi_metrix
from federated import graph_gen,graph_dic,graph_aggregate,bi_graph_dic
import copy
from queue import Queue
from graph_utils import normalize
from analyze_dataset import structure_sim

class Server():
    def __init__(self, model, classifiers, device):
        self.model = model.to(device)
        self.classifiers = [classifier.to(device) for classifier in classifiers]
        self.W = {key: value for key, value in self.model.named_parameters()}

        self.clf_Ws= [{key: value for key, value in clf.named_parameters()}
            for clf in self.classifiers
        ]
        self.model_cache = []
    
    # random select clients
    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))
    
    # compute aggregate 
    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
    

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_pairwise_distances(self, seqs, standardize=False):
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]
    
    def graph_modelembedding(self,clients,graph_batch,mode = 'sum'):
        
        res_embed = []
        for client in clients:
            
            # load client model
            for k in self.W.keys():
                self.W[k].data = copy.deepcopy(client.W[k])
            # set model eval 
            self.model.eval()
            # input the graph batch and get the output embedding
            embed = self.model(graph_batch).detach()
            if mode == 'sum':
                embed = torch.flatten(embed).unsqueeze(0)
            elif mode == 'mean':
                embed = torch.mean(embed,dim = 0).unsqueeze(0)
            #embed = torch.mean(embed,dim = 0)
            res_embed.append(embed)
            
        res_matrix = torch.cat(res_embed,dim = 0)

        return res_matrix


    def graph_build(self, client_dWs, A, args, norm = True):

        # get the client gradient 
        #client_dWs = []
        #for client in clients:
        #    dW = {k:copy.deepcopy(client.dW[k]) for k in self.W.keys()}
        #    client.append(dW)
        
        # compute the client graph if needed
        init_A = A.to(args.device)
        
        if norm:
            init_A = normalize(init_A,'sym')
        #print('initial client matrix A')
        #print(args.initial_graph)
        #print(init_A)

        # generate graph
        res_A,quality = graph_gen(client_dWs,init_A,args)

        #print(res_A)
        #print('graph quality:{:.4f}'.format(quality))

        return res_A
    
    def graph_update(self, clients, clients_dWs, A, args):
        
        res_dWs = graph_aggregate(clients_dWs,A,args)

        targs,sours = [],[]

        for client,res_dW in zip(clients,res_dWs):
            W = {}
            dW = {}

            for k in self.W.keys():
                W[k] = client.W[k]
                dW[k] = copy.deepcopy(res_dW[k])

            targs.append(W)
            sours.append(dW)
        
        group_add(targs,sours)
    
    def tograph_update(self, clients, clients_Ws, A, args):

        res_Ws = graph_aggregate(clients_Ws,A,args)
        for client,res_W in zip(clients,res_Ws):
            client.download_weight(res_W)
        
    def bigraph_update(self, clients, clients_Ws, A,args):

        init_A = A.to(args.device)
        res_Ws,res_A,quality = bi_graph_dic(clients_Ws,init_A,args)
        
        for client,res_W in zip(clients,res_Ws):
            client.download_weight(res_W)
        
        return res_A

    '''
    # this function is for learning client graph with intervals
    def inter_graph_update(self, clients, w_que, A = 0, interval = 2):
        if interval == 1:
            self.graph_update(clients,A)
            return 

        assert isinstance(w_que,Queue)
        
        # 1 get gradient for computing client graph
        old_ws = {}
        if w_que.qsize() == interval:
            old_ws = w_que.get()
        else:
            old_ws = copy_oldclientweights(clients,self.W.keys())
        
        targs = copy_clientweights(clients,self.W.keys())
        
        gclient_dWs = group_sub(targs,old_ws)
        
        # 1.1 prepare the initial client graph
        A = dist_simi_metrix(gclient_dWs)
        A = torch.max(A,dim = 1).values[:,None] - A
        
        # 2 get gradient for update
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = copy.deepcopy(client.dW[k])
            client_dWs.append(dW)

        res_dWs,res_A,quality = graph_dic(client_dWs,client_dWs,A)
        
        print(res_A)
        print('graph_quality: ',quality)
        
        
        targs,sours = [],[]
        for client,res_dW in zip(clients,res_dWs):
            W = {}
            dW = {}

            for k in self.W.keys():
                W[k] = client.W[k]
                dW[k] = copy.deepcopy(res_dW[k])#/interval

            targs.append(W)
            sours.append(dW)
        
        group_add(targs,sours)

        targs = copy_clientweights(clients,self.W.keys())
        w_que.put(targs)

        return w_que
    '''

            






        



        

        


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()

def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0), total_size).clone()
            target[name].data += tmp

def group_add(targets,sources):
    
    for target,sources in zip(targets,sources):
        for name in target:
            target[name].data += sources[name]

    return targets


def group_sub(targets,sources):
    
    for target,source in zip(targets,sources):
        for name in target:
            target[name].data -= source[name]
    
    return targets

def copy_clientweights(clients,keys):

    tar = []
    for client in clients:
        W = {}
        for k in keys:
            W[k] = copy.deepcopy(client.W[k])
        tar.append(W)

    return tar

def copy_oldclientweights(clients,keys):

    tar = []
    for client in clients:
        W = {}
        for k in keys:
            W[k] = copy.deepcopy(client.W_old[k])
        tar.append(W)

    return tar


