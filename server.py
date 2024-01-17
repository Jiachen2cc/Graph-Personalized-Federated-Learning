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
from graph_utils import normalize,flattenw
from analyze_dataset import structure_sim

class Server():
    def __init__(self, model, classifiers, extractors, device):
        self.model = model.to(device)

        self.classifiers = [c.to(device) for c in classifiers]
        self.extractors = [e.to(device) for e in extractors]

        self.W = {key: value for key, value in self.model.named_parameters()}

        self.control = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
        self.ex_Ws = [{key: value for key, value in ex.named_parameters()}
            for ex in self.extractors]

        self.clf_Ws= [{key: value for key, value in clf.named_parameters()}
            for clf in self.classifiers]

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
    
    # Weight aggregation FedStar
    def aggregate_weights_se(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            if '_s' in k:
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
            for targ in targs:
                reduce_add_average(target=targ, sources=sours, total_size=total_size)

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
    
    def multi_embedding(self,clients,graph_batch,mode = 'sum'):
        res_embed = []

        for ex,exW,client,cbatch in zip(self.extractors,self.ex_Ws,clients,graph_batch):

            # load client model & extractor
            for k in self.W.keys():
                self.W[k].data = copy.deepcopy(client.W[k])
            for k in exW.keys():
                exW[k].data = copy.deepcopy(client.W[k])
            self.model.eval()
            ex.eval()
            embed = self.model(ex(cbatch)).detach()
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
         
        #1 flatten corresponding parameters and perform aggregating 
        if args.sharing_mode == 'gradient':
            param_matrix = torch.stack([flattenw(c.dW).detach() for c in clients],dim = 0)
        elif args.sharing_mode == 'difference':
            param_matrix = torch.stack([flattenw(c.dW).detach() for c in clients],dim = 0)
            pmean = torch.mean(param_matrix,dim = 0)[None,:]
            param_matrix -= args.diff_rate*pmean
        elif args.sharing_mode == 'ALA':
            param_matrix = torch.stack([flattenw(c.dW).detach() for c in clients],dim = 0)
        
        #2 aggregate and update the parameters
        aggregated_param = torch.mm(A.to(param_matrix.device),param_matrix)
        for i in range(args.layers - 1):
            aggregated_param = torch.mm(A,aggregated_param)
        resparam = (args.serveralpha * aggregated_param) + ((1 - args.serveralpha) * param_matrix)
        
        if args.sharing_mode == 'ALA':
            meanalpha = 0
            for i in range(len(clients)):
                meanalpha += clients[i].ALA_aggregate(aggregated_param[i,:],args)
            meanalpha /= len(clients)
            print('average accept rate: {:.4f}'.format(meanalpha))
            return
        #3 construct the right form of parameters according to sharing mode
        if args.sharing_mode == 'gradient':
            addparam = torch.stack([flattenw(c.W_old).detach() for c in clients],dim = 0)
        elif args.sharing_mode == 'difference':
            addparam = args.diff_rate*pmean + torch.stack([flattenw(c.W_old).detach() for c in clients],dim = 0)

        resparam += addparam

        for i,c in enumerate(clients):
            c.load_param_matrix(resparam[i,:])

        '''
        #2 reload aggregated parameters
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
        '''
    def ala_tuning(self,train_data,aparam,param,args):
        """

        args:
        param: the parameters of the local model
        aparam: parameters of the corresponding server model
        train data: the train data of the client

        returns:
        tuning aggregation result

        """
        
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

    def scaffold_update(self, clients, local_epoch,args):
        
        # haven't implement trianing with client sampling
        for client in clients:
            for k in self.W.keys():
                # update client control parameters
                client.dcontrol[k].data = (1/(local_epoch*args.lr))*(self.W[k].data - client.W[k].data) - self.control[k].data
                client.control[k].data += client.dcontrol[k].data
                
                # compute dW
                client.dW[k] = client.W[k].data - self.W[k].data
        
        total_size = len(clients)

        reduce_add_average(self.W,[[c.dW,1] for c in clients],total_size)
        reduce_add_average(self.control,[[c.dcontrol,1] for c in clients],total_size)




        



            






        



        

        


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

def reduce_add_average(target, sources, total_size):
    for name in target.keys():
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


