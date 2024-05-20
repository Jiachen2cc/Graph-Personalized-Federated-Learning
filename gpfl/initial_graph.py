# create overall initial graph and round initial graph
import torch
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from torch_geometric.data import Batch,Data
import networkx as nx
import random
import torch.nn.functional as F
from gpfl.utils import marginal,cossim

def property_graph(property):
    
    pfeature = torch.stack(property, dim = 0)
    # feature normalization
    pmean, pstd = torch.mean(pfeature,dim = 0), torch.std(pfeature,dim = 0)
    pfeature = (pfeature-pmean)/pstd
    # cosine similarity
    # row normalization
    pfeature /= (torch.sum(pfeature*pfeature,
            dim = 1)[:,None])**0.5
    # compute
    simi = torch.matmul(pfeature,pfeature.T)
    simi = simi * (simi>=0).float()
    return simi

def distribution_graph(dis,thresh = 0.95):
    # distribution should be a n*d matrix
    dis = F.normalize(dis.T,p = 2, dim = 0)
    d = dis.shape[1]
    sharing = torch.zeros((d,d))
    for i in range(d):
        cur = dis[:,i]
        selected, res = [i],[]
        for idx in range(d):
            if idx == i:
                continue
            if torch.sum(cur*dis[:,idx]) > thresh:
                selected.append(idx)
            else:
                res.append(idx)    
        weight = torch.ones(d)
        if len(res) > 0:
            p = dis[:,torch.tensor(res)].reshape((dis.shape[0],len(res)))
            proj = torch.pinverse(p.T@p)@p.T@cur
            proj = proj*(proj >= 0).float()
            x = p@proj
            simi = max(cossim(cur,x),0)
            # maybe there are more choices for regularization
            proj = (proj/torch.max(proj))*simi
            for idx in range(len(res)):
                weight[res[idx]] = proj[idx]
        sharing[i,:] = weight
    
    return sharing 

def random_graph(size):
    
    return torch.tensor(np.random.random((size,size))).float()


def random_graphbatch(graph_num,feature_dim,seed):
    graphs = []
    random.seed(seed)
    
    for _ in range(graph_num):
        graph =struc_graph(30,feature_dim,0.1,seed)
        graphs.append(graph)
    graphs = Batch.from_data_list(graphs)
    return graphs
        

def struc_graph(graph_size,feature_dim,p,seed = 0):
    # assert graphs_size < feature_dim
    sample = erdos_renyi_graph(graph_size,p,seed,False)
    data = onehotdegree_data(sample,torch.tensor(0),feature_dim)
    return data

def onehotdegree_data(graph: nx.Graph, label, upper_bound = None):
    
    gsize = graph.number_of_nodes()
    #1 turn the nx.Graph.edge into edge_index
    adj_matrix = torch.zeros(gsize,gsize)
    
    edge_index = [[],[]]

    for x,y in graph.edges():
        adj_matrix[x,y] = 1
        edge_index[0].append(x)
        edge_index[1].append(y)
    
    edge_index = torch.LongTensor(edge_index)
    inv_index = edge_index[[-1,0]]

    edge_index = torch.concat([edge_index,inv_index],dim = 1)
    
    #2 compute the degree of the nodes
    degrees = torch.sum(adj_matrix,dim = 1)
    if upper_bound is not None:
        degrees = torch.min(degrees, torch.zeros(gsize) + upper_bound - 1)
    max_deg = int(torch.max(degrees).item())
    index = degrees.unsqueeze(1).long()

    features = torch.zeros(gsize,max_deg+1).scatter_(1,index,1)
    if upper_bound is not None:
        features = torch.zeros(gsize,upper_bound).scatter_(1,index,1)

    #features = torch.zeros((gsize,gsize))

    data = Data(x = features, edge_index = edge_index, y = label)
    
    return data



# for rule-based selector
#   compute similarity based on given features
def f2sim(features):
    features = F.normalize(marginal(features),p = 2,dim = 1)
    sim = torch.matmul(features,features.T)
    sim *= (sim >= 0).float().to(sim.device)
    return sim

#   gradualy remove features
def feature_removal(features,sim):

    if(features.shape[1] == 2):
        return None
    
    odiff = torch.sum((f2sim(features)-sim)**2)
    ndiffs = torch.zeros(features.shape[1])

    for i in range(features.shape[1]):
        subfeature = features[:,torch.arange(features.shape[1]) != i]
        ndiffs[i] = torch.sum((f2sim(subfeature) - sim)**2)

    # remove unhelpful features
    resdiff = ndiffs - odiff
    bestidx = torch.argmin(resdiff)
    if(resdiff[bestidx] >= 0):
        return None
    
    return features[:,torch.arange(features.shape[1]) != bestidx]

def rule_selector(features,sim):
    #preprocess the similarity matrix
    sim *= (sim >= 0).float().to(sim.device)

    while(1):

        removal = feature_removal(features,sim)
        if removal is None:
            break
        features = removal
    #print(features.shape)
    # construct client graph based on selected features
    return f2sim(features)