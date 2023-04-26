from torch_geometric.datasets import TUDataset
import torch
import numpy as np
import random
import copy
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils.random import erdos_renyi_graph
import networkx as nx
import torch.nn.functional as F
# how to design a client graph which can reflect useful information of the dataset 
# for dataset from different domains: node features are not important since they can't be aligned
# structure feature is more important:
'''
features which can be taken into consideration:

1 degree distribution !
corresponding perturbation method
# 2-hop neighbors -> 1-hop neighbors (actually that is triangle enclose)
2 triangle num distribution !
corresponding perturbation method
# triangle enclose
# triangle remove 

algorithm design:
** consider the intersection of the neighborhood

'''

#tudataset = TUDataset(f"./data/TUDataset", "PROTEINS")


#graphs = tudataset[0:2]


#graph[0].x = 0 + copy.deepcopy(graph[0].x[:,[1,2,0]])
#s = graph[0].x + copy.deepcopy(graph[0].x[:,[1,2,0]])
#print(graph[0].x)

def node_feature_perturbation(graphs,noise_type = 'Gaussian',seed = 0, noise_rate = 0.3):
    per_graphs = []
    random.seed(seed)
    np.random.seed(seed)

    if noise_type == 'Gaussian':
        
        for graph in graphs:

            if random.random() > noise_rate:
                # do not perform perturbation
                per_graph = graph.clone()
                per_graphs.append(per_graph)
                continue

            noise = torch.randn(graph.x.shape)
            per_x = graph.x + noise
            
            # add perturbated graphs
            per_graph = graph.clone()
            per_graph.__setitem__('x',per_x)
            per_graphs.append(per_graph)
    
    elif noise_type == 'perm':
        
        feature_dim = graphs[0].x.shape[1]
        perm = np.random.permutation(feature_dim)

        for graph in graphs:

            if random.random() > noise_rate:
                # do not perform perturbation
                per_graph = graph.clone()
                per_graphs.append(per_graph)
                continue

            per_x = copy.deepcopy(graph.x[:,perm])

            # add perturbated graphs
            per_graph = graph.clone()
            per_graph.__setitem__('x',per_x)
            per_graphs.append(per_graph)

    return per_graphs

def structure_perturbation(graphs,noise_type = 'tri_en',seed = 0, noise_rate = 0.3):

    per_graphs = []
    random.seed(seed)
    np.random.seed(seed)

    if noise_type == 'tri_en':
        
        for graph in graphs:

            if random.random() > noise_rate:
                # do not perform perturbation
                per_graph = graph.clone()
                per_graphs.append(per_graph)
                continue
                
            per_edge = tri_enclose(graph)
            
            # add perturbated graphs
            per_graph = graph.clone()
            per_graph.__setitem__('edge_index',per_edge)
            per_graphs.append(per_graph)
    
    elif noise_type == 'edge':
        
        count = 0
        for graph in graphs:

            if random.random() > noise_rate:
                count += 1
                per_graph = graph.clone()
                per_graphs.append(per_graph)
                continue

            per_edge = edge_resample(graph)

            per_graph = graph.clone()
            per_graph.__setitem__('edge_index',per_edge)
            per_graphs.append(per_graph)
        print(count)
    return per_graphs
# turn edge_index into neighbor list
def e2n(gsize, edge_index : torch.Tensor):
    
    nlist = [[] for i in range(gsize)]

    for idx in range(edge_index.shape[1]):

        x,y = edge_index[0,idx].item(),edge_index[1,idx].item()
        nlist[x].append(y)
        nlist[y].append(x)
    
    for i in range(gsize):
        nlist[i] = set(nlist[i])
    
    return nlist

def tri_enclose(graph: Data):

    gsize = len(graph.x)
    nlist = e2n(gsize, graph.edge_index)
    
    U = set(range(gsize))

    add_edges = []
    for i in range(gsize):

        neigh_i = nlist[i]
        nneigh_i = U - (neigh_i|set([i]))
        
        # for not neighbor nodes
        for j in nneigh_i:
            neigh_j = nlist[j]
            inter = neigh_i & neigh_j
            # i,j can be added to generate new triangles
            if len(inter) != 0:
                add_edges.append([i,j])
    
    if len(add_edges) == 0:
        return graph.edge_index
    
    add_edges = torch.LongTensor(add_edges).T
    edges = graph.edge_index
    new_edges = torch.cat((add_edges,edges),dim = 1)

    return new_edges

def edge_resample(graph: Data, pos_p = 0.4, neg_p = 0):

    g = to_networkx(graph,to_undirected = True)

    neg_edges = [edge for edge in nx.non_edges(g)]
    pos_edges = [edge for edge in g.edges()]

    # completeness check
    gsize = g.number_of_nodes()

    assert (len(neg_edges) + len(pos_edges))*2 == gsize*(gsize - 1)
    
    sample_neg = random.sample(neg_edges,int(neg_p * len(neg_edges)))
    sample_pos = random.sample(pos_edges,min(int(pos_p * len(pos_edges)),len(list(g.edges)) - 1))
    
    #print(len(sample_neg))
    #print(len(sample_pos))

    g.remove_edges_from(sample_pos)
    g.add_edges_from(sample_neg)
    
    # complete graph test
    new_edges = torch.LongTensor(list(g.edges)).T
    inv_edges = new_edges[[-1,0]]

    new_edges = torch.concat([new_edges,inv_edges],dim = 1)

    return new_edges

# fptype = ['add','replace'](feature perturbation type)
def node_feature_noise(graphs, noise_rate = 0.3, seed = 0, SNR = 20, fptype = 'add',edge_per = False, remove_rate = 0.2):

    per_graphs = []
    random.seed(seed)
    np.random.seed(seed)

    for graph in graphs:

        # get feature size
        x = graph.x
        noise = torch.normal(0,1,x.shape)

        # 2 types of noise: replace label or added to the label
        # generate standard Gaussian noise
        per_num = int(x.shape[0] * noise_rate)
        per_idx = np.random.choice(list(range(x.shape[0])),per_num,replace = False)
        per_x = copy.deepcopy(x)

        #compute Ps and Pn
        Ps = torch.sum(abs(x)**2)
        Pn = torch.sum(noise**2)
        
        #normailze noise
        ePn = Ps/(10**(SNR/10))
        noise = (noise/(Pn**0.5))*(ePn**0.5)

        if fptype == 'add':
            per_x[per_idx,:] += noise[per_idx,:]

        elif fptype == 'replace':
            per_x[per_idx,:] = noise[per_idx,:]


        per_graph = graph.clone()
        per_graph.__setitem__('x',per_x)

        # perturbate the edges of the graph based on noisy features 
        if edge_per:

            norm_perx = per_x/torch.norm(per_x, dim = 1, keepdim = True)
            simi_matrix = torch.matmul(norm_perx,norm_perx.T)

            per_edge = pertur_edge(graph,simi_matrix,remove_rate)
            per_graph.__setitem__('edge_index',per_edge)

        per_graphs.append(per_graph)

    return per_graphs

# fmask = ['None','zero','one','Gaussian'](feature mask)
# prate the rate for dropping positive edges
# nrate the rate for adding negative edges
def edge_noise(graphs,prate = 0.1,nrate = 0.1,fmask = 'None',seed = 0):

    per_graphs = []
    random.seed(seed)
    np.random.seed(seed)

    for graph in graphs:
        
        # edge pertur
        per_edge = edge_resample(graph,prate,nrate)
        #num_node = graph.x.shape[0]
        #edge_rate = graph.edge_index.shape[1] / ((num_node - 1)*num_node)
        #per_edge = erdos_renyi_graph(num_node,edge_rate,False)
        
        # feature mask
        if fmask == 'zero':
            per_features = torch.zeros(graph.x.shape)
        elif fmask == 'one':
            per_features = torch.ones(graph.x.shape)
        elif fmask == 'Gaussian':
            per_features = torch.normal(0,1,graph.x.shape)
        elif fmask == 'None':
            per_features = graph.x

        per_graph = graph.clone()
        per_graph.__setitem__('edge_index',per_edge)
        per_graph.__setitem__('x',per_features)
        per_graphs.append(per_graph)
    
    return per_graphs
    
def feature_normalize(graphs,ntype = 'gauss_norm'):

    norm_graphs = []

    for g in graphs:
        x = g.x
        
        if ntype == 'F_norm':
            norm_x = torch.norm(x,dim = 1,keepdim = True)
            res = x / norm_x

        elif ntype == 'minmax_norm':
            minx = torch.min(x, dim = 1,keepdim = True).values
            maxx = torch.max(x, dim = 1,keepdim = True).values
            res = (x - minx)/(maxx - minx) 
        
        elif ntype == 'gauss_norm':
            sigma = torch.std(x, dim = 1, keepdim = True)
            meanx = torch.mean(x, dim = 1, keepdim = True)
            res = (x - meanx)/sigma
        
        norm_graph = g.clone()
        norm_graph.__setitem__('x',res)
        norm_graphs.append(norm_graph)
    
    return norm_graphs

def pertur_edge(graph: Data, simi_matrix, min_rate = 0.2):
    # exchange the smallest edges and largest non edges
    g = to_networkx(graph,to_undirected = True)

    edge_list = np.array([[x,y] for (x,y) in g.edges])
    nedge_list = np.array([[x,y] for (x,y) in nx.non_edges(g)])
    
    
    #print(edge_list)
    #print(nedge_list)
    
    edge_score,nedge_score = [],[]

    for x,y in g.edges:
        edge_score.append(simi_matrix[x,y])

    for u,v in nx.non_edges(g):
        nedge_score.append(simi_matrix[u,v])

    edge_score,nedge_score = np.array(edge_score),np.array(nedge_score)

    exchange_num = int(len(edge_list)*min_rate)

    # remove the min edges 
    nomin_index = np.argsort(edge_score)[exchange_num::]

    # find the max non-edges
    max_index = np.argsort(-nedge_score)[:min(exchange_num,len(nedge_score))]

    per_edge = edge_list[nomin_index]
    per_nedge = nedge_list[max_index]
    
    if len(per_nedge) == 0:
        per_edge = torch.LongTensor(per_edge).T
    else:
        per_edge = torch.LongTensor(np.concatenate([per_edge,per_nedge],axis = 0)).T

    inv_per_edge = per_edge[[-1,0]]

    res_edge = torch.concat([per_edge,inv_per_edge],dim = 1)

    return res_edge

def node_downsample(graphs,sample_rate = 0.8,seed = 0):

    per_graphs = []
    random.seed(seed)
    np.random.seed(seed)

    for graph in graphs:

        g = to_networkx(graph,to_undirected = True)
        node_list = list(range(g.number_of_nodes()))
        
        sample_num = max(int(len(node_list) * sample_rate),1)

        sub_node_list = random.sample(node_list,sample_num)
         
        idx_dict = {sub_node_list[i]:i for i in range(sample_num)}


        subg = g.subgraph(sub_node_list)
        
        sub_edges = []
        for x,y in subg.edges():
            sub_edges.append([idx_dict[x],idx_dict[y]])
        
        if len(sub_edges) == 0:
            sub_edges = torch.LongTensor([[0,1],[1,0]])
        else:
            sub_edges = torch.LongTensor(sub_edges).T
            inv_edges = sub_edges[[-1,0]]
            sub_edges = torch.concat([sub_edges,inv_edges],dim = 1)
        
        sub_feature = graph.x[np.array(sub_node_list)]

        #pertur_feature = torch.normal(0,1,sub_feature.shape)

        per_graph = Data(x = sub_feature, edge_index = sub_edges,y = graph.y)
        #per_graph = Data(x = pertur_feature, edge_index = sub_edges,y = graph.y)
        per_graphs.append(per_graph)
    
    return per_graphs


def feature_mix(graphs,mix_type,mix_rate,attr_graphs = None):

    per_graphs = []
    
    if mix_type == 'attr':
        
        max_dim = max(graphs[0].num_node_features,attr_graphs[0].num_node_features)

        for g,attrg in zip(graphs,attr_graphs):
            
            g_pad = torch.concat([g.x,torch.zeros((g.x.shape[0],max_dim - g.x.shape[1]))],dim = 1)
            ag_pad = torch.concat([attrg.x,torch.zeros((g.x.shape[0],max_dim - attrg.x.shape[1]))],dim = 1)
            per_x = (1 - mix_rate)*g_pad + mix_rate*ag_pad
            per_g = g.clone()
            per_g.__setitem__('x',per_x)

            per_graphs.append(per_g)
    
    else:

        for g in graphs:

            if mix_type == 'ones':
                per = torch.ones(g.x.shape)
            elif mix_type == 'Gauss':
                per = torch.normal(0,1,g.x.shape)
            per_x = (1 - mix_rate)*g.x + mix_rate * per
            #per_x = per
            
            per_g = g.clone()
            per_g.__setitem__('x',per_x)

            per_graphs.append(per_g)

    return per_graphs
             
    





    



        




    
    



