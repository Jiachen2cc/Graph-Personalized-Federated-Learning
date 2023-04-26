import torch
from torch_geometric.data import Batch,Data
from networkx.generators.random_graphs import erdos_renyi_graph
#from deepsnap.graph import Graph
import networkx as nx
import random



def random_graphbatch(graph_num,graph_size,feature_dim,threshold = 0.3,seed = 0):
    

    graph_list = []
    
    for _ in range(graph_num):

        # random feature generating
        feature = torch.normal(0,1,size = (graph_size,feature_dim))
        normf = torch.norm(feature,dim = 1,keepdim = True)
        feature /= normf
        
        # generate edge based on node feature similarity
        edge_index = [[],[]]
        simif = torch.matmul(feature,feature.T)

        for i in range(graph_size):
            for j in range(graph_size):
                if simif[i,j] >= threshold:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        
        edge_index = torch.LongTensor(edge_index)
        
        graph = Data(x = feature,edge_index = edge_index)

        graph_list.append(graph)
    
    graph_batch = Batch.from_data_list(graph_list)

    return graph_batch

def struc_graphs(graph_num,graph_size,feature_dim,p,seed = 0):
    # assert graphs_size < feature_dim
    graphs = []

    for _ in range(graph_num):

        sample = erdos_renyi_graph(graph_size,p,seed,False)
        data = onehotdegree_data(sample,torch.tensor(0),feature_dim)
        graphs.append(data)
    graphs = Batch.from_data_list(graphs)
    return graphs
        
        

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

def feature_padding(graphs,max_dim = None):
    
    datasets = []
    if max_dim is None:
        max_dim = max([graph.x.shape[1] for graph in graphs])
    
    for g in graphs:

        resf = torch.zeros((g.x.shape[0], max_dim - g.x.shape[1]))
        ansf = torch.concat([g.x,resf],dim = 1)

        ansgraph = g.clone()
        ansgraph.__setitem__('x',ansf)
        datasets.append(ansgraph)
    
    return datasets



    


def arti_datasets(cd_list, gsize = 30, p_rates = [0.2,0.8], seed = 0):
    
    # cd_list show the class distribution of each client(need to accord with the p_rates)
    # p_rates shows the edge rate of each class
    # create random graphs
     
    client_num = len(cd_list)
    datasets = [[] for i in range(client_num)]

    for cli in range(client_num):
        
        datas = []

        for cidx in range(len(p_rates)):

            res = struc_graphs(cd_list[cli][cidx],gsize,None,p_rates[cidx],seed)
            datas.extend(res)
        
        datasets[cli] = feature_padding(datas)
        
        check_dim = []
        for data in datasets[cli]:
            check_dim.append(data.x.shape[1])
        #print(max(check_dim) - min(check_dim))

    #exit(0)
    return datasets

def toy_datasets(cd_list, gsize = 30, label_num = 3, seed = 0):

    random.seed(seed)

    client_num = len(cd_list)
    datasets = [[] for i in range(client_num)]

    for cli in range(client_num):

        num = cd_list[cli]
        for i in range(num):
            edge_index = []
            for node_idx in range(gsize):
                edge_index.extend([[node_idx,(t+node_idx)%gsize] for t in range(1,gsize)])
            edge_index = torch.LongTensor(edge_index).T

            feature = torch.zeros([gsize,gsize])

            label = torch.tensor(int(label_num*random.random()))

            graph = Data(x = feature, edge_index = edge_index, y = label)
            datasets[cli].append(graph)
        
    return datasets





                    
        

             
    

        



    





