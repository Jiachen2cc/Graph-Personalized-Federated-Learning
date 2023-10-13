import torch
from torch_geometric.utils import to_networkx, degree
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    #print(maxdegree)
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        #print(tuple[0].edge_index)
        deg = degree(idx, tuple[2], dtype=torch.long)
        #print(len(x))
        #print(deg)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    #print(s)
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes/numGraphs, numEdges/numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df

def cross_res_analyze(records):

    # compute average accuracy and the variance
    
    # analyze total mean and client mean
    
    #ele_list = ['best_test_acc','final_train_acc','final_test_acc']

    cnames = records[0].keys()
    ele_list = records[0][list(cnames)[0]].keys()
    cross_meanAccs = {cname: {e:[] for e in ele_list} for cname in cnames}
    # client mean
    for cname in cnames:
        for rec in records:
            for e in ele_list:
                cross_meanAccs[cname][e].append(rec[cname][e])
        for e in ele_list:
            cross_meanAccs[cname][e] = np.mean(np.array(cross_meanAccs[cname][e]))
    
    # total mean
    cross_meanAccs['mean'] = {e:[] for e in ele_list}
    cross_meanAccs['std'] = {e:[] for e in ele_list}
    for e in ele_list:
        rec_meanAccs = []
        for rec in records:
            client_mean = []
            for cname in cnames:
                # we don't count the global model in our mean performance
                if 'global_model' in cname:
                    continue
                client_mean.append(rec[cname][e])
            client_mean = np.mean(np.array(client_mean))
            rec_meanAccs.append(client_mean)
        rec_meanAccs = np.array(rec_meanAccs)
        cross_meanAccs['mean'][e] = np.mean(rec_meanAccs)
        cross_meanAccs['std'][e] = np.std(rec_meanAccs)
    
    df = pd.DataFrame()
    for k,v in cross_meanAccs.items():
        df.loc[k,ele_list] = v
    
    return df

# only preserve serveral strongest link for each client
def graph_truncate(graph,link_num):

    _,idx = torch.topk(graph,link_num,dim = 1)

    mask = torch.zeros(graph.shape)
    for i in range(mask.shape[0]):
        mask[i,idx[i]] = 1

    return graph*mask
    #return mask


def mean_diff(vmatrix,diff_rate,threshold = 0.1):
    
    vmean = torch.mean(vmatrix,dim = 0)

    # compute resd
    resd = vmatrix - vmean
    mean_res = torch.mean(resd**2,dim = 0)
    # compute vmean
    measure = (torch.mean(vmatrix**2,dim = 0)+vmean**2)/2
    #difp = torch.mean(mean_res/torch.mean(vmatrix**2,dim = 0))
    difp = torch.mean(mean_res/measure)
    diff_rate = 1 #if difp >= threshold else 1
    #print(difp)
    return vmatrix-diff_rate*vmean


# in-cluster    link: 1/size(cluster)
# inter-cluster link: 0
def cluster_uniform_graph(num,csize):
    A = torch.zeros((num,num))
    
    cnum = num//csize
    for i in range(cnum):
        A[i*csize:(i+1)*csize,:][:,i*csize:(i+1)*csize] = 1/csize
    
    return A

def get_roc_auc(label,score):

    Fscore = F.softmax(score,dim = 1).cpu()
    label = label.cpu()
    #print(Fscore.shape)
    #print(label.shape)
    #label = F.one_hot(label,Fscore.shape[1]).cpu()
    return 0

    #return roc_auc_score(label,Fscore,average = 'macro')

#   randomly generate weighted graph
def random_con_graph(num):
    return torch.tensor(np.random.random((num,num))).float()


#   compute similarity based on given features
def f2sim(features):
    features = F.normalize(mean_diff(features,1,0),p = 2,dim = 1)
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

    
        

    

