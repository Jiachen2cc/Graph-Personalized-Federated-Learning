import numpy as np
from torch_geometric.data import Data
import torch
from analyze_graphs import *


def distance_matrix(dis_f, elements):

    dism = torch.zeros(len(elements), len(elements))

    for i in range(len(elements)):
        e_i = elements[i]
        for j in range(len(elements)):
            e_j = elements[j]
            dism[i, j] = dis_f(e_i, e_j)

    return dism

# align the length of different distributions
# assume distribution is stored in numpy array


def len_align(distributions):

    max_len = np.max(np.array([len(dis) for dis in distributions]))
    align_diss = []
    for dis in distributions:
        align_dis = np.zeros(max_len)
        align_dis[:len(dis)] += dis
        align_diss.append(align_dis)

    return align_diss


def tri_num_disb(dataset):
    # count triangle num distribution
    tri_nums = np.array([count_tri(graph) for graph in dataset])

    max_tri = np.max(tri_nums)

    tri_dis = np.zeros(max_tri + 1)  # tri_num in [0,max_tri]
    for i in range(max_tri + 1):
        tri_dis[i] = np.sum(tri_nums == i)
    tri_dis /= np.sum(tri_dis)

    return tri_dis

def avg_tri_num(dataset):

    tri_nums = np.array([count_tri(graph) for graph in dataset])
    return np.mean(tri_nums)

def deg_disb(dataset):
    # count degree distribution
    degdis_graphs = [count_degree_distribution(graph) for graph in dataset]

    max_len = np.max(np.array([len(dis) for dis in degdis_graphs]))

    degdis_dataset = np.zeros(max_len)
    for degdis in degdis_graphs:
        degdis_dataset[:len(degdis)] += degdis
    degdis_dataset /= np.sum(degdis_dataset)

    return degdis_dataset

def hop2_disb(dataset):
    hop2dis_graphs = [count_hop2_neighbor(graph) for graph in dataset]

    max_len = np.max(np.array([len(dis) for dis in hop2dis_graphs]))

    hop2dis_dataset = np.zeros(max_len)
    for hop2dis in hop2dis_graphs:
        hop2dis_dataset[:len(hop2dis)] += hop2dis
    hop2dis_dataset /= np.sum(hop2dis_dataset)

    return hop2dis_dataset
    
'''
def structure_sim(disbs):
   
    features_similarity_matrix = {}

    for key in disbs.keys():

        feature_dis = disbs[key]
        align_dis = len_align(feature_dis)

        fdis_matrix = distance_matrix(JS_d,align_dis)
        fsim_matrix = torch.max(fdis_matrix, dim = 1).values[:,None] - fdis_matrix

        features_similarity_matrix[key] = fsim_matrix
    
    return features_similarity_matrix
'''

def structure_sim(fdisbs,eps):

    align_disbs = len_align(fdisbs)
    
    fdis_matrix = distance_matrix(JS_d,align_disbs)
    fsim_matrix = (1+eps)*torch.max(fdis_matrix, dim = 1).values[:,None] - fdis_matrix

    return fsim_matrix

def average_degree(dataset):
    
    all_deg = np.array([graph.edge_index.shape[1]/(graph.x.shape[0]*(graph.x.shape[0] - 1)) for graph in dataset])
    
    return np.mean(all_deg)


def rlabelnode(dataset):

    report = {}

    for graph in dataset:

        label = graph.y.item()
        if label not in report.keys():
            report[label] = []
        report[label].append(graph.x.shape[0])
    
    for label in report.keys():

        mean_node_num = np.mean(np.array(report[label]))
        report[label] = mean_node_num
        
    print(report)

def simplepre(dataset,bound):

    ansnum = 0
    for graph in dataset:

        if graph.x.shape[0] > bound and graph.y.item() == 0:
            ansnum += 1
        elif graph.x.shape[0] <= bound and graph.y.item() == 1:
            ansnum += 1
    
    print(ansnum/len(dataset))


def avg_nodenum(dataset):

    nodenum = 0
    for graph in dataset:
        nodenum += graph.x.shape[0]
    nodenum /= len(dataset)
    
    return nodenum


def homo_analyze(dataset):

    avg_homo,avg_hetero = 0,0

    for graph in dataset:

        homo_edge,hetero_edge = homo_edge_count(graph)
        avg_homo += homo_edge
        avg_hetero += hetero_edge
    
    avg_homo /= len(dataset)
    avg_hetero /= len(dataset)

    homo_rate = avg_homo /(avg_homo + avg_hetero)
    
    print('average homophily edges:{:.4f},homophily edge rate:{:.4f}'.format(avg_homo,homo_rate))
    return avg_homo, homo_rate

def pfeature(dataset):

    res = torch.stack([network_properties(graph) for graph in dataset],dim = 0)
    dfeature = torch.mean(res,dim = 0)
    dstd = torch.std(res,dim = 0)


    return [dfeature,dstd]

def cons_feature(datasets):

    fm = torch.stack([pfeature(d)[0] for d in datasets],dim = 0)
    fstd = torch.stack([pfeature(d)[1] for d in datasets],dim = 0)
    
    fmean = torch.mean(fm,dim = 0)
    #fres = (fm - fmean)**2
    mean_res = torch.std(fm,dim = 0)
    fm = (fm - fmean)/mean_res
    # row normalization
    fm /= (torch.sum(fm*fm,dim = 1)[:,None])**0.5
    # compute
    simi = torch.matmul(fm,fm.T)
    simi = simi * (simi>=0).float()
    #simi /= torch.sum(simi,dim = 1)[:,None]
    #print(simi)
    return simi

def pg_analysis(clients):
    cons_feature([c.train_data for c in clients])
    cons_feature([c.sample_uniform() for c in clients])
    

    fm = torch.stack([pfeature(c.train_data)[0] for c in clients],dim = 0)
    fstd = torch.stack([pfeature(c.train_data)[1] for c in clients],dim = 0)
    
    fmean = torch.mean(fm,dim = 0)
    #fres = (fm - fmean)**2
    mean_res = torch.std(fm,dim = 0)
    fm = (fm - fmean)/mean_res
    # row normalization
    fm /= (torch.sum(fm*fm,dim = 1)[:,None])**0.5
    # compute
    simi = torch.matmul(fm,fm.T)
    simi = simi * (simi>=0).float()
    #simi /= torch.sum(simi,dim = 1)[:,None]
    #print(simi)
    return simi


def get_meanfeature(clients):

    return torch.stack([pfeature(c.train_data)[0] for c in clients],dim = 0)



def distance_norm(distance):

    #res = torch.mean(distance,dim = 0)[None,:] - distance
    res = distance
    res = torch.mean(res,dim = 1)[:,None] - res
    res = res * (res >= 0).float()
    row_sum = torch.sum(res,dim = 1)
    res = res / row_sum[:,None]

    print(res)

def simi_norm(feature):

    df = np.stack(feature,axis = 0) - np.mean(feature,axis = 0)[None,:]
    ndf = df/((np.sum(df*df,axis = 1))**0.5)[:,None]
    #ndis -= np.mean(ndis,axis = 0)[None,:]
    simi = torch.tensor(np.matmul(ndf,ndf.T))
    simi = simi * (simi >= 0).float()
    row_sum = torch.sum(simi,dim = 1)
    simi = simi / row_sum[:,None]

    print(simi)


# design ideal client graph based on label distributions
def label_dis(clients,eps):

    align_disbs = len_align([c.get_labeldis() for c in clients])
    fdis_matrix = distance_matrix(JS_d,align_disbs)
    #fsim_matrix = (1+eps)*torch.max(fdis_matrix, dim = 1).values[:,None] - fdis_matrix
    fsim_matrix = (1+eps)*torch.mean(fdis_matrix, dim = 1)[:,None] - fdis_matrix + 1e-12
    
    #filter 
    mask = (fsim_matrix >= 0).float()
    fsim_matrix = mask * fsim_matrix

    #row normalization
    fsim_matrix /= torch.sum(fsim_matrix,dim = 1)[:,None]
    return fsim_matrix











