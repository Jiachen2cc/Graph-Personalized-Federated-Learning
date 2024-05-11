# this file contains aid function for building the graph

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import copy
#from model_compression import continous_compress,compress_shape
import cvxpy as cp




def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

def nearest_neighbors(X, k, metric):
    # 指定每个node的度数，给出相应的graph
    adj = kneighbors_graph(X, k, metric=metric)
    # 转化为正常矩阵
    adj = np.array(adj.todense(), dtype=np.float32)
    # 添加自环
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    # metric 应该是距离的度量方式
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0]) # 0,1,2,...,n-1 共 n 个 nodes
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))

    return s, d

#分块计算knn以加快速度
def knn_fast(X, k, b):
    #归一化
    X = F.normalize(X, dim=1, p=2)   
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()

    #从0开始，以步长b扫描整个tensor
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        #截取长度为b的tensor
        sub_tensor = X[index:index + b]
        # cosine相似度
        similarities = torch.mm(sub_tensor, X.t())
        #找到每行最大的k+1个相似度的坐标，返回的ind，为列坐标
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)

        #求范数和
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    #归一化
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values

# apply certain nonlinearity function
def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2

# we only don't consider the negative link between client
def no_negtive(adj):

    mask = (adj >= 0).float().cuda()
    p_adj = adj*mask#/torch.max(adj)  

    return p_adj


def normalize(adj, mode, sparse=False):
    EOS = 1e-10
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            #inv_sqrt_degree = torch.diag(inv_sqrt_degree)
            #norm_adj = torch.matmul(inv_sqrt_degree,adj)
            #norm_adj = torch.matmul(norm_adj,inv_sqrt_degree.T)
            return inv_sqrt_degree[:,None] * adj * inv_sqrt_degree[:,None]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            inv_degree = torch.diag(inv_degree)
            norm_adj = torch.matmul(inv_degree,adj)
            return norm_adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def matri2dict(models_state, paramtreix, keys, key_shapes):

    models_dic = copy.deepcopy(models_state)
    for i in range(len(models_dic)):
        pointer = 0
        for k in range(len(key_shapes)):
            num_p = 1
            for n in key_shapes[k]:
                num_p *= n
            models_dic[i][keys[k]] = paramtreix[i][pointer:pointer + num_p].reshape(key_shapes[k])
            pointer += num_p
    return models_dic


def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    #print(filter_mode)
    keys = []
    param_vector = None

    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    #print(keys)
    return param_vector

def state_dict2metrix(models_state):
    parameter_metrix = []
    
    for state_dic in models_state:
        parameter_metrix.append(sd_matrixing(state_dic).clone().detach())
    param_metrix = torch.stack(parameter_metrix)

    return param_metrix

'''
def para2metrix(models_state,cmode,cdim):

    key_shapes = [list(param.data.shape) for _,param in models_state[0].items()]

    param_metrix = state_dict2metrix(models_state)

    if cmode == 'continous':
        compress_param = continous_compress(param_metrix.cpu(),cdim)
    elif cmode == 'shape':
        compress_param = compress_shape(param_metrix.cpu(),key_shapes)
    elif cmode is None:
        # simple avg pool compress
        stri = param_metrix.shape[1]//cdim
        compress_param = torch.avg_pool1d(param_metrix,stri,stri)
    
    return compress_param
'''
def flattenw(w):
    #return torch.cat([v.flatten() for v in w.data()])
    return torch.cat([v.flatten() for v in w.values()])

# graph learning for pFedgraph

def cal_model_cosine_difference(clients,initial_global):
    diff_matrix = torch.zeros((len(clients),len(clients)))
    dws = []
    for i in range(len(clients)):
        dwc = {}
        for key in initial_global.keys():
            if 'graph' in key:
                dwc[key] = clients[i].W[key] - initial_global[key]
        #exit(0)
        dwc = sd_matrixing(dwc)
        dws.append(dwc)
    
    for i in range(len(clients)):
        for j in range(len(clients)):
            diff = - torch.nn.functional.cosine_similarity(dws[i].unsqueeze(0),dws[j].unsqueeze(0))
            if diff < -0.9:
               diff = -1.0
            diff_matrix[i,j] = diff
            
    return diff_matrix
            
            
def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix

def update_graph_matrix(graph_matrix,clients,initial_global,size_freqs,alpha):
    
    diff_matrix = cal_model_cosine_difference(clients,initial_global)
    #print('difference')
    #print(diff_matrix)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix,range(len(clients)),diff_matrix,
                                                    alpha,size_freqs)
    
    return graph_matrix

def graph_aggregate(graph_matrix,clients,device):
    
    paramvector = {i:sd_matrixing(clients[i].W) for i in range(len(clients))}
    cluster_model_vectors = {}
    
    for i in range(len(clients)):
        tmp_state = torch.zeros(paramvector[i].shape)
        cluster_state = torch.zeros(paramvector[i].shape)
        agg_weight = graph_matrix[i]
        
        for nid in range(len(clients)):
            para = paramvector[nid]
            tmp_state += agg_weight[nid] * para
            cluster_state += agg_weight[nid] * para / torch.linalg.norm(para)
        
        cluster_model_vectors[i] = cluster_state
        
        # load models
        clients[i].load_param_matrix(tmp_state.to(device))
        
    
    return cluster_model_vectors
        
        
        
            
            