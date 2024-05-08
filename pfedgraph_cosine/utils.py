import torch
import numpy as np
import copy
import cvxpy as cp
    
def compute_local_test_accuracy(model, dataloader, data_distribution):

    model.eval()

    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif  similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0), weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    # print("model_similarity_matrix" ,model_similarity_matrix)
    return model_similarity_matrix

def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1, similarity_matric):
    # index_clientid = torch.tensor(list(map(int, list(nets_this_round.keys()))))     # for example, client 'index_clientid[0]'s model difference vector is model_difference_matrix[0] 
    for net in nets_this_round.values():
        net.to('cpu')
    index_clientid = list(nets_this_round.keys())
    # model_difference_matrix = cal_model_difference(index_clientid, nets_this_round, nets_param_start, difference_measure)
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lambda_1, fed_avg_freqs)
    # print(f'Model difference: {model_difference_matrix[0]}')
    # print(f'Graph matrix: {graph_matrix}')
    return graph_matrix


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
  
def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def aggregation_by_graph(graph_matrix, nets_this_round, global_w, global_p):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_p))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        # if client_id==0:
        #     print(f'Aggregation weight: {aggregation_weight_vector}. Summation: {aggregation_weight_vector.sum()}')
        
        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        for neighbor_id in nets_this_round.keys():
            net_para = weight_flatten_all({k:v for k,v in nets_this_round[neighbor_id].named_parameters()}).detach()
            cluster_model_state += net_para * (aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))
               
    for client_id in nets_this_round.keys():
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])
    
    return cluster_model_vectors

def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)

def compute_loss(net, test_data_loader):
    net.eval()
    loss, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss += torch.nn.functional.cross_entropy(out, target).item()
            total += x.data.size()[0]
    net.to('cpu')
    return loss / float(total)



import torch
from torch_geometric.utils import to_networkx, degree, to_dense_adj, to_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import sparse as sp


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

    
def init_structure_encoding(args, gs, type_init):
    
    if type_init == 'rw':
        for g in gs:
            # Geometric diffusion features with Random Walk
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE_rw=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE_rw,dim=-1)

            g['stc_enc'] = SE_rw
    
    elif type_init == 'dg':
        for g in gs:
            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = SE_dg
    
    elif type_init == 'rw_dg':
        for g in gs:
            # SE_rw
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE,dim=-1)

            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    return gs

