import random
import numpy as np
from dataprocess.utils import *
from sklearn.model_selection import StratifiedKFold

def label_skew_balance(graphs, num_client, seed, alpha=4):

    random.seed(seed)
    np.random.seed(seed)

    labels = np.array([graph.y.item() for graph in graphs])
    num_classes = labels.max() + 1
    
    # get soft dirichlet label distribution
    class_priors = soft_dirichlet(alpha, num_classes, num_client)
    
    prior_cusum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(labels == i)[0] for i in range(num_classes)]

    # compute the size of each class
    class_amount = [len(idx_list[i]) for i in range(num_classes)]
    
    # compute the size of each local dataset
    client_sample_nums = np.array(cal_num(len(labels), num_client))
    # print(client_sample_nums)

    client_indices = [np.zeros(client_sample_nums[cid]).astype(
        np.int64) for cid in range(num_client)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_client)
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cusum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]
            break
            
    graph_chunks = []

    for i in range(num_client):
        graph_chunks.append([graphs[idx] for idx in client_indices[i]])
    for client_idx in client_indices:
        label_res = np.zeros(np.max(labels)+1).astype(int)
        for idx in client_idx:
            label_res[int(labels[idx])] += 1
        #print(label_res)
    return [client_indices[i] for i in range(num_client)]
    
    


def toy_split(graphs,rate = 0.5):

    labels = np.array([g.y.item() for g in graphs])
    datasets = [[],[]]

    index0 = [k[0] for k in np.argwhere(labels == 0)]
    index1 = [k[0] for k in np.argwhere(labels == 1)]

    num0 = min(int(rate*len(index0)),len(index0) - 10)
    num1 = max(len(labels)//2 - num0,10)
    
    #print(num0,num1)

    c00 = np.random.choice(index0,num0,False)
    c01 = np.random.choice(index1,num1,False)

    datasets[0] = [idx for idx in c00] + [idx for idx in c01]
    #datasets[0] = [graphs[idx] for idx in c00] + [graphs[idx] for idx in c01]

    c10 = set(index0) - set(c00)
    c11 = set(index1) - set(c01)

    #print(len(c10),len(c11))
    datasets[1] = [idx for idx in c10] + [idx for idx in c11]
    #datasets[1] = [graphs[idx] for idx in c10] + [graphs[idx] for idx in c11]
    
    #print('client_0 label 0:{}, label 1:{}'.format(len(c00),len(c01)))
    #print('client_1 label 0:{}, label 1:{}'.format(len(c10),len(c11)))
    
    return datasets


def uniform_split(graphs,num_splits):

    skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = 0)

    labels = np.array([g.y.item() for g in graphs])
    idx_list = []

    clients_idx = []

    for idx in skf.split(np.zeros(len(labels)),labels):
        idx_list.append(idx)

    for idx in range(num_splits):
        _,client_idx = idx_list[idx]
        clients_idx.append(client_idx)
    
    return clients_idx

# uniformly split the sub_chunks
# # assume 2*num_subchunk >= num_client >= num_chunk
def subchunk_split(graphs,sub_chunk_idxs,num_splits):

    all_chunk_idx = []

    for idxs in sub_chunk_idxs:
        sub_graphs = [graphs[i] for i in idxs]
        for cidxs in uniform_split(sub_graphs,num_splits):
            all_chunk_idx.append([idxs[i] for i in cidxs])
    return all_chunk_idx