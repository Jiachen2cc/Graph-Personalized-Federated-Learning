import numpy as np
import random
from sklearn.model_selection import StratifiedKFold


dataset_format = {
    'TuDataset':['AIDS','BZR','COX2','DD','DHFR','ENZYMES','NCI1','PROTEINS',
                 'PRC_MR','NCI-H23','alchemy_full','DBLP_v1','PC-3','MCF-7','MOLT-4','NCI-H23',
                 'OVCAR-8','P388','SF-295','SN12C','SW-620','UACC257','Yeast'],
    'GNNBenchmark':['PATTERN','CLUSTER','MNIST','CIFAR10','TSP','CYCLES'],
}

def cal_num(total,n_split):

    lb = total // n_split

    deta = total - lb * n_split

    prior = [lb for i in range(n_split)]
    if deta == 0:
        return prior
    else:
        for i in range(deta):
            prior[i] = prior[i] + 1
    
    return prior


def balancelabel_downsample(graphs):
    
    random.seed(0)
    labels = [g.y.item() for g in graphs]

    label_num = np.max(labels)+1
    label_dict = {k:[] for k in range(label_num)}

    for idx in range(len(labels)):
        label_dict[labels[idx]].append(idx)

    downnum = min(len(label_dict[k]) for k in range(label_num))

    res = []
    for k in range(label_num):
        res.extend(random.sample(label_dict[k],downnum))
    
    downgraphs = [graphs[idx] for idx in res]
    return downgraphs

def kfold_split(graphs,fold_num,seed):

    skf = StratifiedKFold(n_splits = fold_num, shuffle = True, random_state = seed)
    labels = [graph.y.item() for graph in graphs]
    #print(labels)
    train_idx_list = []
    test_idx_list = []

    for idx in skf.split(np.zeros(len(labels)),labels):
        train_idx,test_idx = idx
        train_idx_list.append(train_idx)
        test_idx_list.append(test_idx)
    
    return train_idx_list, test_idx_list

def soft_dirichlet(alpha, num_classes, num_client, upper = 0.9):
    priors = np.random.dirichlet(
        alpha = [alpha] * num_classes, size = num_client)
    
    for i in range(num_client):
        x = priors[i]
        resv = np.sum(np.maximum(x-upper,0))
        snum = np.sum(x <= upper)
        x[x > upper] = upper
        x[x < upper] += resv/snum
    
    return priors