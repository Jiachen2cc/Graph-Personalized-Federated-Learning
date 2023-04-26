from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from utils import get_maxDegree
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
from torch_geometric.data.datapipes import functional_transform
from sklearn.model_selection import StratifiedKFold
import numpy as np
from perturbations import *
from analyze_dataset import *
from dataset_builder import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

group2datas = {
    'molecules':['MUTAG','BZR','COX2','DHFR','PTC_MR','AIDS','NCI1'],
    'molecules_tiny':['MUTAG','BZR','COX2','DHFR','PTC_MR','NCI1'],
    'small':["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR",                   # small molecules
                    "ENZYMES", "DD", "PROTEINS"],
    'mix':["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",   # small molecules
                    "ENZYMES", "DD", "PROTEINS",                                # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"],
    'biochem':["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"],
    'exp2': ['NCI1','COLLAB'],
    'exp3': ['AIDS','DD','IMDB-MULTI'],
    'exp4': ['AIDS','NCI1','DD','PROTEINS','COLLAB','IMDB-MULTI'],
    'clf_test':['AIDS','DD','PROTEINS'],
    #'easy':['AIDS','BZR','COX2','COLLAB']
    #'easy':['AIDS','BZR','COX2'],
    'easy':['IMDB-BINARY','BZR','COX2'],
    'easy1':['AIDS','IMDB-BINARY'],
    'exp':['COX2'],
    'H1':['NCI1','PROTEINS','IMDB-BINARY'],
    'H2':['NCI1','IMDB-BINARY','IMDB-MULTI'],
    'Htest1':['BZR','ENZYMES'],
    'Htest2':['DD','COX2'],
    'Htest3':['IMDB-BINARY','PROTEINS'],
    'Htest4':['BZR','DD'],
    'Htest5':['BZR','PROTEINS'],
    'Htest6':['COX2','PROTEINS'],
    'Htest7':['IMDB-BINARY','ENZYMES'],
    'Htest8':['BZR','COX2'],
    'Htest9':['PROTEINS','DD'],
    'Htest10':['IMDB-BINARY','BZR'],
    'Htest11':['IMDB-BINARY','DD'],
    'Htest12':['DD','ENZYMES'],
    'stest':['MUTAG','DHFR','NCI1'],
    'ltest':['DHFR','AIDS','IMDB-BINARY'],
    'structure':['IMDB-BINARY','IMDB-MULTI'],
    'scale':['scalefree'],
}

easy_datasets = ['AIDS','BZR','COX2','COLLAB']
nofeature_datasets = ['COLLAB','IMDB-BINARY','IMDB-MULTI']
hasattr_datasets = ['AIDS','BZR','COX2','DHFR','ENZYMES','PROTEINS']

gcfl_param = {
    'PROTEINS': [0.03,0.06],
    'IMDB-BINARY': [0.025,0.045],
    'NCI1':[0.04,0.08],
    'molecules':[0.07,0.28],
    'biochem':[0.07,0.35],
    'mix':[0.08,0.04]
}

def data_process(datapath, data, convert_x = False):

    if data == "COLLAB":
        tudataset = TUDataset(
            f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(
            f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(
            f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    elif data == 'scalefree':
        tudataset = scalefree_data('data/artifdataset')
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr = False)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(
                f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    return tudataset

def load_attr(datapath, data):

    if data not in hasattr_datasets:
        raise Exception('This dataset does not has any attributes')
    
    dataset = TUDataset(f'{datapath}/TUDataset', data, use_node_attr = True)

    graphs = [g for g in dataset]

    return graphs


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

@functional_transform('normalization_feature')
class NormalizationFeature(BaseTransform):

    '''
    Args:
       norm_type: the type of normalization 
       F_norm: x -> x/norm_x
       minmax_norm: x -> (x-min(x))/(max(x)-min(x))
       gauss_norm: x-> (x-mean(x))/sigma(x)
    '''
    def __init__(
        self,
        norm_type: str
    ):
        self.norm_type = norm_type

    
    def __call__(self, data: Data) -> Data:

        x = data.x

        if self.norm_type == 'F_norm':
            norm_x = torch.norm(x,dim = 1,keepdim = True)
            res = x / norm_x
        elif self.norm_type == 'minmax_norm':
            minx = torch.min(x,dim = 1,keepdim = True).values
            maxx = torch.max(x,dim = 1,keepdim = True).values
            res = (x - minx)/(maxx - minx)
        elif self.norm_type == 'gauss_norm':
            sigma = torch.std(x, dim = 1,keepdim = True)
            meanx = torch.mean(x, dim = 1,keepdim = True)
            res = (x - meanx)/sigma
        
        data.x = res

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


def graph_process(data,graphs,args):

    # feature normalize:
    graphs = feature_normalize(graphs)

    # show the label distribution of datasets
    print(data)
    rlabelnode(graphs)
    print(avg_nodenum(graphs))

    # enlarge the heterogeneity gap between datasets
    if args.hetero and data == args.target_dataset:
        #graphs = node_downsample(graphs,sample_rate = args.downsample_rate)
        graphs = edge_noise(graphs,0,0,'Gaussian')
    
    if args.difficulty:

        if args.noise_type == 'feature':
            graphs = node_feature_noise(graphs,args.noise_rate,args.seed,args.SNR,args.fptype,args.per_edge,args.ex_rate)
        
        elif args.noise_type == 'structure':
            print(data)
            print('average degree before structure perturbation')
            print(average_degree(graphs))
            graphs = edge_noise(graphs,args.prate, args.nrate, args.fmask, args.seed)
            print('average degree after structure perturbation')
            print(average_degree(graphs))

        elif args.noise_type == 'node':
            print('node perturbation!')
            node_num = np.mean(np.array([graph.x.shape[0] for graph in graphs]))
            print('average node num before node downsampling',node_num)
            graphs = node_downsample(graphs,args.downsample_rate,args.seed)
            node_num = np.mean(np.array([graph.x.shape[0] for graph in graphs]))
            print('average node num before node downsampling',node_num)

    
    print(len(graphs))

    return graphs

# can only be applied to datasets with two classes, and split two clients
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


def label_balanced_downsample(labels,down_rate,seed = 0):
    
    random.seed(0)

    label_num = np.max(labels)+1
    label_dict = {k:[] for k in range(label_num)}

    for idx in range(len(labels)):
        label_dict[labels[idx]].append(idx)
    
    res = []
    for k in range(label_num):
        sample_num = max(int(down_rate * len(label_dict[k])),10)
        res.extend(random.sample(label_dict[k],sample_num))
    
    return res


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
    '''
    for i in range(num_client - len(sub_chunk_idxs)):
        sub_graphs = [graphs[i] for i in sub_chunk_idxs[i]]
        subx,suby = toy_split(sub_graphs)
        all_chunk_idx.append([sub_chunk_idxs[i][j] for j in subx])
        all_chunk_idx.append([sub_chunk_idxs[i][j] for j in suby])
    for i in range(num_client - len(sub_chunk_idxs),len(sub_chunk_idxs)):
        all_chunk_idx.append(sub_chunk_idxs[i])
    '''
    return all_chunk_idx

def show_label_distribution(graphs):

    labels = np.array([g.y.item() for g in graphs])
    label_num = np.max(labels)+1

    label_dis = [np.sum(labels == i) for i in range(label_num)]

    print(label_dis)







    


    


    





