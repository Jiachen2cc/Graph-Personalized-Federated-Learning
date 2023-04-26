from torch_geometric.data import Dataset,Data
import pickle
import numpy as np
import networkx as nx
from functest import *

'''
class Scalefree(Dataset):
    
    def __init__(self,root,transform = None, pre_transform = None,node_label = 'onehot',graph_label = 'path'):

        super(Scalefree, self).__init__(root,transform,pre_transform)
        # sample root = 'data/artifdataset/scalefree_onehot_path'
        _,node_label,graph_label = root.split('_')
        
        self.raw_file_names = 'scalefree.pkl'
        self.processed_file_names = 'data.pt'
    
    def download():
        pass

    def process():
'''

clustering_bins = np.linspace(0.3, 0.6, 7)
path_bins = np.linspace(1.8, 3.0, 7)

# default node label is onehot degree label
def scalefree_data(path, graph_label = 'path'):

    with open('{}/scalefree.pkl'.format(path),'rb') as f:
        graphs = pickle.load(f)
    
    data_list = []

    for graph in graphs:

        if graph_label == 'path':
            path = nx.average_shortest_path_length(graph)
            label = np.digitize(path,path_bins)
        elif graph_label == 'clustering':
            clustering = nx.average_clustering(graph)
            label = np.digitize(clustering,clustering_bins)
        data_list.append(onehotdegree_data(graph,torch.tensor(label)))
    dataset = feature_padding(data_list)

    return dataset
        





