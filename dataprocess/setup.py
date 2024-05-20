from torch_geometric.datasets import TUDataset,GNNBenchmarkDataset
from torch_geometric.transforms import OneHotDegree
from utils import get_maxDegree
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
from torch_geometric.data.datapipes import functional_transform
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from dataprocess.utils import *
from dataprocess.data_split import *
from dataprocess.data_property import network_properties
import os
import time

class SetUp:
    
    def __init__(self,args):
        self.args = args
        self.graphs = self.load_dataset()
        self.property = self.extract_property()
        self.num_node_features = self.graphs[0].num_node_features
        self.num_graph_labels = max([g.y.item() for g in self.graphs])+1
        self.splited_graphs = self.split_dataset()
        self.show_label_distribution()
        
    
    # load dataset according to our requirements
    def load_dataset(self):
        data,datapath = self.args.data_group, self.args.datapath
        if data == "COLLAB":
            tudataset = TUDataset(
                f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(
                f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(
                f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        #elif data == 'scalefree':
        #    tudataset = scalefree_data('data/artifdataset')
        #elif data in ['ogbg-molhiv']:
        #    tudataset = ogb_process(PygGraphPropPredDataset(name = 'ogbg-molhiv', root = f"{datapath}/"),data)
        elif data in dataset_format['TuDataset']:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr = False)
            if self.args.convert_x:
                maxdegree = get_maxDegree(tudataset)
                tudataset = TUDataset(
                    f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
        elif data in dataset_format['GNNBenchmark']:
            tudataset = GNNBenchmarkDataset(f"{datapath}/Benchmark",data,split = 'train') + GNNBenchmarkDataset(f"{datapath}/Benchmark",data,split = 'val') + GNNBenchmarkDataset(f"{datapath}/Benchmark",data,split = 'test')
            #graphs = [g for g in tudataset]

        graphs = [g for g in tudataset]
        if(data == 'Yeast'):
            graphs = balancelabel_downsample(graphs)
        
    
        return graphs   
    
    
    def split_dataset(self):
        self.args.num_clients *= self.args.num_splits
        # if split method is toy split
        if self.args.split_way == 'toy':
            graphs_chunks_idx = toy_split(self.graphs,self.args.toy_rate)
            if self.args.num_splits > 1:
                graphs_chunks_idx = subchunk_split(self.graphs,graphs_chunks_idx,
                    self.args.num_splits if self.args.num_clients > 1 else self.args.num_splits//2)
        elif self.args.split_way == 'blabel_skew':
            # if split method is label skew 
            graphs_chunks_idx = label_skew_balance(self.graphs, self.args.num_clients, 
                                    self.args.seed, self.args.skew_rate)
        
        splitedData = {}
        for idx, chunks_idx in zip(list(range(self.args.num_clients)),graphs_chunks_idx):
            ds = f'{idx}-{self.args.data_group}'
            chunks = [self.graphs[idx] for idx in chunks_idx]
            chunks_property = [self.property[idx,:] for idx in chunks_idx]
            train_idx_list, test_idx_list = kfold_split(chunks,self.args.fold_num,
                                            self.args.seed)
            splitedData[ds] = (chunks, {'train':train_idx_list,'test':test_idx_list},
                            self.num_node_features,
                            self.num_graph_labels, self.args.data_group,
                            chunks_property)
        
        return splitedData
    
    
    def show_label_distribution(self):
        
        label_all = [np.array([g.y.item() 
                        for g in self.splited_graphs[k][0]]) 
                     for k in self.splited_graphs.keys()]
        label_dis = np.zeros((self.num_graph_labels,len(label_all)))
        for i in range(self.num_graph_labels):
            for j, labels in enumerate(label_all):
                label_dis[i][j] = np.sum(labels == i)
        print(label_dis)

    def extract_property(self):
        # return chose statistics properties of dataset   
        property_path = os.path.join(self.args.propertypath,self.args.data_group+'.pt')
        property_tensor = None
        if not os.path.exists(property_path):
            property_tensor = torch.stack([network_properties(g) for g in self.graphs],dim = 0).cpu()
            torch.save(property_tensor,property_path)
        else:
            property_tensor = torch.load(property_path)
        return property_tensor
        
        