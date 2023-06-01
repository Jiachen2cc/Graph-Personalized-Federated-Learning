import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


# share classifier and center part but not share feature extractor
class featureGIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(featureGIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))
        
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
        '''
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))
        '''
    def forward(self):
        pass

class classiGIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(classiGIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        #self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))
    
    def forward(self):
        pass
    
class serverGIN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                           torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
        #self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.dropout = 0

    def forward(self,data):

        x, edge_index, batch = data.x,data.edge_index,data.batch
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x,edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training = self.training)
        x = global_add_pool(x,batch)
        #x = self.post(x)
        #x = F.dropout(x, self.dropout, training = self.training)
        x = F.log_softmax(x,dim = 1)

        return x

class GINextractor(torch.nn.Module):

    def __init__(self,nfeat,nhid):
        super().__init__()
        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat,nhid))

    def forward(self,data):
        x = data.x
        x = self.pre(x)
        res = data.clone()
        res.__setitem__('x',x)
        return res
        
class GINclassifier(torch.nn.Module):

    def __init__(self,nhid,nclass):
        super(GINclassifier, self).__init__()
        #self.post = torch.nn.Sequential(torch.nn.Linear(nhid,nclass))
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid,nclass))
        

class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
#sys.path.append("models/")
#from mlp import MLP

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model, if there is only a single layer , the model is linear
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model(not linear)
            # one linear layer is followed by a BatchNorm layer
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class serverGraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, hidden_dim, learn_eps, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer) why use this?
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(serverGraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))  #不知道这是啥
        
        #self.pre = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        
        #网络的每个非输出层都是由 num_mlp_layers 层 mlp构成
        for layer in range(self.num_layers-2):
            
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            #self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        #每层都配备了一层linear_prediction ?
        '''
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
        '''

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer) why use this?
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))  #不知道这是啥
        
        self.pre = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
        self.pre_batchnorm = nn.BatchNorm1d(hidden_dim)
        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        
        #网络的每个非输出层都是由 num_mlp_layers 层 mlp构成
        for layer in range(self.num_layers-2):

            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        #每层都配备了一层linear_prediction ?
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph
        #获取将batch拼接后整体大图的邻接矩阵
        #compute the maximum number of neighbors within the graphs in the current minibatch
        #整个batch中出现的最大度数
        #希望把每个节点的度数都pad成一样大小
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g)) # 为每次padding记录起始位置
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)      #最后得到所有graph的整体list(对应graph的段长为len(graph)*(max_deg+[1(added only when learn_eps = False)])



    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix
        #相当于形成了一个大图，包含若干块状的稀疏矩阵
        edge_mat_list = []
        start_idx = [0]
        # add the offset
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])# edge_mat is Tensor
        Adj_block_idx = torch.cat(edge_mat_list, 1)         #按列拼接  2*n
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1]) #共包含多少条边

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]    # the total number of nodes
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)
        
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)


    def __preprocess_graphpool(self, batch_graph): 
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        #不是特别理解是什么意思
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        if layer == 0:
            pooled_rep = self.pre(pooled)
            h = self.pre_batchnorm(pooled_rep)
        else:
            pooled_rep = self.mlps[layer - 1](pooled)
            h = self.batch_norms[layer - 1](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        if layer == 0:
            pooled_rep = self.pre(pooled)
            h = self.pre_batchnorm(pooled_rep)
        else:
            pooled_rep = self.mlps[layer - 1](pooled)
            h = self.batch_norms[layer - 1](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat
        
        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0
    
        #perform pooling over all nodes in each graph in every layer（每一层的特征都会收集）
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer