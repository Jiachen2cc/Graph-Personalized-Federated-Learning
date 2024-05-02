import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GraphConvolution(torch.nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, sparse = False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.sparse = sparse
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
   
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        if self.sparse: 
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'
      

class GAE_dense(torch.nn.Module):
    
    def __init__(self, in_dim, hidden, sparse = False):
        
        super(GAE_dense, self).__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        
        self.layer1 = GraphConvolution(in_dim,hidden,True,sparse)
        self.layer_norm = nn.LayerNorm([hidden])
        self.layer2 = GraphConvolution(hidden,hidden//4,True,sparse)
    
    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
    
    def forward(self, x, A):
        x = self.layer1(x,A)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.layer2(x,A)
        x = torch.mm(x,x.t())
        return x
        
        