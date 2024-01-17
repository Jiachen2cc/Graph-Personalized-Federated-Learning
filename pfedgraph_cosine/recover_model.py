# this file contains models which aim at building the client graph through recontructing masked feature

from torch_geometric.nn import GAE,GATConv
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import torch.nn as nn
#from argument_setting import args
from pfedgraph_cosine.graph_utils import symmetrize,no_negtive,normalize
from pfedgraph_cosine.data_utils import device
#from layers import GCNConv_dense,GraphAttentionLayer
#from graph_contructor import MLP, FullParam, MLP_Diag


class GAE_dense(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim = 16, dropout = 0.5, layers = 2, act = F.relu):
        super().__init__()
 
        self.layers = torch.nn.ModuleList()
        
        self.layers.append(GCNConv_dense(input_dim,hidden_dim))
        #for i in range(layers - 2):
            #self.layers.append(GCNConv_dense(hidden_dim,hidden_dim))
        self.layers.append(GCNConv_dense(hidden_dim,latent_dim))
        
        self.act = act
        self.dropout = dropout
    
    def forward(self, x, A, normal = True,sigmoid = True):

        if normal:
            A = normalize(A,'sym',False)
        
        for i,conv in enumerate(self.layers):
            x = conv(x,A)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x,p = self.dropout)
        
        #x = F.normalize(x,p = 2,dim = 1)
        adj = torch.matmul(x,x.T)
        return torch.sigmoid(adj) if sigmoid else torch.relu(1 - torch.relu(1-adj))
        #print('learned graph before process')
        #print(adj)
        
        #return adj



# 看上去可采用的策略：利用mse_loss对GAE进行初始化(x)
# GAE通常会把dropout设成0  这里暂且设为）0.5
class GCN_DAE(torch.nn.Module):

    def __init__(self, nlayers, in_dim, hidden_dim, nfeatures, dropout, 
    dropout_adj,gen_mode,gen_hidden_dim,gen_latent_dim,gen_layer,
    gen_act = F.relu,act = F.relu):
        super().__init__()
    
        #self.gcn = GCN(in_dim,hidden_dim,nfeatures,nlayers,dropout,act)
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv_dense(in_dim,hidden_dim))
        for _ in range(nlayers - 2):
            self.layers.append(GCNConv_dense(hidden_dim,hidden_dim))
        self.layers.append(GCNConv_dense(hidden_dim,nfeatures))

        self.dropout = dropout
        self.act = act
        
        
        if gen_mode == 'GAE':
            self.graph_gen = GAE_dense(in_dim,gen_hidden_dim,gen_latent_dim,0,gen_layer,gen_act)
        if gen_mode == 'GAEAT':
            self.graph_gen = GAE_AT_dense(in_dim,gen_hidden_dim,gen_latent_dim,0.5)
        '''
        elif gen_mode == 'FullParam':
            self.graph_gen = FullParam(features.cpu(),'none',args.k,args.knn_metric,args.i,False)
        elif gen_mode == 'MLP_D':
            self.graph_gen = MLP_Diag(gen_layer,features.shape[1],args.k,args.knn_metric,args.non_linearity,args.i,False,args.mlp_act)
        elif gen_mode == 'MLP':
            self.graph_gen = MLP(gen_layer,in_dim,gen_hidden_dim,gen_latent_dim,features,10,args.k,args.knn_metric,args.non_linearity,args.i,False,args.mlp_act)
        '''
        self.gen_mode = gen_mode
            
        
        self.adj_dropout = dropout_adj
    
    def init_graph_gen(self,features,pre_A , epochs = 20, lr = 1e-2,weight_decay = 5e-4):

        optimizer = torch.optim.Adam(self.graph_gen.parameters(),lr,weight_decay = weight_decay)

        for i in range(epochs):
            optimizer.zero_grad()
        
            rec_A = self.graph_gen(features, pre_A)
            loss = F.mse_loss(rec_A,pre_A)
            loss.backward()
            optimizer.step() 

    def forward(self,features,x,A,normal = True,sigmoid = True):
        
        #A = A - A*torch.eye(A.shape[0]).to(A.device)
        A = self.graph_gen(features, A, sigmoid = sigmoid)
        #print(A)

        if normal:
            #A_ = symmetrize(A_)
            A_ = normalize(A,'sym')
        #A = F.dropout(A_,self.adj_dropout,training = self.training)
        '''
        for conv in self.layers[:-1]:
            x = conv(x,A_)
            x = self.act(x)
            x = F.dropout(x, self.dropout, training = self.training)
        x = self.layers[-1](x, A_)
        '''
        
        return x,A

        
# fixed 2 layers
class GAE_AT_dense(torch.nn.Module):

    def __init__(self,input_dim,hidden_dim,latent_dim,dropout,alpha = 0.2, nheads = 1):
         
        super(GAE_AT_dense, self).__init__()
        self.attentions = [GraphAttentionLayer(input_dim,hidden_dim,dropout,alpha) for _ in range(nheads)]
        
        '''
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        '''
        self.out_att = GraphAttentionLayer(nheads*hidden_dim,latent_dim,dropout,alpha,concat = False)
        self.dropout = dropout

    def forward(self,x,A,normal = True,sigmoid = True):

        if normal:
            A = normalize(A,'sym',False)
            # set self-loop to 0

        
        #x = F.dropout(x, self.dropout, training=self.training)
        x = torch.concat([att(x,A) for att in self.attentions], dim = 1)
        x = F.elu(self.out_att(x,A))
        x = F.dropout(x, self.dropout, training=self.training)

        #x = F.normalize(x,p = 2, dim = 1)
        # just the same as the traditional GAE
        #adj = torch.sigmoid(torch.matmul(x,x.T))
        adj = torch.sigmoid(torch.matmul(x,x.T))
        #x = F.normalize(x,p=2,dim = 1)
        #adj = torch.matmul(x,x.T)

        return adj

EOS = 1e-10
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class Diag(nn.Module):
    def __init__(self, input_size):
        super(Diag, self).__init__()
        self.W = nn.Parameter(torch.ones(input_size))
        self.input_size = input_size

    def forward(self, input):
        hidden = input @ torch.diag(self.W)
        return hidden


            

        
