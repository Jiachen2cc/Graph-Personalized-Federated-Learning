# this file contains models which aim at building the client graph through recontructing masked feature

from torch_geometric.nn import GAE,GATConv
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
#from argument_setting import args
from graph_utils import symmetrize,no_negtive,normalize
from layers import GCNConv_dense,GraphAttentionLayer
from graph_contructor import MLP, FullParam, MLP_Diag


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
        for conv in self.layers[:-1]:
            x = conv(x,A_)
            x = self.act(x)
            x = F.dropout(x, self.dropout, training = self.training)
        
        x = self.layers[-1](x, A_)

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


            

        
