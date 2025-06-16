import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GCNConv


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
    

# decoupled GNN for FedStar

class serverGIN_dc(torch.nn.Module):
    def __init__(self, n_se, nlayer, nhid):
        super(serverGIN_dc, self).__init__()

        self.embedding_s = torch.nn.Linear(n_se, nhid)
        self.Whp = torch.nn.Linear(nhid + nhid, nhid)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))
            
            
class GIN_dc(torch.nn.Module):
    def __init__(self, nfeat, n_se, nhid, nclass, nlayer, dropout):
        super(GIN_dc, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.embedding_s = torch.nn.Linear(n_se, nhid)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

        self.Whp = torch.nn.Linear(nhid + nhid, nhid)
        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc
        x = self.pre(x)
        s = self.embedding_s(s)
        for i in range(len(self.graph_convs)):
            x = torch.cat((x, s), -1)
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.graph_convs_s_gcn[i](s, edge_index)
            s = torch.tanh(s)

        x = self.Whp(torch.cat((x, s), -1))
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
    
# masked GIN for FedPub

from fedpub.layers import MaskedLinear

class MaskedGIN(torch.nn.Module):
    
    def __init__(self, n_feat, nhid, nclass, nlayer, dropout,
                 l1 = 1e-3, args = None):
        super(MaskedGIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout 
        
        self.pre = torch.nn.Sequential(MaskedLinear(n_feat, nhid, l1, args))
        self.graph_convs = torch.nn.ModuleList()
        for l in range(nlayer):
            nnk = torch.nn.Sequential(
                MaskedLinear(nhid, nhid, l1, args),
                torch.nn.ReLU(),
                MaskedLinear(nhid, nhid, l1, args)
            )
            self.graph_convs.append(GINConv(nnk))

        self.post = torch.nn.Sequential(
            MaskedLinear(nhid, nhid, l1, args),
            torch.nn.ReLU()
        )
        self.readout = torch.nn.Sequential(
            MaskedLinear(nhid, nclass, l1, args)
        )
    
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
        x = F.log_softmax(x, dim = 1)
        
        return x
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
    