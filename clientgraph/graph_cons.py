from clientgraph.model import GAE_dense
from clientgraph.loss import *
import torch.nn.functional as F
import torch.optim as optim
from clientgraph.utils import *
import copy


class graph_constructor:
    
    def __init__(self,feature_dim,args):
        
        # use sparse if the input graph is discrete
        self.model = GAE_dense(feature_dim,128,args.discrete).to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.glr, 
                            weight_decay = args.gweight_decay)
        self.args = args
    
    def preprocess(self, features, pre_A):
        # discretize the input graph in case needed
        if self.args.discrete:
            pre_A = (F.sigmoid(pre_A) >= 0.5).float().to(pre_A.device)
        return features, pre_A
        
    def train(self,features,A):
        
        # reset model parameters
        self.model.reset_parameters()
        self.model.train()
        # write train round
        for e in range(self.args.gc_epoch):
            '''
            mask = get_random_mask(features, 
                    self.args.mask_ratio).to(self.args.device)
            rec_loss, adj = get_loss_masked_features(self.model,
                                features, A, mask, self.args)
            '''
            adj = self.model(features,A)
            adj = torch.sigmoid(adj)
            loss = ent_loss(adj) + self.args.loss_gama*size_loss(adj)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def postprocess(self, adj):
        # may used to get a sparser adj
        
        # compute row wise softmax to normalize the sharing weight
        # adj = F.softmax(adj, dim = 1)
        adj = torch.sigmoid(adj)
        return adj
            
    def gen_graph(self, features, A):
        
        # generate the client graph
        with torch.no_grad():
            adj = self.model(features,A)
            if self.args.discrete:
                adj = (adj >= 0).float().to(adj.device)
        
        return adj
    
    
    def graph_based_aggregation(self, features, A):
        
        # train the graph generator
        features, A = self.preprocess(features, A)
        self.train(features, A)
        client_graph = self.gen_graph(features, A)
        sharing_graph = self.postprocess(client_graph)
        '''
        keys,key_shapes = [],[]
        param_metrix = state_dict2metrix(models_state)
        for key, param in models_state[0].items():
            keys.append(key)
            key_shapes.append(list(param.data.shape))
        
        aggregated_param = torch.mm(sharing_graph, param_metrix)
        for i in range(args.layers - 1):
            aggregated_param = torch.mm(sharing_graph, aggregated_param)
        new_param_matrix = ((self.args.serveralpha * aggregated_param) + 
                            ((1 - self.args.serveralpha) * param_metrix))
        
        models_dic = copy.deepcopy(models_state)
        new_param_matrix = new_param_matrix.to(args.device)
        # reconstract parameter
        for i in range(len(models_dic)):
            pointer = 0
            for k in range(len(keys)):
                num_p = 1
                for n in key_shapes[k]:
                    num_p *= n
                models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
                pointer += num_p
        '''
        return sharing_graph
            
        
        
        
        
        
    
    
    
        
    