import torch
from pfedgraph_cosine.recover_model import GCN_DAE
import torch.nn.functional as F
import torch.optim as optim
import copy
#from argument_setting import args
import numpy as np
#from wdelete import get_random_mask
from pfedgraph_cosine.graph_utils import normalize,matri2dict,state_dict2metrix
#from analyze_client import gradient_ana
import os

def adj_entloss(adj,eps = 1e-6):

    adj_ent = - adj*torch.log(adj+eps) - (1-adj)*torch.log(1-adj+eps)
    loss = torch.mean(adj_ent)
    return loss

def adj_reluloss(adj,eps = 1e-6):

    adj_relu = adj*(1-adj)
    loss = torch.mean(adj_relu)
    return loss

def size_loss(adj,acted = 'relu',eps = 1e-6):

    if acted == 'sigmoid':
        loss = torch.mean(-torch.log(1+eps-adj))
    elif acted == 'relu':
        loss = torch.mean(adj) 
    return loss


def calc_sim(matrix):
    norm_matrix = matrix / torch.norm(matrix,dim = 1, p =2)[:,None]
    sim = torch.mm(norm_matrix,norm_matrix.t())
    return sim

def val_cgraph(A:torch.Tensor):
    return A.var(dim = 1).mean()

def get_random_mask(features,ratio):
    probs = torch.full(features.shape, ratio)
    mask = torch.bernoulli(probs)
    return mask


def graph_gen(models_state, pre_A, args):
    
    key_shapes = []
    for param in models_state[0].items():
        key_shapes.append(list(param.data.shape))

    param_metrix = state_dict2metrix(models_state).to(args.device)

    _,A = generate_adj(param_metrix, pre_A, args)
    #A = generate_adj(param_metrix, pre_A)
    quality = val_cgraph(A)

    return A, quality

def graph_aggregate(models_state, A, args):
    
    keys = []
    key_shapes = []
    #filtered_param = state_dict2metrix(models_state,'batch_norm').to(args.device)
    
    param_metrix = state_dict2metrix(models_state)
    for key, param in models_state[0].items():
        keys.append(key)
        key_shapes.append(list(param.data.shape))
    
    aggregated_param = torch.mm(A, param_metrix)
    for i in range(args.layers - 1):
        aggregated_param = torch.mm(A, aggregated_param)
    new_param_matrix = (args.serveralpha * aggregated_param) + ((1 - args.serveralpha) * param_metrix)

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

    return models_dic

def graph_dic(graph_w,agg_w,pre_A,args,normal = True):
    A,quality = graph_gen(graph_w,pre_A,args)
    #quality = val_cgraph(A)

    res_w = graph_aggregate(agg_w,A,args)

    return res_w, A, quality

def gae_adj(param_metrix,pre_A,args):

    pass

def preprocess_input(features,ini_graph):

    features = F.normalize(features,dim = 1)
    return features,ini_graph

def generate_adj(clients,param_metrix,pre_A,args,model = None):
    '''
    Net = SLAPS(param_metrix.shape[1],32,16,0,F.relu,
    param_metrix.shape[1],64,0.5,F.relu,len(param_metrix)).to(args.device)
    '''
    # preprocess the input parameters
    #param_metrix,pre_A = preprocess_input(param_metrix,pre_A)

    Net = GCN_DAE(1,param_metrix.shape[1],128,param_metrix.shape[1],0.5,0,args.gen_mode,64,32,1).to(args.device)
    if model is not None:
        Net = model
    Net.train()
    #Net.init_graph_gen(param_metrix,pre_A)
    #print(Net.parameters())
    optimizer = optim.Adam(Net.parameters(),lr = args.glr,weight_decay = args.gweight_decay)
    mask = get_random_mask(param_metrix,args.mask_ratio).to(args.device)

    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        
        loss1, adj = get_loss_masked_features(Net,param_metrix,pre_A,mask,args)
        #loss2 = F.binary_cross_entropy(adj, (pre_A >= 1/pre_A.shape[0]).float()) 
        #ploss = gain_loss(adj,gain)
        #aloss = ALA_loss(adj,clients,args)
        #print(aloss)
        #print(ploss)
        #loss2 = F.binary_cross_entropy(adj.view(-1),pre_A.view(-1))
        if args.sigmoid:
            loss = adj_entloss(adj) + args.loss_gama*size_loss(adj,'sigmoid')
        else:
            loss = adj_reluloss(adj) + args.loss_gama*size_loss(adj)
        #loss += 0.1*aloss
        loss.backward()
        #print(adj)
        #print(adj)
        #print(loss)
        
        if e >= 0:
            pass
            #print('loss_used:{:.4f}'.format(loss))
            #print('loss1:{:.4f}, loss2:{:.4f}'.format(loss1,loss2))
            #print('learned client graph')
            #print(adj)
            #gradient_ana(Net)
        
        optimizer.step()
        #print('edge_loss:{:.4f}'.format(loss))
    # get final adj(without noise)
    Net.eval()
    #exit(0)
    # generate normalized predicted client graph
    with torch.no_grad():
        resf,adj = Net(param_metrix,param_metrix,pre_A)
        if args.discrete and args.sigmoid:
            adj = (adj >= 0.5).float().to(adj.device)
        else:
            '''
            mask = (adj >= 0.5).float().to(adj.device)
            adj = adj*mask
            '''
            pass
        if args.sharing_mode == 'ALA':
            pass
            #adj.fill_diagonal_(0)
        #print('the output client graph(before normalization)')
        #print(adj)
        #adj = normalize(adj,'sym')
    #print(pre_A)  
    #print(adj)
    #exit(0)
    return resf,adj.detach().to('cpu'),Net

def bi_graph_dic(models_state,pre_A,args):
    
    keys = []
    key_shapes = []

    for key,param in models_state[0].items():
        keys.append(key)
        key_shapes.append(list(param.data.shape))
    
    param_metrix = state_dict2metrix(models_state).to(args.device)
    learned_param,A,_ = generate_adj(param_metrix, pre_A, args)

    res_param = args.serverbeta * learned_param + (1 - args.serverbeta) * param_metrix
    
    # reshape parameter matrix to models state
    res_models = matri2dict(models_state,res_param,keys,key_shapes)
    quality = val_cgraph(A)

    return res_models,A,quality
    


    


def get_loss_masked_features(model,features,pre_A,mask,args):

    if args.compress_mode == 'discrete':
        masked_features = features*(1-mask)
        logits, Adj = model(features, masked_features,pre_A)
        indices = mask > 0

        loss = F.binary_cross_entropy_with_logits(logits[indices],features[indices],reduction = 'mean')
    
    else:
        noise = torch.normal(0.0, 1.0, size=features.shape).to(args.device)
        masked_features = features + (noise * mask)
        logits, Adj = model(features, masked_features,pre_A,sigmoid = args.sigmoid)
        indices = mask > 0

        loss = F.mse_loss(logits[indices],features[indices],reduction = 'mean')
    
    return loss, Adj


def gain_loss(graph,gain):

    # negative gain causes negative weight 
    # positive gain causes positive weight

    tgain = (gain > 0)[:,None].to(graph.device)
    tgraph = (graph > 1/graph.shape[0])
    target_graph = (tgain & tgraph).float()
    loss = F.binary_cross_entropy(graph,target_graph)

    return loss

def flattenw(w):
    #return torch.cat([v.flatten() for v in w.data()])
    return torch.cat([v.flatten() for v in w.values()])

def ALA_loss(adj,clients,args):
    # construct features
    if args.sharing_mode == 'gradient':
        features = torch.stack([flattenw(c.dW).detach() for c in clients])
        features = torch.matmul(adj,features)
        features += torch.stack([flattenw(c.W_old).detach() for c in clients])


    # perform parameter aggregation based on predicted client graph
    loss = 0
    #loss += F.mse_loss(features,torch.zeros(features.shape).to(adj.device))

    for i,c in enumerate(clients):
        cfeature = features[i,:]
        loss += c.ALA_train(cfeature,args)
    
    return loss
