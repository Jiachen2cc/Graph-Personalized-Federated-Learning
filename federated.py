from webbrowser import get
import torch
from recover_model import GCN_DAE
import torch.nn.functional as F
import torch.optim as optim
import copy
#from argument_setting import args
import numpy as np
from util import get_random_mask
from model_compression import continous_compress, discrete_compress,compress_shape
from graph_utils import normalize,matri2dict,state_dict2metrix
from analyze_client import gradient_ana
import os

def adj_entloss(adj):

    adj_ent = - adj*torch.log(adj) - (1-adj)*torch.log(1-adj)
    loss = torch.mean(adj_ent)
    return loss

def size_loss(adj,mask_act = 'relu'):

    if mask_act == 'sigmoid':
        adj = torch.sigmoid(adj)
    elif mask_act == 'relu':
        adj = torch.nn.ReLU()(adj)
    size_loss = torch.sum(adj)
    return adj


def calc_sim(matrix):
    norm_matrix = matrix / torch.norm(matrix,dim = 1, p =2)[:,None]
    sim = torch.mm(norm_matrix,norm_matrix.t())
    return sim

def val_cgraph(A:torch.Tensor):
    return A.var(dim = 1).mean()


def graph_gen(models_state, pre_A, args):
    
    key_shapes = []
    for param in models_state[0].items():
        key_shapes.append(list(param.data.shape))

    param_metrix = state_dict2metrix(models_state).to(args.device)
    if args.compress_mode == 'continous':
        compress_param = continous_compress(param_metrix.cpu(),args.compress_dim).to(args.device)
    elif args.compress_mode == 'shape':
        compress_param = compress_shape(param_metrix.cpu(),key_shapes).to(args.device)


    _,A = generate_adj(compress_param, pre_A, args)
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

def generate_adj(param_metrix,pre_A,args,model = None):
    '''
    Net = SLAPS(param_metrix.shape[1],32,16,0,F.relu,
    param_metrix.shape[1],64,0.5,F.relu,len(param_metrix)).to(args.device)
    '''
    Net = GCN_DAE(2,param_metrix.shape[1],128,param_metrix.shape[1],0.5,0,args.gen_mode,64,32,2).to(args.device)
    if model is not None:
        Net = model
    #print(pre_A)
    Net.train()
    #Net.init_graph_gen(param_metrix,pre_A)
    #print(Net.parameters())
    optimizer = optim.Adam(Net.parameters(),lr = args.glr,weight_decay = args.gweight_decay)
    mask = get_random_mask(param_metrix,args.mask_ratio).to(args.device)

    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        
        loss1, adj = get_loss_masked_features(Net,param_metrix,pre_A,mask,args)
        loss2 = F.mse_loss(pre_A,adj,reduction='mean') 
        loss3 = adj_entloss(adj)
        
        #print('gc_epoch:{}'.format(e))
        #print('adj during training:',adj.view(10,10))
        # loss = loss1 + loss2
        #loss = loss2 
        loss = loss2 + loss3
        loss.backward()
        '''
        if e >= 0:
            print('loss_used:{:.4f}'.format(loss))
            print('loss1:{:.4f}, loss2:{:.4f}'.format(loss1,loss2))
            print('learned client graph')
            print(adj)
            gradient_ana(Net)
        '''
        optimizer.step()
        #print('edge_loss:{:.4f}'.format(loss))
    # get final adj(without noise)
    Net.eval()
    #exit(0)
    # generate normalized predicted client graph
    with torch.no_grad():
        resf,adj = Net(param_metrix,param_metrix,pre_A)
        adj = normalize(adj,'sym')
        
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
        logits, Adj = model(features, masked_features,pre_A)
        indices = mask > 0

        loss = F.mse_loss(logits[indices],features[indices],reduction = 'mean')
    
    return loss, Adj
