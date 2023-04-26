# anaylze the similarity between the client parameters
from graph_utils import *
import torch
import copy

def cos_sim(embed):
    #print(embed.shape)
    #exit(0)
    norm = torch.norm(embed,dim = 1,keepdim = True)
    norm_embed = embed/norm

    return torch.matmul(norm_embed,norm_embed.T)


def simi_ana(clients,global_state,server_key):
    
    # 1 get the model state
    # only consider the center part of the model
    models_state = []
    for client in clients:
        W = {}
        for k in server_key:
            W[k] = copy.deepcopy(client.W[k])
        models_state.append(W)

    param_metrix = state_dict2metrix(models_state)
    
    similarity = cos_sim(param_metrix)
    return similarity


    if global_state is not None:
        server_p = sd_matrixing(global_state)
        snorm = torch.norm(server_p)
        snormp = server_p/snorm
    #-------get the similarity between local and global model----------
        simi_k = torch.sum((param_metrix * snormp)/torch.norm(param_metrix),dim = 1)
        print(simi_k)
    


def clf_ana(clients,server):

    clf_state = []
    for client,clf_W in zip(clients,server.clf_Ws):
        W = {}
        server_key = clf_W.keys()
        for k in server_key:
            W[k] = copy.deepcopy(client.W[k])
        clf_state.append(W)

    param_metrix = state_dict2metrix(clf_state)
    # normalize the parameters
    print(cos_sim(param_metrix))

def gradient_ana(model):
    
    all_norm = 0
    
    for name,value in model.named_parameters():
        if value.grad is not None:
            grad = value.grad.clone()
            norm = torch.norm(grad)
            print(name,' param vari:{:.4f}'.format(torch.var(value.data)),
            'grad norm:{:.4f}'.format(norm))
            all_norm += norm

    return all_norm

def graph_diff(graphs):
    
    diffs = []

    for idx in range(len(graphs) - 1):

        mean_diff = torch.mean(torch.abs(graphs[idx+1] - graphs[idx]))
        diffs.append(mean_diff)

    print(diffs)
    

        


    


