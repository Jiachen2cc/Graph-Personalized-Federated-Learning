from audioop import avg
#from argument_setting import args
import numpy as np
import torch
import torch.nn.functional as F
from graph_utils import state_dict2metrix


#graphs, num_classes = load_data(args.dataset, args.degree_as_tag)


# init clinet graph based on client parameters distance
def dist_simi_metrix(models_state,eps):

    param_metrix = state_dict2metrix(models_state)

    dist_metrix = torch.zeros(len(param_metrix),len(param_metrix))
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = F.pairwise_distance(
                param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
    dist_metrix = F.normalize(dist_metrix)#.to(args.device)
    
    dist_simi = (1+eps)*torch.max(dist_metrix,dim = 1).values[:,None] - dist_metrix
    #return 1 - dist_metrix
    return dist_simi






