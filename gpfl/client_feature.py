import torch
import torch.nn.functional as F
from gpfl.model_compression import continous_compress,compress_shape
from gpfl.utils import marginal


def cos_sim(embed):
    #print(embed.shape)
    #exit(0)
    norm = torch.norm(embed,dim = 1,keepdim = True)
    norm_embed = embed/norm

    return torch.matmul(norm_embed,norm_embed.T)

def prepare_features(embed,param,args):
    # param to metrix
    cparam = (para2metrix(param,None,args.compress_dim)).to(args.device)
    ofeature = {'embed':embed,'param':cparam}
    choices = {'embed':embed,'param':cparam}

    # apply transforms to parameters | get normalized features(each sample scaled to norm = 1)
    if args.input_choice == 'diff':
        choices = {k: F.normalize(marginal(f,args.diff_rate),p=2,dim=1) for k,f in choices.items()}
    
    # compute similarity
    sims = {k: torch.matmul(f,f.T) for k,f in choices.items()}
    k = args.para_choice if args.para_choice in choices.keys() else 'embed'
    return ofeature[k],choices[k],sims[args.graph_choice]


# helper functions related to parameter - vector conversion
def para2metrix(models_state,cmode,cdim):

    key_shapes = [list(param.data.shape) for _,param in models_state[0].items()]

    param_metrix = state_dict2metrix(models_state)

    if cmode == 'continous':
        compress_param = continous_compress(param_metrix.cpu(),cdim)
    elif cmode == 'shape':
        compress_param = compress_shape(param_metrix.cpu(),key_shapes)
    elif cmode is None:
        # simple avg pool compress
        stri = param_metrix.shape[1]//cdim
        compress_param = torch.avg_pool1d(param_metrix,stri,stri)
    
    return compress_param

def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    #print(filter_mode)
    keys = []
    param_vector = None

    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    #print(keys)
    return param_vector

def state_dict2metrix(models_state):
    parameter_metrix = []
    
    for state_dic in models_state:
        parameter_metrix.append(sd_matrixing(state_dic).clone().detach())
    param_metrix = torch.stack(parameter_metrix)

    return param_metrix