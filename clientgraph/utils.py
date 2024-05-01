import torch
import torch.nn.functional as F

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

def get_random_mask(features,ratio):
    probs = torch.full(features.shape, ratio)
    mask = torch.bernoulli(probs)
    return mask

def state_dict2metrix(models_state):
    parameter_metrix = []
    
    for state_dic in models_state:
        parameter_metrix.append(sd_matrixing(state_dic).clone().detach())
    param_metrix = torch.stack(parameter_metrix)

    return param_metrix

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