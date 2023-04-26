import datetime
import random
import os
import torch
import numpy as np
from collections import namedtuple
from functools import singledispatch
import scipy.sparse as sp


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

def trainable_params(model):
    result = []
    for p in model.parameters():
        if p.requires_grad:
            result.append(p)
    return result

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def state_dict2metrix(models_state):
    parameter_metrix = []
    
    for state_dic in models_state:
        parameter_metrix.append(sd_matrixing(state_dic).clone().detach())
    param_metrix = torch.stack(parameter_metrix)

    return param_metrix

