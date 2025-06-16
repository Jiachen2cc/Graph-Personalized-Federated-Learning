import torch
from collections import OrderedDict
import numpy as np

def cossim(x,y):
    scale = (torch.sum(x**2)*torch.sum(y**2))**0.5
    return torch.sum(x*y)/scale

def extract_detach_model_weight(state_dict):
    return OrderedDict([(k,v.clone().detach().cpu()) for k,v in state_dict.items()])


def set_state_dict(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = v.cuda(gpu_id)
        else:
            _state_dict[k] = v.requires_grad_().cuda(gpu_id)
    return _state_dict

def aggregate_model_weights(
    params,
    agg_weights
):
    res = OrderedDict([(k,None) for k in params[0].keys()])
    for name, _ in res.items():
        weighted_params = torch.stack([p[name]*agg_weights[i] for i,p in enumerate(params)])
        res[name] = torch.sum(weighted_params, dim=0)
    return res