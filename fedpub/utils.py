import torch

def cossim(x,y):
    scale = (torch.sum(x**2)*torch.sum(y**2))**0.5
    return torch.sum(x*y)/scale