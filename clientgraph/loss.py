import torch

def ent_loss(adj, eps = 1e-6):
    
    adj_ent = -adj*torch.log(adj+eps) - (1-adj)*torch.log(1-adj+eps)
    loss = torch.mean(adj_ent)
    
    return loss


def size_loss(adj, eps = 1e-6):
    loss = torch.mean(-torch.log(1+eps-adj))
    return loss