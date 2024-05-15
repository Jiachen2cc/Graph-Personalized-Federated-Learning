import torch

def marginal(feature):
    return feature - torch.mean(feature,dim = 0)

def cossim(x,y):
    scale = (torch.sum(x**2)*torch.sum(y**2))**0.5
    return torch.sum(x*y)/scale

def normalize(adj, mode, sparse=False):
    EOS = 1e-10
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            #inv_sqrt_degree = torch.diag(inv_sqrt_degree)
            #norm_adj = torch.matmul(inv_sqrt_degree,adj)
            #norm_adj = torch.matmul(norm_adj,inv_sqrt_degree.T)
            return inv_sqrt_degree[:,None] * adj * inv_sqrt_degree[:,None]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            inv_degree = torch.diag(inv_degree)
            norm_adj = torch.matmul(inv_degree,adj)
            return norm_adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value