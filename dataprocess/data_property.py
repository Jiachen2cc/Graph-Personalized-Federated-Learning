from torch_geometric.data import Data,Batch
import networkx as nx
import torch
from torch_geometric.utils import to_networkx,degree,to_undirected

def network_properties(graph: Data):
    
    properties = {}
    EPS = 1e-12
    # get basic information
    N = graph.x.shape[0]
    ud = to_undirected(graph.edge_index)
    E = ud.shape[1]
    vd = degree(ud[0])
    
    # average degree
    properties['average degree'] = torch.mean(vd) if len(vd) > 0 else 0
    # average hop2 degree
    #properties['average hop2'] = torch.mean(torch.FloatTensor(hop2_neighbor(graph)))
    
    # density(average degree centrality)
    properties['density'] = E/(N*(N-1)) if N > 1 else 0

    # degree variance
    properties['degree variance'] = torch.mean(vd*vd) - torch.mean(vd)**2 if len(vd) > 0 else 0

    # network entropy
    properties['network entropy'] = torch.mean(vd*torch.log(vd+EPS))/E if len(vd) > 0 else 0

    # scale-free exponent
    properties['negative scale-free exponent'] = -(1/torch.mean(vd)+2) if len(vd) > 0 else 0

    ng = to_networkx(graph,to_undirected = True)

    # closeness centrality
    cc = nx.closeness_centrality(ng)
    properties['closeness centrality'] = sum(cc[k] for k in cc.keys())/N

    # subgraph centrality
    # sc = nx.subgraph_centrality(ng)
    # properties['subgraph centrality'] = sum(sc[k] for k in sc.keys())/N

    # betweeness centrality
    #bc = nx.betweenness_centrality(ng)
    #properties['betweenness centrality'] = sum(bc[k] for k in bc.keys())/N

    # eigenvector centrality
    #ec = nx.eigenvector_centrality(ng,max_iter=1000,tol = 1e-4)
    #properties['eigenvector centrality'] = sum(ec[k] for k in ec.keys())/N

    # current flow closeness centrality
    # cfcc = nx.current_flow_closeness_centrality(ng)
    # properties['current flow closeness centrality'] = sum(cfcc[k] for k in cfcc.keys())/N
    
    res = torch.tensor([properties[k] for k in properties.keys()])

    return res
