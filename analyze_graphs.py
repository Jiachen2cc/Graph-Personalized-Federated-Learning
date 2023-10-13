import numpy as np
from torch_geometric.data import Data,Batch
import torch
from torch_geometric.nn import WLConv
from torch_geometric.utils import to_networkx,degree,to_undirected
import networkx as nx

class WL(torch.nn.Module):
    def __init__(slef,num_layers = 3):
        super().init()
        slef.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])
    
    def forward(self, x, edge_index, batch = None):
        for conv in self.convs:
            x = conv(x,edge_index)
            hist = conv.histogram(x,batch,norm = True)
        return hist


# count the label distribution in specific client dataset
def label_distribution(graphs):
    
    labels = np.array([graph.y.item() for graph in graphs])

    num_classes = labels.max() + 1

    class_dis = [len(np.argwhere(labels == y).flatten()) for y in range(num_classes)]

    return class_dis

# turn edge_index into neighbor list
def e2n(gsize, edge_index : torch.Tensor):
    
    nlist = [[] for i in range(gsize)]

    for idx in range(edge_index.shape[1]):

        x,y = edge_index[0,idx].item(),edge_index[1,idx].item()
        nlist[x].append(y)
        nlist[y].append(x)
    
    for i in range(gsize):
        nlist[i] = set(nlist[i])
    
    return nlist

# count the triangle num in specific graphs
def count_tri(graph: Data):
    
    tri_num = 0
    gsize = len(graph.x)
    nlist = e2n(gsize, graph.edge_index)

    for i in range(gsize):

        neigh_i = nlist[i]
        for j in neigh_i:
            neigh_j = nlist[j]
            inter = neigh_i.intersection(neigh_j)

            tri_num += len(inter)
    
    tri_num = tri_num // 6
    
    return tri_num


# count the degree distribution for given graphs
def count_degree_distribution(graph: Data):

    ud = to_undirected(graph.edge_index)
    vd = degree(ud[0])

    max_deg = torch.max(vd).item()  
    
    deg_dis = np.zeros(max_deg + 1) # deg in [0,max_deg]
    for i in range(max_deg + 1):
        deg_dis[i] = torch.sum(max_deg == i).item()
    
    return deg_dis

# count the 2-hop neighbor number for given graphs

def count_hop2_neighbor(graph: Data):

    hop2_neighbors_num = hop2_neighbor(graph)
    max_hop2 = np.max(hop2_neighbors_num)
    hop2_dis = np.zeros(max_hop2 + 1)
    for i in range(max_hop2 + 1):
        hop2_dis[i] = np.sum(hop2_neighbors_num == i)
    
    return hop2_dis


def hop2_neighbor(graph: Data):

    gsize = len(graph.x)
    nlist = e2n(gsize, graph.edge_index)
    
    hop2_neighbors_num = []
    for i in range(gsize):
        
        neigh_i = nlist[i]
        total_i = neigh_i | set([i])

        all_2neigh = set([])
        for j in neigh_i:
            all_2neigh = all_2neigh | nlist[j]
        
        hop2_neigh = all_2neigh - total_i
        hop2_neighbors_num.append(len(hop2_neigh))

    hop2_neighbors_num = np.array(hop2_neighbors_num)

    return hop2_neighbors_num


def wl_dis(graphs):

    data = Batch.from_data_list(graphs)
    wl = WL(num_layers = 3)

    hist = wl(data.x,data.edge_index,data.batch)
    return hist
    
# measure the similarity between distributions
# compute p//q
# assume this two are of the same shape 1*n
def KL_d(disp,disq,eps = 1e-12):    
    # add eps to avoid nan 
    disp = disp + eps
    disq = disq + eps

    distance = np.sum(disp*np.log10(disp/disq))

    if np.isnan(distance):
        print('KL_bug!')
        exit(0)

    return distance


#like KL,but want to guarantee the symmetry
# assume this two are of the same shape 1*n
def JS_d(disp,disq):
    mid = (disp+disq)/2
    distance = (KL_d(disp,mid) + KL_d(disq,mid))/2
    return distance



# create fake graph to test the relationship between client graph and similarity
def fake_graph(dim,type = 'test1', seed = 0):

    g = torch.zeros((dim,dim))
    
    if type == 'test1':
        for i in range(dim):
            j = (i+1)%dim
            if j != 0:
                g[i,i] = 0.5
                g[i,j] = 0.5
            else:
                g[i,i] = 1

    elif type == 'test2':
        np.random.seed(seed)
        dim1 = int(np.floor(np.random.sample() * (dim-3) + 2))
        g[0:dim1,0:dim1] = 1/dim1
        g[dim1:dim,dim1:dim] = 1/(dim - dim1)

    return g

def homo_edge_count(graph: Data, eps = 1e-5):

    homo_count = 0
    hetero_count = 0

    g = to_networkx(graph,to_undirected = True)

    for x,y in g.edges():
        fx,fy = graph.x[x],graph.x[y]
        if torch.sum(torch.abs(fx-fy)) < eps:
            homo_count += 1
        else:
            hetero_count += 1
    
    return homo_count,hetero_count

def network_properties(graph: Data):
    
    properties = {}
    EPS = 1e-12
    # get basic information
    N = graph.x.shape[0]
    ud = to_undirected(graph.edge_index)
    E = ud.shape[1]
    vd = degree(ud[0])
    
    subkey = ['density','network entropy','closeness centrality']
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


    








    
