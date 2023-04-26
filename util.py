import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from networkx import NetworkXError
#from argument_setting import args


def tag2feature(tags_list):
    tagset = set([])
    for tags in tags_list:
        tagset = tagset.union(set(tags))
    
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    
    features = []
    for tags in tags_list:
        feature = torch.zeros(len(tags),len(tagset))
        feature[range(len(tags)),[tag2index[tag] for tag in tags]] = 1
        features.append(feature)
    return features


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label                 #  label for graph
        self.g = g                         #  the graph itself  
        self.node_tags = node_tags         #
        self.neighbors = []                #  neighbor list for each node?
        self.node_features = node_features             
        self.edge_mat = 0                  #  edge list

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    
    #    dataset: name of dataset
    #    test_proportion: ratio of test train split
    #    seed: random seed for random splitting of dataset
    

    print('loading data')
    g_list = []
    label_dict = {}    #应该是为了把不同形态的label形式统一
    feat_dict = {}     
    
    #文件格式分析
    # first line: the number of graphs
    # n: graph size  l: graph label
    #
    node_feature_flag = False
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())          
        for i in range(n_g):                     
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []           # label of each node
            node_features = []       # feature of each node
            n_edges = 0              # the number of edges
            for j in range(n):
                # row[1] = the degree of the node
                # row[0] = the label of the node
                # so 2 + int(row[1]) 之后就是 attr(如果后面存在的话)
                g.add_node(j)  
                row = f.readline().strip().split()   #第j行是与j th node 相关的信息
                tmp = int(row[1]) + 2                
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    node_features.append(attr)
                # node 的标签是统一的
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])
                

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])       
            
            # figure out whether the node has its feature
            # !!!! No dataset has node feature
            if node_features != []:
                node_features = torch.from_numpy(np.stack(node_features)).float()
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False
            assert len(g) == n
            
            g_list.append(S2VGraph(g, l, node_tags, node_features))   #  g 为图本身，带有edge信息， l 为图的标签   node_tags为节点标签

    #add labels and edge_mat       
    for g in g_list:
        # add an edge list for each node
        
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        
        #g.neighbors = [list(nx.neighbors(g.g,i)) for i in range(len(g.g))]

        # compute max degree
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]          # what ?
            degree_list.append(len(g.neighbors[i]))  
        
        # the max degree in g
        g.max_neighbor = max(degree_list)

        # add label for the graph
        g.label = label_dict[g.label]
        
        # generate edge mat(shape like 2 * num_edges)
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        
        g.edge_mat = torch.LongTensor(edges).transpose(0,1) # like 2 * num(edges)

    ori_features = tag2feature([g.node_tags for g in g_list])
    #改写节点标签（if needed）
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
        degree_features = tag2feature([g.node_tags for g in g_list])
        

    #Extracting unique tag labels (for all the graph)
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    #构造标签数值 -> label_index 的映射
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    
    #采用node标签对应的one-hot label作为特征
    #潜在的问题：存在原生特征的图，利用原生特征是否会更好
    '''
    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        print(g.node_features)
        exit(0)
    '''
    if node_feature_flag:
        pass
    elif not degree_as_tag:
        for g,orif in zip(g_list,ori_features):
            g.node_features = orif
    else:
        for g,orif,degreef in zip(g_list,ori_features,degree_features):
            #if args.mix_tag:
            #    g.node_features = torch.concat([orif,degreef],dim = 1)
            #else:
            g.node_features = degreef
    



    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % g_list[0].node_features.shape[1])

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)



#训练集 & 测试集 split
def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def para_norm(para,p):

    norm = 0

    for var in para.keys():
        norm += (para[var]**p).sum()

    return norm**(1/p)
    
def para_cos_sim(para_x,para_y):

    norm_x,norm_y = para_norm(para_x,2),para_norm(para_y,2)
    inner_xy = 0
    for var in para_x.keys():
        inner_xy += (para_x[var]*para_y[var]).sum()
    
    return inner_xy/(norm_x * norm_y)

def sim_matrix(para_list):
    
    size = len(para_list)
    nodes = list(range(size))
    edges = []
    edges_np = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            # guarantee the weight of each edge is positive(stoer-wanger alogorithm need this condition)
            sim = (para_cos_sim(para_list[i],para_list[j])+1)/2
            edges.append((i,j,sim))
            edges_np[i,j] = sim
    
    #print(edges_np)

    G1 = nx.Graph()
    G1.add_nodes_from(nodes)
    G1.add_weighted_edges_from(edges)

    return G1

# wx - wy
def state_diff(wx,wy):
    w_deta = {}
    
    for key in wx.keys():
        w_deta[key] = wx[key] - wy[key]
    
    return w_deta

def get_random_mask(features,ratio):
    probs = torch.full(features.shape, ratio)
    mask = torch.bernoulli(probs)
    return mask


def load_multi_data(datasets,degree_as_tag):

    classes_list = []
    mul_graph_list = []
    for data in datasets:
        graphs,num_classes = load_data(data,degree_as_tag)
        classes_list.append(num_classes)
        mul_graph_list.append(graphs)
    
    if len(set(classes_list)) > 1:
        print('please align the classes num!')
        print(classes_list)
        exit(0)
    
    return mul_graph_list,classes_list[0]

            

        
        
    
