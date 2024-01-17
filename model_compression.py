#from argument_setting import args
import torch
import sklearn.decomposition as decomp
import numpy as np
from sklearn.cluster import k_means
import os


# slice X into serveral pieces n*ndim
def segments(X,ndim):
    
    slice_num = int(np.floor(X.shape[1]/ndim))
    
    for i in range(slice_num):
        yield X[:,i*ndim:(i+1)*ndim]
    
    if slice_num * ndim < X.shape[1]:
        yield X[:,slice_num*ndim:X.shape[1]]

# seems can bring slight improvements 
def continous_compress(X,ndim = 1000):
    
    # assume X.shape[0] < X.shape[1] since X.shape[0] : num_clients  X.shape[1]: num_parameters
    # so pca asks compress_dim <= X.shape[0]
    pca_max = X.shape[0]

    pca_segments = int(np.floor(ndim/pca_max))
    if pca_max*pca_segments < ndim:
        pca_segments += 1
    
    seg_dim = int(np.floor(X.shape[1] / pca_segments))
    #print(pca_segments)
    #print(seg_dim)
    parts = []
    for x in segments(X,seg_dim):
        dim = min(x.shape)
        method = decomp.PCA(dim)
        res = torch.from_numpy(method.fit_transform(x))
        parts.append(res)
    res = torch.concat(parts,dim = 1).float()
    #print(type(res))

    
    return res


def discrete_compress(X,ndim = 10):
    
    slice_dim = int(np.float(X.shape[1]/ndim))

    parts = []
    for x in segments(X,slice_dim):
        _,label,_ = k_means(x,2,max_iter = 10)
        label = torch.unsqueeze(torch.from_numpy(label),0)
        #print(label.shape)
        parts.append(label)
    res = torch.concat(parts,dim = 1).float()
    
    return res

def compress_shape(X,key_shapes):

    parts = []
    index = 0
    sample_num = X.shape[0]
    for shape in key_shapes:
        # extract specific layer
        vol = np.prod(np.array(shape),0)
        part = X[:,index:index+vol]
        index += vol
        
        # if layer is one-dim, we don't make any compression
        if len(shape) == 1:
            parts.append(part)
            continue
        
        # compress this layer according to its smallest dim
        dim = np.min(np.array(shape))
        method = decomp.PCA(1)
        cparts = []
        for idx in range(sample_num):
            cpart = part[idx,:]
            cpart = cpart.view(dim,-1)
            res = torch.from_numpy(method.fit_transform(cpart)).reshape(1,dim)
            cparts.append(res)
        part = torch.concat(cparts,dim = 0)
        parts.append(part)
    
    res = torch.concat(parts, dim = 1).float()

    return res
