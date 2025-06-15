from fedpub.misc.utils import from_networkx
from gpfl.initial_graph import random_graphbatch

def process_fedpub(
    client,
    server,
    args
):
    # 1. initialize 
    
    # 1.1 generate graphs
    graph_batch = random_graphbatch(
        20,
        client[0].data[0].x.shape[1],
        seed = 0
    )
    
    
    # 1.2 initialize models
    
    
    
    # 2. train model
    
    # 2.1 round start
    
    
    # 2.2 calculate the similarity matrix based on the functional embedding
    
    
    # 2.3 update local model parameters 
    
    
    # 2.4 backward, update parameters and masks
    
    
    
    # 3. finish communication, collect information 
    
    
    