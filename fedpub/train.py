import torch

from fedpub.misc.utils import from_networkx
from gpfl.initial_graph import random_graphbatch
from fedpub.utils import cossim
from client import Client_GC
from training import analyze_train


def local_train_fedpub(
    local_clients : list[Client_GC],
    cur_round,
    args
):
    """ perform customized local training under FedPub setting

    Args:
        local_models (_type_): the client local models (shallow copy)
        train_local_dls: local dataloader for training
        args (_type_): _description_
    """
    
    for i, client in enumerate(local_clients):
        train_local_dl = client.dataLoader['train']
        client.model.train()
        iterator = iter(train_local_dl)
        for _, batch in enumerate(iterator):
            batch = batch.to(args.device)
            client.optimizer.zero_grad()
            target = batch.y
            out = client.model(batch)
            loss = client.model.loss(out, target)
            for name, param in client.model.state_dict():
                if 'mask' in name:
                    loss += torch.norm(param.float(), 1) * args.fedpub_l1
                elif 'conv' in name or 'readout' in name:
                    if cur_round == 1: 
                        continue
                    ## prev_w may need to be fixed
                    loss += torch.norm(param.float() - client.prev_w[name], 2) * args.fedpub_loc_l2
            loss.backward()
            client.optimizer.step()
    
    
    

def process_fedpub(
    clients,
    server,
    args
):
    # 1. initialize 
    
    # 1.1 generate graphs
    proxy_graph = random_graphbatch(
        20,
        clients[0].data[0].x.shape[1],
        seed = 0
    )
    
    
    # 1.2 initialize models
    # client and server already contains initalized models
    
    num_client = len(clients)
    # 2. train model
    for i in range(1, args.num_round + 1):
        # 2.1 local train 
        # for c in clients:
        #     c.compute_weight_update(args.local_epoch)
        local_train_fedpub(
            clients,
            i,
            args
        )
        # 2.2 generate functional embedding
        embed = server.graph_modelembedding(clients, proxy_graph.to(args.device), "sum")
        # embed is a (n,k) tensor, where n is the number of clients and k is the embedding dimension
        
        # 2.3 calculate the similarity matrix based on the functional embedding
        sim_matrix = torch.empty((num_client, num_client))
        for i in range(num_client):
            for j in range(num_client):
                sim_matrix[i,j] = 1 - cossim(embed[i], embed[j])
        # apply exp over sim weights
        sim_matrix = torch.exp(args.norm_scale * sim_matrix)
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, None]
        
        # 2.4 backward, update parameters and masks
        
        # 1. aggregate client model weights by sim_matrix weight
        
        
        # 2. set model weights here
        
        # 2.5 evaluation
        for c in clients:
            c.model.cuda()
            c.evaluate()
        
    
    # 3. finish communication, collect information 
    allAccs = analyze_train(clients, args)
    return allAccs
    
    
    