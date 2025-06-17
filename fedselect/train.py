import copy
from fedselect.lottery_ticket import init_mask_zeros, delta_update
from collections import OrderedDict
from client import Client_GC
from fedselect.pflopt.optimizers import MaskLocalAltSGD
import torch
from broadcast import (
    add_masks,
    add_server_weights,
    div_server_weights,
    broadcast_server_to_client_initialization
)
from training import analyze_train

def local_train(
    client : Client_GC,
    mask,
    args,
    clip_grad_norm = False,
    max_grad_norm = 3.5
):
    optimizer = MaskLocalAltSGD(client.model.parameters(), mask, lr = args.lr)
    # epoch = args.local_epoch
    train_loss_1 = 0
    iterator = iter(client.dataLoader['train'])
    
    for _, batch in enumerate(iterator):
        batch = batch.to(args.device)
        optimizer.zero_grad()
        output = client.model(batch)
        loss = client.model.loss(output, batch.y)
        train_loss += loss.item()
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(client.model.parameters(), max_grad_norm)
        optimizer.step()
        
    train_loss_1 /= len(client.dataLoader['train'])
    optimizer.toggle()
    
    return train_loss_1

        
    
    




def process_fedselect(
    clients,
    server,
    args
):
    # 1. do some initializations
    initial_state_dict = copy.deepcopy(server.model.state_dict())
    # 2. FL stage
    com_rounds = args.num_rounds
    client_accuracies = [{i:0 for i in clients} for _ in range(args.num_rounds)]
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in range(len(clients))}
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in range(len(clients))}
    client_masks = {i: None for i in range(len(clients))}
    client_masks_prev = {i: init_mask_zeros(server.model) for i in range(len(clients))}
    server_accumulate_mask = OrderedDict()
    server_weights = OrderedDict()
    
    lth_iters = args.fedselect_lth_epoch_iters
    prune_rate = args.fedselect_prune_percent / 100
    prune_target = args.fedselect_prune_target / 100
    lottery_ticket_convergence = []
    
    for round_num in range(com_rounds):
        round_loss = 0
        for i in range(len(clients)):
            client = client[i]
            client_mask = client_masks_prev.get(i)
            loss = local_train(client,client_mask,args)
            round_loss += loss
            if round_num < com_rounds - 1:
                # update server
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                server_weights = add_server_weights(
                    server_weights, client.model.state_dict(), client_mask
                )
                
            client_state_dicts[i] = copy.deepcopy(client.model.state_dict())
            client_masks[i] = copy.deepcopy(client_mask)
            
            if round_num % lth_iters == 0 and round_num != 0:
                client_mask = delta_update(
                    prune_rate,
                    client_state_dicts[i],
                    client_state_dict_prev[i],
                    client_masks_prev[i],
                    bound=prune_target,
                    invert=True,
                )
                client_state_dict_prev[i] = copy.deepcopy(client_state_dicts[i])
                client_masks_prev[i] = copy.deepcopy(client_mask)
                
        round_loss /= len(clients)
        
        # compute acc matrix ? not sure if used
        if round_num < com_rounds - 1:
            
            server_weights = div_server_weights(server_weights, server_accumulate_mask)
            # Server broadcasts non lottery ticket parameters u_i to every device
            for i in range(clients):
                client_state_dicts[i] = broadcast_server_to_client_initialization(
                    server_weights, client_masks[i], client_state_dicts[i]
                )
            server_accumulate_mask = OrderedDict()
            server_weights = OrderedDict()

        
        # evaluation
        for c in clients:
            c.model.cuda()
            c.evaluate()
    
    allAccs = analyze_train(clients, args)