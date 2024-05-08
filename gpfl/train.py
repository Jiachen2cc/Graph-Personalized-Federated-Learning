
def prepare_initial(clients):
    
    # choice 1 acquire property value and construct graph based on simi
    property = ...
    
    # choice 2 random initialize
    random = ...
    
    return ...
    

def process_gpfl(clients,server,args):
    
    #1 initialization
    ...
    
    # loop over each communication round
    for i in range(args.num_rounds):
        
        # 1 local train
        for c in clients:
            c.train()
            
        # 2 prepare client feature initial graph
        cfeature = ...
        init_A = ...
        
        # 3 construct client graph
        A = ...
        
        # 4 graph-guided model aggregation
        
        # evaluate and record important info
    
    # evaluate train stage
    return ...    
    
        
        