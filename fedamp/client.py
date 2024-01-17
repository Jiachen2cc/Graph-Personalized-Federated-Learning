import torch
import copy as cp
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
#from analyze_dataset import tri_num_disb,deg_disb,hop2_disb
import numpy as np
import random
from pfedgraph_cosine.data_utils import label_balanced_downsample
from pfedgraph_cosine.utils import get_roc_auc
import torch.nn.functional as F
from pfedgraph_cosine.graph_utils import sd_matrixing

class Client_GC():
    def __init__(self, model, client_id, client_name, dataset_name, data, split_idx, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.dname = dataset_name

        self.data = data
        self.split_idx = split_idx

        self.train_size = None
        self.train_data = None
        self.test_data = None
        self.dataLoader = {'train': None, 'test': None}

        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}
        self.control = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dcontrol = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.weights = {k:torch.ones_like(self.W[k]).to(args.device) for k in self.W.keys()}
        self.wm = 1

        self.gconvNames = None

        self.train_stats = {
            'trainingLosses': [],
            'trainingAccs': [],
            'testLosses': [],
            'testAccs': []
        }

        self.eval_stats = {
            'testLosses': [],
            'testAccs': [],
            'testRocs':[],
        }

        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.
        
        '''
        self.f_dict = {
            'triangle_disb': tri_num_disb,
            'degree_disb': deg_disb,
            'hop2_disb': hop2_disb
        }
        '''
    
    # designed for global model training: merge local datasets to build global dataset
    def merge_data(self,local_trains,local_tests,Batchsize):
        
        train,test = [],[]
        for ltrain,ltest in zip(local_trains,local_tests):
            train += ltrain
            test += ltest
        self.train_size = len(train)
        self.dataLoader['train'] = DataLoader(train, Batchsize, True)
        self.dataLoader['test'] = DataLoader(test, Batchsize, True)

        self.train_data = cp.deepcopy(train)
        self.test_data = cp.deepcopy(test)
    
    # sample a uniform subdataset from the original dataset
    def sample_uniform(self):
        # classify the train dataset according to the label 
        labels = np.array([g.y.item() for g in self.train_data])
        label_num = np.max(labels)+1
        subgraph = [np.where(labels == i)[0] for i in range(label_num)]
        sample_size = min([len(sub) for sub in subgraph])
        
        #sample_idx = np.concatenate([np.random.choice(sub,sample_size) if len(sub) < sample_size else np.random.choice(sub,sample_size,replace = False) for sub in subgraph])
        sample_idx = np.concatenate([np.random.choice(sub,sample_size) for sub in subgraph])
        return [self.train_data[idx] for idx in sample_idx]
        


    def split_traintest(self,fold_idx,Batchsize,args):

        train_idx, test_idx = self.split_idx['train'][fold_idx], self.split_idx['test'][fold_idx]
        train = [self.data[idx] for idx in train_idx]
        test = [self.data[idx] for idx in test_idx]
        
        # label balanced dataset downsample
        #if args.data_down < 1 and 'global_model' not in self.name:
        '''
        if args.data_down < 1:
            labels = np.array([g.y.item() for g in train])
            train_idx = label_balanced_downsample(labels,args.data_down,args.seed)
            train = [train[idx] for idx in train_idx]
        '''
        self.train_size = len(train)
        
        #print(self.name,self.train_size,len(test),len(self.data))
        self.dataLoader['train'] = DataLoader(train, Batchsize, True)
        self.dataLoader['test'] = DataLoader(test, Batchsize, True)
        
        self.train_data = cp.deepcopy(train)
        self.test_data = cp.deepcopy(test)

        #return struc_f
    
    def get_labeldis(self):

        labels = np.array([g.y.item() for g in self.train_data])
        ldis = [np.sum(labels == i) for i in range(1,int(np.max(labels)+1))]
        return ldis

    def dict_extend(self,x,y):
        
        for k in x.keys():
            x[k].extend(y[k])
        
        return x
    
    def structure_feature_analysis(self,fname):
        assert self.train_data != None

        sfeature = self.f_dict[fname](self.train_data)

        return sfeature

    def download_from_server(self, args, server):
        self.gconvNames = server.W.keys()
        if args.Federated_mode == 'fedstar':
            for k in server.W:
                if '_s' in k:
                    self.W[k].data = server.W[k].data.clone()
        else:
            for k in server.W:
                self.W[k].data = server.W[k].data.clone()
    
    def download_weight(self,W):
        self.gconvNames = W.keys()
        for k in W:
            self.W[k].data = W[k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device,self.train_data)
        #print(train_stats)
        self.train_stats = self.dict_extend(self.train_stats,train_stats)
        #print(self.train_stats)

        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()
    


    def compute_weight_update(self, local_epoch):
        """ For GCFL """
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device,self.train_data)
        self.train_stats = self.dict_extend(self.train_stats, train_stats)


        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)


        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate(self,record = True):

        valloss,valacc,roc_auc_score = eval_gc(self.model, self.dataLoader['test'], self.args.device)
        
        if record:
            self.eval_stats['testLosses'].append(valloss)
            self.eval_stats['testAccs'].append(valacc)
            self.eval_stats['testRocs'].append(roc_auc_score)

        return valloss,valacc,roc_auc_score
    
    def load_param_matrix(self,param):

        # reshape parameter tensor back to model parameters
        pointer = 0
        for k in self.W.keys():
            kshape = self.W[k].data.shape
            num_p = 1
            for n in kshape:
                num_p *= n
            self.W[k].data = param[pointer:pointer+num_p].view(kshape)
            pointer += num_p
        
        

    def ALA_aggregate(self,server_dW,args):
        
        # create temp model
        model_t = cp.deepcopy(self.model)
        param_t = {k:v for k,v in model_t.named_parameters()}

        # compute loss
        optimizer = torch.optim.SGD(model_t.parameters(),0)

        # load the aggregated gradient in server
        serverd = cp.deepcopy(self.dW)
        pointer = 0
        loss = 0
        #loss = F.mse_loss(param,torch.zeros(param.shape).to(param.device))
        for k in serverd.keys():
            kshape = serverd[k].data.shape
            num_p = 1
            for n in kshape:
                num_p *= n
            serverd[k].data = server_dW[pointer:pointer + num_p].view(kshape)
            pointer += num_p 
        
        locald = cp.deepcopy(self.dW)
        localp = cp.deepcopy(self.W_old)
        
        # weight initialization & param initialization
        #weights = {k:torch.ones_like(localp[k]).to(args.device) for k in localp.keys()}
        for k in serverd.keys():
            param_t[k].data = localp[k] + locald[k] + (serverd[k] - locald[k]) * self.weights[k]

        # initialize the parameters 
        # initialize 
        # sample ratio% of training data and compute loss
        #sample_idx = label_balanced_downsample(np.array([g.y.item() for g in self.train_data]),args.ala_ratio,args.seed)
        #sample = [self.train_data[idx] for idx in sample_idx]
        sample_loader = DataLoader(self.train_data,args.batch_size,True)
        
        # perform training 
        losses = []
        cnt = 0
        model_t.train()

        for i in range(args.ala_round):

            loss = 0

            for _, batch in enumerate(self.dataLoader['train']):
                optimizer.zero_grad()
                batch.to(args.device)
                pred = model_t(batch)
                tloss = model_t.loss(pred,batch.y)
                tloss.backward()
                loss += tloss*batch.num_graphs
                optimizer.step()
                
                '''
                print('accept loss {:.4f}'.format(tloss))
                for k in serverd.keys():
                    print('gradient norm')
                    print(torch.norm(param_t[k].grad))
                    print('difference norm')
                    print(torch.norm((serverd[k] - locald[k])*param_t[k].grad))
                '''
                # update weight 
                
                for k in serverd.keys():
                    self.weights[k].data = torch.clamp(
                        self.weights[k].data + (args.ala_lr/args.lr) * param_t[k].grad * (serverd[k] - locald[k]), 0, 1)
                    #print('weigth diff {:.4f}'.format(torch.norm(args.ala_lr * param_t[k].grad * (serverd[k] - locald[k]))))
                #self.wm = sum([torch.sum(weights[k]) for k in serverd.keys()])/(pointer+1)
                # update param
                for k in serverd.keys():
                    param_t[k].data = localp[k] + locald[k] + (serverd[k] - locald[k]) * self.weights[k]
                
            loss /= len(self.train_data)
            #print('current loss {:.4f}'.format(loss))
            weight_mean = sum([torch.sum(self.weights[k]) for k in serverd.keys()])/(pointer+1)
            #print('mean accept weight {:.4f}'.format(weight_mean))
            losses.append(loss)
            cnt += 1
        
        self.download_weight({k: param_t[k] for k in serverd.keys()})
        # compute the average accept rate
        weight_mean = sum([torch.sum(self.weights[k]) for k in serverd.keys()])/(pointer+1)
        #print(weight_mean)
        #exit(0)
        return weight_mean





    def local_train_prox(self, local_epoch, mu):
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        """ For FedProx """
        train_stats = train_gc_prox(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device,
                               self.gconvNames, self.W, mu, self.W_old)
        self.train_stats = self.dict_extend(self.train_stats, train_stats)
        
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()
        
    def local_train_fedgraph(self, local_epoch, cluster_model = None, lam = 0.01):
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)
        
        train_stats = train_gc_fedgraph(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device,
                               cluster_model, lam)
        self.train_stats = self.dict_extend(self.train_stats, train_stats)
        
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()
        
        

    def evaluate_prox(self, mu):
        return eval_gc_prox(self.model, self.dataLoader['test'], self.args.device, self.gconvNames, mu, self.W_old)


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm

def train_gc(model, dataloaders, optimizer, local_epoch, device, train_data):
    losses_train, accs_train, losses_test, accs_test = [], [], [], []
    train_loader, test_loader = dataloaders['train'],dataloaders['test']
    
    
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0
        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        #loss_v, acc_v = eval_gc(model, val_loader, device)
        loss_tt, acc_tt,_ = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        #losses_val.append(loss_v)
        #accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)
    
    '''
    for epoch in range(local_epoch):
  
        model.train()
        selected_idx = np.random.permutation(len(train_data))[:args.batchsize]
        batch_graph = [train_data[idx] for idx in selected_idx]
        loader = DataLoader(batch_graph,args.batchsize)
        for _,batch in enumerate(loader):
            break
        batch.to(device)
        pred,label = model(batch),batch.y
        acc = (pred.max(dim = 1)[1].eq(label).sum().item())/len(label)
        loss = model.loss(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tt, acc_tt = eval_gc(model, test_loader, device)

        losses_train.append(loss)
        accs_train.append(acc)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)
    '''
    #print('train loss',losses_train)
    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, #'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}

def eval_gc(model, test_loader, device):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    preds,labels = [],[]
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        preds.append(pred)
        labels.append(F.one_hot(label,pred.shape[1]))
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    preds = torch.concat(preds,dim = 0)
    labels = torch.concat(labels,dim = 0)
    #print(preds.shape)
    #print(labels.shape)

    roc_auc_score = get_roc_auc(labels,preds)
    #roc_auc_score = 0

    return total_loss/ngraphs, acc_sum/ngraphs, roc_auc_score


def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox

def train_gc_prox(model, dataloaders, optimizer, local_epoch, device, gconvNames, Ws, mu, Wt):
    losses_train, accs_train,losses_test, accs_test = [], [], [], []
    convGradsNorm = []
    train_loader,test_loader = dataloaders['train'],dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        #loss_v, acc_v = eval_gc(model, val_loader, device)
        loss_tt, acc_tt,_ = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        #losses_val.append(loss_v)
        #accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, #'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test, 'convGradsNorm': convGradsNorm}


def train_gc_fedgraph(model, dataloaders, optimizer, local_epoch, device, cluster_model = None, lam = 0.01):
    losses_train, accs_train,losses_test, accs_test = [], [], [], []
    convGradsNorm = []
    train_loader,test_loader = dataloaders['train'],dataloaders['test']
    
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label) #+ mu / 2. * _prox_term(model, gconvNames, Wt)
            # regulation term
            if cluster_model is not None:
                flatten_model = sd_matrixing({k:v for k,v in model.named_parameters()})
                loss += lam * torch.dot(cluster_model, flatten_model)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        #loss_v, acc_v = eval_gc(model, val_loader, device)
        loss_tt, acc_tt,_ = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

        #convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, #'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test} #, 'convGradsNorm': convGradsNorm}

def eval_gc_prox(model, test_loader, device, gconvNames, mu, Wt):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs