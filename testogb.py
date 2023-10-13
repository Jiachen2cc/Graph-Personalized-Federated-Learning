from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

#dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = 'data/')
dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = 'data/')

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx['train']], batch_size = 32, shuffle = True)

atom_encoder = AtomEncoder(emb_dim = 100)
print(atom_encoder(dataset[0].x))
#print(atom_encoder(dataset[0].x))
print(dataset[0].y[0])
print(len(dataset))

