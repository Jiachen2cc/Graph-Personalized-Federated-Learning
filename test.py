import numpy as np
import random
import torch
import networkx as nx
import torch.nn.functional as F

x = torch.tensor([[0.4694,-0.2829,-1.5093],
                  [-1.1356,1.2127,-0.1735]])

print(F.normalize(x,p=2,dim =1 ))