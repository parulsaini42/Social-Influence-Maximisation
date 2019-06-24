import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import numpy as np

class CustomDataset:
    def __init__(self,args,graph,vec):
    	self.args = args
    	self.data = vec
    	self.to_tensor = transforms.ToTensor()
    	self.num= len(graph.nodes())
    	# Calculate len
    	self.data_len = len(self.data.index)
        
    def __getitem__(self, index):
        inp_as_np = np.asarray(self.data.iloc[index][0:self.num]).reshape(1,self.num)
        fin_as_np = np.asarray(self.data.iloc[index][self.num:]).reshape(1,self.num)
        # Transform to tensor
        inp_as_tensor = torch.from_numpy(inp_as_np).type('torch.FloatTensor').squeeze()
        fin_as_tensor = torch.from_numpy(fin_as_np).type('torch.FloatTensor').squeeze()
        
        return (inp_as_tensor, fin_as_tensor)

    def __len__(self):
        return self.data_len



