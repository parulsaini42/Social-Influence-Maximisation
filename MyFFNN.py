#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import networkx as nx
import random
import numpy as np


# In[3]:



edges = pd.read_csv('chain_graph.csv')
edge_list=edges.values.tolist()
for e in edge_list:
        if len(set(e))==len(e):
            continue
        else:
            edge_list.remove(e)
        
G = nx.from_edgelist(edge_list)


# Read json file with seed set and activations

# In[4]:


import json
# read file
with open('chain.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)


# Create binary vector for input(1X400) and final vector(1X400) and store as a row in a dataframe(1X800)

# In[7]:


global seed
global active

def func1(f):
    if f in seed:
        return 1
    else:
        return 0
def func2(f):
    if f in active:
        return 1
    else:
        return 0
l=[]
for i in range(1,len(obj)):
    in_vec=list(G.nodes())
    fin_vec=list(G.nodes())
    seed=obj[i]['seed set'] 
    active=obj[i]['sequence']
    inp=list(map(func1,in_vec))
    fin=list(map(func2,fin_vec))
    inp.extend(fin)
    l.append(inp)
  
  
df=pd.DataFrame(l,columns = [str(x) for x in range(0,2*len(G.nodes()))]) 
df.head()


# Export to csv

# In[ ]:


export_csv = df.to_csv (r'C:\Users\Parul\Data\chain_vector.csv', index = None, header=True) 


# In[9]:


from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn


# Feed into neural network creating custom dataset

# In[11]:



class CustomDataset():
    def __init__(self, csv_path,graph):
       
        self.data = pd.read_csv(csv_path)
        self.to_tensor = transforms.ToTensor()
        self.num= len(graph.nodes())
        # Calculate len
        self.data_len = len(self.data.index)
        
    def __getitem__(self, index):
        inp_as_np = np.asarray(self.data.iloc[index][0:self.num]).reshape(1,400).astype('uint8')
        fin_as_np = np.asarray(self.data.iloc[index][self.num:]).reshape(1,400).astype('uint8')
        # Transform to tensor
        inp_as_tensor = self.to_tensor(inp_as_np)
        fin_as_tensor = self.to_tensor(fin_as_np)
        
        return (inp_as_tensor, fin_as_tensor)

        

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Call dataset
    dataset = CustomDataset('chain_vector.csv',G)
    # Define data loader
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
    trainset, valset = random_split(dataset,lengths)

    train_loader = DataLoader(trainset, batch_size=100, shuffle=True)
    val_loader = DataLoader(valset, batch_size=100, shuffle=True)
    
    


# In[12]: NN Model



class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


# In[13]:



input_dim = 1*400
hidden_dim = 400
output_dim = 1*400
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
#loss
criterion = nn.CrossEntropyLoss()


# In[14]:


#optimizer
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


# In[51]:


print(len(list(model.parameters())))


# In[52]:


print(list(model.parameters())[0].size())


# In[53]:


print(list(model.parameters())[1].size())


# In[54]:


print(list(model.parameters())[2].size())


# In[55]:


print(list(model.parameters())[3].size())


# In[15]:


# 100 iterations and 3 epochs
batch_size = 100
n_iters = 300
num_epochs = n_iters / (len(train_loader) / batch_size)
num_epochs = int(num_epochs)



# In[17]:



'''
STEP 7: TRAIN THE MODEL
'''
iter = 0
for epoch in range(num_epochs):
    for i, (in_, fin_) in enumerate(train_loader):
        # Load images with gradient accumulation capabilities
        in_ = in_.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(in_)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, fin_)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for in_, fin_ in val_loader:
                # Load images with gradient accumulation capabilities
                in_ = fin_.requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(in_)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += fin_.size(0)

                # Total correct predictions
                correct += (predicted == fin_).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


# In[ ]:





# In[ ]:




