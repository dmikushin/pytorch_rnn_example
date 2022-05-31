#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


# In[48]:


class RNNCell(nn.Module):
    
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(RNNCell, self).__init__()
        self.Wx = torch.randn(hiddenSize, inputSize) # input weights
        self.Wh = torch.randn(hiddenSize, hiddenSize) # hidden weights
        self.Wy = torch.randn(outputSize,recurhiddenSizerentSize) # output weights
        self.h = torch.zeros(hiddenSize,1) # initial hidden state
        self.bh = torch.zeros(hiddenSize,1) # hidden state bias
        self.by = torch.zeros(outputSize,1) # output bias

    def forward(self, x):
        self.h = torch.tanh(self.bh + torch.matmul(self.Wx, x) + torch.matmul(self.Wh,self.h))
        output = nn.Softmax(self.by + torch.matmul(self.Wy,self.h))
        
        return output, self.h


# In[49]:


X = torch.sin(torch.linspace(0,100,100000))
plt.plot(X)
plt.ylabel('Sin x')
plt.xlabel('x')


# In[50]:


class RNNData(Dataset):
    def __init__(self, X, sequenceLength):
        'Initialization'
        self.X = X
        self.sequenceLength = sequenceLength

    def __len__(self):
        'Denotes the total number of samples'
        return int(torch.floor(torch.tensor(len(self.X)/self.sequenceLength)))
    
    def __getitem__(self, index):
        sequence = self.X[index:index+self.sequenceLength]
        y = self.X[index+self.sequenceLength+1]
        return sequence, y


# In[51]:


#hyperparameters
batchSize = 100 
sequenceLength = 50
numLayers = 1
hiddenSize = 4
learningRate = 0.01
epochs = 100


# In[52]:


data = RNNData(X,sequenceLength)
dataLoader = DataLoader(data, batch_size=batchSize, shuffle=True)
for x,y in dataLoader:
    print(x)
    print(y)
    break


# In[53]:


# create our RNN based network with an RNN followed by a linear layer
class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super().__init__()
        self.RNN = nn.RNN(input_size=inputSize, 
                          hidden_size=hiddenSize, 
                          num_layers=numLayers, 
                          nonlinearity='tanh', 
                          batch_first=True) #inputs and outputs are  (batch, seq, feature)
        self.linear = nn.Linear(hiddenSize,1)
        
    def forward(self,x,hState):
        x, h = self.RNN(x,hState)
        out = self.linear(x[:,-1,:]) # gets last output
        return out


# In[54]:


# create our network instance, pick loss function and optimizer
model = RNN(1,hiddenSize,numLayers)
lossFn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)


# In[55]:


# check output to see if everything is setup correctly
ytest = model(torch.randn(batchSize,sequenceLength,1),torch.zeros([numLayers, batchSize, hiddenSize]))
ytest.shape


# In[56]:


# train the model!
model.train()
lossHistory = []
for epoch in range(epochs):
    lossTotal = 0
    for x,y in dataLoader:
        hState = torch.zeros([numLayers, batchSize, hiddenSize])
        yhat= model(x.reshape([batchSize,sequenceLength, 1]),hState)
        
        loss = lossFn(yhat.view(-1),y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        lossTotal +=loss
    lossHistory.append(lossTotal)
    print(lossTotal.item())
        
plt.plot(lossHistory)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[57]:


print(X[:sequenceLength])
print(X[sequenceLength+1])


# In[58]:


model.eval()
model(X[:sequenceLength].reshape(1,sequenceLength,1),torch.zeros([numLayers, 1, hiddenSize]))


# In[ ]:




