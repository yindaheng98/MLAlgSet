from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(7, 1)

    def forward(self, x):
        x=self.fc(x)
        x=torch.sigmoid(x)
        return x

from PimaDataset import test_loader,getTrainLoaders,getDropLoaders

batch_size=1
epochs=5
log_interval=0

from Model import Model,process

res=''

print('dropS')
train_loaders,test_loaders=getDropLoaders(batch_size,True)
train_loader,test_loader=train_loaders['dropS'],test_loaders['dropS']
res+=process("dropS",Net,train_loader,test_loader)

print('dropI')
train_loaders,test_loaders=getDropLoaders(batch_size,True)
train_loader,test_loader=train_loaders['dropI'],test_loaders['dropI']
res+=process("dropI",Net,train_loader,test_loader)

res=res[0:-1]+','

