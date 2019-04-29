from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x=self.fc1(x)
        x=torch.sigmoid(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)
        return x

from PimaDataset import test_loader,getTrainLoaders,getDropLoaders

batch_size=1
epochs=5
log_interval=0

from Model import Model,process

res=''

train_loaders=getTrainLoaders(batch_size,True)
for k in train_loaders:
    print(k)
    train_loader=train_loaders[k]
    res+=process(k,Net,train_loader,test_loader)

print('drop')
train_loaders,test_loaders=getDropLoaders(batch_size,True)
train_loader,test_loader=train_loaders['drop'],test_loaders['drop']
res+=process("drop",Net,train_loader,test_loader)

res=res[0:-1]+','

