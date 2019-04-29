from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(6, 1)

    def forward(self, x):
        x=self.fc(x)
        x=torch.sigmoid(x)
        return x

from PimaDataset import test_loader,getTrainLoaders,getDropLoaders

batch_size=1
epochs=5
log_interval=0

from Model import Model,process2

train_loaders,test_loaders=getDropLoaders(batch_size,True)
train_loader,test_loader=train_loaders['dropIS'],test_loaders['dropIS']
targets,outputs=process2(Net,train_loader,test_loader)
