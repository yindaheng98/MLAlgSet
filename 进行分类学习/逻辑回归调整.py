from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x=self.fc(x)
        x=torch.sigmoid(x)
        return x
    
from PimaDataset import test_loader,getTrainLoaders,getDropLoaders
from TrainTest import train,test,device

batch_size=1
epochs=5
log_interval=0
def process():
    model = Net().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.35)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer,log_interval,test_loader)
        #train(model, train_loader, optimizer,log_interval,test_loader)
        #train(model, train_loader, optimizer,log_interval,test_loader)
        test(model, test_loader,thr=0.555)

train_loaders=getTrainLoaders(batch_size,True)
for k in train_loaders:
    print(k)
    train_loader=train_loaders[k]
    process()
