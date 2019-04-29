import torch
class Model:
    def __init__(self,model,device,optimizer,loss):
        self.device=device
        self.model=model
        self.optimizer=optimizer
        self.loss=loss

    def test(self,test_loader,thr=0.5):
        self.model.eval()
        test_loss = 0
        TP,FN,FP,TN=0,0,0,0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.float().to(self.device), target.float().to(self.device)
                output = self.model(data)
                output = output.reshape([1,-1])>=thr
                TP+=((output==1)&(target>0.5)).sum()
                FN+=((output==0)&(target>0.5)).sum()
                FP+=((output==1)&(target<0.5)).sum()
                TN+=((output==0)&(target<0.5)).sum()
            
        total=len(test_loader.dataset)
        correct = float(TP+TN)
        return('{"AC":%.6f,"TP":%d,"FN":%d,"FP":%d,"TN":%d},'%(correct/total,TP,FN,FP,TN))

    def train(self,train_loader,**kwargs):
        if kwargs!={}:
            if not('log_interval' in kwargs and kwargs['log_interval']!=0):
                kwargs['log_interval']=50
            if not 'thr' in kwargs:
                kwargs['thr']=0.5
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(self.device), target.float().to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if kwargs!={}:
                if batch_idx % kwargs['log_interval'] == 0:
                    print('Loss: {:.6f}'.format(loss.item()))
                    print('train set:',test(self.model,train_loader,kwargs['thr']))
                    if 'test_loader' in kwargs:
                        print('test set:',self.test(self.model,kwargs['test_loader'],kwargs['thr']))

import torch.nn as nn
import torch.optim as optim
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")

def process(name,Net,train_loader,test_loader):
    res='"%s":['%name
    m = Net().to(device)
    optimizer=optim.Adagrad(m.parameters(), lr=0.35)
    for _ in range(5):
        model=Model(m,device,optimizer,nn.MSELoss())
        model.train(train_loader)
        r=model.test(test_loader,thr=0.555)
        print(r)
        res+=r
    res=res[0:-1]+'],'
    return res

import numpy as np
def process2(Net,train_loader,test_loader):
    m = Net().to(device)
    optimizer=optim.Adagrad(m.parameters(), lr=0.35)
    model=Model(m,device,optimizer,nn.MSELoss())
    model.train(train_loader)
    model.model.to('cpu')
    targetss=[]
    outputss=[]
    with torch.no_grad():
        for _ in range(5):
            targets=[]
            outputs=[]
            for data, target in test_loader:
                data, target = data.float(), target.float()
                targets=target.numpy() if targets==[] else np.concatenate([targets,target.numpy()])
                outputs=model.model(data).numpy() if outputs==[] else np.concatenate([outputs,model.model(data).numpy()])
            targetss.append(list(targets))
            outputss.append(list(outputs.reshape([-1])))
    return targetss,outputss
    
