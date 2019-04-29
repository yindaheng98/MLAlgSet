import torch
import torch.nn as nn
import torch.optim as optim
from Model import Model,device
def process(name,Net,train_loader,test_loader):
    res='"%s":['%name
    m = Net().to(device)
    optimizer=optim.Adagrad(m.parameters(), lr=0.35, weight_decay=1e-5)
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
    optimizer=optim.Adagrad(m.parameters(), lr=0.35, weight_decay=1e-5)
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
