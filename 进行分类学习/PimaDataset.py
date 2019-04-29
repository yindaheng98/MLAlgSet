import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class PimaDataset(Dataset):
    def __init__(self,csv_file):
        data = pd.read_csv(csv_file,engine='python')
        self.len=len(data)
        self.X=np.array(data.drop(columns=['Outcome']))
        self.y=np.array(data['Outcome'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx,:],self.y[idx]

from PimaDatafiles import datafiles,test_file

test_loader=DataLoader(PimaDataset(test_file),batch_size=1, shuffle=False)

def getTrainLoaders(batch_size,shuffle):
    train_loaders={}
    for k in datafiles:
        train_loaders[k]=DataLoader(PimaDataset(datafiles[k]),batch_size=batch_size, shuffle=shuffle)
    return train_loaders

from PimaDatafiles import datafiles_drop

def getDropLoaders(batch_size,shuffle):
    train_loaders_drop={}
    test_loaders_drop={}
    for k in datafiles_drop:
        train_loaders_drop[k]=DataLoader(PimaDataset(datafiles_drop[k][0]),batch_size=batch_size,shuffle=shuffle)
        test_loaders_drop[k]=DataLoader(PimaDataset(datafiles_drop[k][1]),batch_size=1,shuffle=False)
    return train_loaders_drop,test_loaders_drop

#调用了PimaDatafiles，将数据集转化为pytorch风格
#供逻辑回归和人工神经网络使用
