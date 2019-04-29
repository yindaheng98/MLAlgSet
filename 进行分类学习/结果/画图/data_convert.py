def merge(d):
    r={}
    for k in d[0]:
        r[k]=[]
        for dd in d:
            r[k].append(dd[k])
        rr=r[k]
        r[k]={}
        r[k]['max']=max(rr)
        r[k]['min']=min(rr)
        r[k]['avg']=float(sum(rr))/len(rr)
    return r

def convert(jdata):
    for k in jdata:
        jdata[k]=merge(jdata[k])
    return jdata

import os
import json
jdatas={}
data_path='../'
for filelist in os.walk(data_path):
    if filelist[0]==data_path:
        for file in filelist[2]:
            if file[-5:]=='.json':
                with open(data_path+file,'r') as jf:
                    jdata=json.load(jf)
                    jdatas[file[0:-5]]=convert(jdata)

import pandas as pd
cdatas={}
cdatas['随机']=pd.DataFrame.from_dict(jdatas,orient='index').stack().apply(lambda r:pd.DataFrame.from_dict(r).stack())
file='sklearn结果合集/sklearn结果合集.json'
with open(data_path+file,'r') as jf:
    cdatas['非随机']=pd.DataFrame.from_dict(json.load(jf),orient='index').stack().apply(lambda r:pd.Series(r))

things_to_drop=['mean']
for i in range(10):
    things_to_drop.append('knn%d'%(i+1))
for i in range(5):
    things_to_drop.append('pmm%d'%(i+1))

rename_dict={'em':'EM填补',
             'mice':'MICE填补',
             'knn':'KNN填补',
             'pmm':'PMM填补',
             'drop':'删除数据缺失行',
             'dropS':'仅删除SkinThickness列',
             'dropI':'仅删除Insulin列',
             'dropIS':'两列都删除',
             }

for k in cdatas:
    cdatas[k]=cdatas[k].rename(index= {'knn5':'knn','pmm3':'pmm'},level=1)
    cdatas[k]=cdatas[k].rename(index= rename_dict,level=1)
    cdatas[k]=cdatas[k].drop(things_to_drop,level=1)

rename_dict={'GaussianNB':'高斯朴素贝叶斯',
             'entropy':'ID3决策树',
             'gini':'CART决策树',
             'knn':'KNN分类器',
             'linear':'线性核SVM',
             'poly':'多项式核SVM',
             'rbf':'径向基核SVM'}
cdatas['非随机']=cdatas[k].rename(index= rename_dict,level=0)
cdatas_swap={
    '随机':cdatas['随机'].swaplevel(),
    '非随机':cdatas['非随机'].swaplevel()}
