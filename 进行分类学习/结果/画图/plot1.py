from data_convert import cdatas
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['font.size']=12

import numpy as np
plt.figure(figsize=(6,12),dpi=300)

dw=0.04
d=0
data1=cdatas['非随机'].loc[pd.IndexSlice[:,:],'AC']
data2=cdatas['随机'].loc[pd.IndexSlice[:,:],pd.IndexSlice['avg','AC']]
datas=pd.concat([data1,data2])
tick_label=list(datas.index.get_level_values(0).drop_duplicates())
for k in datas.index.get_level_values(1).drop_duplicates():
    d+=dw
    data=datas.loc[pd.IndexSlice[:,k]]
    x=np.arange(len(data))*0.4+d
    '''
    txerr=[]
    txerr.append(np.array(cdatas['随机'].loc[pd.IndexSlice[:,k],pd.IndexSlice['max','AC']]))
    txerr.append(np.array(cdatas['随机'].loc[pd.IndexSlice[:,k],pd.IndexSlice['min','AC']]))
    xerr=[np.array(data[0:(len(data)-len(txerr[0]))]),np.array(data[0:(len(data)-len(txerr[1]))])]
    xerr[0]=np.concatenate([xerr[0],txerr[0]])
    xerr[1]=np.concatenate([xerr[1],txerr[1]])
    xerr[0]=np.abs(xerr[0]-data)
    xerr[1]=np.abs(xerr[1]-data)
    '''
    plt.barh(x,data,dw,label=k)
    for dat,y in zip(data,x):
        plt.text(dat+0.008,y-0.5*dw,'%.2f%%'%(dat*100), ha='center', va= 'bottom',fontsize=6)

plt.yticks(x-3.5*dw,tick_label)
plt.legend(fontsize=10,loc='upper left')
plt.xlabel('测试集上的预测正确率')
plt.xlim([0.65,0.88])
plt.tight_layout()
plt.savefig('图/f1.pdf')
