from data_convert import cdatas_swap as cdatas
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['font.size']=10

import numpy as np
plt.figure(figsize=(6,9),dpi=120)

dw=0.08
d=0
datas=cdatas['随机'].loc[pd.IndexSlice[:,:],pd.IndexSlice['avg','AC']]
tick_label=list(datas.index.get_level_values(0).drop_duplicates())
for k in datas.index.get_level_values(1).drop_duplicates():
    d+=dw
    data=datas.loc[pd.IndexSlice[:,k]]
    x=np.arange(len(data))*0.6+d
    txerr=[]
    txerr.append(np.array(cdatas['随机'].loc[pd.IndexSlice[:,k],pd.IndexSlice['max','AC']]))
    txerr.append(np.array(cdatas['随机'].loc[pd.IndexSlice[:,k],pd.IndexSlice['min','AC']]))
    xerr=[np.array(data[0:(len(data)-len(txerr[0]))]),np.array(data[0:(len(data)-len(txerr[1]))])]
    xerr[0]=np.concatenate([xerr[0],txerr[0]])
    xerr[1]=np.concatenate([xerr[1],txerr[1]])
    xerr[0]=np.abs(xerr[0]-data)
    xerr[1]=np.abs(xerr[1]-data)
    plt.hlines(x, txerr[0], txerr[1])
    plt.plot(data, x, 'o',label=k)
    for dat,y,tx0,tx1 in zip(data,x,txerr[0],txerr[1]):
        plt.text(tx0+0.007,y-0.16*dw,'%.2f%%'%(tx0*100), ha='center', va= 'bottom',fontsize=6)
        plt.text(tx1-0.006,y-0.16*dw,'%.2f%%'%(tx1*100), ha='center', va= 'bottom',fontsize=6)
        plt.text(dat,y+0.1*dw,'%.2f%%'%(dat*100), ha='center', va= 'bottom',fontsize=6)

plt.yticks(x-2*dw,tick_label)
plt.legend(fontsize=10,loc='upper right')
plt.xlabel('测试集上的预测正确率')
plt.xlim([0.75,1])
plt.tight_layout()
plt.savefig('图/f4.pdf')

