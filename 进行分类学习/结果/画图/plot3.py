from data_convert import cdatas_swap as cdatas
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['font.size']=12

import numpy as np
plt.figure(figsize=(6,12),dpi=300)

dw=0.025
d=0
data1=cdatas['非随机'].loc[pd.IndexSlice[:,:],'AC']
data2=cdatas['随机'].loc[pd.IndexSlice[:,:],pd.IndexSlice['avg','AC']]
datas=pd.concat([data1,data2])
tick_label=list(datas.index.get_level_values(0).drop_duplicates())
for k in datas.index.get_level_values(1).drop_duplicates():
    d+=dw
    data=datas.loc[pd.IndexSlice[:,k]]
    x=np.arange(len(data))*0.4+d
    plt.barh(x,data,dw,label=k)
    for dat,y in zip(data,x):
        plt.text(dat+0.008,y-0.5*dw,'%.2f%%'%(dat*100), ha='center', va= 'bottom',fontsize=6)

plt.yticks(x-3.5*dw,tick_label)
plt.legend(fontsize=10,loc='upper left')
plt.xlabel('测试集上的预测正确率')
plt.xlim([0.65,0.88])
plt.tight_layout()
plt.savefig('图/f3.pdf')
