with open('../ROC.txt','r') as f:
    datas=eval(f.read())
import numpy as np
def convert(data):
    target=np.array(data['targets'])
    output=np.array(data['outputs'])
    TPRs=[]
    FPRs=[]
    for thr in np.linspace(0,1,100):
        TP=np.sum((output>=thr)&(target>0.5))
        FN=np.sum((output<thr)&(target>0.5))
        FP=np.sum((output>=thr)&(target<0.5))
        TN=np.sum((output<thr)&(target<0.5))
        TPRs.append(TP/(TP+FN))
        FPRs.append(FP/(FP+TN))
    return {'TPR':TPRs,'FPR':FPRs}
for k in datas:
    datas[k]=convert(datas[k])

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['font.size']=10
for k in datas:
    plt.plot(datas[k]['FPR'],datas[k]['TPR'],label=k)

plt.xlim([0,1])
plt.ylim([0,1.004])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.tight_layout()
plt.savefig('图/f5.pdf')

