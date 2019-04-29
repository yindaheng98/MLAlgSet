from ann.dropout.ann1 import res as res1
from ann.dropout.ann2 import res as res2
from ann.dropout.ann3 import res as res3
import json
res=res1+res2+res3
res='{'+res[0:-1]+'}'
with open("结果/神经网络+Dropout.json","w") as f:
    f.writelines(res)

#汇集"ann/dropout/"目录下的加了dropout的人工神经网络模型的结果，输出为json文件
