from ann.ann1 import res as res1
from ann.ann2 import res as res2
from ann.ann3 import res as res3
import json
res=res1+res2+res3
res='{'+res[0:-1]+'}'
with open("结果/神经网络.json","w") as f:
    f.writelines(res)

#汇集"ann/"目录下的人工神经网络模型的结果，输出为json文件
