from logreg.logreg1 import res as res1
from logreg.logreg2 import res as res2
from logreg.logreg3 import res as res3
import json
res=res1+res2+res3
res='{'+res[0:-1]+'}'
with open("结果/逻辑回归.json","w") as f:
    f.writelines(res)

#汇集"logreg/"目录下的逻辑回归模型的结果，输出为json文件
