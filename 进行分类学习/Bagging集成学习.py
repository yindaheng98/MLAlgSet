from PimaDatatable import datatables
from skTrain import sklearn_processes as process
import numpy as np
np.random.seed(0)

from sklearn.ensemble import RandomForestClassifier
res=process(lambda :RandomForestClassifier(n_estimators=500,n_jobs=-1),datatables,5)
res=res[0:-1]
with open('结果/随机森林.json','w') as f:
    f.writelines(res)
