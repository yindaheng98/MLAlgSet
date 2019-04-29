from PimaDatatable import datatables
from skTrain import sklearn_processes as process
import numpy as np


res='{'

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
np.random.seed(0)
bl=DecisionTreeClassifier(criterion='gini',max_depth=2)
res+='"CART+AdaBoost":'+process(lambda :AdaBoostClassifier(bl,n_estimators=200,algorithm="SAMME.R"),datatables,1)

from sklearn.svm import SVC
np.random.seed(0)
tol=1e-9
res+='"linear":'+process(lambda :SVC(kernel='linear',tol=tol,shrinking=False),datatables,1)
res+='"rbf":'+process(lambda :SVC(kernel='rbf',gamma='auto',tol=tol,shrinking=False),datatables,1)
res+='"poly":'+process(lambda :SVC(kernel='poly',gamma='auto',tol=tol,shrinking=False),datatables,1)

from sklearn.neighbors import KNeighborsClassifier
np.random.seed(0)
res+='"knn":'+process(lambda :KNeighborsClassifier(n_neighbors=10),datatables,1)
    
from sklearn.tree import DecisionTreeClassifier
np.random.seed(0)
res+='"gini":'+process(lambda :DecisionTreeClassifier(criterion='gini'),datatables,1)
res+='"entropy":'+process(lambda :DecisionTreeClassifier(criterion='entropy'),datatables,1)

from sklearn.naive_bayes import GaussianNB
np.random.seed(0)
res+='"GaussianNB":'+process(lambda :GaussianNB(),datatables,1)

res=res[0:-1]+'}'
with open('结果/sklearn结果合集/sklearn结果合集.json','w') as f:
    f.writelines(res)
