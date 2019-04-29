from PimaDatatable import datatables
from skTrain import sklearn_processes as process
import numpy as np
np.random.seed(0)

from sklearn.svm import SVC
tol=1e-9
svc1=SVC(kernel='linear',tol=tol,shrinking=False)
svc2=SVC(kernel='rbf',gamma='auto',tol=tol,shrinking=False)
svc3=SVC(kernel='poly',gamma='auto',tol=tol,shrinking=False)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)

from sklearn.tree import DecisionTreeClassifier
dt1=DecisionTreeClassifier(criterion='gini')
dt2=DecisionTreeClassifier(criterion='entropy')

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
from sklearn.ensemble import VotingClassifier
e=[("svc1",svc1),
   ("svc2",svc2),
   ("svc3",svc3),
   ("knn",knn),
   ("dt1",dt1),
   ("dt2",dt2),
   ("nb",nb)]
res=process(lambda :VotingClassifier(estimators=e, voting="hard"),datatables,5)
res=res[0:-1]
with open('结果/Stacking集成学习.json','w') as f:
    f.writelines(res)
