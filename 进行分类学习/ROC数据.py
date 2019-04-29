ls={}
from logreg.logreg_ROC import targets,outputs
ls['逻辑回归']={'targets':targets,'outputs':outputs}
from ann.ann_ROC import targets,outputs
ls['神经网络']={'targets':targets,'outputs':outputs}
from ann.dropout.ann_ROC import targets,outputs
ls['神经网络+dropout']={'targets':targets,'outputs':outputs}
from ann.regularization.ann_ROC import targets,outputs
ls['神经网络+L2正则化']={'targets':targets,'outputs':outputs}
with open("结果/ROC.txt","w") as f:
    f.writelines(str(ls))


