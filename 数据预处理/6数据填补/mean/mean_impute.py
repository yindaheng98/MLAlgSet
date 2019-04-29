import pandas as pd
diabetes=pd.read_csv('../../4+5数据集的删除和划分和规范化/train_norm_impute_no.csv',engine='python')
import impyute as impy
imputed=impy.imputation.cs.mean(diabetes)

imputed.columns=diabetes.columns
imputed['Outcome']=diabetes['Outcome']
imputed.to_csv('train_impute_mean.csv',index=False)
