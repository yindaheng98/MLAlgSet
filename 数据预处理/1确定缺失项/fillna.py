import pandas as pd
diabetes=pd.read_csv('diabetes.txt')
import numpy as np
diabetes['Glucose'].replace(0,np.nan,inplace=True)
diabetes['BloodPressure'].replace(0,np.nan,inplace=True)
diabetes['SkinThickness'].replace(0,np.nan,inplace=True)
diabetes['Insulin'].replace(0,np.nan,inplace=True)
diabetes['BMI'].replace(0,np.nan,inplace=True)
diabetes.to_csv('diabetes.csv',index=False)
