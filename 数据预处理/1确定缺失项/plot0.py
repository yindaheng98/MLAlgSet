import pandas as pd
diabetes=pd.read_csv('diabetes.txt')
diabetes['Outcome'].loc[diabetes['Outcome']==1]='Diabete'
diabetes['Outcome'].loc[diabetes['Outcome']==0]='no Diabete'
diabetes.rename(columns={'DiabetesPedigreeFunction':"Diabetes\nPedigree\nFunction",
                         'BloodPressure':"Blood\nPressure",
                         'SkinThickness':"Skin\nThickness"},inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams['axes.labelsize']=23
rcParams['legend.fontsize']=23
rcParams['legend.shadow']=True
rcParams['legend.markerscale']=2.0
sns.pairplot(diabetes,hue='Outcome',markers="+",
             plot_kws=dict(s=10,linewidth=0.5))
plt.savefig('1.png')
