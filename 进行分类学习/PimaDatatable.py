from PimaDatafiles import test_file,datafiles,datafiles_drop
import pandas as pd

def parseTable(file):
    data=pd.read_csv(file,engine='python')
    table={}
    table['X']=data.drop(columns=['Outcome'])
    table['y']=data['Outcome']
    return table

datatables={}

for k in datafiles:
    datatable={}
    datatable['train']=parseTable(datafiles[k])
    datatable['test']=parseTable(test_file)
    datatables[k]=datatable

for k in datafiles_drop:
    datatable={}
    datatable['train']=parseTable(datafiles_drop[k][0])
    datatable['test']=parseTable(datafiles_drop[k][1])
    datatables[k]=datatable

#引用了PimaDatafiles中的文件路径，将预处理数据集转化为sklearn风格
#供knn、决策树、集成学习使用
