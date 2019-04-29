test_file='../数据预处理/4+5数据集的删除和划分和规范化/test_norm_impute.csv'

data_folder=u'../数据预处理/6数据填补/'
datafiles={
    'em':data_folder+'em/train_impute_em.csv',
    'mean':data_folder+'mean/train_impute_mean.csv',
    'mice':data_folder+'mice/train_impute_mice.csv'}

for i in range(1,11):
    datafiles['knn%d'%i]=data_folder+'knn/diabetes_knn_%d.csv'%i
for i in range(1,6):
    datafiles['pmm%d'%i]=data_folder+'pmm/diabetes_pmm%d.csv'%i

data_folder_drop=u'../数据预处理/4+5数据集的删除和划分和规范化/'
datafiles_drop={
    'drop':[data_folder_drop+'train_norm.csv',
            data_folder_drop+'test_norm.csv'],
    'dropI':[data_folder_drop+'train_normI.csv',
            data_folder_drop+'test_normI.csv'],
    'dropS':[data_folder_drop+'train_normS.csv',
            data_folder_drop+'test_normS.csv'],
    'dropIS':[data_folder_drop+'train_normIS.csv',
            data_folder_drop+'test_normIS.csv']}

#这个文件是所有数据预处理结果的相对路径
#其他所有和读取数据有关的操作都引用了这个文件
#如果要直接使用此文件中的路径进行读取，调用此文件的.py必须和此文件在同一目录
