function process_knn_impute(filename,k)
diabetes=importdata(filename);
rep=diabetes.data==0;
rep(:,[1,9])=0;
diabetes.data(rep)=NaN;

diabetes.data=knnimpute(diabetes.data,k);

diabetes=array2table(diabetes.data,'VariableNames',diabetes.textdata);
writetable(diabetes,sprintf('diabetes_knn_%d.csv',k));
end