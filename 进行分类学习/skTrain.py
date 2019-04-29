def sklearn_process(model,data):
    model.fit(data['train']['X'],data['train']['y'])
    prediction=model.predict(data['test']['X'])
    correction=data['test']['y']
    AC=(correction==prediction).sum()/len(correction)
    TP=(prediction[correction==prediction]==1).sum()
    FN=(prediction[correction!=prediction]==1).sum()
    FP=(prediction[correction!=prediction]==0).sum()
    TN=(prediction[correction==prediction]==0).sum()
    return '{"AC":%f,"TP":%d,"FN":%d,"FP":%d,"TN":%d},'%(AC,TP,FN,FP,TN)

def sklearn_processes(Model,datas,n=1):
    r='{'
    for ke in datas:
        r+='"%s":['%ke if n>1 else '"%s":'%ke
        data=datas[ke]
        for _ in range(n):
            model=Model()
            r+=sklearn_process(model,data)
        r=r[0:-1]+('],' if n>1 else ',')
    r=r[0:-1]+'},'
    return r
