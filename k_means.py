from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#データをロード
iris_data = datasets.load_iris()

features=iris_data.data.tolist()

targets=iris_data.target.tolist()


def k_means(k,X,y,maxiter=300):
    #上限値の設定
    for count in range(maxiter):
        old_center=X[:k]

        new_center=[]

        columns=[]
        
        #初期条件の決定
        for i in range(k):
            columns.append([])
    
    
        for i in X:
            
            distance=np.sum((np.array(old_center) -np.array(i))**2,axis=1)
            
            columns[np.argsort(distance)[0]].append(i)
         
        #重心の計算
        for column in columns:
            new_center.append(np.sum(np.array(column),axis=0)/len(column))
        
        if new_center is old_center:
            break
        else:
            old_center=new_center
    return columns


if __name__==__main__:
    
    
    k_means(3,features,300)