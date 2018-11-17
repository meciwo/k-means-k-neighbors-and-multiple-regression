from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#データをロード
iris_data = datasets.load_iris()

features=iris_data.data.tolist()
targets=iris_data.target.tolist()

def NearestNeibors(k,X,y):
    """k:近傍点の数
    　　　　　X：データの特徴量
         y：正解ラベル"""
    right = 0
   
    #一つ抜き法で検証
    for test_X in X:
        
        distance_list=[]
        labels_list=[]
        for train_X in X:
        
        #ユークリッド距離の二乗を求める
            distance = np.sum(((np.array(test_X)-np.array(train_X)))**2)
            distance_list.append([[distance],train_X])
        
        distance_list.sort(key=lambda x:x[0])
        
        for neibor in distance_list[:k+1]:
            label=targets[features.index(neibor[1])]
            labels_list.append(label)
        
        count = np.bincount(labels_list[1:])
        pred_y = np.argmax(count)
        
        if labels_list[0]==pred_y:
            right+=1
            
        
    accuracy = right/len(X)
        
    return accuracy

if __name=="__main__":
    accuracies=[]
    for k in range(1,31):
        accuracies.append(NearestNeibors(k,features,targets))
    print(accuracies)
    
        
