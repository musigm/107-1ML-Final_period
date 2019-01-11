#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
from sklearn.cross_validation import train_test_split
from sklearn import cluster, datasets, metrics

# 開啟 CSV 檔案
with open('winequality-red.csv', newline='') as csvfile:
  # 讀取 CSV 檔案內容
    wines = csv.DictReader(csvfile)
    wines_x=[]
    wines_y=[]
    for wine in wines:
        #wines_x.append([wine['fixed acidity'], wine['volatile acidity'], wine['citric acid'],
        #    wine['residual sugar'], wine['free sulfur dioxide'], wine['total sulfur dioxide'], 
        #    wine['chlorides'], wine['density'], wine['pH'], 
        #    wine['sulphates'], wine['alcohol']])
        #wines_y.append(wine['quality'])
        
        wines_x.append([ wine['total sulfur dioxide'], wine['pH'], 
            wine['sulphates']])
        wines_y.append(wine['quality'])

# 切分訓練與測試資料
train_X, test_X, train_Y, test_Y = train_test_split(wines_x, wines_y, test_size = 0.3)

#建立分類器 - KMeans
kmeans_fit = cluster.KMeans(n_clusters = 6).fit(wines_x)
cluster_labels = kmeans_fit.labels_

# 印出績效
silhouette_avg = metrics.silhouette_score(wines_x, cluster_labels)
print(silhouette_avg)

