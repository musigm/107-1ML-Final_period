#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
from sklearn import cross_validation, svm, preprocessing, metrics
from sklearn.cross_validation import train_test_split

# 開啟 CSV 檔案
with open('winequality-red.csv', newline='') as csvfile:
  # 讀取 CSV 檔案內容
    wines = csv.DictReader(csvfile)
    wines_x=[]
    wines_y=[]
    for wine in wines:
        wines_x.append([wine['fixed acidity'], wine['volatile acidity'], wine['citric acid'],
            wine['residual sugar'], wine['free sulfur dioxide'], wine['total sulfur dioxide'], 
            wine['chlorides'], wine['density'], wine['pH'], 
            wine['sulphates'], wine['alcohol']])
        wines_y.append(wine['quality'])
        
        #wines_x.append([ wine['total sulfur dioxide'], wine['pH'], 
        #    wine['sulphates']])
        #wines_y.append(wine['quality'])
    
# 切分訓練與測試資料
train_X, test_X, train_Y, test_Y = train_test_split(wines_x, wines_y, test_size = 0.3)

#建立分類器 - SVM
svc = svm.SVC()
svc_fit = svc.fit(train_X, train_Y)

# 預測
test_Y_predicted = svc.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_Y, test_Y_predicted)
print(accuracy)


# In[ ]:




