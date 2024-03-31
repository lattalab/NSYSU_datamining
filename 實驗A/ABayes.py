#!/usr/bin/env python
# coding: utf-8

# In[5]:


#實驗A
from sklearn.naive_bayes import GaussianNB          # 高斯貝氏分類器 GaussianNB
from sklearn import datasets    
from sklearn.model_selection import train_test_split  #訓練模型
from sklearn.metrics import accuracy_score, recall_score  #算準確率跟召回率

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

fp = pd.read_csv("train_data.csv")
fn = pd.read_csv("test_data.csv")

#Set up training data
#means it will select all rows,    “: -1 ” means that it will ignore last column
#as feature
X = fp.iloc[:,:-1]
# ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one
#as classification outcome
y = fp.iloc [:, -1]

# 拆分成訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model=GaussianNB()                         # 這裡使用高斯貝氏分類器
model.fit(X_train,y_train)

#Set up test data
X_t = fn.iloc[:,:-1]
Y_t = fn.iloc[:,-1]
predictions = model.predict(X_t) #把test丟進訓練模型預測

print("在實驗A中:")
# 計算準確率
accuracy = accuracy_score(Y_t, predictions)
print("Accuracy:", accuracy)
# 計算召回率
recall = recall_score(Y_t, predictions)
print("Recall:", recall)


# In[ ]:




