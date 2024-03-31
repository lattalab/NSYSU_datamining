import pandas as pd
import numpy as np
import random
from process import *
import scatter

# 手寫KNN

# test data中選一筆資料跟train data的每一筆做歐幾里得距離，再找鄰近的k個鄰居
# 歐幾里得距離
unknown_class = []
predict_label = [0 for i in range(len(row__test_list))] # 初始化預測結果

def distance():
    dis = dict() # create a empty dictionary
    for i in range(len(row__test_list)):
        for j in range(len(no_outlier)):
            dis[ (j) ] = no_outlier[j][0],(np.linalg.norm(np.array(row__test_list[i]) - np.array(no_outlier[j][1])))

        dis1 = sorted(dis.items(), key = lambda x:x[1][1]) #排序
        d1 = dis1[0:8] #取前五個 ， k= 5
        # 分類
        cal(d1,i)

        
def cal(x,idx): #看k=5的鄰居，是哪類(做分類)
    labels = {int(i+1):0 for i in range(9)}
    for j in x:
        labels[ j[1][0] ] +=1 # 知道label分類結果
    labels = sorted(labels.items(), key = lambda x:x[1],reverse=True) #排序
    # print(labels)
    if labels[0][1] < 5:
        unknown_class.append(idx)
    else:
        predict_label[idx] = labels[0][0]

    
print(unknown_class)
print(predict_label)
    
count = 0
for i in range(len(predict_label)):
    if (int(fn_outcome[i]) == predict_label[i]) or (int(fn_outcome[i])>=9 and predict_label[i] ==0):
        count +=1
    
print("accuracy of knn: ", count/len(predict_label))