import pandas as pd
import numpy as np
import random

fp = open("train_data.csv")
fn = open("test_data.csv")
fp_label = open("train_label.csv")
fn_label = open("test_label.csv")

# 儲存label結果
fp_outcome = "".join(fp_label.readlines()).split("\n")
del fp_outcome [-1]
fn_outcome = "".join(fn_label.readlines()).split("\n")
del fn_outcome [-1]

# 讀training data and test data
# test dataset
all_test = []
for line in fn:
    a = line.replace(","," ").replace("\n","").split(" ")
    a2 = list(map(lambda x:np.nan if x == '' else float(x), a)) # 填補缺失值
    all_test.append(a2)

# train dataset
all_train = []
for line in fp:
    a = line.replace(","," ").replace("\n","").split(" ")
    a2 = list(map(lambda x:np.nan if x == '' else float(x), a)) # 填補缺失值
    all_train.append(a2)

## 標準化
def Zscore(all_data):
    avg_test = []
    # get every column
    column = zip(*all_data)
    column_list = list(map(list,column)) # 原始資料
    for i in range(len(column_list)):
        # calculate mean(), with ignoring np.nan
        avg_test.append(np.nanmean(column_list[i])) 
        # 將缺失值替換為指定值
        arr_filled = np.nan_to_num(column_list[i], nan=avg_test[i])
        column_list[i] = list(arr_filled) # 存進column (處理資料完畢)

    # 找每個特徵之標準差
    std_test = []
    for i in range(len(column_list)):
        std_test.append(np.std(column_list[i]))

    # 標準化資料(Z-Score)
    zscore_column = []
    for i in range(len(column_list)):
        new_column = []
        for elements in column_list[i]:
                if std_test[i] == 0:
                    new_column.append(elements)
                else:
                    new_column.append((elements-avg_test[i])/std_test[i])
        zscore_column.append(new_column)

    return zscore_column

std_test_column = Zscore(all_test)
std_train_column = Zscore(all_train)

# get every row (標準化完的資料)
row_test = zip(*std_test_column)
row__test_list = list(map(list,row_test)) # 原始資料
row_train = zip(*std_train_column)
row__train_list = list(map(list,row_train)) # 原始資料

## pick Outiler

label_1 = [row__train_list[i] for i in range(len(row__train_list)) if fp_outcome[i] == '1']
from sklearn.ensemble import IsolationForest
# Create an Isolation Forest instance
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
# Fit the model to your data
isolation_forest.fit(label_1)
# Predict outliers
outlier_predictions = isolation_forest.predict(label_1)
# Get indices of outliers
outlier_indices = np.where(outlier_predictions == -1)[0]
# Delete the outlier data points from the 'data' array
data_without_outliers = np.delete(label_1, outlier_indices, axis=0).tolist()
# 以 標籤,一列資料 儲存 剩餘label
label_other = [(int(fp_outcome[i]),row__train_list[i]) for i in range(len(row__train_list)) if fp_outcome[i] != '1']

# 結合所有資料
no_outlier = []
for i in range(len(data_without_outliers)):
    no_outlier.append( (1,data_without_outliers[i]) )
no_outlier += label_other
no_outlier = sorted(no_outlier)

## Oversampling 

# 資料前處理 :ADASYN
def ADASYN(Max,Min,l1,l2,label):
    # 生成G
    beta = 0.1
    G = (Max-Min)*beta

    def ADASYN_neighbor(l1,l2): # 找鄰居算比例
        Ratio = [] # ri
        for i in l2: # 找鄰居
            ratio = []
            # 以 (label, 距離) 的形式存起來
            for j in l1:
                ratio.append((1,np.linalg.norm(np.array(i) - np.array(j))))
            for j in l2:
                ratio.append((2,np.linalg.norm(np.array(i) - np.array(j))))
            ratio = sorted(ratio,key=lambda x:x[1])  
            
            neighbor = ratio[1:6] # 找五個最近鄰居 (不取0 因為那個是自己)
            
            # r = majority/num of neighbor
            for i in range(5):
                count1 =0 ; count2 =0
                if neighbor[i][0] ==1:
                    count1 +=1
                else:
                    count2 +=1
            
            r = count1/5
            Ratio.append(r)

        Sum = sum(Ratio)
        Gi = list(map(lambda x: int(G*(x/Sum)), Ratio)) # 生成Gi
        return Gi # 回傳Gi

    gi_list = ADASYN_neighbor(l1,l2)
    
    si = []
    def smote(l2,gi):
        for i in l2: # 選中的點
            minority_neighbor = []
            for j in l2: # 找鄰居
                minority_neighbor.append((l2.index(j),np.linalg.norm(np.array(i) - np.array(j))))
            minority_neighbor = sorted(minority_neighbor, key=lambda x:x[1])
            if label == 7:
                minority_neighbor = minority_neighbor[1:3]
            else:
                minority_neighbor = minority_neighbor[1:4]
            
            if label == 7: # 隨機選一個鄰居
                rand =random.randrange(2)
            else:
                rand =random.randrange(3) 
            
            xi = minority_neighbor[rand][0] # 存鄰居的編號
            
            for w in range(gi[l2.index(i)]): # 生成新的點
                lamda = random.uniform(0.0, 1.25) # 從[0, 1.5] 隨機選一個float
                si.append((label,list((i + (np.array(l2[xi]) -np.array (i))* lamda)))) 
        return si # 總生成點

    return smote(l2,gi_list)

label_time = {i:fp_outcome.count(str(i)) for i in range(1,9)} # 標籤:相對個數
label_time[1] -= len(outlier_indices) # 扣掉outiler
# label = 1
l1 = [no_outlier[i][1] for i in range(len(no_outlier)) if no_outlier[i][0] == 1] 
sort_lable_time = dict(sorted(label_time.items(), key=lambda x:x[1],reverse=True))

# 將Oversampling 生成的資料 合併到 train_data
majority = list(sort_lable_time.values())[0]
for key,value in sort_lable_time.items():
    if key == 1 or key ==8:
        continue
    l2 = [no_outlier[i][1] for i in range(len(no_outlier)) if no_outlier[i][0] == key]
    temp_row = ADASYN(majority,value,l1,l2,key)
    no_outlier += temp_row

l8 = [no_outlier[i][1] for i in range(len(no_outlier)) if no_outlier[i][0] == 8]
for i in range(15):
    no_outlier.append((8,l8[0]))
