import pandas as pd
import numpy as np
import process
import random
import KNN_sklearn
from sklearn.manifold import TSNE # 降維用的

thepicture = TSNE(n_components=2, random_state=0)# 降維
Unknown = pd.DataFrame({
    i:process.row__test_list[i] for i in KNN_sklearn.unknown
}).T
Unknown_index = {
  i:Unknown.loc[i].tolist()  for i in KNN_sklearn.unknown
} # 將每列轉成資料 (unknown index:那行資料(特徵279))
X_tsne = thepicture.fit_transform(Unknown).tolist()
idx = [random.randrange(0,len(X_tsne)) for i in range(5)] # 隨機選群心
center = [X_tsne[i] for i in idx]

# 1. init
label = [[],[],[],[],[]] # 存入分群結果
# create a empty dictionary
# 對應群心index : 距離
dis = dict() 
for k in X_tsne:
    count = 0
    for j in center:
        dis[ count ] = np.linalg.norm(np.array(k) - np.array(j))
        count +=1
    
    # 找離哪個群心最近
    temp_list = sorted((dis.items()) ,key= lambda x:x[1])
    if temp_list[0][1] != 0: # 代表不是自己
        Min = temp_list[0]
    else:
        Min = temp_list[1]

    origin_idx = KNN_sklearn.unknown[(X_tsne.index(k))]
    # 以原本的unknown index:二維資料
    label[ Min[0] ].append((origin_idx,k)) # 放入最近的群心 (第一次分群)

def gravity_center(all_cluster): # 設定新群心
    # 將列表中的陣列相加
    new_g = []
    for l in all_cluster:
        sum_array = np.array([0.0,0.0])
        for tup in l:
            sum_array += tup[1]

        new_g.append((sum_array/len(l)).tolist())
    return new_g # updata center


def Distance(unknown,center): # 算data跟重心的距離
    dis = dict() # create a empty dictionary
    label = [[],[],[],[],[]] # record 分群
    for k in unknown:
        count =0
        for j in center:
            dis[ count ] = np.linalg.norm(np.array(k) - np.array(j))
            count +=1

        # 找離哪個群心最近
        temp_list = sorted((dis.items()) ,key= lambda x:x[1])
        if temp_list[0][1] != 0: # 代表不是自己
            Min = temp_list[0]
        else:
            Min = temp_list[1]

        origin_idx = KNN_sklearn.unknown[(X_tsne.index(k))]
        # 以原本的unknown index:二維資料
        label[ Min[0] ].append((origin_idx,k)) # 放入最近的群心
    return label

# 2. loop
while(True): 
    old_center = center
    c = gravity_center(label) # 更新重心
    label = Distance(X_tsne, c) # 更新分群

    # Check if the center are equal
    is_equal = np.array_equal(old_center, c)
    if is_equal: # 終止條件，群不動
        break
    center = c

# 紀錄 未知類別:個數
number = {i:process.fn_outcome.count(str(i)) for i in range(9,13+1)}
number = sorted(number.items(), key = lambda x:x[1]) #排序

# 自己分群的資料
length = {i:len(label[i]) for i in range(5)}
length = sorted(length.items(), key = lambda x:x[1]) # 排序
# key:哪個分類(9~13)， value: unknown test data的資料(有降維)
assign_label = {number[i][0]:label[length[i][0]] for i in range(5)} 
print(number)
print(length)
print(assign_label)

# 更新label
for key,value in assign_label.items():
    for tup in value:
        KNN_sklearn.predict_label[tup[0]] = key

kmeans_predict_label = KNN_sklearn.predict_label
print("kmeans:\n", kmeans_predict_label)
print()

# 混淆矩陣
from sklearn.manifold import TSNE # 降維用的
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
def show_confusion_and_Cluster():
    true_value = [int(i) for i in process.fn_outcome]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    cm = confusion_matrix(y_true=true_value, y_pred=kmeans_predict_label, labels=labels) # 建立confusion matrix

    fig, axes = plt.subplots(1,2,figsize=(18, 16)) 
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(1,2,1)
    # 印出confusion matrix、指定藍色跟透明度
    axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.3) 

    # 設定軸上的刻度值
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=12)
    axes[0].set_yticklabels(labels, fontsize=12)

    # 在每個格子內印出數值
    for i in range(cm.shape[0]): 
        for j in range(cm.shape[1]): 
            axes[0].text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large') 

    # 將x軸刻度位置設為底部
    axes[0].xaxis.set_ticks_position('bottom')
    # 設定座標標籤、標題
    plt.xlabel('Predictions', fontsize=18) 
    plt.ylabel('Actuals', fontsize=18) 
    plt.title('Confusion Matrix', fontsize=18) 

    # 計算每個類別的真實樣本數量
    true_samples = np.sum(cm, axis=1) # 一列

    # 計算每個類別的正確預測數量
    correct_predictions = np.diag(cm) # 得到對角線

    # 計算每個類別的準確率
    accuracies = np.nan_to_num(correct_predictions / true_samples, nan=0)

    # 印出每個類別的準確率
    for i, accuracy in enumerate(accuracies):
        print(f"Class {i+1} accuracy: {accuracy}")

    print("Total accuracy: ",np.sum(correct_predictions)/len(kmeans_predict_label))

    print()

show_confusion_and_Cluster()
