import pandas as pd
import numpy as np
import process
import random
import KNN_sklearn

# 1.initial

# kmeans ++ 找群心
random.shuffle(KNN_sklearn.unknown)
center = [] # 重心
center.append(KNN_sklearn.unknown[0])

# 主要做Kmeans++
def prob(center_idx): 
    distance = [] # 平方歐幾里得
    for i in range(len(KNN_sklearn.unknown)):
        distance.append( (KNN_sklearn.unknown[i],
            np.linalg.norm(
            np.array(process.row__test_list[center[center_idx]]) - process.row__test_list[i])**2)
            )
    
    sum = 0
    for i in distance:
        sum += i[1]

    probability = [] # 算每個點的機率
    for i in distance:
        probability.append((i[0],i[1]/sum))

    sum_prob = [0] # 累積機率
    sum =0
    for i in probability:
        sum += i[1]
        sum_prob.append(sum)

    count = 0 
    while count<4: # 找出 4個新群心
        rand = random.uniform(0.0,1.0) # 隨機選擇一個值
        for i in range(len(sum_prob)-1):
            if sum_prob[i] < rand and  sum_prob[i+1] > rand: # 找出相應範圍
                if not(probability[i][0] in center):
                    center.append(probability[i][0]) # 放入相應的新群心
                    count += 1
                    break

prob(0)

print("Initial center: \n",center)


label = [[],[],[],[],[]]
dis = dict() # create a empty dictionary
for k in KNN_sklearn.unknown:
    for j in center:
        dis[ (j) ] = np.linalg.norm(np.array(process.row__test_list[k]) - np.array(process.row__test_list[j]))
    Min = min(dis , key=dis.get) # 得到最小值value的key
    label[ center.index(Min) ].append(k) # 放入最近的群心 (第一次分群)
center = [process.row__test_list[i] for i in center] # center轉為數據
initial_center = center.copy()

def gravity_center(x): # 設定新群心
    sum = np.zeros(len(process.row__test_list[0]))
    for i in x:
        sum += np.array(process.row__test_list[i])
    return list(sum/len(x))

def Distance(unknown,center): # 算data跟重心的距離
    dis = dict() # create a empty dictionary
    label = [[],[],[],[],[]] # record 分群
    for k in unknown:
        count =0
        for j in center:
            dis[ count ] = np.linalg.norm(np.array(process.row__test_list[k]) - np.array(j))
            count +=1
        Min = min(dis , key=dis.get) # 得到最小值value的key
        label[ Min ].append(k)
    return label

# 2. loop
while(True): 
    old_center = center
    c = [gravity_center(label[i]) for i in range(5)] # 更新重心
    label = Distance(KNN_sklearn.unknown, c) # 更新分群
    if old_center == c: # 終止條件，群不動
        break
    center = c

# 紀錄 未知類別:個數
number = {i:process.fn_outcome.count(str(i)) for i in range(9,13+1)}
number = sorted(number.items(), key = lambda x:x[1]) #排序

# 自己分群的資料
length = {i:len(label[i]) for i in range(5)}
length = sorted(length.items(), key = lambda x:x[1]) # 排序
# key:哪個分類(9~13)， value: unknown test data的編號
assign_label = {number[i][0]:label[length[i][0]] for i in range(5)} 

# 更改原predict label 的資料
for key,value in assign_label.items():
    for i in value:
        KNN_sklearn.predict_label[i] = key

kmeans_predict_label = KNN_sklearn.predict_label
print("After kmeans we predict the label:\n", kmeans_predict_label)
print()

# 顯示混淆矩陣跟後來分群後的結果(只看那五群)
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

    plt.subplot(1,2,2)
    thepicture = TSNE(n_components=2, random_state=0)# 降維
    # Test_all 收集所有的 unknown data
    Test_all = pd.DataFrame({
        i:process.row__test_list[i] for i in KNN_sklearn.unknown
        }).T

    # 設定標籤 對應 顏色
    unknown_label = pd.Series([
            kmeans_predict_label[i] for i in KNN_sklearn.unknown
        ]).T
    colors = {
          9:'violet' ,10:'green' ,11:'orange',
          12:'brown', 13:'blue'
          }
    l = unknown_label.map(colors)
    X_tsne = thepicture.fit_transform(Test_all)
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l)
    plt.title('Cluster',fontsize=18)

    # 印出圖
    plt.show()

show_confusion_and_Cluster()
