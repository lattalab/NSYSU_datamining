import pandas as pd
import numpy as np

# 用套件的KNN

#創造 KNN 的訓練的模型
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import process

Train_new = pd.DataFrame({
    i:process.no_outlier[i][1] for i in range(len(process.no_outlier))
}).T
Train_new_label = pd.Series(
    [process.no_outlier[i][0] for i in range(len(process.no_outlier))])


# 分割訓練集 跟 測試集
# X: 特徵 y: 結果 
X_train, X_test, y_train, y_test = train_test_split(Train_new, Train_new_label, test_size=0.3, random_state=42)
clf = neighbors.KNeighborsClassifier(n_neighbors=14) # 套模型
clf.fit(X_train, y_train)
pred = clf.predict_proba(process.row__test_list) # 得到八個分類的預測機率
predict = clf.predict(process.row__test_list) # 套件預測的label

unknown = []
predict_label = []

for i in range(len(pred)):
    Max = np.max(pred[i])
    if Max < 0.7:
        unknown.append(i)
        predict_label.append(0) # 0 represent unknown class
    else:
        predict_label.append(predict[i])

print("Unknown: \n",unknown)
print("\nNow Knn predict: \n",predict_label)

count = 0
for i in range(len(predict_label)):
    if (int(process.fn_outcome[i]) == predict_label[i]) or (int(process.fn_outcome[i])>=9 and predict_label[i] ==0):
        count +=1
    
print("\nAccuracy of knn: ", count/len(predict_label))
print()