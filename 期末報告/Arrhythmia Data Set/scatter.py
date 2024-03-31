import process
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # 降維用的
# 利用不同資料處理方法，並利用將資料降維，以散步圖顯示處理效果

# 創建一個4x5 = 20的子圖佈局
fig, axes = plt.subplots(4, 5, figsize=(12, 12))
plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.9, hspace=0.7, wspace=0.8)
plt.suptitle('Deal with Train data', fontsize=16)
thepicture = TSNE(n_components=2, random_state=0)# 降維

# 1. original train data
# 設立原始資料的dataframe
Train_all = pd.DataFrame({
    i:process.row__train_list[i] for i in range(len(process.row__train_list))
}).T
X_tsne = thepicture.fit_transform(Train_all)


# 可視化降維後的數據
colors = {1:'red', 2:'green', 3:'blue', 4:'orange', 
          5:'purple', 6:'yellow', 7:'cyan', 8:'magenta'}
l = [colors[int(process.fp_outcome[i])] for i in range(len(process.row__train_list))]

plt.subplot(4,5,1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l)
plt.title('No adjust',fontsize=12)

# 1-2 刪除label 1 outlier後的train data
oldTrain_new = pd.DataFrame({
    i:process.oldtrain_no_outlier[i][1] for i in range(len(process.oldtrain_no_outlier))
}).T
oldTrain_new_label = pd.Series([process.oldtrain_no_outlier[i][0] for i in range(len(process.oldtrain_no_outlier))])
X_tsne = thepicture.fit_transform(oldTrain_new)
l1_2 = oldTrain_new_label.map(colors)
plt.subplot(4,5,2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l1_2)
plt.title('After delete outlier',fontsize=12)

# 2. 自己做完 ADASYN的資料
# 將train data 跟 train label 轉成 dataframe
Train_new = pd.DataFrame({
    i:process.no_outlier[i][1] for i in range(len(process.no_outlier))
}).T
Train_new_label = pd.Series([process.no_outlier[i][0] for i in range(len(process.no_outlier))])

X_tsne = thepicture.fit_transform(Train_new)
l2 = [colors[process.no_outlier[i][0]] for i in range(len(process.no_outlier))]
plt.subplot(4,5,3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l2)
plt.title('After own ADASYN(no outiler)',fontsize=12)

# 3. try to deal with outiler
# 刪除標準化後 >3 或 <-3 的資料 的那行
# Train_all['Label'] = list(process.fp_outcome) # 放入標籤資料
# cond = (Train_all['Label'] == 1)
# Train_1 = Train_all[cond]
# cond2 = (Train_1 >3) | (Train_1 <-3) 
# cond3 = Train_1[cond2].any(axis =1)
# Train_1 = Train_1.drop(Train_1[cond3].index)

# # 與其它標籤結合
# Train_other = Train_all[~cond]
# result = pd.concat([Train_1, Train_other])

# # # 以下其實是印 挑除outiler的資料(續寫3)
# # l3 = result['Label'].map(colors)
# # result.reset_index(drop=True,inplace=True)
# # result = result.drop(columns='Label')
# # X_tsne = thepicture.fit_transform(result)

# # plt.subplot(3,3,4)
# # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l3)
# # plt.title('After picking outlier for label_1',fontsize=12)

# 4. 挑除outiler的資料後 用BorderlineSMOTE
from imblearn.over_sampling import BorderlineSMOTE
borderline = BorderlineSMOTE(random_state=42)
X_resampled, y_resampled = borderline.fit_resample(Train_new, Train_new_label)

l4 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,4)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l4)
plt.title('After BorderlineSMOTE(no outiler)',fontsize=12)

# 5. 挑除outiler的資料後 SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(Train_new, Train_new_label)
l5 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,5)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l5)
plt.title('After SMOTE(no outiler)',fontsize=12)

# # 6. 挑除outiler的資料後 ADASYN
# 有BUG:
# RuntimeError: Not any neigbours belong to the majority class. 
# This case will induce a NaN case with a division by zero. 
# ADASYN is not suited for this specific dataset. Use SMOTE instead.

# from imblearn.over_sampling import ADASYN
# adasyn = ADASYN(random_state=42,n_neighbors=13)
# X_resampled, y_resampled = adasyn.fit_resample(Train_new, Train_new_label)
# l6 = y_resampled.map(colors)
# X_tsne = thepicture.fit_transform(X_resampled)
# plt.subplot(3,3,4)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l6)
# plt.title('Train data after ADASYN(no outlier)')

# 7. 挑除outiler的資料後 SVMSMOTE
from imblearn.over_sampling import SVMSMOTE
svm = SVMSMOTE(random_state=42)
X_resampled, y_resampled = svm.fit_resample(Train_new, Train_new_label)
l7 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,6)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l7)
plt.title('After SVMSMOTE(no outiler)',fontsize=12)

# 8. 挑除outiler的資料後 KMeansSMOTE
# 有BUG 但我不會修
# invalid value encountered in divide
# cluster_weights = cluster_sparsities / cluster_sparsities.sum()
# File "C:\Users\vlva8\AppData\Local\Programs\Python\Python311\Lib\site-packages\imblearn\over_sampling\_smote\cluster.py", line 286, in _fit_resample
# math.ceil(n_samples * cluster_weights[valid_cluster_idx])
# ValueError: cannot convert float NaN to integer

# from imblearn.over_sampling import KMeansSMOTE
# kmeansSmote = KMeansSMOTE(random_state=42)
# X_resampled, y_resampled = kmeansSmote.fit_resample(Train_new, Train_new_label)
# l8 = y_resampled.map(colors)
# X_tsne = thepicture.fit_transform(X_resampled)
# plt.subplot(3,3,4)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l8)
# plt.title('Train data after KMeansSMOTE(no outlier)')

# 9. 挑除outiler的資料後 SMOTEN
from imblearn.over_sampling import SMOTEN
smoten = SMOTEN(random_state=42)
X_resampled, y_resampled = smoten.fit_resample(Train_new, Train_new_label)
l9 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,7)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l9)
plt.title('After SMOTEN(no outiler)',fontsize=12)

# 10. 挑除outiler的資料後 RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
rand = RandomOverSampler(random_state=42)
X_resampled, y_resampled = rand.fit_resample(Train_new, Train_new_label)
l10 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,8)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l10)
plt.title('After RandomOverSampler(no outiler)',fontsize=12)

# Mix sampling (= Oversampling + Undersampling)

# 11. 挑除outiler的資料後 SMOTEENN
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(Train_new, Train_new_label)
l11 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,9)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l11)
plt.title('After SMOTEENN(no outiler)',fontsize=12)

# 12. 挑除outiler的資料後 SMOTETomek
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(Train_new, Train_new_label)
l12 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,10)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l12)
plt.title('After SMOTETomek(no outiler)',fontsize=12)

# Try ADASYN + 不同的Undersampling方法

# 13. ClusterCentroids
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids()
X_resampled, y_resampled = cc.fit_resample(Train_new, Train_new_label)
l13 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,15)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l13)
plt.title('+ ClusterCentroids',fontsize=12)

# 14. CondensedNearestNeighbour
from imblearn.under_sampling import CondensedNearestNeighbour
cnn = CondensedNearestNeighbour()
X_resampled, y_resampled = cnn.fit_resample(Train_new, Train_new_label)
l14 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,11)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l14)
plt.title('+ CondensedNearestNeighbour',fontsize=12)

# 15. EditedNearestNeighbours
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(Train_new, Train_new_label)
l15 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,12)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l15)
plt.title('+ EditedNearestNeighbours',fontsize=12)

# 16. RepeatedEditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
renn = RepeatedEditedNearestNeighbours()
X_resampled, y_resampled = renn.fit_resample(Train_new, Train_new_label)
l16 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,13)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l16)
plt.title('+ RepeatedENN',fontsize=12)

# 17. RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
allknn = AllKNN()
X_resampled, y_resampled = allknn.fit_resample(Train_new, Train_new_label)
l17 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,14)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l17)
plt.title('+ AllKNN',fontsize=12)

# # 18. InstanceHardnessThreshold
# from imblearn.under_sampling import InstanceHardnessThreshold
# iht = InstanceHardnessThreshold()
# X_resampled, y_resampled = iht.fit_resample(Train_new, Train_new_label)
# l18 = pd.Series(y_resampled).map(colors)
# X_tsne = thepicture.fit_transform(X_resampled)
# plt.subplot(4,5,15)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l18)
# plt.title('ADASYN + InstanceHardnessThreshold(no outiler)',fontsize=12)

# 19. NearMiss
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X_resampled, y_resampled = nm.fit_resample(Train_new, Train_new_label)
l19 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,16)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l19)
plt.title('+ NearMiss',fontsize=12)

# 20. NeighbourhoodCleaningRule
from imblearn.under_sampling import NeighbourhoodCleaningRule
nm = NeighbourhoodCleaningRule()
X_resampled, y_resampled = nm.fit_resample(Train_new, Train_new_label)
l19 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,17)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l19)
plt.title('+ NeighbourhoodCleaningRule',fontsize=12)

# 21. OneSidedSelection
from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection()
X_resampled, y_resampled = oss.fit_resample(Train_new, Train_new_label)
l20 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,18)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l20)
plt.title('+ OneSidedSelection',fontsize=12)

# 22. RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
rs = RandomUnderSampler()
X_resampled, y_resampled = rs.fit_resample(Train_new, Train_new_label)
l21 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,19)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l21)
plt.title('+ RandomUnderSampler',fontsize=12)

# 23. TomekLinks
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(Train_new, Train_new_label)
l22 = pd.Series(y_resampled).map(colors)
X_tsne = thepicture.fit_transform(X_resampled)
plt.subplot(4,5,20)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=l22)
plt.title('+ TomekLinks',fontsize=12)

plt.show() # 印出圖