
import pandas as pd 
import numpy as np
from operator import add
from functools import reduce
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import xgboost
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import mean_squared_error
import shap


data = pd.read_excel('41467_2019_12281_MOESM3_ESM.xlsx', header = 1)

data = data.dropna(subset=['Wt_Efficiency', 'eSpCas 9_Efficiency', 'SpCas9-HF1_Efficiency'])

wt_target = data['Wt_Efficiency']
wt_target = wt_target.values
esp_target = data['eSpCas 9_Efficiency']
esp_target = esp_target.values
sp_target = data['SpCas9-HF1_Efficiency']
sp_target = sp_target.values

train = data['21mer']


train = train.values
for i in range(len(train)):
    train[i] = train[i] +'GG'




ntmap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
         }
def get_seqcode(seq):
    
    return np.array(reduce(add, map(lambda c: ntmap[c], seq.upper()))).reshape(
        (1, len(seq), -1))
seq = []
for i in range(len(train)):
    seq.append(get_seqcode(train[i]))

def change(seq):
    seq = list(seq)
    
    for i in range(23):
        if seq[i] == "A":
            seq[i] = 0
        elif seq[i] == 'T':
            seq[i] = 1
        elif seq[i] == 'G':
            seq[i] = 2
            
        else:
            seq[i] = 3
    return seq
seq_num = []
for i in range(len(train)):
    seq_num.append(change(train[i]))


seq = np.array(seq)

seq = seq.reshape(seq.shape[0],23*4)
seq_num = np.array(seq_num)
wt_target = np.array(wt_target)
esp_target = np.array(esp_target)
sp_target =  np.array(sp_target)

target = np.where(wt_target>0.66,2,wt_target)
target = np.where((0.33<target) &(target <=0.66),1,target)
target = np.where(target<0.33,0,target)
target = pd.DataFrame(target)

import pickle
with open('esp_seq_data_array.pkl', 'rb') as f:
    data = pickle.load(f)
    


train = data[0]
train_biofeat = data[1]

target = data[2]


# target2 = np.where(target2>0.66,2,target2)
# target2 = np.where((0.33<target2) &(target2 <=0.66),1,target2)
# target2 = np.where(target2<0.33,0,target2)
# target2 = pd.DataFrame(target2)





# reducer = umap.UMAP()

# pca = PCA(n_components=2)

# embedding = reducer.fit_transform(seq_num)
# umapdf = pd.DataFrame(data = embedding
#              , columns = ['umap_component_1', 'umap_component_2'])


# principalComponents = pca.fit_transform(train_biofeat)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal_component_1', 'principal_component_2'])
# finalDf = pd.concat([principalDf, target2],axis =1)
# group = finalDf.groupby(0)
# finalumapDf = pd.concat([umapdf, target], axis = 1)
# group3 = finalumapDf.groupby(0)
# plt.subplot(2,1,1)




# for name, group in group:
#     plt.title('PCA')
#     plt.plot(group.principal_component_1, group.principal_component_2,   marker='o',  linestyle='',)

# plt.subplot(2,1,2)

# for name, group in group3:
#     plt.title('UMAP')
#     plt.plot(group.umap_component_1, group.umap_component_2,   marker='o',  linestyle='',)

# plt.show()
X_train, X_test, y_train, y_test = train_test_split(train_biofeat,target,test_size=0.2, random_state = 11)

X2_train, X2_test, y2_train, y2_test = train_test_split(train,target,test_size=0.2, random_state = 11)

Linear_regression2 = LinearRegression()
Linear_regression2.fit(X_train, y_train)
bio = Linear_regression2.predict(X_train)


Linear_regression3 = LinearRegression()
Linear_regression3.fit(X2_train, y2_train)
seq1 = Linear_regression3.predict(X2_train)
mse1 = mean_squared_error(y_train, bio)
mse2 = mean_squared_error(y2_train, seq1)
breakpoint()

for i in range(11):
        
    fet1 = X_train[:,i].reshape(-1,1)
    reg = LinearRegression().fit(fet1, y_train)

    
    plt.plot(fet1, y_train,'o')
    predict = reg.predict(fet1)
    plt.plot(fet1,predict)
    plt.show()
coef = reg.coef_()
predict = reg.predict(X_train)

breakpoint()
print("encode :", reg.score(X_test,y_test))

ridge = Ridge(alpha =0.5)
ridge_2 = Ridge(alpha =0.5)
reg_2 = ridge.fit(X_train, y_train)

print(reg_2.score(X_test,y_test))







xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb_model_2 = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

xgb_model_3 = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb= xgb_model.fit(X_train,y_train)
# xgbn = xgb_model_2.fit(X2_train,y2_train)
# xgbb = xgb_model_3.fit(X2_train,y2_train)

xgboost.plot_importance(xgb_model)
print(xgb.score(X_test, y_test))
# print(xgbb.score(X2_test, y2_test))



# with open("seqencocde.csv", 'w') as file:
#   writer = csv.writer(file)
#   writer.writerow(seq)

# train_seq = pd.read_csv('seqencode.csv')


# X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=test_size, random_state=random_state)
breakpoint()