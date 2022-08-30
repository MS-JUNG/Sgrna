
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
t =  df.loc[:, features].values
df_t = pd.DataFrame(t)

x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
u = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
t = StandardScaler().fit_transform(t)
u = StandardScaler().fit_transform(u)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2'])
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal_component_1'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

reducer = umap.UMAP()

tsne = TSNE(n_components=2).fit_transform(df_t)
embedding = reducer.fit_transform(u)
umapdf = pd.DataFrame(data = embedding
             , columns = ['umap_component_1', 'umap_component_2'])

tsnedf = pd.DataFrame(data = tsne
             , columns = ['tsne_component_1', 'tsne_component_2'])

finaltsDf = pd.concat([tsnedf, df[['target']]], axis = 1)
finalumapDf = pd.concat([umapdf, df[['target']]], axis = 1)
breakpoint()
group = finalDf.groupby('target')
group2 = finaltsDf.groupby('target')
group3 = finalumapDf.groupby('target')
plt.subplot(2,2,1)




for name, group in group:
    plt.title('PCA')
    plt.plot(group.principal_component_1, group.principal_component_2,   marker='o',  linestyle='',)

plt.subplot(2,2,2)
for name, group in group2:
    plt.title('TSNE')
    plt.plot(group.tsne_component_1, group.tsne_component_2,   marker='o',  linestyle='',)
# for name, group in groups:
plt.subplot(2,2,3)

for name, group in group3:
    plt.title('UMAP')
    plt.plot(group.umap_component_1, group.umap_component_2,   marker='o',  linestyle='',)
#ax.plot(group.principal_component_1,   marker='o',  linestyle='',)


plt.show()



# plt.scatter(finalDf['principal component 1'],finalDf['principal component 2'])
