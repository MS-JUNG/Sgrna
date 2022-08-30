import pickle
from tkinter import RIDGE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import xgboost
import matplotlib.pyplot as plt
with open('esp_seq_data_array.pkl', 'rb') as f:
    data = pickle.load(f)
    breakpoint()



train = data[0]
train_biofeat = data[1]
target = data[2]
def load_data(X,X_biofeat,y, test_size = 0.15,random_state=40):
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=test_size, random_state=random_state)

    X_train_biofeat, X_test_biofeat, y_train, y_test = train_test_split(
       X_biofeat, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, X_train_biofeat, X_test_biofeat, y_train, y_test


X_train, X_test, X_train_biofeat, X_test_biofeat, y_train, y_test = load_data(train, train_biofeat, target,random_state=33) 
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train,y_train))
ridge = Ridge(alpha =0.5)
reg_2 = ridge.fit(X_train, y_train)
print(reg_2.score(X_train,y_train))
xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
import shap
xgb_model.fit(X_train_biofeat,y_train)
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train_biofeat)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
xgboost.plot_importance(xgb_model)
print(xgb_model.score(X_train_biofeat, y_train))
breakpoint()
