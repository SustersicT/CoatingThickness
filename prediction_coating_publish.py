import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

df = pd.read_excel (r'Dataset_coating_liter_exp.xlsx', sheet_name='coating') 
dataset = df.fillna("0")
dataset = dataset.iloc[:,1:25]

features = dataset.iloc[:,1:23].to_numpy() # extract features
target = dataset.iloc[:,23:25].to_numpy() # extract target


# coding into values

# call function for coding strings into numbers
features = coding(features)

'''----------------plot output to examine outliers--------------------------'''
# descriptive statistics output - target
print ("Mean of Film thickness is: ", target.mean())
print ("Standard deviation of Film thickness is: ", target.std())
print ("Minimum of Film thickness is: ", target.min())
print ("Maximum of Film thickness is: ", target.max())

plt.plot(target,'r+')
plt.title('Film tickness (nm)')
plt.xlabel('instance number')
plt.ylabel('film thickness')
plt.ylim((0,630))


bin = [len(target[target< 100]), len(target[np.logical_and(target > 100, target < 200)]), \
       len(target[np.logical_and(target > 200, target < 300)]), \
       len(target[np.logical_and(target > 300, target < 400)]), \
       len(target[np.logical_and(target > 400, target < 500)]), \
       len(target[np.logical_and(target > 500, target < 2000)]), \
       len(target[target>2000])]

# plot category frequences
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
langs = ['x<100', '100<x<200','200<x<300','300<x<400','400<x<500','500<x<2000','>2000']
ax.bar(langs,bin)
plt.xlabel('categories')
plt.ylabel('number of instances')
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.show()
fig.savefig('Thickness_categories.tiff', dpi=300, bbox_inches="tight")

#instance 3 is outlier, remove it
target = np.delete(target, [3], 0)
features = np.delete(features, [3], 0)


'''----------------------------- ML ----------------------------------------'''
# Training Phase
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.10, random_state=3)
train_y=train_y.ravel()
test_y=test_y.ravel()

'''---------------------- Linear Regression model -------------------------'''
model_linear = LinearRegression()
model_linear.fit(train_x,train_y)
linear_train_pred = model_linear.predict(train_x)
linear_test_pred = model_linear.predict(test_x)
print("Linear regression")
print("R2 score - test: \t{:0.3f}".format(r2_score(test_y, linear_test_pred)))
print("MSE - test: \t{:0.6f}".format(mean_squared_error(test_y, linear_test_pred)))
print("MAE - test: \t{:0.6f}".format(mean_absolute_error(test_y, linear_test_pred)))

''' ----------------------------- SVM------------- -------------------------'''
model_SVR = SVR(kernel='rbf',gamma='scale')
model_SVR.fit(train_x,train_y)
SVR_train_pred = model_SVR.predict(train_x)
SVR_test_pred = model_SVR.predict(test_x)
print("Support Vector Regressor")
print("R2 score - test: \t{:0.3f}".format(r2_score(test_y, SVR_test_pred)))
print("MSE - test: \t{:0.6f}".format(mean_squared_error(test_y, SVR_test_pred)))
print("MAE - test: \t{:0.6f}".format(mean_absolute_error(test_y, SVR_test_pred)))

'''------------------------- Random Forest ------ -------------------------'''
model_forest = RandomForestRegressor(n_estimators=8,max_depth=4)
model_forest.fit(train_x,train_y)
forest_train_pred = model_forest.predict(train_x)
forest_test_pred = model_forest.predict(test_x)
print("Random Forest Regressor")
print("R2 score - test: \t{:0.3f}".format(r2_score(test_y, forest_test_pred)))
print("MSE - test: \t{:0.6f}".format(mean_squared_error(test_y, forest_test_pred)))
print("MAE - test: \t{:0.6f}".format(mean_absolute_error(test_y, forest_test_pred)))


''' ------------------------- Extra Tree Regresson-------------------------'''
model_trees = ExtraTreesRegressor()
model_trees.fit(train_x,train_y)
trees_train_pred = model_trees.predict(train_x)
trees_test_pred = model_trees.predict(test_x)
print("Extra Tree Regressor")
print("R2 score - test: \t{:0.3f}".format(r2_score(test_y, trees_test_pred)))
print("MSE - test: \t{:0.6f}".format(mean_squared_error(test_y, trees_test_pred)))
print("MAE - test: \t{:0.6f}".format(mean_absolute_error(test_y, trees_test_pred)))

# print feature importance
print(model_trees.feature_importances_)


# Figure True versus Predicted  values
fig = plt.figure()
plt.scatter(test_y, trees_test_pred)
plt.ylabel('Predicted thickness [nm]')
plt.xlabel('Actual thickness [nm]')
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.show()
fig.savefig('True_vs_Predicted.tiff', dpi=300, bbox_inches="tight")


'''---------------Select K best---------------------------------------------'''
feature_new = SelectKBest(f_regression, k=5).fit_transform(features, target.ravel())



'''-----------statistical models---------------------------------------------'''
lm = sm.OLS(target, sm.add_constant(features)).fit()
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(lm, alpha  = 0.05, ax = ax, criterion="cooks")
plt. ylabel('Studentized residuals', fontsize=25)
plt. xlabel('H leverage', fontsize=25)
fig.savefig('Influence_removed.tiff', dpi=300, bbox_inches="tight")