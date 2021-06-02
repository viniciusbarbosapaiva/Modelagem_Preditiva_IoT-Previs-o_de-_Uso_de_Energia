import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='white', palette='deep')
width = 0.35
import os
import scipy.stats as sc
from datetime import datetime

#Functions
def autolabel_without_pct(rects,ax): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=90)

#Importing dataset
path = os.getcwd()
document = os.listdir();document
train = pd.read_csv(document[3], sep=',')
test = pd.read_csv(document[2], sep=',')

df = pd.concat([train,test])
df.info()
df.shape

#After concatinate a datasets, reset index
df['date'][0] #Two value will appear
df.reset_index(inplace=True, drop=True)
df['date'][0] #One value will appear

##Statistical Analysis
# Distribution of Dataset
df.columns
stat = df.describe()
types = [np.dtype(df[i]) for i in df[df.columns]]
types = pd.DataFrame(types)
types.info()

#Creating list of numerical index
numerical = np.array([])
numerical = [i for i in np.arange(0,len(df.columns)) for j in np.unique(types)[(np.unique(types)=='float64') | (np.unique(types)=='int64')]  if np.dtype(df.iloc[:,i])==j]          

#Calculating the median
median = df.iloc[:,numerical].apply(lambda x: np.median(x))
median = pd.DataFrame(np.reshape(np.array(median),(1,len(numerical))), columns=[df.columns[numerical[i]] for i in np.arange(0,len(numerical))],index=['median'])
stat = pd.concat([stat,median])

#Calculating a diference between mean and median
mean_minus_median = stat.loc['mean',:] - stat.loc['median',:]
mean_minus_median = pd.DataFrame(np.reshape(np.array(mean_minus_median), (1,len(numerical))), columns=[df.columns[numerical[i]] for i in np.arange(0,len(numerical))],index=['normal_dist'])
stat = pd.concat([stat,mean_minus_median])

##Calculating the skew
skew = df.iloc[:,numerical].apply(lambda x: sc.skew(x))
skew = pd.DataFrame(np.reshape(np.array(skew),(1,len(numerical))), columns= [df.columns[numerical[i]] for i in np.arange(0,len(numerical))],index=['skew'])
stat = pd.concat([stat,skew])

#Defining normal function
def normal_dist(dataframe):
    temp = dataframe
    normal  = np.array([])
    not_normal = np.array([])  
    for i in np.arange(0, temp.shape[1]):
        if temp.iloc[-1,i] >= -0.6 and temp.iloc[-1,i] <= 0.6:
            normal = np.append(normal,temp.columns[i])
        else:
            not_normal = np.append(not_normal,temp.columns[i])
    print('For normally distributed data, the skewness should be about zero.')        
    print('Columns {}, perhaps will have normal distribution because the {} values are between -0.6 and 0.6'.format(normal, temp.index[-1]))    
    print(80*'-')    
    print('Columns {}, perhaps will not have normal distribution because the {} are not between values -0.6 and 0.6'.format(not_normal, temp.index[-1]))
    return normal,not_normal
normal, not_normal = normal_dist(stat)    

#plot histogramns
len(normal)
normal
fig = plt.figure(figsize=(10,10))
for i in np.arange(1,len(normal)+1):
    ax = fig.add_subplot(7, 4,i)
    ax.hist(df.loc[:,normal[i-1]])
    ax.set_title('Distribution of {}'.format(normal[i-1]))
plt.tight_layout() 

len(not_normal)
fig2 = plt.figure(figsize=(10,10))
for i in np.arange(1,len(not_normal)+1):
    ax = fig2.add_subplot(2,3, i)
    ax.hist(df.loc[:,not_normal[i-1]])
    ax.set_title('Distribution of {}'.format(not_normal[i-1]))
plt.tight_layout()  

#Feature engineering
df_feature = df.copy()
df_feature.columns
df_feature.info()

#Drop WeekStatus Column
df_feature.drop('WeekStatus', axis=1, inplace =True)

#Converting Date column in datetime type
from calendar import month_name
df_feature['date'] = df_feature['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df_feature.info()

#Creating the month column and converting to a month name
df_feature['month'] = df_feature['date'].dt.month
df_feature['month'].unique()
type(df_feature['month'][0])
df_feature['month'] = df_feature['month'].apply(lambda x: month_name[x])

#Creating hour columns
df_feature['hour'] = df_feature['date'].dt.hour

#Creating day columns
df_feature['day'] = df_feature['date'].dt.day

#Creating PLot
numerical_columns = df_feature.iloc[:,numerical].columns
def line_plot(data,x,y,hue):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111) 
    sns.lineplot(data=data, x=x,y=y, hue=hue, ci=None, ax=ax, markers=True)
    ax.set_title('Plot of {} by {}'.format(y,x))
    ax.grid(b=True, which='major', linestyle='--')

for i in numerical_columns:
    line_plot(df_feature,'hour',i, 'Day_of_week')
    
for i in numerical_columns:
    line_plot(df_feature,'hour',i, 'month')

sns.pairplot(df_feature, vars=numerical_columns)

#Define X and y
X = df_feature.drop(['NSM','date'], axis=1)
y = df_feature['NSM']

#Get Dummies variables and avoiding dummy trap
X = pd.get_dummies(X,drop_first=True)

#Applying Statmodel (p_value <=0.05)
import statsmodels.api as sm

Xc = sm.add_constant(X)
model = sm.OLS(y, Xc)
model_v1 = model.fit()
model_v1.summary()

# Feature selection with extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(X, y)

print(X.columns)
print(modelo.feature_importances_) 

mean_feature = np.mean(modelo.feature_importances_)
label = X.columns
y_mean = [mean_feature] * len(label) 
ind = np.arange(0,len(label))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_title('The Feature Importances Plot')
rect = ax.bar(label,np.round(modelo.feature_importances_,3), label='Data')
ax.grid(b=True, which='major', linestyle='--')
ax.set_xlabel('Model Features')
ax.set_ylabel('Level of Importance')
ax.set_xticks(ind)
ax.tick_params(axis='x', labelrotation=90)
autolabel_without_pct(rect,ax)
ax.plot(label,y_mean, color='red', label='Mean', linestyle='--')
ax.legend()
plt.plot()

#Selecting the most relevant features
features_importance = dict(zip(X.columns,modelo.feature_importances_))
features_importance = pd.DataFrame(features_importance, index=[0])
features_importance_names = [features_importance.columns[i] for i in np.arange(0,len(features_importance.columns)) if features_importance.iloc[0,i] > mean_feature]
X = X[features_importance_names]

#Splitting the Dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
y_test = pd.DataFrame(sc_y.fit_transform(np.array(y_test.iloc[:]).reshape(len(y_test),1)), columns=[y_test.name])
y_train = pd.DataFrame(sc_y.transform(np.array(y_train.iloc[:]).reshape(len(y_train),1)), columns=[y_train.name])

#### Model Building ####
### Comparing Models
## Multiple Linear Regression Regression
from sklearn.linear_model import LinearRegression
k = X_test.shape[1]
n = len(X_test)
lr_regressor = LinearRegression(fit_intercept=True)
lr_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = lr_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

## Ridge Regression
from sklearn.linear_model import Ridge
rd_regressor = Ridge(alpha=50)
rd_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = rd_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Ridge Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Lasso Regression
from sklearn.linear_model import Lasso
la_regressor = Lasso(alpha=500)
la_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = la_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Lasso Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Polynomial Regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lr_poly_regressor = LinearRegression(fit_intercept=True)
lr_poly_regressor.fit(X_poly, y_train)

# Predicting Test Set
y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(X_test))
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Suport Vector Regression 
'Necessary Standard Scaler '
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = svr_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Support Vector RBF', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = dt_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(X_train,y_train)

# Predicting Test Set
y_pred = rf_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Ada Boosting
from sklearn.ensemble import AdaBoostRegressor
ad_regressor = AdaBoostRegressor()
ad_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = ad_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = gb_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Xg Boosting
from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = xgb_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Ensemble Voting regressor
from sklearn.ensemble import VotingRegressor
voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),
                                                ('rd', rd_regressor),
                                                ('la', la_regressor),
                                                ('lr_poly', lr_poly_regressor),
                                                ('svr', svr_regressor),
                                                ('dt', dt_regressor),
                                                ('rf', rf_regressor),
                                                ('ad', ad_regressor),
                                                ('gr', gb_regressor),
                                                ('xg', xgb_regressor)])

for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,
            rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Ensemble Voting', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)  

#The Best Classifier
print('The best regressor is:')
print('{}'.format(results.sort_values(by='Adj. R2 Score',ascending=False).head(5)))

#Applying K-fold validation
from sklearn.model_selection import cross_val_score
def display_scores (scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard:', scores.std())

lin_scores = cross_val_score(estimator= rf_regressor, X=X, y=y, 
                             scoring= 'neg_mean_squared_error',cv=10) # Era X_train e y_train. Passei para X e y.
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#Validation of model. Analyzing the loss function
from yellowbrick.regressor import ResidualsPlot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,1,1)
residual = ResidualsPlot(rf_regressor, ax=ax1)
residual.fit(X_train, y_train.values.flatten())  
residual.score(X_test, y_test.values.flatten())  
residual.show() 


