# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:07:40 2020

@author: ucegbe
"""


###########################################################################################################

###### PLEASE RUN CODE LINE BY LINE AS IT MAY MALFUNCTION IF HIGHLIGHTED AND RUN AT ONCE#########

###########################################################################################################







import statsmodels.api as sm
import math 
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.correlation import plot_corr
from seaborn import pairplot
from tqdm import tnrange, tqdm_notebook
import itertools
import time


##### IMPORT DATA#####
df = pd.read_csv (r'C:\Users\ucegbe\Downloads\Q4_data.csv')
df.info()


###Visualize data
df.hist(bins=50, figsize=(20,15))

names=df.columns

plt.plot(df[names[0]],'r.')
plt.plot(df[names[1]],'r.')
plt.title("Fracturing fluid Volume")
plt.plot(df[names[2]],'ro')
plt.plot(df[names[3]],'ro')
plt.plot(df[names[4]],'ro')
plt.plot(df[names[5]],'ro')
plt.plot(df[names[6]],'ro')
plt.plot(df[names[7]],'ro')
plt.plot(df[names[8]],'ro')
plt.plot(df[names[9]],'ro')
plt.plot(df[names[10]],'ro')
plt.plot(df[names[11]],'ro')
plt.title("latitude")
plt.plot(df[names[12]],'ro')
plt.title("longitude")
plt.plot(df[names[13]],'r.')
plt.title("Gas Production")
plt.plot(df[names[14]],'ro')
plt.boxplot(df[names[13]])


#### Getting the index number of anomalies in each identified features#####
names=df.columns
a=df.index[df["Gas prod/mth"] == max(df["Gas prod/mth"])].tolist()
b=df.index[df["Frac_fluid_vol(bbls)"] == max(df["Frac_fluid_vol(bbls)"])].tolist()
c=df.index[df["Long."] == max(df["Long."])].tolist()
d=df.index[df["Latitude"] == max(df["Latitude"])].tolist()
a,b,c,d
#### Dropping the rows containing these anomalies ########
df.drop(99, inplace = True)
df.drop(93, inplace = True)
df.drop(22, inplace = True)

### Training and testing Data ######
train_data, test_data = train_test_split(df, test_size=0.3, random_state=668)
## Divide for predictores and response
train_y=np.log(train_data["Gas prod/mth"])  #### taking the log of response 
train_y=pd.DataFrame(train_y)
train_x=train_data.drop("Gas prod/mth", axis=1)




test_y=test_data["Gas prod/mth"]  
test_x=test_data.drop("Gas prod/mth", axis=1)
test_y=pd.DataFrame(test_y)
np.min(test_y["Gas prod/mth"])


#### LOOKING FOR CORRELATION####
corr_matrix = train_data.corr()
corr_matrix["Gas prod/mth"].sort_values(ascending=False) ### with the dependent
plot_corr(corr_matrix,xnames=corr_matrix.columns)


#####Scaling Data#####
scaler=MinMaxScaler()
scaler.fit(train_x)

train_x=scaler.transform(train_x)
Column_names=np.delete(names,(13),axis=0)
train_x=pd.DataFrame(train_x, columns=Column_names)

test_x=scaler.transform(test_x)
test_x=pd.DataFrame(test_x, columns=Column_names)

t=test_y
tr=train_x
#####SUBSET SELECTION####
def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    m=len(Y)
    model_k = LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    R_squared_adj = 1 - ( (1 - R_squared)*(m-1)/(m-len(X.columns) -1))
    return RSS, R_squared,R_squared_adj

k = 14

remaining_features = list(train_x.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()
#Looping over k = 1 to k = 14 features in Predictors
for i in range(1,k+1):
    best_RSS = np.inf
    
    for combo in itertools.combinations(remaining_features,1):

            RSS = fit_linear_reg(train_x[list(combo) + features],train_y)   #Store temp result 

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]

    #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()

display([(i,features_list[i], round(RSS_list[i])) for i in range(1,5)])


##### FINDING THE OPTIMAL SUBSET ######
df11= pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
df11['numb_features'] = df11.index

m = len(train_y)
p = 14
hat_sigma_squared = (1/(m - p -1)) * min(df11['RSS'])

#Computing
df11['C_p'] = (1/m) * (df11['RSS'] + 2 * df11['numb_features'] * hat_sigma_squared )
df11['AIC'] = (1/(m*hat_sigma_squared)) * (df11['RSS'] + 2 * df11['numb_features'] * hat_sigma_squared )
df11['BIC'] = (1/(m*hat_sigma_squared)) * (df11['RSS'] +  np.log(m) * df11['numb_features'] * hat_sigma_squared )
df11['R_squared_adj'] = 1 - ( (1 - df11['R_squared'])*(m-1)/(m-df11['numb_features'] -1))


### PLot ####
variables = ['C_p', 'AIC','BIC','R_squared_adj']
%matplotlib inline
plt.style.use('ggplot')
fig = plt.figure(figsize = (18,6))

for i,v in enumerate(variables):
    ax = fig.add_subplot(1, 4, i+1)
    ax.plot(df11['numb_features'],df11[v], color = 'lightblue')
    ax.scatter(df11['numb_features'],df11[v], color = 'darkblue')
    if v == 'R_squared_adj':
        ax.plot(df11[v].idxmax(),df11[v].max(), marker = 'x', markersize = 20)
    else:
        ax.plot(df11[v].idxmin(),df11[v].min(), marker = 'x', markersize = 20)
    ax.set_xlabel('Number of predictors')
    ax.set_ylabel(v)

fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
plt.show()
min(df11['BIC'])
df11.index[df11['AIC'] == min(df11['AIC'])].tolist()
##### Running linear regression with Significant Feattures #####
train_x=tr
test_x=t


Sig_Features=df11['features'][7]

Sig_Features

train_x=train_x[Sig_Features]
test_x.columns
test_x=test_x[Sig_Features]
lin_model=fit_linear_reg(train_x,train_y)
pd.DataFrame(lin_model, index=("RSS","R_squared","R_Squared_adj"), columns=["Values"])







m=len(train_y)
lin_reg = LinearRegression()
linear_M=lin_reg.fit(train_x, train_y)
RSS = mean_squared_error(train_y,linear_M.predict(train_x))
np.sqrt(RSS)
R_squared = linear_M.score(train_x,train_y)
R_squared_adj = 1 - ( (1 - R_squared)*(m-1)/(m-len(train_x.columns) -1))

lin_predictions = lin_reg.predict(test_x)
lin_predictions=np.exp(lin_predictions)     ####Converting predicted response to original form
lin_predictions=pd.DataFrame(lin_predictions)
test_y=pd.DataFrame(test_y)

RSS = mean_squared_error(test_y,linear_M.predict(test_x))
np.sqrt(RSS)

##### Cross Validation#####
from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, train_x, train_y,scoring="neg_mean_squared_error", cv=10)  ### 10 subsets
rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

#### Calculating Average Error ######
MAPE=[]

for i in range(0,len(test_y)):
    dd=abs(test_y.iloc[i,0]-lin_predictions.iloc[i,0])/test_y.iloc[i,0]
    MAPE.append(dd)
np.mean(MAPE)


###### Plotting predicted vs actual response #####
a=np.array([[0,110000],[0,110000]])
Average_error=np.mean(Results)
plt.plot(a)
fig1 = plt.figure(figsize = (8,6))
ax = fig1.add_subplot(1, 2,1)
ax.scatter(test_y, lin_predictions)
ax.plot(a[0],a[1], color="darkblue")
plt.ylabel("Predicted")
plt.xlabel("Actual")


### residual plot
Residuals=[]
for i in range(0,len(test_y)):
    dd=(test_y.iloc[i,0]-lin_predictions.iloc[i,0])
    Residuals.append(dd)
Residuals=pd.DataFrame(Residuals)
len(Residuals)
len(test_y)

plt.scatter(test_y,Residuals)
plt.ylabel("Actual")
plt.xlabel("Residual")




###### RIDGE REGRESSOR#######
d = {'alpha': [0, 0.001,0.1,10,100], 'RIDGE-RMSE': [3, 4,5,6,4]}
tab=pd.DataFrame(data=d)

### Linear Regression with ridge using cholesky solver
from sklearn.linear_model import Ridge
for i in range(0,5):
    ridge_reg = Ridge(alpha=tab.iloc[i,0], solver="sag")
    ridge_reg.fit(train_x, train_y)
    ridge_pred=ridge_reg.predict(test_x)
    
    lin_mse_ridge = mean_squared_error(test_y, ridge_pred)
    lin_rmse_ridge = np.sqrt(lin_mse_ridge)
    
    tab.iloc[i,1]=lin_rmse_ridge
tab





