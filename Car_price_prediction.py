#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 17:56:16 2021

@author: bash
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
df = pd.read_csv("CarPrice_Assignment.csv")


# what kind of car appear the most in the datasets
plt.title('cars', fontsize=15)
snb.countplot(df['carbody'])
plt.show()


# what type of fuel is used most
fueltype = df['fueltype'].value_counts()
total = df['fueltype'].value_counts().sum()
percentgram = fueltype/total

plt.title('most used fuel', fontsize = 15)
plt.pie(percentgram, labels=['gas', 'diesel'],autopct='%1.1f%%');
plt.show()

# what is the most common insurance risk classification
df['symboling'].value_counts().sort_values().plot.bar()
plt.title('insurance risk rating', fontsize=15)
plt.xlabel('risk rating', fontsize= 15)
plt.ylabel('total')
plt.show()

# Reparing the data
# corrolation between features
plt.figure(figsize=(12,7))
corrolation= df.corr()
snb.heatmap(corrolation, annot = True)
plt.show()


# analyzing the corrolation between the target variable(price) and the other columns
corral = df.corr()['price'].drop('price')
d= corral.sort_values()
print(d)

# Transfoeming categorical variables into numerical variable, so that the variables can also enter the model that predict which ones are best for the algorithms
df['fueltype']=df['fueltype'].map({'gas':'0','diesel':'1'})
df['aspiration']=df['aspiration'].map({"std":'0', 'turbo':'1'})
df['doornumber']=df['doornumber'].map({'two':'2', 'four':'4'})
df['carbody']=df['carbody'].map({'convertible':'1', 'hatchback':'1','sedan':'2', 'wagon':'3','hardtop':'4'})
df['drivewheel']=df['drivewheel'].map({'rwd':'0', 'fwd':'1', '4wd':'2'})
df['enginelocation']=df['enginelocation'].map({'front':'0', 'rear':'1'})
df['cylindernumber']=df['cylindernumber'].map({'four':'4','six':'6','five':'5','three':'3','twelve':'12','two':'2','eight':'8'})

# Transforming object into int32

df['fueltype']=df['fueltype'].astype(int)
df['aspiration']=df['aspiration'].astype(int)
df['doornumber']=df['doornumber'].astype(int)
df['carbody']=df['carbody'].astype(int)
df['drivewheel']=df['drivewheel'].astype(int)
df['enginelocation']=df['enginelocation'].astype(int)
df['cylindernumber']=df['cylindernumber'].astype(int)

# only numerical values

numerical_col = df.select_dtypes(include=['int32','int64','float'])
print(numerical_col.head())

# Seperating training data and test data

from sklearn.model_selection import train_test_split
x = df[['symboling','aspiration','carbody','fueltype','enginelocation','cylindernumber','doornumber','drivewheel','wheelbase','carlength','carwidth','carheight',
        'curbweight','enginesize','stroke','peakrpm','citympg','horsepower','compressionratio','highwaympg','boreratio']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=55,test_size=0.5)

print('{0:0.2f}% are training data'.format((len(x_train)/len(df.index)* 100)))
print('{0:0.2f}% are testing data'.format((len(x_test)/len(df.index)* 100)))

# Linear Regresion
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("""model\t\t\t\t R2  \t\t RMSE  \t\t  MAE""")
print("""Linear regression \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_lr),mean_squared_error(y_test,pred_lr),
                                                                              mean_absolute_error(lr.predict(x_test),y_test)))
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
Rfr = RandomForestRegressor(n_estimators= 10, random_state=50)
Rfr.fit(x_train, y_train)
pred_Rfr = Rfr.predict(x_test)

print("""Model\t\t\t\t R2  \t\t RMSE  \t\t  MAE""")
print("""Random Forest \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test, pred_Rfr),mean_squared_error(y_test,pred_Rfr),
                                                                  mean_absolute_error(Rfr.predict(x_test),y_test)))
#k nearest neighbor
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)

print("""Model\t\t\t\t R2  \t\t RMSE  \t\t  MAE""")
print("""K Nearest neighbor \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_knn),
                                                                       mean_squared_error(y_test,pred_knn),mean_absolute_error(knn.predict(x_test),y_test)))
# Support vector regression
from sklearn.svm import SVR
svm = SVR()
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)

print("""Model\t\t\t\t R2  \t\t RMSE  \t\t  MAE""")
print("""Support vector regression \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_svm),
      
          
                                                                    mean_squared_error(y_test,pred_svm),mean_absolute_error(svm.predict(x_test),y_test)))
# Evaluating cross validation
from sklearn.model_selection import cross_val_score

print('Linear regression')
lr_score = cross_val_score(lr, x_train, y_train , cv=5, scoring='r2')
print(lr_score)
print("mean", lr_score.mean())


print('random forest')
Rfr_score = cross_val_score(Rfr, x_train, y_train, cv = 5 , scoring='r2')
print(Rfr_score)
print('mean', Rfr_score.mean())

print('k nearest neighbor')
knn_score = cross_val_score(knn, x_train, y_train, cv = 5, scoring='r2')
print(knn_score)
print('mean', knn_score.mean())

print('Support vector ')
svm_score = cross_val_score(svm, x_train, y_train, cv =5, scoring='r2')
print(svm_score)
print('mean', svm_score.mean())

# comparing and evaluating the model
# table summary for better viewing


results = pd.DataFrame([{'Algorithm':'Linear regression','original': r2_score(y_test, pred_lr), 'cross validation': lr_score.mean()},
                        {'Algorithm': 'Random forest','original': r2_score(y_test,pred_Rfr), 'cross validation': Rfr_score.mean()},
                        {'Algorithm': 'k nearest neighbor','original': r2_score(y_test,pred_knn), 'cross validation': knn_score.mean()},
                        {'Algorithm': 'Support vector regression','original': r2_score(y_test, pred_svm), 'cross validation': svm_score.mean()}])

print(results.sort_values(by=['cross validation'], ascending=False))


# Saving the model
import pickle
filename = 'Rfr_modelo.sav'
pickle.dump(Rfr, open(filename,'wb'))

# Loading the model and forecasting with new datasets
# (X_test, Y_test must be new datasets prepared with the proper cleanup and transformation procedure)

load_model = pickle.load(open(filename, 'rb'))
results = load_model.predict(x_test[:100])
plt.figure(figsize=(12,8))

plt.title('Real values vs Predicted values')
plt.ylabel('Sales Value')
plt.plot(results) #x_test
plt.plot(y_test.values[:100]) #y_test. 100 first values

plt.legend(['Predictions', 'Real Values'])
plt.show()
