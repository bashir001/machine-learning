#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:47:49 2021

@author: bash
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as snb
import pandas as pd 

df = pd.read_csv('California_Houses.csv')
corralation = df.corr()['Median_House_Value'].drop('Median_House_Value')
print(corralation.sort_values())
plt.figure(figsize=(12,7))
snb.heatmap(df.corr(), annot=True, linewidths = 5, cmap='PiYG')
plt.show()

df.plot(kind='scatter',
        x='Median_Income',
        y='Median_House_Value',
        figsize=(8,6));
df.plot(kind='scatter',
         x='Tot_Rooms',
         y='Median_House_Value',
         figsize=(8,6));
df.plot(kind='scatter',
         x= 'Median_Age',
         y='Median_House_Value',
         figsize=(8,6))
plt.show()


x = df[['Median_Income','Median_Age','Distance_to_SanJose']]
y = df['Median_House_Value']


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 10, test_size=0.4)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred_lr = lr.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("""model\t\t\t\t R2  \t\t RMSE  \t\t  MAE""")
print("""Linear regression \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_lr),
                                                                      mean_squared_error(y_test,pred_lr).round(2),
                                              mean_absolute_error(lr.predict(x_test),y_test)))               

print(lr.score(x_train,y_train).round(2)*100)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=5, random_state=1)
rfr.fit(x_train,y_train)
pred_rfr = rfr.predict(x_test)
print("""model\t\t\t\t R2  \t\t RMSE  \t\t  MAE """)
print("""Random forest \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_rfr),
                                                                  mean_squared_error(y_test,pred_rfr).round(2),
                   mean_absolute_error(rfr.predict(x_test),y_test)))

print('accuracy {0:0.2f}%'.format(rfr.score(x_train,y_train).round(2)*100))

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 5, n_jobs=3)
knn.fit(x_train,y_train)
pred_knn = knn.predict(x_test)
print("""model\t\t\t\t R2  \t\t RMSE  \t\t  MAE """)
print("""K Nearest neighbor \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_knn),
                                                                       mean_squared_error(y_test,pred_knn).round(2),
                   mean_absolute_error(knn.predict(x_test),y_test)))
print('accuracy {0:0.2f}%'.format(knn.score(x_train,y_train).round(2)*100))


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=1)
dt.fit(x_train,y_train)
pred_dt = dt.predict(x_test)
print("""model\t\t\t\t R2  \t\t RMSE  \t\t  MAE """)
print("""Decision tree \t\t {:.2f} \t\t {:.4} \t {:.2f}""".format(r2_score(y_test,pred_dt),
                                                                  mean_squared_error(y_test,pred_dt).round(2),
                   mean_absolute_error(knn.predict(x_test),y_test)))
print('accuracy {0:0.2f}%'.format(dt.score(x_train,y_train).round(2)*100))

from sklearn.model_selection import cross_val_score
print('Linear regression')
lr_score = cross_val_score(lr, x_train, y_train , cv=5, scoring='r2')
print(lr_score)
print("mean", lr_score.mean())

print('Random forest')
rfr_score = cross_val_score(rfr, x_train, y_train,cv=5, scoring='r2')
print(rfr_score)
print('mean;', rfr_score.mean())

print('k nearest neighbor')
knn_score = cross_val_score(knn, x_train, y_train, cv = 5, scoring = 'r2')
print(knn_score)
print('mean;', knn_score.mean())


print('Decision Tree')
dt_score = cross_val_score(dt, x_train, y_train, cv = 5, scoring = 'r2')
print(dt_score)
print('mean;', dt_score.mean())

# comparing and evaluating the model
# table summary for better viewing


results = pd.DataFrame([{'Algorithm':'Linear regression','original': r2_score(y_test, pred_lr), 'cross validation': lr_score.mean()},
                        {'Algorithm': 'Random forest','original': r2_score(y_test,pred_rfr), 'cross validation': rfr_score.mean()},
                        {'Algorithm': 'k nearest neighbor','original': r2_score(y_test,pred_knn), 'cross validation': knn_score.mean()},
                        {'Algorithm': 'Decision tree','original': r2_score(y_test, pred_dt), 'cross validation': dt_score.mean()}])

print(results.sort_values(by=['cross validation'], ascending=False))

# Saving the model
import pickle
filename = 'Rfr_model.sav'
pickle.dump(rfr, open(filename,'wb'))

# Loading the model and forecasting with new datasets
# (X_test, Y_test must be new datasets prepared with the proper cleanup and transformation procedure)

load_model = pickle.load(open(filename, 'rb'))
results = load_model.predict(x_test[:100])
plt.figure(figsize=(12,8))

plt.title('Real values vs Predicted values')
plt.ylabel('Median house value')
plt.plot(results) #x_test
plt.plot(y_test.values[:100]) #y_test. 100 first values

plt.legend(['Predictions', 'Real Values'])
plt.show()
