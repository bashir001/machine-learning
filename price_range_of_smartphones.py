#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:40:28 2021

@author: bash
"""
#import the necessary modules/packages
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")
x_train = train.drop('price_range', axis = 1)
y_train = train['price_range']




#split the data into training and testing
x_test = test.drop('id', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state = 5)



sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
pred = model.predict(x_test)

A = model.score(x_test, y_test)
perc = A*100

#to find the correlated features 
coef_df = pd.DataFrame(train.columns.delete(0))
coef_df.columns = ['Features']
coef_df['correlation'] = pd.Series(model.coef_[0])
coef_df.sort_values(by='correlation', ascending=False)
print("CORRELATED FEATURES")
print(coef_df)
print("LOGISTIC REGRESSION")
print("Accuracy:", perc,'%')

#neural network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver='adam', random_state=3)
mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)
mlp_score = mlp.score(x_test, y_test)
mlp_per = mlp_score * 100
print("NEURAL NETWORK")
print('Accuracy:', mlp_per,'%')



#svm
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_score = svc.score(x_test, y_test)
svc_per = svc_score * 100
print("SUPPORT VECTOR MACHINE")
print("Accuracy:", svc_per,'%')




#k nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 25)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_score = knn.score(x_test, y_test)
knn_per = knn_score * 100
print('K NEAREST NEIGHBOR')
print('Accuracy:',knn_per,'%')



#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
rfc_score = rfc.score(x_test, y_test)
rfc_per = rfc_score * 100
print("RANDOM FOREST")
print("accuracy:",rfc_per,'%')

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_pred =dtc.predict(x_test)
dtc_score = dtc.score(x_test, y_test)
dtc_per = dtc_score * 100
print("DECISION TREE")
print("Accuracy:", dtc_per,'%')


#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred = gnb.predict(x_test)
gnb_score = gnb.score(x_test, y_test)
gnb_per = gnb_score * 100
print("NAIVE BAYES")
print("Accuracy:", gnb_per,'%')



#perceptron
from sklearn.linear_model import Perceptron
p = Perceptron()
p.fit(x_train, y_train)
p_pred = p.predict(x_test)
p_score = p.score(x_test, y_test)
p_per = p_score * 100
print("PERCEPTRON")
print("Accuracy:", p_per,'%')
print('confusion matrics:')
print(confusion_matrix(y_test, pred))


models = pd.DataFrame({'models:': ['Logistic regression','Neural Network', 'support vector machine', 'KNN', 'Random Forest', 'Decision Tree', 'Naive bayes', 'Perceptron'],'Accuracy:': [perc, mlp_per, svc_per, knn_per, rfc_per, dtc_per, gnb_per, p_per]})
models.sort_values
print(models)
