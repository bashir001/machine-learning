import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df = pd.read_csv("titanic (1).csv")
data = pd.get_dummies(df, columns =['Siblings/Spouses', 'Pclass', 'Sex', 'Age', 'Fare'])
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#applying knn methods
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 25)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

#showing accuracy precision and recall
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(pred)
for x in range(len(pred)):
	if pred[x] == 1:
		print(x, end="\t")

#finding the value of k
error_rate = []
for i in range(1, 40):
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(x_train, y_train)
	pred_i = knn.predict(x_test)
	error_rate.append(np.mean(pred_i != y_test))

#plotting the error rate vs. k means
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker = 0, markerfacecolor = 'red', markersize = 10)
plt.title("error_rate vs. k value")
plt.xlabel("k")
plt.ylabel("error_rate")
plt.show()

