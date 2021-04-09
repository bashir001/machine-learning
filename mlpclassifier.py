from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd 
df = pd.read_csv('titanic (1).csv')
df.Sex.replace({'male':0, 'female':1},inplace=True)
df['Male']=df['Sex']=='Male'
x = df[['Pclass','Age','Sex','Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=3)
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver='adam', random_state=3)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x)
print("accuracy:", mlp.score(x_test, y_test))
print(mlp.predict(x).shape)
print((y == y_pred).sum())
print(y.shape[0])