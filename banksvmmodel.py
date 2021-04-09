import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("bank-full.csv", sep=';')
df.columns = [col.replace('"', '') for col in df.columns]
b = df['y'].value_counts()
df = df.dropna()
df.drop(columns=['day','campaign','default','balance','loan','pdays','previous','duration','month'], inplace=True)
df.y.replace({'no':0, 'yes':1},inplace=True)
df= pd.get_dummies(df, columns=['age','job','marital','housing','education','poutcome'])
df['y'].replace('yes',1)
df['y'].replace('no', 0)
x = df.iloc[:,1:]
y = df.iloc[:,7]
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
model = RandomForestClassifier(n_estimators=50)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
for x in range(len(pred)):
	if pred[x] == 1:
		print(x, end="\t")
		print("accuracy:", format(model.score(x_test,y_test)))
result = confusion_matrix(y_test, pred)
print("confusion matrics:", format(result))
result1 = classification_report(y_test, pred)
print("classification report")
print(result1)
result2 = accuracy_score(y_test, pred)
print("accuracy score")
print(result2)
print(df['y'].value_counts())

#saving and loading model as pickle file

#You can now visualize individual trees. The code below visualizes the first decision tree.



 
fn=df.x
cn=df.y
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')








#######To access the single decision tree from the random forest in scikit-learn use estimators_ attribute:

#rf = RandomForestClassifier()
###### first decision tree
#rf.estimators_[0]
####Then you can use standard way to visualize the decision tree:

###you can print the tree representation, with sklearn export_text
###export to graphiviz and plot with sklearn export_graphviz method
###plot with matplotlib with sklearn plot_tree method
####use dtreeviz package for tree plotting



####You can draw a single tree:

#from sklearn.tree import export_graphviz
#from IPython import display
#from sklearn.ensemble import RandomForestRegressor

#m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
#m.fit(X_train, y_train)

#str_tree = export_graphviz(m, 
 #  out_file=None, 
 #  feature_names=X_train.columns, # column names
 #  filled=True,        
 #  special_characters=True, 
 #  rotate=True, 
 #  precision=0.6)

#display.display(str_tree)