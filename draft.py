import pandas as pd
import numpy as np
import graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
import seaborn as sns
import matplotlib.pyplot as plt

#auxiliary functions

def interests(x):
    if x<(-0.5):
        return 1 #diferentes
    elif x<0.5 and x>=(-0.5):
        return 2 #pouco parecidos
    elif x<(0.5) and x>=0:
        return 3 #aproximados
    else:
        return 4 #iguais

def prob(x):
    if x<=3:
        return 1 #pouco
    elif x>3 and x<=7:
        return 2 #alguma
    else:
        return 3 #muita

def freq(x):
    if x<=3:
        return 1 #semanalmente
    elif x>3 and x<6:
        return 2 #mensalmente
    else:
        return 3 #raramente

def ages(x):
    if x<=5:
        return 1 
    if x>5 and x<=10:
        return 2
    else:
        return 3

def goal(x):
    if x<3:
        return 1
    else:
        return 2

#load data
data = pd.read_csv('speedDating_trab.csv')

#solve NaN
data['met'] = data['met'].fillna(0)
data = data[data['like'].notna()]
data = data[data['prob'].notna()]
data = data[data['int_corr'].notna()]
data = data[data['match'].notna()]
data = data[data['age'].notna()]
data = data[data['age_o'].notna()]
data['date'] = data['date'].fillna(np.ceil(data['date'].mean()))
data['go_out'] = data['go_out'].fillna(np.floor(data['go_out'].mean()))
data['length'] = data['length'].fillna(np.floor(data['length'].mean()))
data = data.drop(data.columns[0], axis=1)
data = data.drop(columns='id')
data = data.drop(columns='partner')

age = data['age']
median = age.median()
min = age.min()
max = age.max()
print("Age data statistics")
print("Avegare age " + str(median))
print("Minimum age " + str(min))
print("Maximum age " + str(max))

fig, ax = plt.subplots(figsize=(20,5))

sns.countplot(x=age)
plt.savefig('1.png')

fig, ax = plt.subplots()


sns.countplot(x='match', data = data)
plt.savefig('2.png')

fig, ax = plt.subplots()
sns.countplot(x='match', hue='goal',data = data)
plt.savefig('3.png')

fig, ax = plt.subplots()
sns.countplot(x='match', hue='date',data = data)
plt.savefig('4.png')

fig, ax = plt.subplots()
sns.countplot(x='match', hue='go_out', data = data)
plt.savefig('5.png')

fig, ax = plt.subplots()
sns.countplot(x='match', hue='length', data = data)
plt.savefig('6.png')


fig, ax = plt.subplots(figsize=(10,10))

plot = sns.stripplot(x='length', y = 'like', hue='match', data =data, size=6)
plt.savefig('7.png')

fig, ax = plt.subplots(figsize=(10,10))

plot = sns.stripplot(x='date', y = 'like', hue='match', data =data, size=6)
plt.savefig('8.png')


#reclassify info
data['like'] = data['like'].map(prob)
data['prob'] = data['prob'].map(prob)
data['int_corr'] = data['int_corr'].map(interests)
data['date'] = data['date'].map(freq)
data['go_out'] = data['go_out'].map(freq)
data['goal'] = data['goal'].map(goal)
data['age_difference'] = data.apply(lambda row: np.abs(row.age - row.age_o), axis=1)
data['age_difference'] = data['age_difference'].map(ages)
data = data.drop(columns='age')
data = data.drop(columns='age_o')

#make tests
X = data.drop('match', axis=1)
y = data['match']

print('Holdout')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
match = np.array(y_test)

#decision tree
dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dot_data = tree.export_graphviz(dtree, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("DTree") 

print('Tree Depth')
print(dtree.tree_.max_depth)
print('Decision Tree Confusion Matrix')
predT = dtree.predict(X_test)
print(confusion_matrix(match, predT))
print("Number of mislabeled points out of a total %d points : %d\n" % (X_test.shape[0], (y_test != predT).sum()))
print('Cross Validation')
dtree = tree.DecisionTreeClassifier()
scores = cross_val_score(dtree, X, y, cv=10) #compute scores 10 times with different splits every time
accuracy = scores.mean()
print("%0.2f accuracy with a standard deviation of %0.2f\n\n" % (accuracy, scores.std()))
predT = cross_val_predict(gaussianNB, X, y, cv=10)
print('Precision | Recall | F-beta | Support')
print(precision_recall_fscore_support(y_test, predT, average='micro'))
print('Error Rate: %f' %(1-accuracy))

#naive bayes

print('Gaussian Naive Bayes')
gaussianNB = GaussianNB()
gaussianNB.fit(X_train, y_train)
predGNB = gaussianNB.predict(X_test)
print(confusion_matrix(match, predGNB))
print("Number of mislabeled points out of a total %d points : %d\n" % (X_test.shape[0], (y_test != predGNB).sum()))
print('Cross Validation')
gaussianNB = GaussianNB()
scores = cross_val_score(gaussianNB, X, y, cv=10) #compute scores 10 times with different splits every time
accuracy = scores.mean()
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))  
predT = cross_val_predict(gaussianNB, X, y, cv=10)
print('Precision | Recall | F-beta | Support')
print(precision_recall_fscore_support(y_test, predT, average='micro'))
print('Error Rate: %f' %(1-accuracy))
