import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data

#auxiliary functions

def interests(x):
    if x<(-0.5):
        return 1 #diferentes
    elif x<0 and x>=(-0.5):
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
    if x>10:
        return 3

#load data
data = pd.read_csv('speedDating_trab.csv')

#solve NaN
data['met'] = data['met'].fillna(0)
data = data[data['like'].notna()]
data = data[data['prob'].notna()]
data = data[data['int_corr'].notna()]
data = data[data['match'].notna()]
#tmp solutions
data = data[data['age'].notna()]
data = data[data['age_o'].notna()]
data['date'] = data['date'].fillna(np.ceil(data['date'].mean()))
data['go_out'] = data['go_out'].fillna(np.ceil(data['go_out'].mean()))
#tmp solutions
data['length'] = data['length'].fillna(np.ceil(data['length'].mean()))
data = data.drop(data.columns[0], axis=1)
data = data.drop(columns='id')
data = data.drop(columns='partner')

#reclassify info
data['like'] = data['like'].map(prob)
data['prob'] = data['prob'].map(prob)
data['int_corr'] = data['int_corr'].map(interests)
data['date'] = data['date'].map(freq)
data['go_out'] = data['go_out'].map(freq)
data['age_difference'] = data.apply(lambda row: np.abs(row.age - row.age_o), axis=1)
data['age_difference'] = data['age_difference'].map(ages)
data = data.drop(columns='age')
data = data.drop(columns='age_o')

#make tests
X = data.drop('match', axis=1)
y = data['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
match = np.array(y_test)

#decision tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
dot_data = StringIO()
export_graphviz(tree, out_file=dot_data)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

print('Decision Tree Confusion Matrix')
predT = tree.predict(X_test)
confusion_matrix(match, predT)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != predT).sum()))


#naive bayes

print('Gaussian Naive Bayes')
gaussianNB = GaussianNB()
gaussianNB.fit(X_train, y_train)
predGNB = gaussianNB.predict(X_test)
confusion_matrix(match, predGNB)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != predGNB).sum()))
    
