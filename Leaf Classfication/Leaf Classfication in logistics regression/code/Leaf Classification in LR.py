import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
train = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Data/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#C Inverse of regularization strength; must be a positive float. Like in support vector machines, 
#smaller values specify stronger regularization.
#tol float, optional Tolerance for stopping criteria the smaller tol, the longer the algorithm will run
params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005]}
#L2 regularization on least squares
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf = GridSearchCV(log_reg, params, scoring='log_loss', n_jobs=1, cv=5)
clf.fit(x_train, y_train)#Parameter setting that gave the best results on the hold out data
print("best params: " + str(clf.best_params_))
for params, mean_score, scores in clf.grid_scores_:
  print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
  print(scores)

test = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Data/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

y_test = clf.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')
