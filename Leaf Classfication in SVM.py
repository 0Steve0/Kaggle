import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in logistics regression/data/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

test = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in logistics regression/data/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

svc = SVC(probability=True)
svc.fit(x_train, y_train)
y_test = svc.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')