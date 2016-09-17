import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout,core
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#load train dataset
dataframe = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in DNN with keras/data/train.csv")
dataset = dataframe.values
X = dataset[:,2:].astype(float)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = dataset[:,1]
encoder = LabelEncoder()
le=encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#convert integers to dummy variables 
dummy_y = np_utils.to_categorical(encoded_Y)

#load test dataset
test = pandas.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in logistics regression/data/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

# create model
model = Sequential()
model.add(Dense(384, input_dim=192, init='uniform', activation='relu'))
model.add(Dense(384, init='uniform', activation='relu'))
model.add(Dense(99, init='uniform', activation='sigmoid'))

# Compile and fit model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, nb_epoch=100, batch_size=20)

#make predictions and submission
predictions=model.predict_proba(x_test)
submission = pandas.DataFrame(predictions, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')

 

