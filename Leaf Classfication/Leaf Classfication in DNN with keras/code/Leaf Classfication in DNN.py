import numpy as np
#np.random.seed(100)
import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout,core
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
#load train dataset
dataframe = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Data/train.csv")
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
test = pandas.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Data/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

# create model
model = Sequential()
model.add(Dense(1024,input_dim=192,  init='glorot_normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(99, activation='softmax'))

# Compile and fit model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X, dummy_y,nb_epoch=100, batch_size=128,verbose=1, validation_split=0.1)


#make predictions and submission
predictions=model.predict_proba(x_test)
submission = pandas.DataFrame(predictions, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')

 

