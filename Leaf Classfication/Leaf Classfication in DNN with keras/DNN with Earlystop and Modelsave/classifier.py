import numpy as np
np.random.seed(2016)
import os
#import cv2
import math
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize, imshow
from keras.regularizers import l2


restore_from_last_checkpoint = 0

def load_train_csv(encoder,scaler):
    data = pd.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in DNN with keras/data/train.csv")
    data=data.drop('id',1)
    y = data.pop('species')
    encoder = LabelEncoder().fit(y)
    y = encoder.transform(y)
    scaler = StandardScaler().fit(data)
    X = scaler.transform(data)
    y = np_utils.to_categorical(y)
    X,y = shuffle(X,y)
    return X,y,encoder,scaler

def load_test_csv(scaler):
    data = pd.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in DNN with keras/data/test.csv")
    species = pd.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in DNN with keras/data/train.csv").species.unique()
    species.sort()
    Id = data.pop('id')
    scaler = StandardScaler().fit(data)
    X = scaler.transform(data)
    return X, species, Id


def create_model(input_shape):
    model = Sequential()
    model.add(Dense(384,input_dim = input_shape, activation = 'relu'))
    model.add(Dense(384,activation = 'relu'))
    model.add(Dense(99, activation='softmax'))

    #rmsprop = RMSprop(lr=0.0003, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_cross_validation():
    batch_size = 20
    nb_epoch = 300
    random_state = 51
    encoder = LabelEncoder()
    scaler = StandardScaler()
    #input_dim = 64
    # load the training and validation data sets
    X_train, Y_train, encoder, scaler = load_train_csv(encoder,scaler)
    joblib.dump(scaler, 'C:/Users/sound/Desktop/Kaggle/Leaf Classfication/DNN with Early Stop and Model Save/scaler.pkl')
    weights_path = os.path.join('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/DNN with Early Stop and Model Save', 'weights.h5')

    model = create_model(X_train.shape[1])
    if not os.path.isfile(weights_path) or restore_from_last_checkpoint == 0:
        
        callbacks = [ 
            EarlyStopping(monitor='val_loss', patience=20, verbose=0),
            ModelCheckpoint(weights_path, monitor='loss', save_best_only=True, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1,validation_split=0.2,
              callbacks=callbacks)
        

def run_test():
    weights_path = os.path.join('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/DNN with Early Stop and Model Save', 'weights.h5')
    scaler = joblib.load('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/DNN with Early Stop and Model Save/scaler.pkl')
    X_test, species, index= load_test_csv(scaler)
    #input_dim = 64
    model = create_model(X_test.shape[1])
    model.load_weights(weights_path)
    predicts=model.predict_proba(X_test)
    y_pred = pd.DataFrame(predicts, index=index, columns = species)
    y_pred.to_csv('submission.csv')
    #fp = open('submission.csv', 'w')
    #fp.write(y_pred.to_csv()) 


if __name__ == '__main__':
    run_cross_validation()
    run_test()