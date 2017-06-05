import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
def main():
    # loading training data
    print('Loading training data')
    traindata = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Data/train.csv')#read csv file into dataframe
    x_tr = traindata.values[:, 2:]#The other columns are the pixels of each image and we have 28,000 images
    y_tr = traindata.values[:, 1]#The first column is the label that drawn by user
    
    le = LabelEncoder().fit(traindata['species'])
    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr)
    
    print('Loading test data')
    testdata = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Data/test.csv')
    
    #x_test = testdata.values[:, 1:].astype(float)
    #x_test = scaler.transform(x_test)
    x_test = testdata.drop(['id'], axis=1).values
    x_test = scaler.transform(x_test)
    test_ids = testdata.pop('id')
    print('Start learning...')
    random_forest = RandomForestClassifier(n_estimators=1000)
    random_forest.fit(x_tr, y_tr)
    y_pred = random_forest.predict_proba(x_test)
    
    submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    main()