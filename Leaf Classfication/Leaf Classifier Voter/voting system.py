import pandas
import numpy as np
import math 
from sklearn.preprocessing import LabelEncoder
train = pandas.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in logistics regression/data/train.csv')
le = LabelEncoder().fit(train['species'])
test = pandas.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in logistics regression/data/test.csv')
test_ids = test.pop('id')
dataframe1 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/submission1.csv")
dataset1 = dataframe1.values
dataframe2 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/submission2.csv")
dataset2 = dataframe2.values
dataframe3 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/sample_submission.csv")
dataframe3=dataframe3.drop('id',1)
dataset3 = dataframe3.values
#print(dataset3[0,:])
for j in range(0,593):
  biggest=dataset1[j,1]
  secondbiggest=dataset2[j,1]
  for i in range(1,99):
    if dataset1[j,i]>biggest:
       biggest=dataset1[j,i]
    if dataset2[j,i]>secondbiggest:
       secondbiggest=dataset2[j,i]
  if((biggest>=secondbiggest)&(biggest>0.99)):
      dataset3[j,:]=dataset1[j,1:]
  if((biggest<secondbiggest)&(secondbiggest>0.99)):
      dataset3[j,:]=dataset2[j,1:]
  else:
      dataset3[j,:]=dataset2[j,1:]
      
submission = pandas.DataFrame(dataset3, index=test_ids, columns=le.classes_)
submission.to_csv('voting.csv')