import numpy
import pandas
import math 
dataframe1 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in DNN with keras/output/submission with 0.00731.csv")
dataset1 = dataframe1.values
dataframe2 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Leaf Classfication/Leaf Classfication in DNN with keras/output/submission with 0.022.csv")
dataset2 = dataframe2.values
for j in range(0,593):
  biggest=dataset1[j,1]
  secondbiggest=dataset2[j,1]
  for i in range(1,99):
    if dataset1[j,i]>biggest:
       biggest=dataset1[j,i]
       id1=i
    if dataset2[j,i]>biggest:
       secondbiggest=dataset2[j,i]
       id2=i
  if(biggest<0.9)&(secondbiggest>0.99):
      print(biggest)
      print(id1)
      print(secondbiggest)
      print(id2)
      print(dataset1[j,0])
      print(dataset2[j,0])
      

