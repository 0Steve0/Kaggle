import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from matplotlib import pyplot as plt
#import seaborn as sns
#%matplotlib
data_path = "C:/Users/sound/Desktop/Kaggle/Santender Recomendation/data/"
train = pd.read_csv(data_path+"train_ver2.csv")
#test = pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])
print("Number of rows in train : ", train.shape[0])
#print("Number of rows in test : ", test.shape[0])
print(train.head())