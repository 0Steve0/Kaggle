import pandas
import numpy as np
import math 

dataframe1 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Digit Recoginer/Digit Recognier Voting System/mnist-vggnet with 0.99300.csv")
dataset1 = dataframe1.values
dataframe2 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Digit Recoginer/Digit Recognier Voting System/mnist-vggnet with 0.99214.csv")
dataset2 = dataframe2.values
dataframe3 = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Digit Recoginer/Digit Recognier Voting System/mnist-vggnet with 0.99014.csv")
dataset3 = dataframe3.values
outcome = pandas.read_csv("C:/Users/sound/Desktop/Kaggle/Digit Recoginer/Digit Recognier Voting System/outcome.csv")
outcome=outcome.drop('ImageId',1)
outcomeset= outcome.values
print(outcomeset[0,0])
for j in range(0,27999):
  result1=dataset1[j,1]
  result2=dataset2[j,1]
  result3=dataset3[j,1]
  if((result1==result2)&(result1==result3)):
      outcomeset[j,0]=result1
  if((result1==result2)&(result1!=result3)):
      outcomeset[j,0]=result1
  if((result1==result3)&(result1!=result2)):
      outcomeset[j,0]=result1
  if((result2==result3)&(result1!=result2)):
      outcomeset[j,0]=result2
  if((result1!=result2)&(result2!=result3)&(result1!=result3)):
      outcomeset[j,0]=result1
      
#submission = pandas.DataFrame(dataset3, index=test_ids, columns=le.classes_)
#submission.to_csv('voting.csv')

np.savetxt('voting.csv', np.c_[range(1,len(outcomeset)+1),outcomeset], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')