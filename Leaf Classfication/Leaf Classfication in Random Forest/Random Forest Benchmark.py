#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# load training data
traindata = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Leaf Classfication/data/train.csv')
x_train = traindata.values[:, 2:]
y_train = traindata.values[:, 1]

#set the number of trees in random forest
num_trees = [10, 50, 100, 200, 300, 400, 500]
#calculate the cross validation scores and std
cr_val_scores = list()
cr_val_scores_std = list()
for n_tree in num_trees:
  recognizer = RandomForestClassifier(n_tree)
  cr_val_score = cross_val_score(recognizer, x_train, y_train)
  cr_val_scores.append(np.mean(cr_val_score))
  cr_val_scores_std.append(np.std(cr_val_score))
  
#plot cross_val_score and std
sc_array = np.array(cr_val_scores)
std_array = np.array(cr_val_scores_std) 
plt.plot(num_trees, cr_val_scores)
plt.plot(num_trees, sc_array + std_array, 'b--')
plt.plot(num_trees, sc_array - std_array, 'b--')
plt.ylabel('cross_val_scores')
plt.xlabel('num_of_trees')
plt.savefig('random_forest_benchmark.png')

