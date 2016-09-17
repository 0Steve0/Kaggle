import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

def main():
    # loading training data
    print('Loading training data')
    data = pd.read_csv('C:/Users/sound/Desktop/Kaggle/Digit Recognier in Random Forest/data/train.csv')#read csv file into dataframe
    x_tr = data.values[:, 1:].astype(float)#The other columns are the pixels of each image and we have 28,000 images
    y_tr = data.values[:, 0]#The first column is the label that drawn by user

    scores = list()#list data type and it can be change
    scores_std = list()

    print('Start learning...')
    n_trees = [10, 20, 50, 100]#the number of random trees
    for n_tree in n_trees:
        print(n_tree)
        recognizer = RandomForestClassifier(n_tree)
        score = cross_val_score(recognizer, x_tr, y_tr)
        scores.append(np.mean(score))
        scores_std.append(np.std(score))#compute the standard deviation 

    sc_array = np.array(scores) #change a list to array
    std_array = np.array(scores_std)#numpy array 
    print('Score: ', sc_array)
    print('Std  : ', std_array)

    #plt.figure(figsize=(4,3))
    plt.plot(n_trees, scores)
    plt.plot(n_trees, sc_array + std_array, 'b--')
    plt.plot(n_trees, sc_array - std_array, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('# of trees')
    plt.savefig('cv_trees.png')
    # plt.show()


if __name__ == '__main__':
    main()