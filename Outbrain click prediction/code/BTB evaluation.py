import pandas as pd
import numpy as np 

reg =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # trying anokas idea of regularization
eval = True

train = pd.read_csv("C:/Users/sound/Desktop/Kaggle/Outbrain click prediction/data/clicks_train.csv")


ids = train.display_id.unique()#find the unique display of an array
ids = np.random.choice(ids, size=len(ids)//10, replace=False) #generate a random sample from a given 1-D array

valid = train[train.display_id.isin(ids)] #vaild size is 0.1 of total size 
train = train[~train.display_id.isin(ids)] #train size is 0.9 of total size

print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()

for i in range(0,21):
    print(reg[i])
    def get_prob(k):
        if k not in cnt:
            return 0
        return cnt[k]/(float(cntall[k]) + reg[i])

    def srt(x):
        ad_ids = map(int, x.split())
        ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
        return " ".join(map(str,ad_ids))
       

    from ml_metrics import mapk

    y = valid[valid.clicked==1].ad_id.values
    y = [[_] for _ in y]
    p = valid.groupby('display_id').ad_id.apply(list)
    p = [sorted(x, key=get_prob, reverse=True) for x in p]

    print (mapk(y, p, k=12))