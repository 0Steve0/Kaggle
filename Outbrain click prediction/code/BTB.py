import pandas as pd
import numpy as np 

train = pd.read_csv("C:/Users/sound/Desktop/Kaggle/Outbrain click prediction/data/clicks_train.csv")
#count all the clicked ad id in train 
cnt = train[train.clicked==1].ad_id.value_counts()
print(cnt)
#count all the ad id in train including the duplicate one 
cntall = train.ad_id.value_counts()
print(cntall)

#calculate the probability of ad id
#given a specific ad id calculate its probability

def get_prob(k):
    if k not in cnt:
        return 0
    if((cnt[k]/float(cntall[k]))>0.9):
       print(cnt[k]/float(cntall[k]))
    return cnt[k]/float(cntall[k])
#print(count)
print(get_prob) #give 0 0 times of click
#print(get_prob(273567)) give 1 1 times of click and exist 1 time in train 

#sort the ad_ids based on the probability that exist in train id
def srt(x):
    ad_ids = map(int, x.split())#splite the raw input 1 2 3 4 >>[1,2,3,4]
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)#sort under the probability
    return " ".join(map(str,ad_ids))#join return the string to display
 
 
#subm = pd.read_csv("C:/Users/sound/Desktop/Kaggle/Outbrain click prediction/data/sample_submission.csv") 
#python supports the creation of anonymous functions at runtime
#subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))#applies function along input axis of DataFrame 
#subm.to_csv("subm_1prob.csv", index=False)