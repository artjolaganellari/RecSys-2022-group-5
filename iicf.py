#!/usr/bin/env python
# coding: utf-8

# # Item-item collaborative filtering
# RecSys Challenge 2022 - Group 5

# ## General Data Preparation

# Let us read in the different datasets.

# In[1]:


import pandas as pd
import os
import numpy as np


# In[2]:


base_path_train = "~/shared/data/project/training"

items_df = pd.read_csv(os.path.join(base_path_train, "item_features.csv"))
purchase_df = pd.read_csv(os.path.join(base_path_train, "train_purchases.csv"))
session_df = pd.read_csv(os.path.join(base_path_train, "train_sessions.csv"))


# In[3]:


items_df


# In[4]:


items_df.item_id.nunique()


# In[5]:


purchase_df


# In[6]:


session_df


# Now we combine the views inside a session and the purchases of this session in one dataframe, with the column `was_bought` indicating whether the item was only viewed or bought.

# In[7]:


purchase_df_processed = purchase_df.copy()
purchase_df_processed["was_bought"] = True

session_df_processed = session_df.copy()
session_df_processed["was_bought"] = False
df_processed = pd.concat([purchase_df_processed, session_df_processed]).sort_values(["session_id", "date"])
df_processed


# Now we denormalize the item features table, to have a more handy representation of the item features

# In[8]:


items_processed_df = items_df.pivot_table(values='feature_value_id', index='item_id', columns='feature_category_id').reset_index()
items_processed_df.index.names = ['index']
items_processed_df.columns = ["item_id"] + [f"item_feature_{x}" for x in list(range(73))]
items_processed_df


# The item features can now be merged to the combined dataset with session views and purchases from above.
# Also NULL values are filled by 0.

# In[9]:


df_processed = df_processed.merge(items_processed_df, how="left", on="item_id")
df_processed["was_bought"] = df_processed["was_bought"].astype(float)
df_processed


# In[10]:


items_processed_df = items_processed_df.fillna(0)
items_processed_df


# In[11]:


item_id2index = dict(zip(items_processed_df.item_id, items_processed_df.index))


# In[12]:


all_items = list(items_processed_df["item_id"])


# In[13]:


items_processed_array = np.array(items_processed_df.drop("item_id",axis=1))
items_processed_array[item_id2index[2]]


# In[14]:


items_processed_array[item_id2index[2]].shape


# Next we read in the candidate items.

# In[15]:


candidate_items = list(pd.read_csv("candidate_items.csv")["item_id"])
candidate_items[:10]


# ## Item-item collaborative filtering specific data preparation

# Similar to assignment 1 from the course, we will prepare our data for item-item collaborative filtering.
# 
# For this we also include the test data, in order to be able to do the predictions in the end for these test sessions.
# 
# In order to handle implicit feedback, we will set the rating of bought items to 1 and the rating of seen but not bought items to 0.5. The idea behind this is that seen items shall have a higher rating then non-seen items (which were not of interest in this session). This way, we capture how interesting an item was with respect to a session.

# First we load necessary modules.

# In[16]:


import os
import csv
import pandas as pd
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import sklearn.preprocessing as pp


# Now we read in the test data and set the rating of seen items to 0.5.

# In[17]:


base_path_test = "~/shared/data/project/test"


test_df = pd.read_csv(os.path.join(base_path_test, "test_sessions.csv"))


# In[18]:


test_df["rating"]=0.5
test_df = test_df.drop("date", axis=1)


# In[19]:


test_sessions = test_df.session_id.unique()
len(test_sessions)


# For the training data we set the rating of seen (but not bought) items to 0.5, and the rating of purchased items to 1.

# In[20]:


df_processed.loc[df_processed.was_bought==True, "rating"] = 1
df_processed.loc[df_processed.was_bought==False, "rating"] = 0.5
df_processed = pd.concat([df_processed, test_df])


# In[21]:


ratings_raw = df_processed[["session_id", "item_id", "rating"]]
ratings_raw


# As in assignement 1, a ratings dataframe is built. To make life easier, we convert the session and item IDs to consecutive numbers.

# In[22]:


itemIds = ratings_raw.item_id.unique()
itemIds.sort()
sessionIds = ratings_raw.session_id.unique()
sessionIds.sort()

## create internal ids for movies and users, that have consecutive indexes starting from 0
itemId_to_itemIDX = dict(zip(itemIds, range(0, itemIds.size)))
itemIDX_to_itemId = dict(zip(range(0, itemIds.size), itemIds))

sessionId_to_sessionIDX = dict(zip(sessionIds, range(0, sessionIds.size )))
sessionIDX_to_sessionId = dict(zip(range(0, sessionIds.size), sessionIds))

## drop timestamps
ratings = pd.concat([ratings_raw['session_id'].map(sessionId_to_sessionIDX), ratings_raw['item_id'].map(itemId_to_itemIDX), ratings_raw['rating']], axis=1)
ratings.columns = ['session', 'item', 'rating']

ratings.head()


# In[23]:


test_sessions_idx = [sessionId_to_sessionIDX[s] for s in test_sessions]
candidate_items_idx = [itemId_to_itemIDX[i] for i in candidate_items if i in itemId_to_itemIDX.keys()]


# Now we build the session-item matrix `R`, and compute the item and session averages.

# In[24]:


R = sp.csr_matrix((ratings.rating, (ratings.session, ratings.item)))
R_dok = R.todok()

m = R.shape[0]
n = R.shape[1]
numRatings = R.count_nonzero()

print("There are", m, "sessions,", n, "items and", numRatings, "ratings.")


# In[25]:


item_sums = R.sum(axis=0).A1 ## matrix converted to 1-D array via .A1
item_cnts = (R != 0).sum(axis=0).A1
item_avgs = item_sums / item_cnts
print("item_avgs", item_avgs)


# In[26]:


session_sums = R.sum(axis=1).A1 ## matrix converted to 1-D array via .A1
session_cnts = (R != 0).sum(axis=1).A1
session_avgs = session_sums / session_cnts
print("session_avgs", session_avgs)


# ## Computing item similarities

# As indicated in lecture 4, slide 8, it makes sense to normalize user vectors when working with implicit data. Thus, we will do that in the next step.

# In[27]:


# normalizing
norms = np.array(np.sqrt(R.T.multiply(R.T).sum(axis=0)))
norms[norms == 0.0] = 0.00001 # avoid dividing by 0
norms = norms.reshape((1,m))

nc = np.repeat(norms, session_cnts)
nc = nc.reshape(R.data.shape)

R.data /= nc


# Similar to assignment 1, we write a function computing the item similarities for a given item ID. 

# In[28]:


def compute_item_similarities(i_id):
    iI = np.empty((n,))

    # YOUR CODE HERE
    Rc = R.copy()
    
    # mean-centering
    #mc = np.repeat(item_avgs, item_cnts)
    #Rc.data -= mc
    
    # extracting user similarities
    i = Rc[:,i_id]
    iI = np.array(Rc.T.dot(i).todense())
    iI = iI.reshape((n,))
    
    return iI


# In[29]:


iI = compute_item_similarities(121)
iI[10]


# To make our computations faster, we precompute item-similarities for the candidate items. The results are stored in a dictionary mapping the item ID to its item similarities.

# In[30]:


iI_dict = {}

for i in candidate_items_idx:
    iI = compute_item_similarities(i)
    iI_dict[i] = iI


# ## Prediction of implicit feedback for a specific (user, item) pair

# In the next step, we write a function which returns a rating prediction (prediction of implicit feedback). This is done according to lecture 10, slide 14: The "prediction for a target item is the sum of similarities of all current session items".

# In[31]:


## a default value
with_deviations = True

def predict_rating(u_id, i_id, have_rated_idx):
    iI = iI_dict[i_id]
    
    # create new iI and fill it with 0 and then fill it with iI for items having been rated by the session (i.e. which were seen)
    iI_new = np.full(iI.shape, 0)
    iI_new[have_rated_idx] = iI[have_rated_idx]
    
    prediction = iI_new.sum()
    
    return prediction


# In[32]:


import time

have_rated_idx = np.array([j_id for j_id in range(n) if (4, j_id) in R_dok])
start = time.time()
print(predict_rating(4, candidate_items_idx[1], have_rated_idx))
end = time.time()
end - start


# ## Prediction on test data

# Now we will write a function, which for a given session and each candidate item computes the predicted rating according to the above function. This score will then be used for ranking our candidate items.

# In[33]:


def predict_session(s):
    candidate_rank_dict = {}
    have_rated_idx = np.array([j_id for j_id in range(n) if (s, j_id) in R_dok])
    for i in candidate_items_idx:
        
        p = predict_rating(s, i, have_rated_idx)
        
        candidate_item = itemIDX_to_itemId[i]
        candidate_rank_dict[candidate_item] = p

    candidate_rank_df = pd.DataFrame(candidate_rank_dict.items(), columns = ["item_id", "score"])
    candidate_rank_df = candidate_rank_df.sort_values("score", ascending=False).head(100).reset_index(drop=True)
    candidate_rank_df["rank"] = candidate_rank_df.index + 1
    session_id = sessionIDX_to_sessionId[s]
    candidate_rank_df["session_id"] = session_id
    candidate_rank_df = candidate_rank_df.drop("score", axis=1)[["session_id", "item_id", "rank"]]
    return candidate_rank_df


# This function is applied for all test sessions and the result is written to a csv file.

# In[34]:


start = time.time()
candidate_rank_df_list = []
for s in test_sessions_idx:
    
    candidate_rank_df = predict_session(s)
    candidate_rank_df_list.append(candidate_rank_df)

candidate_rank_df = pd.concat(candidate_rank_df_list)
end = time.time()
print(end - start)

candidate_rank_df


# In[35]:


candidate_rank_df.to_csv("results_iicf.csv", index=False)


# In[36]:


# len(candidate_rank_df[candidate_rank_df.score == 0]) / len(candidate_rank_df)


# In[ ]:




