#!/usr/bin/env python
# coding: utf-8

# # A content based approach
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


# ## Recommender using a content based approach
# 
# In this notebook, the goal will be to recommend the most similar items to the ones seen in the session.
# For this, we first define our distance function between two items. The computed distance is simply the number of non-equal features (as the feature values are only categorical).

# In[16]:


def item_dist(item_id1, item_id2):
    item1_row = items_processed_array[item_id2index[item_id1]]
    item2_row = items_processed_array[item_id2index[item_id2]]
    
    diff = item1_row - item2_row
    dist = np.sum(diff != 0)
    return dist


# In[17]:


item_dist(3,4)


# In[18]:


(items_processed_array[item_id2index[3]] != items_processed_array[item_id2index[4]]).sum()


# In[19]:


items_processed_array[item_id2index[4]]


# Now we precomute the distances for each pair `(item, candidate_item)` and store it in a dictionary `item_dist_dict`.

# In[20]:


def precompute_distances():
    item_dist_dict = {}
    for candidate_item in candidate_items:
        for item in all_items:
            item_dist_dict[(item, candidate_item)] = item_dist(item, candidate_item)
    return item_dist_dict

item_dist_dict = precompute_distances()


# ## Prediction on test data

# Let us read in the test data set.

# In[21]:


import pandas as pd
import os
import numpy as np

base_path_test = "~/shared/data/project/test"


test_df = pd.read_csv(os.path.join(base_path_test, "test_sessions.csv"))

test_df


# Now we define a function which recommends the top 100 candidate items for a given session.
# 
# To do so, for each candidate item and each item in the session, the distance is retrieved from the `item_dist_dict`. In the end, all those distances will be summed up for the session. Thus, if a session has `k` items, we retrieve for each candidate item `k` distances and take the sum of these `k` distances. This way, we get a distance between the session and each candidate item.
# 
# Finally we order the candidate items by ascending distance between session and candidate item. Thus, we will get the closest items to that session.

# In[22]:


def compute_session_prediction_test(session_id):
    session_df_trial = test_df[test_df.session_id==session_id]
    items_in_session = list(session_df_trial["item_id"])
    items_in_session
    candidate_rank_dict = {}
    for candidate_item in candidate_items:
        distance = 0
        for item_in_session in items_in_session:
            distance += item_dist_dict[(item_in_session, candidate_item)]
        candidate_rank_dict[candidate_item] = distance

    candidate_rank_df = pd.DataFrame(candidate_rank_dict.items(), columns = ["item_id", "score"])
    candidate_rank_df = candidate_rank_df.sort_values("score", ascending=True).head(100).reset_index(drop=True)
    candidate_rank_df["rank"] = candidate_rank_df.index + 1
    candidate_rank_df["session_id"] = session_id
    candidate_rank_df = candidate_rank_df.drop("score", axis=1)
    
    return candidate_rank_df


# To finally perform our prediction on the test dataset, we iterate over all sessions and perform the ranking described above.

# In[23]:


def predict_cb():
    session_ids = test_df.session_id.unique()
    
    out_df = compute_session_prediction_test(session_ids[0])
    for session_id in session_ids[1:]:
        candidate_rank_df = compute_session_prediction_test(session_id)
        out_df = pd.concat([out_df, candidate_rank_df])
    
    return out_df[["session_id", "item_id", "rank"]]


# In[24]:


from datetime import datetime


# Let us run the prediction and write the results to a csv file..

# In[25]:


print(datetime.now())
out_df = predict_cb()
print(datetime.now())
out_df


# In[26]:


out_df.to_csv("results_content_based.csv", index=False)


# In[ ]:




