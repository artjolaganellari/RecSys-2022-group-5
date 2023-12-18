#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[19]:


import pandas as pd
import os
import numpy as np


# In[20]:


base_path_train = "~/shared/data/project/training"


items_df = pd.read_csv(os.path.join(base_path_train, "item_features.csv"))
purchase_df = pd.read_csv(os.path.join(base_path_train, "train_purchases.csv"))
session_df = pd.read_csv(os.path.join(base_path_train, "train_sessions.csv"))


# In[21]:


items_df


# In[22]:


items_df.item_id.nunique()


# In[23]:


purchase_df


# In[24]:


session_df


# In[25]:


purchase_df_processed = purchase_df.copy()
purchase_df_processed["was_bought"] = True

session_df_processed = session_df.copy()
session_df_processed["was_bought"] = False
df_processed = pd.concat([purchase_df_processed, session_df_processed]).sort_values(["session_id", "date"])
df_processed


# In[26]:


items_processed_df = items_df.pivot_table(values='feature_value_id', index='item_id', columns='feature_category_id').reset_index()
items_processed_df.index.names = ['index']
items_processed_df.columns = ["item_id"] + [f"item_feature_{x}" for x in list(range(73))]
items_processed_df


# In[27]:


df_processed = df_processed.merge(items_processed_df, how="left", on="item_id")
df_processed["was_bought"] = df_processed["was_bought"].astype(float)
df_processed


# In[28]:


items_processed_df = items_processed_df.fillna(0)
items_processed_df


# In[29]:


item_id2index = dict(zip(items_processed_df.item_id, items_processed_df.index))


# In[30]:


all_items = list(items_processed_df["item_id"])


# In[31]:


items_processed_array = np.array(items_processed_df.drop("item_id",axis=1))
items_processed_array[item_id2index[2]]


# In[32]:


candidate_items = list(pd.read_csv("candidate_items.csv")["item_id"])
candidate_items[:10]


# ## Recommender using item-item CF

# In[33]:


def item_dist(item_id1, item_id2):
    item1_row = items_processed_array[item_id2index[item_id1]]
    item2_row = items_processed_array[item_id2index[item_id2]]
    
    diff = item1_row[1:] - item2_row[1:]
    dist = np.sum(diff != 0)
    return dist


# In[34]:


item_dist(3,4)


# In[35]:


def precompute_distances():
    item_dist_dict = {}
    for candidate_item in candidate_items:
        for item in all_items:
            item_dist_dict[(item, candidate_item)] = item_dist(item, candidate_item)
    return item_dist_dict

item_dist_dict = precompute_distances()


# In[36]:


def compute_session_prediction(session_id):
    session_df_trial = session_df[session_df.session_id==session_id]
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
    return candidate_rank_df


# In[37]:


candidate_rank_df = compute_session_prediction(19)
candidate_rank_df


# In[38]:


recommended_items_iicf = list(candidate_rank_df["item_id"])


# # Reading in test data

# In[39]:


import pandas as pd
import os
import numpy as np

base_path_test = "~/shared/data/project/test"


test_df = pd.read_csv(os.path.join(base_path_test, "test_sessions.csv"))

test_df


# In[53]:


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


# In[64]:


def predict_iicf():
    session_ids = test_df.session_id.unique()
    
    out_df = compute_session_prediction_test(session_ids[0])
    for session_id in session_ids[1:]:
        candidate_rank_df = compute_session_prediction_test(session_id)
        out_df = pd.concat([out_df, candidate_rank_df])
    
    return out_df[["session_id", "item_id", "rank"]]


# In[65]:


from datetime import datetime


# In[66]:


print(datetime.now())
out_df = predict_iicf()
print(datetime.now())
out_df


# In[67]:


out_df.to_csv("results_content_based.csv", index=False)


# In[ ]:




