#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
import os
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm


# # Preparations

# In[2]:


# Load the data
base_path_train = "~/shared/data/project/training"

items_df = pd.read_csv(os.path.join(base_path_train, "item_features.csv"))
purchase_df = pd.read_csv(os.path.join(base_path_train, "train_purchases.csv"))
session_df = pd.read_csv(os.path.join(base_path_train, "train_sessions.csv"))

base_path_test = "~/shared/data/project/test"

test_df = pd.read_csv(os.path.join(base_path_test, "test_sessions.csv"))


# In[3]:


# Extend the purchases and sessions dataframes with bought information
purchase_df_processed = purchase_df.copy()
purchase_df_processed["was_bought"] = 1

session_df_processed = session_df.copy()
session_df_processed["was_bought"] = 0


# In[4]:


# Combine the purchases and sessions dataframes into one
df_processed = pd.concat([purchase_df_processed, session_df_processed]).sort_values(["session_id", "date"])
df_processed


# In[5]:


# Add a new column "categorized_feature" to items_df which holds the category ID and the feature value together as one value 
items_df["categorized_feature"] = items_df["feature_category_id"] * 10000 + items_df["feature_value_id"]
items_df


# In[6]:


# The "categorized feature" should be pivoted in order to have one item with its features / row
items_processed_df = items_df.pivot_table(values='categorized_feature', index='item_id', columns='feature_category_id').reset_index()
items_processed_df.index.names = ['index']
column_list = [f"item_feature_{x+1}" for x in list(range(73))]

items_processed_df.columns = ["item_id"] + column_list
items_processed_df


# In[7]:


# The session data should be extended with the "categorized features"
# The goal is to compare the sessions based on the viewed items' features
df_processed2 = df_processed.merge(items_processed_df, how="left", on="item_id")
df_processed2["was_bought"] = df_processed2["was_bought"].astype(float)
df_processed2


# In[8]:


# Create a distinct list of item features per session to be able to compare the sessions based on the item features 
df_processed2_melted = df_processed2\
.melt(id_vars=['session_id'], value_vars=column_list)[["session_id","value"]]\
.dropna().drop_duplicates()

df_processed2_melted


# In[9]:


# Prepare the test dataset
test_df_merged = test_df.merge(items_processed_df, how="left", on="item_id")
test_df_merged


# In[10]:


# Create a distinct list of item features per session to be able to compare the sessions based on the item features 
test_df_melted = test_df_merged\
.melt(id_vars=['session_id'], value_vars=column_list)[["session_id","value"]]\
.dropna().drop_duplicates()

test_df_melted


# # Prototype approach with the session_id 3234

# In[11]:


# The "target" session(s) are outer joined with the "train" sessions on the "categorized feature" column
joined = test_df_melted[test_df_melted["session_id"] == 3234].merge(df_processed2_melted, how="outer", on="value")
joined.rename(columns={'value': 'categorized_feature'}, inplace=True)
joined


# In[12]:


# The joined dataframe will be grouped on the "train" sessions, so we will have the 
#  * count of the common distinct categorized features between "target" and "train" sessions ( = CCFta)
#  * count of the distinct categorized features in the "train" sessions ( = CFtr )
# the count of the distinct categorized features in the "target" session(s) are knew/should be done separately. ( = CFta)
#
# The similarity is calculated as "CCFta" / ("CFtr + CFta - CCFta") 
#
# The dataframe ordered by similarity

joined_grouped = joined.groupby(['session_id_y']).count()

joined_grouped.rename(columns={'session_id_x': 'count_common_features'}, inplace=True)
joined_grouped.rename(columns={'categorized_feature': 'count_features_id_y'}, inplace=True)

joined_grouped['session_id_x'] = 3234
joined_grouped['count_features_id_x'] = test_df_melted[test_df_melted["session_id"] == 3234].shape[0] 

joined_grouped['similarity'] = \
    joined_grouped['count_common_features'] \
    /   (joined_grouped['count_features_id_y'] \
        + joined_grouped['count_features_id_x'] \
        - joined_grouped['count_common_features']\
        )

joined_grouped.sort_values(by=['similarity'], inplace=True, ascending=False)
joined_grouped


# In[13]:


# The first 100 rows will be taken from the ordered dataframe which holds the similarities between the "target" and "train" sessions

top100 = joined_grouped.head(100)
top100 = top100.reset_index()
top100.rename(columns={'session_id_y': 'session_id'}, inplace=True)
top100.insert(0, 'rank', range(1, 1 + len(top100)))
top100


# In[14]:


# The top 100 similar sessions needs to be extended with the purchase informations

results = top100.merge(purchase_df_processed, how="inner", on="session_id")
results


# In[15]:


# The recommendations are

recommendation = results[['session_id_x','item_id','rank']].copy()
recommendation.columns = ['session_id','item_id','rank']
# recommendation
recommendation.to_csv("results_uucf.csv", index=False)


# # Full approach

# In[ ]:


# This approach runs out from memory

# joined = test_df_melted.merge(df_processed2_melted, how="outer", on="value")


# In[23]:


# This approach runs out of time as it would need about 166 hours to complete as each session needs 12 seconds, and there is 50k sessions

# start = time.time()

# li_df = []

# sessions = test_df_melted.session_id.unique()

# for session_id in sessions[:3]:
#     joined = test_df_melted[test_df_melted['session_id'] == session_id].merge(df_processed2_melted, how="outer", on="value")
#     joined.rename(columns={'value': 'categorized_feature'}, inplace=True)

#     processed = joined.groupby(['session_id_y']).count()
    
#     processed.rename(columns={'session_id_x': 'count_common_features'}, inplace=True)
#     processed.rename(columns={'categorized_feature': 'count_features_id_y'}, inplace=True)
    
#     processed['session_id_x'] = session_id
#     processed['count_features_id_x'] = test_df_melted[test_df_melted["session_id"] == session_id].shape[0]

#     processed['similarity'] = processed['count_common_features'] / (processed['count_features_id_y'] + test_df_melted[test_df_melted['session_id'] == session_id].shape[0] - processed['count_common_features'])
#     processed.sort_values(by=['similarity'], inplace=True, ascending=False)
    
#     top100 = processed.head(100)
#     top100 = top100.reset_index()
#     top100.rename(columns={'session_id_y': 'session_id'}, inplace=True)
#     top100.insert(0, 'rank', range(1, 1 + len(top100)))
    
#     tmp = top100.merge(purchase_df_processed, how="inner", on="session_id")
    
#     li_df.append(tmp)

# result = pd.concat(li_df, axis=0, ignore_index=True)
    
# end = time.time()

# print(end - start)

# result[['session_id_x','item_id','rank']]


# In[ ]:


# This approach runs out from disk space

# df_processed2_melted.to_csv("yourdata2.csv")
# df2_key = df_processed2_melted.value

# creating a empty bucket to save result
# df_result = pd.DataFrame(columns=(test_df_melted.columns.append(df_processed2_melted.columns)).unique())
# df_result.to_csv("df3.csv",index_label=False)

# deleting df2 to save memory
# del(df_processed2_melted)

# def preprocess(x):
#     tmp=pd.merge(test_df_melted, x, how="outer", on="value")
#     tmp.to_csv("df3.csv",mode="a",header=False,index=False)

# reader = pd.read_csv("yourdata2.csv", chunksize=1000) # chunksize depends with you colsize

# [preprocess(r) for r in reader]

