#!/usr/bin/env python
# coding: utf-8

# # An LSTM based approach
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


# Next we read in the candidate items.

# In[14]:


candidate_items = list(pd.read_csv("candidate_items.csv")["item_id"])
candidate_items[:10]


# ## LSTM specific Data Preparation
# 
# ### Item sequence preparation

# Similarly to predicting the next word of a sentence, we can predict the "next item" in a session. For this, we see the viewed (but not purchased) items as a given sequence of input words, and the purchased items as output words. Now we want to transform the data into a form reflecting this idea.
# 
# First, let us prepare the training data for the LSTM. For this, we select a subset of the training dataset, so that RAM is not exceeded.

# In[15]:


session_ids_cand = df_processed[(df_processed.was_bought==1) & (df_processed.item_id.isin(candidate_items))].session_id


# In[16]:


#df_w2v = df_processed[df_processed.session_id < 10000].sort_values(["session_id", "date"])[["session_id", "item_id", "was_bought"]]
df_w2v = df_processed[(df_processed.session_id.isin(session_ids_cand)) & (df_processed.session_id < 50000)].sort_values(["session_id", "date"])[["session_id", "item_id", "was_bought"]]
df_w2v


# In the next step, we just map the item IDs to a subsequent list of numbers (without wholes)

# In[17]:


item_list = list(df_w2v.item_id.sort_values().unique())
item_dict = {v: k for k, v in dict(zip(range(len(item_list)), item_list)).items()}
item_dict[4]


# In[18]:


item_dict_rev = {v: k for k, v in item_dict.items()}


# In the next step, we extract the viewed items (corresponding to the previous words) and purchased items (corresponding to the next words).

# In[19]:


def get_values(x):
    x = x["item_id"]
    return [item_dict[x] for x in list(x.values.ravel())]

all_words = df_w2v.groupby('session_id').apply(get_values).to_list()

def get_values(x):
    x = x["item_id"]
    return [item_dict[x] for x in list(x.values.ravel())]

prev_words = df_w2v[(df_w2v.was_bought==0)].groupby('session_id').apply(get_values).to_list()

def get_values(x):
    x = x["item_id"]
    return item_dict[x.values.ravel()[0]]

next_words = df_w2v[(df_w2v.was_bought==1)].groupby('session_id').apply(get_values).to_list()


# In[20]:


prev_words[:5]


# In[21]:


next_words[:5]


# In[22]:


unique_words = [item_dict[x] for x in df_w2v["item_id"].unique()]
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
len(unique_words)


# Sessions do usually have different lengths. Thus we take the longest session as standard session length and pad the shorter sessions with zeros in the beginning, as follows:

# In[23]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[24]:


max_seq_length = max([len(x) for x in prev_words])
input_seqs = np.array(pad_sequences(prev_words, maxlen=max_seq_length, padding='pre'))

print(max_seq_length)
print(input_seqs[:5])


# ### Embedding Matrix Preparation with Word2Vec
# 
# To reprensent each item by some feature vector, we use Word2Vec embedding. From this we build an embedding matrix.

# In[25]:


#get_ipython().system('pip install gensim')


# In[26]:


from gensim.models.word2vec import Word2Vec


# In[27]:


w2v = Word2Vec(all_words, min_count=1)


# In[28]:


#list the vocabulary words
words = list(w2v.wv.index_to_key)

print(words[:5])


# In[29]:


my_dict = dict({})
for idx, key in enumerate(w2v.wv.index_to_key):
    my_dict[key] = w2v.wv[key]
    # Or my_dict[key] = model.wv.get_vector(key)
    # Or my_dict[key] = model.wv.word_vec(key, use_norm=False)


# In[30]:


embeddings_matrix = np.zeros((len(words), 100))

for i, word in enumerate(words):
    embedding_vector = my_dict[word]
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


# In[31]:


embeddings_matrix


# In[32]:


embeddings_matrix.shape


# ## Defining and training the LSTM

# Let us define our LSTM model.

# In[33]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD


# In[34]:


model = tf.keras.Sequential(
    [tf.keras.layers.Embedding(input_dim = len(words), output_dim=100, weights=[embeddings_matrix], input_length=max_seq_length, trainable=False),
     tf.keras.layers.LSTM(256),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(len(words) , activation='softmax')
    ])

opt = SGD(lr=10**(-6))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Also, we convert our output (i.e. the purchased items) to a categorical object.

# In[35]:


next_words = np.array(next_words)
outputs = tf.keras.utils.to_categorical(next_words, num_classes=len(words))
outputs.shape


# Now, we train our model.

# In[36]:


tf.config.run_functions_eagerly(True)


# In[37]:


history = model.fit(input_seqs, outputs, epochs=10, validation_split=0.2, verbose=1, batch_size=256)


# ## Prediction on test data

# Let us read in the test data set.

# In[38]:


import pandas as pd
import os
import numpy as np

base_path_test = "~/shared/data/project/test"


test_df = pd.read_csv(os.path.join(base_path_test, "test_sessions.csv"))

test_df


# In order to not exceed the RAM, we split the test dataframe into sub-dataframes, which will be predicted batch by batch.

# In[39]:


session_id_limits = list(range(0, test_df.session_id.max() + 10000000, 10000000))


# In[40]:


test_dfs = []
for i in range(len(session_id_limits)-1):
    session_id_limit1 = session_id_limits[i]
    session_id_limit2 = session_id_limits[i+1]
    test_df_small = test_df[(test_df.session_id >= session_id_limit1) & (test_df.session_id < session_id_limit2)]
    test_dfs.append(test_df_small)


# In[41]:


len(test_dfs)


# In[42]:


def get_values(x):
    x = x["item_id"]
    return [item_dict[x] for x in list(x.values.ravel()) if x in item_dict.keys()]


# In[43]:


candidate_items_keys = [item_dict[x] for x in candidate_items if x in item_dict.keys()]
candidate_items_keys[:4]


# Now we predict using our LSTM model. For this we need to prepare the data in the same fashion as above.
# From the result, we extract the 100 items with the largest scores.

# In[44]:


candidate_rank_dfs = []
j = 0
for test_df_small in test_dfs:
    print(j)
    j+=1
    
    # data preparation
    prev_words_test = test_df_small.groupby('session_id').apply(get_values).to_list()
    input_seqs_test = np.array(pad_sequences(prev_words_test, maxlen=max_seq_length, padding='pre'))
    
    # lstm prediction
    preds = model.predict(input_seqs_test)

    # sorting the predictions by highest score
    arr = preds.argsort().astype("float32")
    # retrieving only the candidate items
    cond = np.isin(arr.astype(int), candidate_items_keys)
    arr[~cond] = np.nan
    arrlist = arr.tolist()
    
    # creating the ranks
    for i in range(len(test_df_small.session_id.unique())):
        rank_dict = {}
        session_id  = test_df_small.session_id.unique()[i]

        # get the scores from the lecture
        scores = arrlist[i]
        y = [x for x in scores if x>=0 and x in item_dict.keys()] # x>=0 removes the nan values
        ranked_list = [item_dict_rev[x] for x in y[-100:]] # get the top 100 items

        for num, item in enumerate(ranked_list):
            rank_dict[item] = len(ranked_list) - num # this is just mapping the item to its rank
        
        candidate_rank_df = pd.DataFrame(rank_dict.items(), columns = ["item_id", "rank"]).sort_values("rank")
        candidate_rank_df["session_id"] = session_id
        candidate_rank_dfs.append(candidate_rank_df)
    
candidate_rank_df = pd.concat(candidate_rank_dfs)
candidate_rank_df = candidate_rank_df[["session_id", "item_id", "rank"]].reset_index(drop=True)
candidate_rank_df


# In[45]:


candidate_rank_df


# Finally, we write the result to a csv file.

# In[46]:


candidate_rank_df.to_csv("results_lstm_word2vec.csv", index=False)


# ## Final checks

# Number of predicted sessions.

# In[47]:


len(candidate_rank_df.session_id.unique())


# Number of recommendations of non-candidate items. Should be 0.

# In[48]:


(~candidate_rank_df.item_id.isin(candidate_items)).sum()


# Number of sessions to predict.

# In[49]:


len(test_df.session_id.unique())


# List of items ranked first at least once.

# In[50]:


candidate_rank_df[candidate_rank_df["rank"] == 1].item_id.unique()


# In[ ]:




