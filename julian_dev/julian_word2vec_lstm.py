#!/usr/bin/env python
# coding: utf-8

# # Preparation

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


# In[7]:


purchase_df_processed = purchase_df.copy()
purchase_df_processed["was_bought"] = True

session_df_processed = session_df.copy()
session_df_processed["was_bought"] = False
df_processed = pd.concat([purchase_df_processed, session_df_processed]).sort_values(["session_id", "date"])
df_processed


# In[8]:


items_processed_df = items_df.pivot_table(values='feature_value_id', index='item_id', columns='feature_category_id').reset_index()
items_processed_df.index.names = ['index']
items_processed_df.columns = ["item_id"] + [f"item_feature_{x}" for x in list(range(73))]
items_processed_df


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


candidate_items = list(pd.read_csv("candidate_items.csv")["item_id"])
candidate_items[:10]


# # Word2vec

# In[15]:


session_ids_cand = df_processed[(df_processed.was_bought==1) & (df_processed.item_id.isin(candidate_items))].session_id


# In[16]:


#df_w2v = df_processed[df_processed.session_id < 10000].sort_values(["session_id", "date"])[["session_id", "item_id", "was_bought"]]
df_w2v = df_processed[(df_processed.session_id.isin(session_ids_cand)) & (df_processed.session_id < 50000)].sort_values(["session_id", "date"])[["session_id", "item_id", "was_bought"]]
df_w2v


# In[17]:


item_list = list(df_w2v.item_id.sort_values().unique())
item_dict = {v: k for k, v in dict(zip(range(len(item_list)), item_list)).items()}
item_dict[4]


# In[18]:


item_dict_rev = {v: k for k, v in item_dict.items()}


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


# In[23]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[24]:


max_seq_length = max([len(x) for x in prev_words])
input_seqs = np.array(pad_sequences(prev_words, maxlen=max_seq_length, padding='pre'))

print(max_seq_length)
print(input_seqs[:5])


# In[ ]:





# In[ ]:





# In[25]:


from gensim.models.word2vec import Word2Vec


# In[26]:


w2v = Word2Vec(all_words, min_count=1)


# In[27]:


#list the vocabulary words
words = list(w2v.wv.index_to_key)

print(words[:5])


# In[28]:


my_dict = dict({})
for idx, key in enumerate(w2v.wv.index_to_key):
    my_dict[key] = w2v.wv[key]
    # Or my_dict[key] = model.wv.get_vector(key)
    # Or my_dict[key] = model.wv.word_vec(key, use_norm=False)


# In[29]:


embeddings_matrix = np.zeros((len(words), 100))

for i, word in enumerate(words):
    embedding_vector = my_dict[word]
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


# In[30]:


embeddings_matrix


# In[31]:


embeddings_matrix.shape


# In[32]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD


# In[33]:


model = tf.keras.Sequential(
    [tf.keras.layers.Embedding(input_dim = len(words), output_dim=100, weights=[embeddings_matrix], input_length=max_seq_length, trainable=False),
     tf.keras.layers.LSTM(256),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(len(words) , activation='softmax')
    ])

opt = SGD(lr=10**(-6))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[34]:


input_seqs.shape


# In[35]:


next_words = np.array(next_words)


# In[36]:


next_words.shape


# In[37]:


next_words


# In[38]:


outputs = tf.keras.utils.to_categorical(next_words, num_classes=len(words))
outputs.shape


# In[39]:


tf.config.run_functions_eagerly(True)


# In[40]:


history = model.fit(input_seqs, outputs, epochs=10, validation_split=0.2, verbose=1, batch_size=256)


# In[41]:


df_w2v_test = df_processed[(df_processed.session_id > 50000) & (df_processed.session_id < 51000)].sort_values(["session_id", "date"])[["session_id", "item_id", "was_bought"]]
df_w2v_test


# In[42]:


def get_values(x):
    x = x["item_id"]
    return [item_dict[x] for x in list(x.values.ravel()) if x in item_dict.keys()]

prev_words_test = df_w2v_test[(df_w2v_test.was_bought==0)].groupby('session_id').apply(get_values).to_list()


# In[43]:


prev_words_test


# In[44]:


input_seqs_test = np.array(pad_sequences(prev_words_test, maxlen=max_seq_length, padding='pre'))


# In[45]:


input_seqs_test


# In[46]:


preds = model.predict(input_seqs_test)


# In[47]:


preds.shape


# In[48]:


preds


# In[49]:


input_seqs_test.shape


# In[50]:


pd.Series(np.argmax(preds, axis=-1)).unique()


# In[51]:


preds.argsort()[:,-100:]


# In[52]:


arr = preds.argsort().astype("float32")


# In[53]:


arr


# In[54]:


len(candidate_items)


# In[55]:


candidate_items_keys = [item_dict[x] for x in candidate_items if x in item_dict.keys()]
candidate_items_keys[:4]


# In[56]:


len(candidate_items_keys)


# In[57]:


cond = np.isin(arr.astype(int), candidate_items_keys)
cond


# In[58]:


arr[~cond] = np.nan
arr


# In[59]:


df_w2v_test.session_id.unique()[:3]


# In[60]:


arrlist = arr.tolist()
candidate_rank_dfs = []

for i in range(3):
    rank_dict = {}
    session_id  = df_w2v_test.session_id.unique()[i]
    
    my_list = arrlist[i]
    y = [x for x in my_list if x>=0 and x in item_dict.keys()]
    ranked_list = [item_dict_rev[x] for x in y[-100:]]
    
    for num, item in enumerate(ranked_list):
        rank_dict[item] = len(ranked_list) - num
    candidate_rank_df = pd.DataFrame(rank_dict.items(), columns = ["item_id", "rank"]).sort_values("rank")
    candidate_rank_df["session_id"] = session_id
    candidate_rank_dfs.append(candidate_rank_df)
    
candidate_rank_df = pd.concat(candidate_rank_dfs)
candidate_rank_df = candidate_rank_df[["session_id", "item_id", "rank"]].reset_index(drop=True)
candidate_rank_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Test data

# In[61]:


import pandas as pd
import os
import numpy as np

base_path_test = "~/shared/data/project/test"


test_df = pd.read_csv(os.path.join(base_path_test, "test_sessions.csv"))

test_df.session_id.max()


# In[62]:


session_id_limits = list(range(0, test_df.session_id.max() + 10000000, 10000000))


# In[63]:


test_dfs = []
for i in range(len(session_id_limits)-1):
    session_id_limit1 = session_id_limits[i]
    session_id_limit2 = session_id_limits[i+1]
    test_df_small = test_df[(test_df.session_id >= session_id_limit1) & (test_df.session_id < session_id_limit2)]
    test_dfs.append(test_df_small)


# In[64]:


len(test_df)


# In[65]:


def get_values(x):
    x = x["item_id"]
    return [item_dict[x] for x in list(x.values.ravel()) if x in item_dict.keys()]


# In[ ]:





# In[66]:


candidate_rank_dfs = []
for test_df_small in test_dfs:
    
    prev_words_test = test_df_small.groupby('session_id').apply(get_values).to_list()
    
    input_seqs_test = np.array(pad_sequences(prev_words_test, maxlen=max_seq_length, padding='pre'))

    input_seqs_test

    preds = model.predict(input_seqs_test)

    preds.shape

    preds

    input_seqs_test.shape

    np.argmax(preds, axis=-1)

    preds.argsort()[:,-100:]

    arr = preds.argsort().astype("float32")

    arr

    cond = np.isin(arr.astype(int), candidate_items_keys)
    cond

    arr[~cond] = np.nan
    arr

    test_df.session_id.unique()[:3]

    arrlist = arr.tolist()

    
    for i in range(len(test_df_small.session_id.unique())):
        rank_dict = {}
        session_id  = test_df_small.session_id.unique()[i]

        my_list = arrlist[i]
        y = [x for x in my_list if x>=0 and x in item_dict.keys()]
        ranked_list = [item_dict_rev[x] for x in y[-100:]]

        for num, item in enumerate(ranked_list):
            rank_dict[item] = len(ranked_list) - num
        candidate_rank_df = pd.DataFrame(rank_dict.items(), columns = ["item_id", "rank"]).sort_values("rank")
        candidate_rank_df["session_id"] = session_id
        candidate_rank_dfs.append(candidate_rank_df)
    #display(len(pd.concat(candidate_rank_dfs).session_id.unique()))
    #display(pd.concat(candidate_rank_dfs))

candidate_rank_df = pd.concat(candidate_rank_dfs)
candidate_rank_df = candidate_rank_df[["session_id", "item_id", "rank"]].reset_index(drop=True)
candidate_rank_df


# In[67]:


candidate_rank_df


# In[68]:


len(candidate_rank_df.session_id.unique())


# In[69]:


(~candidate_rank_df.item_id.isin(candidate_items)).sum()


# In[70]:


len(test_df.session_id.unique())


# In[71]:


candidate_rank_df.to_csv("results_lstm.csv", index=False)


# In[76]:


candidate_rank_df[candidate_rank_df["rank"] == 1].item_id.unique()


# In[ ]:




