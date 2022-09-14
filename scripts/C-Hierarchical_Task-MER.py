#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input args (4): model-name, nº epochs, random seed, random exec
model_name = "mbert_galen"
epochs = 95
random_seed = 1
random_exec = 1

"""
Estimated hyper-parameters for the remaining models:
mBERT: epochs=70, random_seed=1

BETO-Galén: epochs=69, random_seed=0
BETO: epochs=89, random_seed=2

XLM-R-Galén: epochs=63, random_seed=2
XLM-R: epochs=88, random_seed=1
"""

import sys
if len(sys.argv) > 1:
    model_name = sys.argv[-4]
    epochs = int(sys.argv[-3])
    random_seed = int(sys.argv[-2])
    random_exec = int(sys.argv[-1])

hier_iob_exec = random_exec
    
print("Model name:", model_name, 
      "| nº epochs:", epochs,
      "| random seed:", random_seed,
      "| random exec:", random_exec)


# In[ ]:


root_path = "../"


# In[ ]:


from transformers import BertTokenizerFast, XLMRobertaTokenizerFast

# All variables that depend on model_name
if model_name == 'beto':
    model_path = root_path + "models/" + "BERT/pytorch/BETO/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif model_name == 'beto_galen':
    model_path = root_path + "models/" + "BERT/pytorch/BETO-Galen/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif model_name == 'mbert':
    model_path = root_path + "models/" + "BERT/pytorch/mBERT/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif model_name == 'mbert_galen':
    model_path = root_path + "models/" + "BERT/pytorch/mBERT-Galen/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif model_name == 'xlmr':
    model_path = root_path + "models/" + "XLM-R/pytorch/xlm-roberta-base/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
else: # default
    model_path = root_path + "models/" + "XLM-R/pytorch/XLM-R-Galen/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)


# In[1]:


utils_path = root_path + "utils/"
corpus_path = root_path + "datasets/cantemist_v6/"
sub_task_path = "cantemist-norm/"
ss_corpus_path = root_path + "datasets/Cantemist-SSplit-text/"
dev_gs_path = corpus_path + "dev-set2/" + sub_task_path
test_gs_path = corpus_path + "test-set/" + sub_task_path


# In[2]:


import tensorflow as tf

# Auxiliary components
import sys
sys.path.insert(0, utils_path)
from nlp_utils import *

print(sys.path)

# Hyper-parameters
type_tokenizer = "transformers"

subtask = 'norm'
subtask_ann = subtask + '-iob_code'
text_col = "raw_text"
SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = epochs
LR = 3e-5

GREEDY = True
IGNORE_VALUE = -100
code_strat = 'o'
ANN_STRATEGY = "word-all"
EVAL_STRATEGY = "word-prod"
mention_strat = "prod"
LOGITS = False

tf.random.set_seed(random_seed)

JOB_NAME = "c_hier_task_iob_" + model_name + "_" + str(random_exec)
print("\n" + JOB_NAME)


# ## Load text

# ### Training corpus

# In[3]:


train_path = corpus_path + "train-set/" + sub_task_path
train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and f.split('.')[-1] == "txt"]
n_train_files = len(train_files)
train_data = load_text_files(train_files, train_path)
dev1_path = corpus_path + "dev-set1/" + sub_task_path
train_files.extend([f for f in os.listdir(dev1_path) if os.path.isfile(dev1_path + f) and f.split('.')[-1] == "txt"])
train_data.extend(load_text_files(train_files[n_train_files:], dev1_path))
df_text_train = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in train_files], 'raw_text': train_data})


# ### Development corpus

# In[4]:


dev_path = corpus_path + "dev-set2/" + sub_task_path
dev_files = [f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and f.split('.')[-1] == "txt"]
dev_data = load_text_files(dev_files, dev_path)
df_text_dev = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in dev_files], 'raw_text': dev_data})


# ### Test corpus

# In[5]:


test_path = corpus_path + "test-set/" + sub_task_path
test_files = [f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and f.split('.')[-1] == 'txt']
test_data = load_text_files(test_files, test_path)
df_text_test = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in test_files], 'raw_text': test_data})


# ## Process annotations

# In[6]:


# Training corpus


# In[7]:


train_ann_files = [train_path + f for f in os.listdir(train_path) if f.split('.')[-1] == "ann"]
train_ann_files.extend([dev1_path + f for f in os.listdir(dev1_path) if f.split('.')[-1] == "ann"])


# In[8]:


df_codes_train_ner = process_brat_norm(train_ann_files).sort_values(["doc_id", "start", "end"])


# In[9]:


assert ~df_codes_train_ner[["doc_id", "start", "end"]].duplicated().any()


# In[10]:


# Development corpus


# In[11]:


dev_ann_files = [dev_path + f for f in os.listdir(dev_path) if f.split('.')[-1] == "ann"]


# In[12]:


df_codes_dev_ner = process_brat_norm(dev_ann_files).sort_values(["doc_id", "start", "end"])


# In[13]:


assert ~df_codes_dev_ner[["doc_id", "start", "end"]].duplicated().any()


# ### Remove overlapping annotations

# In[14]:


# Training corpus


# In[15]:


df_codes_train_ner_final = eliminate_overlap(df_ann=df_codes_train_ner)


# In[16]:


# Development corpus


# In[17]:


df_codes_dev_ner_final = eliminate_overlap(df_ann=df_codes_dev_ner)


# ## Creation of annotated sequences

# In[18]:


train_dev_codes = sorted(set(df_codes_dev_ner_final["code"].values).union(set(df_codes_train_ner_final["code"].values))) 


# In[19]:


print(len(train_dev_codes))


# In[20]:


# Create IOB-2 and Clinical-Coding label encoders as dict (more computationally efficient)
iob_lab_encoder = {"B": 0, "I": 1, "O": 2}
iob_lab_decoder = {0: "B", 1: "I", 2: "O"}

code_lab_encoder = {}
code_lab_decoder = {}
i = 0
for code in train_dev_codes:
    code_lab_encoder[code] = i
    code_lab_decoder[i] = code
    i += 1
    
if code_strat.upper() == "O":    
    code_lab_encoder["O"] = i
    code_lab_decoder[i] = "O"


# In[21]:


print(len(iob_lab_encoder), len(iob_lab_decoder))


# In[22]:


print(len(code_lab_encoder), len(code_lab_decoder))


# In[23]:


# Text classification (later ignored)


# In[24]:


train_dev_codes = sorted(set(df_codes_dev_ner["code"].values).union(set(df_codes_train_ner["code"].values))) 


# In[25]:


print(len(train_dev_codes))


# In[26]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb_encoder = MultiLabelBinarizer()
mlb_encoder.fit([train_dev_codes])


# ### Training corpus

# Only training texts with NER annotations are considered:

# In[27]:


train_doc_list = sorted(set(df_codes_train_ner_final["doc_id"]))


# In[28]:


# Sentence-Split data


# In[29]:


ss_sub_corpus_path = ss_corpus_path + "training/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_train = load_ss_files(ss_files, ss_sub_corpus_path)


# In[30]:


train_ind, train_att, train_type, train_y, train_text_y, train_frag, train_start_end_frag,                 train_word_id = ss_create_input_data_ner(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_codes_train_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore text-level
                                    doc_list=train_doc_list, ss_dict=ss_dict_train,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[31]:


train_iob_y = train_y[0]


# ### Development corpus
# 
# Only development texts with NER annotations are considered:

# In[33]:


dev_doc_list = sorted(set(df_codes_dev_ner_final["doc_id"]))


# In[34]:


# Sentence-Split data


# In[35]:


ss_sub_corpus_path = ss_corpus_path + "development/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_dev = load_ss_files(ss_files, ss_sub_corpus_path)


# In[36]:


dev_ind, dev_att, dev_type, dev_y, dev_text_y, dev_frag, dev_start_end_frag,                 dev_word_id = ss_create_input_data_ner(df_text=df_text_dev, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_train_ner_final, # ignore text-level
                                    doc_list=dev_doc_list, ss_dict=ss_dict_dev,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[37]:


dev_iob_y = dev_y[0]


# ### Test corpus
# 
# All test texts are considered:

# In[39]:


test_doc_list = sorted(set(df_text_test["doc_id"]))


# In[40]:


# Sentence-Split data


# In[41]:


ss_sub_corpus_path = ss_corpus_path + "test-background/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_test = load_ss_files(ss_files, ss_sub_corpus_path)


# In[42]:


test_ind, test_att, test_type, test_y, test_text_y, test_frag, test_start_end_frag,                 test_word_id = ss_create_input_data_ner(df_text=df_text_test, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore text-level
                                    doc_list=test_doc_list, ss_dict=ss_dict_test,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[43]:


test_iob_y = test_y[0]


# ### Training & Development corpus
# 
# We merge the previously generated datasets:

# In[45]:


# Indices
train_dev_ind = np.concatenate((train_ind, dev_ind))


# In[46]:


print(train_dev_ind.shape)


# In[47]:


# Attention
train_dev_att = np.concatenate((train_att, dev_att))


# In[48]:


print(train_dev_att.shape)


# In[49]:


# Type
train_dev_type = np.concatenate((train_type, dev_type))


# In[50]:


print(train_dev_type.shape)


# In[51]:


# IOB-2 y
train_dev_iob_y = np.concatenate((train_iob_y, dev_iob_y))


# In[52]:


print(train_dev_iob_y.shape)


# ## Fine-tuning

# In[55]:


# Set memory growth


# In[56]:


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)


# In[57]:


for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# In[58]:


from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification

if model_name.split('_')[0] in ('beto', 'mbert'):
    model = TFBertForTokenClassification.from_pretrained(model_path, from_pt=True)
    
else: # default
    model = TFXLMRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)


# In[59]:


# [IOB]


# In[60]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

iob_num_labels = len(iob_lab_encoder)

input_ids = Input(shape=(SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int64')

out_seq = model.layers[0](input_ids=input_ids, attention_mask=attention_mask)[0] # take the output sub-token sequence 

# IOB-2
out_iob = Dense(units=iob_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_iob_model = Activation(activation='softmax', name='iob_output')(out_iob)

model = Model(inputs=[input_ids, attention_mask], outputs=out_iob_model)


# In[61]:


print(model.summary())


# In[62]:


print(model.input)


# In[63]:


print(model.output)


# In[64]:


df_dev_gs = format_ner_gs(dev_gs_path, subtask=subtask)
df_test_gs = format_ner_gs(test_gs_path, subtask=subtask)


# In[66]:


import tensorflow_addons as tfa
import time

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'iob_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE)}
loss_weights = {'iob_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

start_time = time.time()

history = model.fit(x={'input_ids': train_dev_ind, 'attention_mask': train_dev_att}, 
                    y={'iob_output': train_dev_iob_y}, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                    verbose=2)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print()


# ## Evaluation

# ### Development

# In[67]:


start_time = time.time()


# In[68]:


iob_dev_preds = model.predict({'input_ids': dev_ind, 'attention_mask': dev_att})


# In[69]:


np.save(file="dev_preds_" + JOB_NAME + ".npy", arr=iob_dev_preds)


# In[70]:


df_dev_preds = ner_preds_brat_format(doc_list=dev_doc_list, fragments=dev_frag, 
                                    preds=[iob_dev_preds],
                                    start_end=dev_start_end_frag, word_id=dev_word_id, 
                                    lab_decoder_list=[iob_lab_decoder], 
                                    df_text=df_text_dev, 
                                    text_col=text_col, strategy=EVAL_STRATEGY, subtask='ner',
                                    type_tokenizer=type_tokenizer)


# In[71]:


print(calculate_ner_metrics(gs=df_dev_gs, pred=format_ner_pred_df(gs_path=dev_gs_path, df_preds=df_dev_preds, 
                                                                  subtask='ner'),
                            subtask='ner'))


# In[72]:


end_time = time.time()
print("--- Dev evaluation: %s seconds ---" % (end_time - start_time))
print()


# ### Test

# In[73]:


start_time = time.time()


# In[74]:


iob_test_preds = model.predict({'input_ids': test_ind, 'attention_mask': test_att})


# In[75]:


np.save(file="test_preds_" + JOB_NAME + ".npy", arr=iob_test_preds)


# In[76]:


df_test_preds = ner_preds_brat_format(doc_list=test_doc_list, fragments=test_frag, 
                                      preds=[iob_test_preds], 
                                      start_end=test_start_end_frag, word_id=test_word_id, 
                                      lab_decoder_list=[iob_lab_decoder], 
                                      df_text=df_text_test, 
                                      text_col=text_col, strategy=EVAL_STRATEGY, subtask='ner',
                                      type_tokenizer=type_tokenizer)


# In[77]:


print(calculate_ner_metrics(gs=df_test_gs, pred=format_ner_pred_df(gs_path=test_gs_path, df_preds=df_test_preds, 
                                                                   subtask='ner'),
                            subtask='ner'))


# In[78]:


end_time = time.time()
print("--- Test evaluation: %s seconds ---" % (end_time - start_time))


# In[ ]:


# Save preds DataFrame


# In[ ]:


df_test_preds.to_csv("df_test_preds_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')


# In[ ]:


# Save word-level preds (subsequently used in ensemble)


# In[ ]:


import pickle

doc_word_preds, doc_word_start_end = seq_ner_preds_brat_format(doc_list=test_doc_list, fragments=test_frag, 
                           arr_start_end=test_start_end_frag, arr_word_id=test_word_id, arr_preds=[iob_test_preds], 
                           strategy=EVAL_STRATEGY, type_tokenizer=type_tokenizer)
with open("test_word_preds_" + JOB_NAME + ".pck", "wb") as f:
    pickle.dump(doc_word_preds, f)

if random_exec == 1:
    with open("test_word_start_end_" + model_name + ".pck", "wb") as f:
        pickle.dump(doc_word_start_end, f)

