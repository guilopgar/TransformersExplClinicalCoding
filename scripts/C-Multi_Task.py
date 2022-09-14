#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning on Cantemist-NORM

# In[ ]:


# Input args (4): model-name, nº epochs, random seed, random exec
model_name = "mbert_galen"
epochs = 80
random_seed = 2
random_exec = 1

"""
Estimated hyper-parameters for the remaining models:
mBERT: epochs=97, random_seed=3

BETO-Galén: epochs=98, random_seed=2
BETO: epochs=91, random_seed=1

XLM-R-Galén: epochs=91, random_seed=3
XLM-R: epochs=94, random_seed=1
"""

import sys
if len(sys.argv) > 1:
    model_name = sys.argv[-4]
    epochs = int(sys.argv[-3])
    random_seed = int(sys.argv[-2])
    random_exec = int(sys.argv[-1])
    
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
subtask_ann = subtask + '-iob_code_suf'
text_col = "raw_text"
SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = epochs
LR = 3e-5

GREEDY = True
IGNORE_VALUE = -100
code_strat = 'o'
ANN_STRATEGY = "word-all"
EVAL_STRATEGY = "word-sum_norm"
mention_strat = "sum"
LOGITS = False

tf.random.set_seed(random_seed)

JOB_NAME = "c_multi_task_" + model_name + "_" + str(random_exec)
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


df_codes_train_ner["code_pre"] = df_codes_train_ner["code"].apply(lambda x: x.split('/')[0])
df_codes_train_ner["code_suf"] = df_codes_train_ner["code"].apply(lambda x: '/'.join(x.split('/')[1:]))


# In[10]:


assert ~df_codes_train_ner[["doc_id", "start", "end"]].duplicated().any()


# In[11]:


# Development corpus


# In[12]:


dev_ann_files = [dev_path + f for f in os.listdir(dev_path) if f.split('.')[-1] == "ann"]


# In[13]:


df_codes_dev_ner = process_brat_norm(dev_ann_files).sort_values(["doc_id", "start", "end"])


# In[14]:


df_codes_dev_ner["code_pre"] = df_codes_dev_ner["code"].apply(lambda x: x.split('/')[0])
df_codes_dev_ner["code_suf"] = df_codes_dev_ner["code"].apply(lambda x: '/'.join(x.split('/')[1:]))


# In[15]:


assert ~df_codes_dev_ner[["doc_id", "start", "end"]].duplicated().any()


# ### Remove overlapping annotations

# In[16]:


# Training corpus


# In[17]:


df_codes_train_ner_final = eliminate_overlap(df_ann=df_codes_train_ner)


# In[18]:


# Development corpus


# In[19]:


df_codes_dev_ner_final = eliminate_overlap(df_ann=df_codes_dev_ner)


# ## Creation of annotated sequences

# In[20]:


train_dev_codes_pre = sorted(set(df_codes_dev_ner_final["code_pre"].values).union(set(df_codes_train_ner_final["code_pre"].values))) 


# In[21]:


print(len(train_dev_codes_pre))


# In[22]:


train_dev_codes_suf = sorted(set(df_codes_dev_ner_final["code_suf"].values).union(set(df_codes_train_ner_final["code_suf"].values))) 


# In[23]:


len(train_dev_codes_suf)


# In[24]:


# Create IOB-2 and Clinical-Coding label encoders as dict (more computationally efficient)
iob_lab_encoder = {"B": 0, "I": 1, "O": 2}
iob_lab_decoder = {0: "B", 1: "I", 2: "O"}

# Code-pre
code_pre_lab_encoder = {}
code_pre_lab_decoder = {}
i = 0
for code in train_dev_codes_pre:
    code_pre_lab_encoder[code] = i
    code_pre_lab_decoder[i] = code
    i += 1
    
if code_strat.upper() == "O":    
    code_pre_lab_encoder["O"] = i
    code_pre_lab_decoder[i] = "O"

# Code-suf
code_suf_lab_encoder = {}
code_suf_lab_decoder = {}
i = 0
for code in train_dev_codes_suf:
    code_suf_lab_encoder[code] = i
    code_suf_lab_decoder[i] = code
    i += 1
    
if code_strat.upper() == "O":    
    code_suf_lab_encoder["O"] = i
    code_suf_lab_decoder[i] = "O"


# In[25]:


print(len(iob_lab_encoder), len(iob_lab_decoder))


# In[26]:


print(len(code_pre_lab_encoder), len(code_pre_lab_decoder))


# In[27]:


print(len(code_suf_lab_encoder), len(code_suf_lab_decoder))


# In[28]:


# Text classification (later ignored)


# In[29]:


train_dev_codes = sorted(set(df_codes_dev_ner_final["code"].values).union(set(df_codes_train_ner_final["code"].values))) 


# In[30]:


print(len(train_dev_codes))


# In[31]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb_encoder = MultiLabelBinarizer()
mlb_encoder.fit([train_dev_codes])


# In[32]:


# "Impossible" codes mask


# In[33]:


train_dev_codes_pre_o_mask = np.ones(len(train_dev_codes_pre)+1)
train_dev_codes_pre_o_mask[-1] = -1

# Mask only "O" suf label
train_dev_codes_pre_suf_mask = np.ones(shape=(len(train_dev_codes_pre), len(train_dev_codes_suf) + 1)) # "O" label
train_dev_codes_pre_suf_mask[:, -1] = -1


# In[34]:


print(train_dev_codes_pre_o_mask.shape)


# In[35]:


print(train_dev_codes_pre_suf_mask.shape)


# ### Training corpus

# Only training texts with NER annotations are considered:

# In[36]:


train_doc_list = sorted(set(df_codes_train_ner_final["doc_id"]))


# In[37]:


# Sentence-Split data


# In[38]:


ss_sub_corpus_path = ss_corpus_path + "training/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_train = load_ss_files(ss_files, ss_sub_corpus_path)


# In[39]:


train_ind, train_att, train_type, train_y, train_text_y, train_frag, train_start_end_frag,                 train_word_id = ss_create_input_data_ner(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_codes_train_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore text-level
                                    doc_list=train_doc_list, ss_dict=ss_dict_train,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[40]:


train_iob_y, train_code_pre_y, train_code_suf_y = train_y


# ### Development corpus
# 
# Only development texts with NER annotations are considered:

# In[41]:


dev_doc_list = sorted(set(df_codes_dev_ner_final["doc_id"]))


# In[42]:


# Sentence-Split data


# In[43]:


ss_sub_corpus_path = ss_corpus_path + "development/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_dev = load_ss_files(ss_files, ss_sub_corpus_path)


# In[44]:


dev_ind, dev_att, dev_type, dev_y, dev_text_y, dev_frag, dev_start_end_frag,                 dev_word_id = ss_create_input_data_ner(df_text=df_text_dev, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_train_ner_final, # ignore text-level
                                    doc_list=dev_doc_list, ss_dict=ss_dict_dev,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[45]:


dev_iob_y, dev_code_pre_y, dev_code_suf_y = dev_y


# ### Test corpus
# 
# All test texts are considered:

# In[46]:


test_doc_list = sorted(set(df_text_test["doc_id"]))


# In[47]:


# Sentence-Split data


# In[48]:


ss_sub_corpus_path = ss_corpus_path + "test-background/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_test = load_ss_files(ss_files, ss_sub_corpus_path)


# In[49]:


test_ind, test_att, test_type, test_y, test_text_y, test_frag, test_start_end_frag,                 test_word_id = ss_create_input_data_ner(df_text=df_text_test, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore text-level
                                    doc_list=test_doc_list, ss_dict=ss_dict_test,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[50]:


test_iob_y, test_code_pre_y, test_code_suf_y = test_y


# ### Training & Development corpus
# 
# We merge the previously generated datasets:

# In[51]:


# Indices
train_dev_ind = np.concatenate((train_ind, dev_ind))


# In[52]:


print(train_dev_ind.shape)


# In[53]:


# Attention
train_dev_att = np.concatenate((train_att, dev_att))


# In[54]:


print(train_dev_att.shape)


# In[55]:


# Type
train_dev_type = np.concatenate((train_type, dev_type))


# In[56]:


print(train_dev_type.shape)


# In[57]:


# IOB-2 y
train_dev_iob_y = np.concatenate((train_iob_y, dev_iob_y))


# In[58]:


print(train_dev_iob_y.shape)


# In[59]:


# Clinical-Coding-prefix y
train_dev_code_pre_y = np.concatenate((train_code_pre_y, dev_code_pre_y))


# In[60]:


print(train_dev_code_pre_y.shape)


# In[61]:


# Clinical-Coding-suffix y
train_dev_code_suf_y = np.concatenate((train_code_suf_y, dev_code_suf_y))


# In[62]:


print(train_dev_code_suf_y.shape)


# ## Fine-tuning

# In[63]:


# Set memory growth


# In[64]:


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)


# In[65]:


for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# In[65]:


from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification

if model_name.split('_')[0] in ('beto', 'mbert'):
    model = TFBertForTokenClassification.from_pretrained(model_path, from_pt=True)
    
else: # default
    model = TFXLMRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)


# In[66]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

iob_num_labels = len(iob_lab_encoder)
code_pre_num_labels = len(code_pre_lab_encoder)
code_suf_num_labels = len(code_suf_lab_encoder)

input_ids = Input(shape=(SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int64')

out_seq = model.layers[0](input_ids=input_ids, attention_mask=attention_mask)[0] # take the output sub-token sequence 

# IOB-2
out_iob = Dense(units=iob_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_iob_model = Activation(activation='softmax', name='iob_output')(out_iob)
# Code-pre
out_code_pre = Dense(units=code_pre_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_code_pre_model = Activation(activation='softmax', name='code_pre_output')(out_code_pre)
# Code-suf
out_code_suf = Dense(units=code_suf_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_code_suf_model = Activation(activation='softmax', name='code_suf_output')(out_code_suf)

model = Model(inputs=[input_ids, attention_mask], outputs=[out_iob_model, out_code_pre_model, out_code_suf_model])


# In[67]:


print(model.summary())


# In[68]:


print(model.input)


# In[69]:


print(model.output)


# In[70]:


df_dev_gs = format_ner_gs(dev_gs_path, subtask=subtask)
df_test_gs = format_ner_gs(test_gs_path, subtask=subtask)


# In[71]:


import tensorflow_addons as tfa
import time

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'iob_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE),
        'code_pre_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE),
        'code_suf_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE)}
loss_weights = {'iob_output': 1, 'code_pre_output': 1, 'code_suf_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

start_time = time.time()

history = model.fit(x={'input_ids': train_dev_ind, 'attention_mask': train_dev_att}, 
                    y={'iob_output': train_dev_iob_y, 'code_pre_output': train_dev_code_pre_y, 
                       'code_suf_output': train_dev_code_suf_y}, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                    verbose=2)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print()


# ## Evaluation

# ### Development

# In[72]:


start_time = time.time()


# In[73]:


dev_preds = model.predict({'input_ids': dev_ind, 
                           'attention_mask': dev_att})
iob_dev_preds = dev_preds[0]
code_pre_dev_preds = dev_preds[1]
code_suf_dev_preds = dev_preds[2]


# In[74]:


# Evaluate NER performance
df_dev_preds_iob = ner_preds_brat_format(
    doc_list=dev_doc_list, fragments=dev_frag, preds=[iob_dev_preds], 
    start_end=dev_start_end_frag, word_id=dev_word_id, 
    lab_decoder_list=[iob_lab_decoder], 
    df_text=df_text_dev, text_col=text_col, strategy=EVAL_STRATEGY, 
    subtask='ner', type_tokenizer=type_tokenizer
)


# In[75]:


print(calculate_ner_metrics(gs=df_dev_gs, pred=format_ner_pred_df(gs_path=dev_gs_path, df_preds=df_dev_preds_iob, 
                                                                  subtask='ner'),
                            subtask='ner'))


# In[76]:


df_dev_preds = ner_preds_brat_format(doc_list=dev_doc_list, fragments=dev_frag, 
                                    preds=[iob_dev_preds, code_pre_dev_preds, code_suf_dev_preds],
                                    start_end=dev_start_end_frag, word_id=dev_word_id, 
                                    lab_decoder_list=[iob_lab_decoder, code_pre_lab_decoder, code_suf_lab_decoder], 
                                    df_text=df_text_dev, 
                                    text_col=text_col, strategy=EVAL_STRATEGY, subtask=subtask_ann + '_mask',
                                    mention_strat=mention_strat, type_tokenizer=type_tokenizer,
                                    codes_pre_suf_mask=train_dev_codes_pre_suf_mask,
                                    codes_pre_o_mask=train_dev_codes_pre_o_mask)


# In[77]:


print(calculate_ner_metrics(gs=df_dev_gs, pred=format_ner_pred_df(gs_path=dev_gs_path, df_preds=df_dev_preds, 
                                                                  subtask=subtask),
                            subtask=subtask))


# In[78]:


end_time = time.time()
print("--- Dev evaluation: %s seconds ---" % (end_time - start_time))
print()


# ### Test

# In[79]:


start_time = time.time()


# In[80]:


test_preds = model.predict({'input_ids': test_ind, 
                            'attention_mask': test_att})
iob_test_preds = test_preds[0]
code_pre_test_preds = test_preds[1]
code_suf_test_preds = test_preds[2]


# In[82]:


np.save(file="iob_test_preds_" + JOB_NAME + ".npy", arr=iob_test_preds)


# In[81]:


# Evaluate NER performance
df_test_preds_iob = ner_preds_brat_format(
    doc_list=test_doc_list, fragments=test_frag, preds=[iob_test_preds], 
    start_end=test_start_end_frag, word_id=test_word_id, 
    lab_decoder_list=[iob_lab_decoder], 
    df_text=df_text_test, text_col=text_col, strategy=EVAL_STRATEGY, 
    subtask='ner', type_tokenizer=type_tokenizer
)


# In[82]:


print(calculate_ner_metrics(gs=df_test_gs, pred=format_ner_pred_df(gs_path=test_gs_path, df_preds=df_test_preds_iob, 
                                                                  subtask='ner'),
                            subtask='ner'))


# In[85]:


# Save final results DF
df_test_preds_iob.to_csv("df_test_preds_ner_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')


# In[83]:


df_test_preds = ner_preds_brat_format(doc_list=test_doc_list, fragments=test_frag, 
                                      preds=[iob_test_preds, code_pre_test_preds, code_suf_test_preds], 
                                      start_end=test_start_end_frag, word_id=test_word_id, 
                                      lab_decoder_list=[iob_lab_decoder, code_pre_lab_decoder, code_suf_lab_decoder], 
                                      df_text=df_text_test, 
                                      text_col=text_col, strategy=EVAL_STRATEGY, subtask=subtask_ann + '_mask',
                                      mention_strat=mention_strat, type_tokenizer=type_tokenizer,
                                      codes_pre_suf_mask=train_dev_codes_pre_suf_mask,
                                      codes_pre_o_mask=train_dev_codes_pre_o_mask)


# In[84]:


print(calculate_ner_metrics(gs=df_test_gs, pred=format_ner_pred_df(gs_path=test_gs_path, df_preds=df_test_preds, 
                                                                   subtask=subtask),
                            subtask=subtask))


# In[85]:


end_time = time.time()
print("--- Test evaluation: %s seconds ---" % (end_time - start_time))


# In[89]:


# Save final results DF
df_test_preds.to_csv("df_test_preds_norm_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')

