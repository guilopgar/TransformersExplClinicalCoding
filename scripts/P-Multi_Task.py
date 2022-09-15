#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input args (4): model-name, nº epochs, random seed, random exec
model_name = "mbert_galen"
epochs = 87
random_seed = 0
random_exec = 1

"""
Estimated hyper-parameters for the remaining models:
mBERT: epochs=90, random_seed=3

BETO-Galén: epochs=96, random_seed=0
BETO: epochs=100, random_seed=2

XLM-R-Galén: epochs=93, random_seed=3
XLM-R: epochs=87, random_seed=0
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
corpus_path = root_path + "datasets/final_dataset_v4_to_publish/"
ss_corpus_path = root_path + "datasets/CodiEsp-SSplit-text/"
dev_gs_path = corpus_path + "dev/devX.tsv"
test_gs_path = corpus_path + "test/testX.tsv"


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
subtask_ann = subtask + '-iob_cont_disc_code_suf'
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

TYPE_ANN = 'PROCEDIMIENTO'
CODE_SEP = '.' if TYPE_ANN == 'DIAGNOSTICO' else ''
TYPE_TASK = TYPE_ANN[0].lower()

tf.random.set_seed(random_seed)

JOB_NAME = TYPE_TASK + "_multi_task_" + model_name + "_" + str(random_exec)
print("\n" + JOB_NAME)


# In[ ]:


codes_d_path = root_path + "datasets/final_dataset_v4_to_publish/codiesp_codes/codiesp-" + TYPE_TASK.upper() + "_codes.tsv"


# ## Load text

# ### Training corpus

# In[3]:


train_path = corpus_path + "train/text_files/"
train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and f.split('.')[-1] == "txt"]
train_data = load_text_files(train_files, train_path)
df_text_train = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in train_files], 'raw_text': train_data})


# ### Development corpus

# In[4]:


dev_path = corpus_path + "dev/text_files/"
dev_files = [f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and f.split('.')[-1] == "txt"]
dev_data = load_text_files(dev_files, dev_path)
df_text_dev = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in dev_files], 'raw_text': dev_data})


# ### Test corpus

# In[5]:


test_path = corpus_path + "test/text_files/"
test_files = [f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and f.split('.')[-1] == 'txt']
test_data = load_text_files(test_files, test_path)
df_text_test = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in test_files], 'raw_text': test_data})


# ## Process annotations

# ### Training corpus

# In[6]:


df_codes_train_ner = pd.read_table(corpus_path + "train/trainX.tsv", sep='\t', header=None)
df_codes_train_ner.columns = ["doc_id", "type", "code", "word", "location"]
df_codes_train_ner = df_codes_train_ner[~df_codes_train_ner[['doc_id', 'type', 'location']].duplicated(keep='first')]
df_codes_train_ner['disc'] = df_codes_train_ner['location'].apply(lambda x: ';' in x)


# Select one type of annotations:

# In[7]:


df_codes_train_ner = df_codes_train_ner[df_codes_train_ner['type'] == TYPE_ANN]


# Split discontinuous annotations:

# In[8]:


df_codes_train_ner_final = process_labels_norm_prueba(df_ann=df_codes_train_ner[["doc_id", "type", "code", "word", "location"]])


# Remove annotations of zero length:

# In[9]:


df_codes_train_ner_final['length'] = df_codes_train_ner_final.apply(lambda x: x['end'] - x['start'], axis=1)
df_codes_train_ner_final = df_codes_train_ner_final[df_codes_train_ner_final['length'] > 0]


# Remove long-overlapping annotations with different codes (Cont vs disc):

# In[10]:


df_codes_train_ner_final = eliminate_overlap_code(df_ann=df_codes_train_ner_final)


# Separate continuous and discontinuous annotations:

# In[11]:


# Continiuous
df_codes_train_ner_final_cont = df_codes_train_ner_final[df_codes_train_ner_final['disc'] == 0].copy()
df_codes_train_ner_final_cont['disc'] = df_codes_train_ner_final_cont['disc'].astype(bool)


# In[12]:


# Discontinuous
df_codes_train_ner_final_disc = restore_disc_ann(df_ann=df_codes_train_ner[df_codes_train_ner['disc']], 
                    df_ann_final=df_codes_train_ner_final[df_codes_train_ner_final['disc'] == 1])


# Remove long-overlapping annotations of the same disc-type:

# In[13]:


# Continuous


# In[14]:


df_codes_train_ner_final_cont = eliminate_overlap(df_ann=df_codes_train_ner_final_cont)


# In[15]:


# Discontinuous


# In[16]:


df_codes_train_ner_final_disc['start'] = df_codes_train_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_train_ner_final_disc['end'] = df_codes_train_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[-1]))


# In[17]:


df_codes_train_ner_final_disc = eliminate_overlap(df_ann=df_codes_train_ner_final_disc)


# Concatenate continuous and discontinuous annotations:

# In[18]:


# Concat
cols_concat = ['doc_id', 'type', 'code', 'word', 'location', 'start', 'end', 'disc']
df_codes_train_ner_final = pd.concat([df_codes_train_ner_final_cont[cols_concat], 
                                      df_codes_train_ner_final_disc[cols_concat]])


# Now, we remove the right-to-left (text wise) discontinuous annotations:

# In[19]:


df_codes_train_ner_final['direction'] = df_codes_train_ner_final.apply(check_ann_left_right_direction, axis=1)


# In[20]:


df_codes_train_ner_final = df_codes_train_ner_final[df_codes_train_ner_final['direction']]


# We only select the annotations fully contained in a single sentence:

# In[21]:


# Sentence-Split data
ss_sub_corpus_path = ss_corpus_path + "train/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_train = load_ss_files(ss_files, ss_sub_corpus_path)


# In[22]:


df_mult_sent_train, df_one_sent_train, df_no_sent_train = check_ann_span_sent(df_ann=df_codes_train_ner_final, 
                                                                             ss_dict=ss_dict_train)


# In[23]:


df_codes_train_ner_final = df_one_sent_train.copy()


# In[24]:


print(df_codes_train_ner_final.disc.value_counts())


# In[25]:


df_codes_train_ner_final.sort_values(['doc_id', 'start', 'end'], inplace=True)


# In[26]:


# Code splitting


# In[27]:


if TYPE_TASK == 'd':
    df_codes_train_ner_final["code_pre"] = df_codes_train_ner_final["code"].apply(lambda x: x.split('.')[0])
    df_codes_train_ner_final["code_suf"] = df_codes_train_ner_final["code"].apply(lambda x: None if not '.' in x else x.split('.')[1])
else:
    df_codes_train_ner_final["code_pre"] = df_codes_train_ner_final["code"].apply(lambda x: x[:4])
    df_codes_train_ner_final["code_suf"] = df_codes_train_ner_final["code"].apply(lambda x: None if len(x) < 7 else x[4:7])


# In[28]:


print(df_codes_train_ner_final.shape)


# ### Development corpus

# In[29]:


df_codes_dev_ner = pd.read_table(corpus_path + "dev/devX.tsv", sep='\t', header=None)
df_codes_dev_ner.columns = ["doc_id", "type", "code", "word", "location"]
df_codes_dev_ner = df_codes_dev_ner[~df_codes_dev_ner[['doc_id', 'type', 'location']].duplicated(keep='first')]
df_codes_dev_ner['disc'] = df_codes_dev_ner['location'].apply(lambda x: ';' in x)


# Select one type of annotations:

# In[30]:


df_codes_dev_ner = df_codes_dev_ner[df_codes_dev_ner['type'] == TYPE_ANN]


# Split discontinuous annotations:

# In[31]:


df_codes_dev_ner_final = process_labels_norm_prueba(df_ann=df_codes_dev_ner[["doc_id", "type", "code", "word", "location"]])


# Remove annotations of zero length:

# In[32]:


df_codes_dev_ner_final['length'] = df_codes_dev_ner_final.apply(lambda x: x['end'] - x['start'], axis=1)
df_codes_dev_ner_final = df_codes_dev_ner_final[df_codes_dev_ner_final['length'] > 0]


# Remove long-overlapping annotations with different codes (Cont vs disc):

# In[33]:


df_codes_dev_ner_final = eliminate_overlap_code(df_ann=df_codes_dev_ner_final)


# Separate continuous and discontinuous annotations:

# In[34]:


# Continiuous
df_codes_dev_ner_final_cont = df_codes_dev_ner_final[df_codes_dev_ner_final['disc'] == 0].copy()
df_codes_dev_ner_final_cont['disc'] = df_codes_dev_ner_final_cont['disc'].astype(bool)


# In[35]:


# Discontinuous
df_codes_dev_ner_final_disc = restore_disc_ann(df_ann=df_codes_dev_ner[df_codes_dev_ner['disc']], 
                    df_ann_final=df_codes_dev_ner_final[df_codes_dev_ner_final['disc'] == 1])


# Remove long-overlapping annotations of the same disc-type:

# In[36]:


# Continuous


# In[37]:


df_codes_dev_ner_final_cont = eliminate_overlap(df_ann=df_codes_dev_ner_final_cont)


# In[38]:


# Discontinuous


# In[39]:


df_codes_dev_ner_final_disc['start'] = df_codes_dev_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_dev_ner_final_disc['end'] = df_codes_dev_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[-1]))


# In[40]:


df_codes_dev_ner_final_disc = eliminate_overlap(df_ann=df_codes_dev_ner_final_disc)


# Concatenate continuous and discontinuous annotations:

# In[41]:


# Concat
cols_concat = ['doc_id', 'type', 'code', 'word', 'location', 'start', 'end', 'disc']
df_codes_dev_ner_final = pd.concat([df_codes_dev_ner_final_cont[cols_concat], 
                                      df_codes_dev_ner_final_disc[cols_concat]])


# Now, we remove the right-to-left (text wise) discontinuous annotations:

# In[42]:


df_codes_dev_ner_final['direction'] = df_codes_dev_ner_final.apply(check_ann_left_right_direction, axis=1)


# In[43]:


df_codes_dev_ner_final = df_codes_dev_ner_final[df_codes_dev_ner_final['direction']]


# We only select the annotations fully contained in a single sentence:

# In[44]:


# Sentence-Split data
ss_sub_corpus_path = ss_corpus_path + "dev/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_dev = load_ss_files(ss_files, ss_sub_corpus_path)


# In[45]:


df_mult_sent_dev, df_one_sent_dev, df_no_sent_dev = check_ann_span_sent(df_ann=df_codes_dev_ner_final, 
                                                                             ss_dict=ss_dict_dev)


# In[46]:


df_codes_dev_ner_final = df_one_sent_dev.copy()


# In[47]:


print(df_codes_dev_ner_final.disc.value_counts())


# In[48]:


df_codes_dev_ner_final.sort_values(['doc_id', 'start', 'end'], inplace=True)


# In[49]:


# Code splitting


# In[50]:


if TYPE_TASK == 'd':
    df_codes_dev_ner_final["code_pre"] = df_codes_dev_ner_final["code"].apply(lambda x: x.split('.')[0])
    df_codes_dev_ner_final["code_suf"] = df_codes_dev_ner_final["code"].apply(lambda x: None if not '.' in x else x.split('.')[1])
else:
    df_codes_dev_ner_final["code_pre"] = df_codes_dev_ner_final["code"].apply(lambda x: x[:4])
    df_codes_dev_ner_final["code_suf"] = df_codes_dev_ner_final["code"].apply(lambda x: None if len(x) < 7 else x[4:7])


# In[51]:


print(df_codes_dev_ner_final.shape)


# ## Creation of annotated sequences

# In[52]:


train_dev_codes_pre = sorted(set(df_codes_dev_ner_final["code_pre"].values).union(set(
    df_codes_train_ner_final["code_pre"].values
))) 


# In[53]:


print(len(train_dev_codes_pre))


# In[54]:


train_dev_codes_suf = sorted(set(df_codes_dev_ner_final[df_codes_dev_ner_final['code_suf'].apply(lambda x: x is not None)]["code_suf"].values).union(set(df_codes_train_ner_final[df_codes_train_ner_final['code_suf'].apply(lambda x: x is not None)]["code_suf"].values))) 


# In[55]:


len(train_dev_codes_suf)


# In[56]:


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

# Add "O" label to code-suf, since some codes do not have suffix
code_suf_lab_encoder["O"] = i
code_suf_lab_decoder[i] = "O"


# In[57]:


print(len(iob_lab_encoder), len(iob_lab_decoder))


# In[58]:


print(len(code_pre_lab_encoder), len(code_pre_lab_decoder))


# In[59]:


print(len(code_suf_lab_encoder), len(code_suf_lab_decoder))


# In[60]:


# Text classification (later ignored)


# In[61]:


train_dev_codes = sorted(set(df_codes_dev_ner_final["code"].values).union(set(df_codes_train_ner_final["code"].values))) 


# In[62]:


print(len(train_dev_codes))


# In[63]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb_encoder = MultiLabelBinarizer()
mlb_encoder.fit([train_dev_codes])


# In[64]:


# "Impossible" codes mask


# In[65]:


train_dev_codes_pre_o_mask = np.ones(len(train_dev_codes_pre)+1)
train_dev_codes_pre_o_mask[-1] = -1

train_dev_codes_pre_suf_mask = np.ones(shape=(len(train_dev_codes_pre), len(train_dev_codes_suf) + 1)) # "O" label


# In[66]:


print(train_dev_codes_pre_o_mask.shape)


# In[67]:


print(train_dev_codes_pre_suf_mask.shape)


# ### Training corpus

# Only training texts with NER annotations are considered:

# In[68]:


train_doc_list = sorted(set(df_codes_train_ner_final["doc_id"]))


# In[69]:


train_ind, train_att, train_type, train_y, train_text_y, train_frag, train_start_end_frag,                 train_word_id = ss_create_input_data_ner(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_codes_train_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore text-level
                                    doc_list=train_doc_list, ss_dict=ss_dict_train,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, iob_lab_encoder, 
                                                      code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[70]:


train_iob_cont_y, train_iob_disc_y, train_code_pre_y, train_code_suf_y = train_y


# ### Development corpus
# 
# Only development texts with NER annotations are considered:

# In[71]:


dev_doc_list = sorted(set(df_codes_dev_ner_final["doc_id"]))


# In[72]:


dev_ind, dev_att, dev_type, dev_y, dev_text_y, dev_frag, dev_start_end_frag,                 dev_word_id = ss_create_input_data_ner(df_text=df_text_dev, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_train_ner_final, # ignore text-level
                                    doc_list=dev_doc_list, ss_dict=ss_dict_dev,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, iob_lab_encoder, 
                                                      code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[73]:


dev_iob_cont_y, dev_iob_disc_y, dev_code_pre_y, dev_code_suf_y = dev_y


# ### Test corpus
# 
# All test texts are considered:

# In[74]:


test_doc_list = sorted(set(df_text_test["doc_id"]))


# In[75]:


# Sentence-Split data


# In[76]:


ss_sub_corpus_path = ss_corpus_path + "test/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_test = load_ss_files(ss_files, ss_sub_corpus_path)


# In[77]:


test_ind, test_att, test_type, test_y, test_text_y, test_frag, test_start_end_frag,                 test_word_id = ss_create_input_data_ner(df_text=df_text_test, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore text-level
                                    doc_list=test_doc_list, ss_dict=ss_dict_test,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, iob_lab_encoder, 
                                                      code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[78]:


test_iob_cont_y, test_iob_disc_y, test_code_pre_y, test_code_suf_y = test_y


# ### Training & Development corpus
# 
# We merge the previously generated datasets:

# In[79]:


# Indices
train_dev_ind = np.concatenate((train_ind, dev_ind))


# In[80]:


print(train_dev_ind.shape)


# In[81]:


# Attention
train_dev_att = np.concatenate((train_att, dev_att))


# In[82]:


print(train_dev_att.shape)


# In[83]:


# IOB-2 Continuous y
train_dev_iob_cont_y = np.concatenate((train_iob_cont_y, dev_iob_cont_y))


# In[84]:


print(train_dev_iob_cont_y.shape)


# In[85]:


# IOB-2 Discontinuous y
train_dev_iob_disc_y = np.concatenate((train_iob_disc_y, dev_iob_disc_y))


# In[86]:


print(train_dev_iob_disc_y.shape)


# In[87]:


# Clinical-Coding-prefix y
train_dev_code_pre_y = np.concatenate((train_code_pre_y, dev_code_pre_y))


# In[88]:


print(train_dev_code_pre_y.shape)


# In[89]:


# Clinical-Coding-suffix y
train_dev_code_suf_y = np.concatenate((train_code_suf_y, dev_code_suf_y))


# In[90]:


print(train_dev_code_suf_y.shape)


# ## Fine-tuning

# In[91]:


# Set memory growth


# In[92]:


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)


# In[93]:


for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# In[94]:


from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification

if model_name.split('_')[0] in ('beto', 'mbert'):
    model = TFBertForTokenClassification.from_pretrained(model_path, from_pt=True)
    
else: # default
    model = TFXLMRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)


# In[95]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

iob_num_labels = len(iob_lab_encoder)
code_pre_num_labels = len(code_pre_lab_encoder)
code_suf_num_labels = len(code_suf_lab_encoder)

input_ids = Input(shape=(SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int64')

out_seq = model.layers[0](input_ids=input_ids, attention_mask=attention_mask)[0] # take the output sub-token sequence 

# IOB-2 Continuous
out_iob_cont = Dense(units=iob_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_iob_cont_model = Activation(activation='softmax', name='iob_cont_output')(out_iob_cont)
# IOB-2 Discontinuous
out_iob_disc = Dense(units=iob_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_iob_disc_model = Activation(activation='softmax', name='iob_disc_output')(out_iob_disc)

# Code-pre
out_code_pre = Dense(units=code_pre_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_code_pre_model = Activation(activation='softmax', name='code_pre_output')(out_code_pre)
# Code-suf
out_code_suf = Dense(units=code_suf_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_code_suf_model = Activation(activation='softmax', name='code_suf_output')(out_code_suf)

model = Model(inputs=[input_ids, attention_mask], outputs=[out_iob_cont_model, out_iob_disc_model, out_code_pre_model, out_code_suf_model])


# In[96]:


print(model.summary())


# In[97]:


print(model.input)


# In[98]:


print(model.output)


# In[99]:


df_dev_gs = format_codiesp_x_gs(dev_gs_path)
df_test_gs = format_codiesp_x_gs(test_gs_path)

valid_codes_D = set(pd.read_csv(codes_d_path, sep='\t', header=None, 
                                  usecols=[0])[0].tolist())
valid_codes_D = set([x.lower() for x in valid_codes_D])


# In[101]:


import tensorflow_addons as tfa
import time

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'iob_cont_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE),
        'iob_disc_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE),
        'code_pre_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE),
        'code_suf_output': TokenClassificationLoss(from_logits=LOGITS, ignore_val=IGNORE_VALUE)}
loss_weights = {'iob_cont_output': 1, 'iob_disc_output': 1, 
                'code_pre_output': 1, 'code_suf_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

start_time = time.time()

history = model.fit(x={'input_ids': train_dev_ind, 'attention_mask': train_dev_att}, 
                    y={'iob_cont_output': train_dev_iob_cont_y, 'iob_disc_output': train_dev_iob_disc_y, 
                       'code_pre_output': train_dev_code_pre_y, 'code_suf_output': train_dev_code_suf_y}, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                    verbose=2)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print()


# ## Evaluation

# ### Development

# In[102]:


start_time = time.time()


# In[103]:


dev_preds = model.predict({'input_ids': dev_ind, 
                           'attention_mask': dev_att})
iob_cont_dev_preds = dev_preds[0]
iob_disc_dev_preds = dev_preds[1]
code_pre_dev_preds = dev_preds[2]
code_suf_dev_preds = dev_preds[3]


# In[104]:


# Evaluate NER performance
df_dev_preds_iob = ner_preds_brat_format(
    doc_list=dev_doc_list, fragments=dev_frag, preds=[iob_cont_dev_preds, iob_disc_dev_preds], 
    start_end=dev_start_end_frag, word_id=dev_word_id, 
    lab_decoder_list=[iob_lab_decoder], 
    df_text=df_text_dev, text_col=text_col, strategy=EVAL_STRATEGY, 
    subtask='norm-iob_cont_disc', type_tokenizer=type_tokenizer
)


# In[105]:


# Adapt to CodiEsp format
df_dev_preds_iob['label_pred'] = TYPE_ANN
df_dev_preds_iob['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_dev_preds_iob.iterrows()]
df_dev_preds_iob['code'] = 'n23' if TYPE_ANN == 'DIAGNOSTICO' else 'bn20'
df_dev_preds_iob = df_dev_preds_iob[['clinical_case', 'pos_pred', 'label_pred', 'code']]


# In[106]:


print(calculate_codiesp_ner_metrics(df_gs=df_dev_gs[df_dev_gs['label_gs'] == TYPE_ANN], 
                            df_pred=format_codiesp_x_pred_df(df_run=df_dev_preds_iob,
                                                             valid_codes=valid_codes_D)))


# In[107]:


df_dev_preds = ner_preds_brat_format(doc_list=dev_doc_list, fragments=dev_frag, 
                                    preds=[iob_cont_dev_preds, iob_disc_dev_preds, 
                                           code_pre_dev_preds, code_suf_dev_preds],
                                    start_end=dev_start_end_frag, word_id=dev_word_id, 
                                    lab_decoder_list=[iob_lab_decoder, code_pre_lab_decoder, code_suf_lab_decoder], 
                                    df_text=df_text_dev, 
                                    text_col=text_col, strategy=EVAL_STRATEGY, subtask=subtask_ann + '_mask',
                                    mention_strat=mention_strat, type_tokenizer=type_tokenizer,
                                    code_sep=CODE_SEP,
                                    codes_pre_suf_mask=train_dev_codes_pre_suf_mask,
                                    codes_pre_o_mask=train_dev_codes_pre_o_mask)


# In[108]:


# Adapt to CodiEsp format
df_dev_preds['label_pred'] = TYPE_ANN
df_dev_preds['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_dev_preds.iterrows()]
df_dev_preds = df_dev_preds.rename(columns={'code_pred': 'code'})
df_dev_preds = df_dev_preds[['clinical_case', 'pos_pred', 'label_pred', 'code']]


# In[109]:


print(calculate_codiesp_x_metrics(df_gs=df_dev_gs[df_dev_gs['label_gs'] == TYPE_ANN], 
                            df_pred=format_codiesp_x_pred_df(df_run=df_dev_preds,
                                                             valid_codes=valid_codes_D)))


# In[110]:


end_time = time.time()
print("--- Dev evaluation: %s seconds ---" % (end_time - start_time))
print()


# ### Test

# In[111]:


start_time = time.time()


# In[112]:


test_preds = model.predict({'input_ids': test_ind, 
                           'attention_mask': test_att})
iob_cont_test_preds = test_preds[0]
iob_disc_test_preds = test_preds[1]
code_pre_test_preds = test_preds[2]
code_suf_test_preds = test_preds[3]


# In[113]:


np.save(file="iob_cont_test_preds_" + JOB_NAME + ".npy", arr=iob_cont_test_preds)
np.save(file="iob_disc_test_preds_" + JOB_NAME + ".npy", arr=iob_disc_test_preds)


# In[114]:


df_test_preds_iob = ner_preds_brat_format(
    doc_list=test_doc_list, fragments=test_frag, preds=[iob_cont_test_preds, iob_disc_test_preds], 
    start_end=test_start_end_frag, word_id=test_word_id, 
    lab_decoder_list=[iob_lab_decoder], 
    df_text=df_text_test, text_col=text_col, strategy=EVAL_STRATEGY, 
    subtask='norm-iob_cont_disc', type_tokenizer=type_tokenizer
)


# In[115]:


# Adapt to CodiEsp format
df_test_preds_iob['label_pred'] = TYPE_ANN
df_test_preds_iob['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_test_preds_iob.iterrows()]
df_test_preds_iob['code'] = 'n23' if TYPE_ANN == 'DIAGNOSTICO' else 'bn20'
df_test_preds_iob = df_test_preds_iob[['clinical_case', 'pos_pred', 'label_pred', 'code']]


# In[116]:


print(calculate_codiesp_ner_metrics(df_gs=df_test_gs[df_test_gs['label_gs'] == TYPE_ANN], 
                            df_pred=format_codiesp_x_pred_df(df_run=df_test_preds_iob,
                                                             valid_codes=valid_codes_D)))


# In[117]:


# Save final results DF
df_test_preds_iob.to_csv("df_test_preds_ner_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')


# In[118]:


df_test_preds = ner_preds_brat_format(doc_list=test_doc_list, fragments=test_frag, 
                                    preds=[iob_cont_test_preds, iob_disc_test_preds, 
                                           code_pre_test_preds, code_suf_test_preds],
                                    start_end=test_start_end_frag, word_id=test_word_id, 
                                    lab_decoder_list=[iob_lab_decoder, code_pre_lab_decoder, code_suf_lab_decoder], 
                                    df_text=df_text_test, 
                                    text_col=text_col, strategy=EVAL_STRATEGY, subtask=subtask_ann + '_mask',
                                    mention_strat=mention_strat, type_tokenizer=type_tokenizer,
                                    code_sep=CODE_SEP,
                                    codes_pre_suf_mask=train_dev_codes_pre_suf_mask,
                                    codes_pre_o_mask=train_dev_codes_pre_o_mask)


# In[119]:


# Adapt to CodiEsp format
df_test_preds['label_pred'] = TYPE_ANN
df_test_preds['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_test_preds.iterrows()]
df_test_preds = df_test_preds.rename(columns={'code_pred': 'code'})
df_test_preds = df_test_preds[['clinical_case', 'pos_pred', 'label_pred', 'code']]


# In[120]:


print(calculate_codiesp_x_metrics(df_gs=df_test_gs[df_test_gs['label_gs'] == TYPE_ANN], 
                            df_pred=format_codiesp_x_pred_df(df_run=df_test_preds,
                                                             valid_codes=valid_codes_D)))


# In[121]:


end_time = time.time()
print("--- Test evaluation: %s seconds ---" % (end_time - start_time))


# In[122]:


# Save final results DF
df_test_preds.to_csv("df_test_preds_norm_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')

