#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Input args (4): model-name, nº epochs, random seed, random exec
model_name = "mbert_galen"
hier_iob_exec = 1
epochs = 68
random_seed = 1
random_exec = 1

"""
Estimated hyper-parameters for the remaining models:
mBERT: epochs=90, random_seed=1

BETO-Galén: epochs=81, random_seed=0
BETO: epochs=91, random_seed=3

XLM-R-Galén: epochs=77, random_seed=0
XLM-R: epochs=82, random_seed=2
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


# In[2]:


root_path = "../"


# In[3]:


from transformers import BertTokenizerFast, XLMRobertaTokenizerFast

# All variables that depend on model_name
if model_name == 'beto':
    model_path = root_path + "models/" + "BERT/pytorch/BETO/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    MENTION_START_TOK_ID = 10
    MENTION_END_TOK_ID = 11
    ens_multi_model_name = ["beto_beto_galen", "beto_mbert_xlmr"]
    
elif model_name == 'beto_galen':
    model_path = root_path + "models/" + "BERT/pytorch/BETO-Galen/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    MENTION_START_TOK_ID = 10
    MENTION_END_TOK_ID = 11
    ens_multi_model_name = ["beto_beto_galen", "beto_galen_mbert_galen_xlmr_galen"]
    
elif model_name == 'mbert':
    model_path = root_path + "models/" + "BERT/pytorch/mBERT/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    MENTION_START_TOK_ID = 10
    MENTION_END_TOK_ID = 11
    ens_multi_model_name = ["mbert_mbert_galen", "beto_mbert_xlmr"]
    
elif model_name == 'mbert_galen':
    model_path = root_path + "models/" + "BERT/pytorch/mBERT-Galen/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    MENTION_START_TOK_ID = 10
    MENTION_END_TOK_ID = 11
    ens_multi_model_name = ["mbert_mbert_galen", "beto_galen_mbert_galen_xlmr_galen"]
    
elif model_name == 'xlmr':
    model_path = root_path + "models/" + "XLM-R/pytorch/xlm-roberta-base/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<mention>', '</mention>']})
    MENTION_START_TOK_ID, MENTION_END_TOK_ID = tokenizer.additional_special_tokens_ids
    ens_multi_model_name = ["xlmr_xlmr_galen", "beto_mbert_xlmr"]
    
else: # default
    model_path = root_path + "models/" + "XLM-R/pytorch/XLM-R-Galen/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<mention>', '</mention>']})
    MENTION_START_TOK_ID, MENTION_END_TOK_ID = tokenizer.additional_special_tokens_ids
    ens_multi_model_name = ["xlmr_xlmr_galen", "beto_galen_mbert_galen_xlmr_galen"]


# In[4]:


utils_path = root_path + "utils/"
corpus_path = root_path + "datasets/final_dataset_v4_to_publish/"
ss_corpus_path = root_path + "datasets/CodiEsp-SSplit-text/"
dev_gs_path = corpus_path + "dev/devX.tsv"
test_gs_path = corpus_path + "test/testX.tsv"


# In[5]:


import tensorflow as tf

# Auxiliary components
sys.path.insert(0, utils_path)
from nlp_utils import *

print(sys.path)

# Hyper-parameters
type_tokenizer = "transformers"

subtask = 'norm'
subtask_ann = subtask + '-iob_cont_disc'
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
TYPE_TASK = TYPE_ANN[0].lower()
CODE_SEP = '.' if TYPE_ANN == 'DIAGNOSTICO' else ''

MENTION_LIMIT_EMB_VALUE = 1

tf.random.set_seed(random_seed)

RES_DIR = root_path + "results/CodiEsp/final_exec/"

df_iob_preds_name = TYPE_TASK + "_hier_task_iob_" + model_name + "_" + str(hier_iob_exec)
EMPTY_SAMPLES = False

JOB_NAME = TYPE_TASK + "_hier_task_cls_" + model_name + "_" + str(random_exec)
print("\n" + JOB_NAME)

## Additional predictions
# Ensemble
RES_DIR_ENS = RES_DIR + "ensemble/"
ens_model_name_arr = [model_name] + ens_multi_model_name
df_iob_preds_name_ens = TYPE_TASK + "_hier_task_iob_"

# Multi-task
multi_iob_exec = random_exec
df_iob_preds_name_multi = "ner_" + TYPE_TASK + "_multi_task_" + model_name + "_" + str(multi_iob_exec)


# In[6]:


codes_d_path = root_path + "datasets/final_dataset_v4_to_publish/codiesp_codes/codiesp-" + TYPE_TASK.upper() + "_codes.tsv"


# ## Load text
# 
# Firstly, all text files from training and development Cantemist corpora are loaded in different dataframes.
# 
# Also, NER-annotations are loaded.

# ### Training corpus

# In[7]:


train_path = corpus_path + "train/text_files/"
train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and f.split('.')[-1] == "txt"]
train_data = load_text_files(train_files, train_path)
df_text_train = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in train_files], 'raw_text': train_data})


# ### Development corpus

# In[8]:


dev_path = corpus_path + "dev/text_files/"
dev_files = [f for f in os.listdir(dev_path) if os.path.isfile(dev_path + f) and f.split('.')[-1] == "txt"]
dev_data = load_text_files(dev_files, dev_path)
df_text_dev = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in dev_files], 'raw_text': dev_data})


# ### Test corpus

# In[9]:


test_path = corpus_path + "test/text_files/"
test_files = [f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and f.split('.')[-1] == 'txt']
test_data = load_text_files(test_files, test_path)
df_text_test = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in test_files], 'raw_text': test_data})


# ## Process NER annotations
# 
# We load and pre-process the NER annotations in BRAT format available for the CodiEsp-X subtask.

# ### Training corpus

# In[10]:


df_codes_train_ner = pd.read_table(corpus_path + "train/trainX.tsv", sep='\t', header=None)
df_codes_train_ner.columns = ["doc_id", "type", "code", "word", "location"]
df_codes_train_ner = df_codes_train_ner[~df_codes_train_ner[['doc_id', 'type', 'location']].duplicated(keep='first')]
df_codes_train_ner['disc'] = df_codes_train_ner['location'].apply(lambda x: ';' in x)


# Select one type of annotations:

# In[11]:


df_codes_train_ner = df_codes_train_ner[df_codes_train_ner['type'] == TYPE_ANN]


# Split discontinuous annotations:

# In[12]:


df_codes_train_ner_final = process_labels_norm_prueba(df_ann=df_codes_train_ner[["doc_id", "type", "code", "word", "location"]])


# Remove annotations of zero length:

# In[13]:


df_codes_train_ner_final['length'] = df_codes_train_ner_final.apply(lambda x: x['end'] - x['start'], axis=1)
df_codes_train_ner_final = df_codes_train_ner_final[df_codes_train_ner_final['length'] > 0]


# Separate continuous and discontinuous annotations:

# In[14]:


# Continiuous
df_codes_train_ner_final_cont = df_codes_train_ner_final[df_codes_train_ner_final['disc'] == 0].copy()
df_codes_train_ner_final_cont['disc'] = df_codes_train_ner_final_cont['disc'].astype(bool)


# In[15]:


# Discontinuous
df_codes_train_ner_final_disc = restore_disc_ann(df_ann=df_codes_train_ner[df_codes_train_ner['disc']], 
                    df_ann_final=df_codes_train_ner_final[df_codes_train_ner_final['disc'] == 1])


# In[16]:


df_codes_train_ner_final_disc['start'] = df_codes_train_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_train_ner_final_disc['end'] = df_codes_train_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[-1]))


# Concatenate continuous and discontinuous annotations:

# In[17]:


# Concat
cols_concat = ['doc_id', 'type', 'code', 'word', 'location', 'start', 'end', 'disc']
df_codes_train_ner_final = pd.concat([df_codes_train_ner_final_cont[cols_concat], 
                                      df_codes_train_ner_final_disc[cols_concat]])


# Now, we remove the right-to-left (text wise) discontinuous annotations:

# In[18]:


df_codes_train_ner_final['direction'] = df_codes_train_ner_final.apply(check_ann_left_right_direction, axis=1)


# In[19]:


df_codes_train_ner_final = df_codes_train_ner_final[df_codes_train_ner_final['direction']]


# We only select the annotations fully contained in a single sentence:

# In[20]:


# Sentence-Split data
ss_sub_corpus_path = ss_corpus_path + "train/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_train = load_ss_files(ss_files, ss_sub_corpus_path)


# In[21]:


df_mult_sent_train, df_one_sent_train, df_no_sent_train = check_ann_span_sent(df_ann=df_codes_train_ner_final, 
                                                                             ss_dict=ss_dict_train)


# In[22]:


df_codes_train_ner_final = df_one_sent_train.copy()


# In[23]:


print(df_codes_train_ner_final.disc.value_counts())


# In[24]:


df_codes_train_ner_final.sort_values(['doc_id', 'start', 'end'], inplace=True)


# In[25]:


# Code splitting


# In[26]:


if TYPE_TASK == 'd':
    df_codes_train_ner_final["code_pre"] = df_codes_train_ner_final["code"].apply(lambda x: x.split('.')[0])
    df_codes_train_ner_final["code_suf"] = df_codes_train_ner_final["code"].apply(lambda x: None if not '.' in x else x.split('.')[1])
else:
    df_codes_train_ner_final["code_pre"] = df_codes_train_ner_final["code"].apply(lambda x: x[:4])
    df_codes_train_ner_final["code_suf"] = df_codes_train_ner_final["code"].apply(lambda x: None if len(x) < 7 else x[4:7])


# ### Development corpus

# In[27]:


df_codes_dev_ner = pd.read_table(corpus_path + "dev/devX.tsv", sep='\t', header=None)
df_codes_dev_ner.columns = ["doc_id", "type", "code", "word", "location"]
df_codes_dev_ner = df_codes_dev_ner[~df_codes_dev_ner[['doc_id', 'type', 'location']].duplicated(keep='first')]
df_codes_dev_ner['disc'] = df_codes_dev_ner['location'].apply(lambda x: ';' in x)


# Select one type of annotations:

# In[28]:


df_codes_dev_ner = df_codes_dev_ner[df_codes_dev_ner['type'] == TYPE_ANN]


# Split discontinuous annotations:

# In[29]:


df_codes_dev_ner_final = process_labels_norm_prueba(df_ann=df_codes_dev_ner[["doc_id", "type", "code", "word", "location"]])


# Remove annotations of zero length:

# In[30]:


df_codes_dev_ner_final['length'] = df_codes_dev_ner_final.apply(lambda x: x['end'] - x['start'], axis=1)
df_codes_dev_ner_final = df_codes_dev_ner_final[df_codes_dev_ner_final['length'] > 0]


# Separate continuous and discontinuous annotations:

# In[31]:


# Continiuous
df_codes_dev_ner_final_cont = df_codes_dev_ner_final[df_codes_dev_ner_final['disc'] == 0].copy()
df_codes_dev_ner_final_cont['disc'] = df_codes_dev_ner_final_cont['disc'].astype(bool)


# In[32]:


# Discontinuous
df_codes_dev_ner_final_disc = restore_disc_ann(df_ann=df_codes_dev_ner[df_codes_dev_ner['disc']], 
                    df_ann_final=df_codes_dev_ner_final[df_codes_dev_ner_final['disc'] == 1])


# In[33]:


df_codes_dev_ner_final_disc['start'] = df_codes_dev_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[0]))
df_codes_dev_ner_final_disc['end'] = df_codes_dev_ner_final_disc['location'].apply(lambda x: int(x.split(' ')[-1]))


# Concatenate continuous and discontinuous annotations:

# In[34]:


# Concat
cols_concat = ['doc_id', 'type', 'code', 'word', 'location', 'start', 'end', 'disc']
df_codes_dev_ner_final = pd.concat([df_codes_dev_ner_final_cont[cols_concat], 
                                      df_codes_dev_ner_final_disc[cols_concat]])


# Now, we remove the right-to-left (text wise) discontinuous annotations:

# In[35]:


df_codes_dev_ner_final['direction'] = df_codes_dev_ner_final.apply(check_ann_left_right_direction, axis=1)


# In[36]:


df_codes_dev_ner_final = df_codes_dev_ner_final[df_codes_dev_ner_final['direction']]


# We only select the annotations fully contained in a single sentence:

# In[37]:


# Sentence-Split data
ss_sub_corpus_path = ss_corpus_path + "dev/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_dev = load_ss_files(ss_files, ss_sub_corpus_path)


# In[38]:


df_mult_sent_dev, df_one_sent_dev, df_no_sent_dev = check_ann_span_sent(df_ann=df_codes_dev_ner_final, 
                                                                             ss_dict=ss_dict_dev)


# In[39]:


df_codes_dev_ner_final = df_one_sent_dev.copy()


# In[40]:


print(df_codes_dev_ner_final.disc.value_counts())


# In[41]:


df_codes_dev_ner_final.sort_values(['doc_id', 'start', 'end'], inplace=True)


# In[42]:


# Code splitting


# In[43]:


if TYPE_TASK == 'd':
    df_codes_dev_ner_final["code_pre"] = df_codes_dev_ner_final["code"].apply(lambda x: x.split('.')[0])
    df_codes_dev_ner_final["code_suf"] = df_codes_dev_ner_final["code"].apply(lambda x: None if not '.' in x else x.split('.')[1])
else:
    df_codes_dev_ner_final["code_pre"] = df_codes_dev_ner_final["code"].apply(lambda x: x[:4])
    df_codes_dev_ner_final["code_suf"] = df_codes_dev_ner_final["code"].apply(lambda x: None if len(x) < 7 else x[4:7])


# ## Creation of annotated sequences
# 
# We create the corpus used to fine-tune the transformer model on a NER task. In this way, we split the texts into sentences, and convert them into sequences of subtokens. Also, each generated subtoken is assigned a NER label in IOB-2 format.

# In[44]:


train_dev_codes_pre = sorted(set(df_codes_dev_ner_final["code_pre"].values).union(set(
    df_codes_train_ner_final["code_pre"].values
)))


# In[45]:


len(train_dev_codes_pre)


# In[46]:


train_dev_codes_suf = sorted(set(df_codes_dev_ner_final[df_codes_dev_ner_final['code_suf'].apply(lambda x: x is not None)]["code_suf"].values).union(set(df_codes_train_ner_final[df_codes_train_ner_final['code_suf'].apply(lambda x: x is not None)]["code_suf"].values))) 


# In[47]:


len(train_dev_codes_suf)


# In[48]:


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


# In[49]:


print(len(iob_lab_encoder), len(iob_lab_decoder))


# In[50]:


print(len(code_pre_lab_encoder), len(code_pre_lab_decoder))


# In[51]:


print(len(code_suf_lab_encoder), len(code_suf_lab_decoder))


# In[52]:


# Text classification (later ignored)


# In[53]:


train_dev_codes = sorted(set(df_codes_dev_ner_final["code"].values).union(set(df_codes_train_ner_final["code"].values))) 


# In[54]:


print(len(train_dev_codes))


# In[55]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb_encoder = MultiLabelBinarizer()
mlb_encoder.fit([train_dev_codes])


# ### Training corpus

# Only training texts with NER annotations are considered:

# In[56]:


train_doc_list = sorted(set(df_codes_train_ner_final["doc_id"]))


# In[57]:


train_ind, train_att, train_type, train_y, train_text_y, train_frag, train_start_end_frag,                 train_word_id = ss_create_input_data_ner(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore
                                    doc_list=train_doc_list, ss_dict=ss_dict_train,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, iob_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# ### Development corpus
# 
# Only development texts with NER annotations are considered:

# In[58]:


dev_doc_list = sorted(set(df_codes_dev_ner_final["doc_id"]))


# In[59]:


dev_ind, dev_att, dev_type, dev_y, dev_text_y, dev_frag, dev_start_end_frag,                 dev_word_id = ss_create_input_data_ner(df_text=df_text_dev, text_col=text_col, 
                                    df_ann=df_codes_train_ner_final, df_ann_text=df_codes_train_ner_final, # ignore
                                    doc_list=dev_doc_list, ss_dict=ss_dict_dev,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, iob_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# ### Test corpus
# 
# All test texts are considered:

# In[60]:


test_doc_list = sorted(set(df_text_test["doc_id"]))


# In[61]:


# Sentence-Split data


# In[62]:


ss_sub_corpus_path = ss_corpus_path + "test/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_test = load_ss_files(ss_files, ss_sub_corpus_path)


# In[63]:


test_ind, test_att, test_type, test_y, test_text_y, test_frag, test_start_end_frag,                 test_word_id = ss_create_input_data_ner(df_text=df_text_test, text_col=text_col, 
                                    df_ann=df_codes_dev_ner_final, df_ann_text=df_codes_dev_ner_final, # ignore
                                    doc_list=test_doc_list, ss_dict=ss_dict_test,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, iob_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# ### Training & Development corpus
# 
# We merge the previously generated datasets:

# In[64]:


# Indices
train_dev_ind = np.concatenate((train_ind, dev_ind))


# In[65]:


print(train_dev_ind.shape)


# In[66]:


# Attention
train_dev_att = np.concatenate((train_att, dev_att))


# In[67]:


print(train_dev_att.shape)


# In[68]:


# Additional Embedding & CLS samples


# In[69]:


subtask_ann = subtask + '-iob_code_suf'


# In[70]:


df_codes_train_ner_final = df_codes_train_ner_final.rename(columns={"doc_id": "clinical_case"})


# In[71]:


train_cls_y, train_cls_ind, df_train_cls_ann = create_cls_emb_y_samples(
    df_ann=df_codes_train_ner_final, doc_list=train_doc_list, arr_frag=train_frag,
    arr_start_end=train_start_end_frag, arr_word_id=train_word_id, arr_ind=train_ind,
    seq_len=SEQ_LEN, empty_samples=EMPTY_SAMPLES, subtask=subtask_ann
)


# In[72]:


train_cls_code_pre_y = np.array([code_pre_lab_encoder[sample[0][-1][0]] for sample in train_cls_y])
train_cls_code_suf_y = np.array([code_suf_lab_encoder[sample[0][-1][1] if sample[0][-1][1] is not None else "O"]                                  for sample in train_cls_y])
train_cls_emb_y = np.array([sample[1] for sample in train_cls_y])


# In[73]:


df_codes_dev_ner_final = df_codes_dev_ner_final.rename(columns={"doc_id": "clinical_case"})


# In[74]:


dev_cls_y, dev_cls_ind, df_dev_cls_ann = create_cls_emb_y_samples(
    df_ann=df_codes_dev_ner_final, doc_list=dev_doc_list, arr_frag=dev_frag,
    arr_start_end=dev_start_end_frag, arr_word_id=dev_word_id, arr_ind=dev_ind,
    seq_len=SEQ_LEN, empty_samples=EMPTY_SAMPLES, subtask=subtask_ann
)


# In[75]:


dev_cls_code_pre_y = np.array([code_pre_lab_encoder[sample[0][-1][0]] for sample in dev_cls_y])
dev_cls_code_suf_y = np.array([code_suf_lab_encoder[sample[0][-1][1] if sample[0][-1][1] is not None else "O"]                                for sample in dev_cls_y])
dev_cls_emb_y = np.array([sample[1] for sample in dev_cls_y])


# In[76]:


# Insert mention-delimiter tokens


# In[77]:


def create_ind_emb_mention_sep(arr_ind, arr_emb, tokenizer=tokenizer,
                               ann_tok_start=MENTION_START_TOK_ID, 
                               ann_tok_end=MENTION_END_TOK_ID, 
                               emb_val_limit=MENTION_LIMIT_EMB_VALUE):
    """
    Generate new indices and embeddings arrays by separating the annotated mentions
    from the remaining text using 2 special starting and end tokens around
    each mention.
    """
    
    arr_ind_sep = []
    arr_emb_sep = []
    assert arr_ind.shape == arr_emb.shape
    for i_frag in range(len(arr_ind)):
        frag_ind = arr_ind[i_frag]
        frag_emb = arr_emb[i_frag]
        seq_len = len(frag_ind)
        frag_ind_sep = []
        frag_emb_sep = []
        left = 0
        while frag_ind[left] != tokenizer.sep_token_id:
            if frag_emb[left] == 1:
                frag_ind_sep.append(ann_tok_start)
                frag_ind_sep.append(frag_ind[left])
                frag_emb_sep += [emb_val_limit, 1]
                right = left + 1
                while frag_ind[right] != tokenizer.sep_token_id and frag_emb[right] == 1:
                    frag_ind_sep.append(frag_ind[right])
                    frag_emb_sep.append(1)
                    right += 1
                frag_ind_sep.append(ann_tok_end)
                frag_emb_sep.append(emb_val_limit)

                left = right 
            else:
                frag_ind_sep.append(frag_ind[left])
                frag_emb_sep.append(0)
                left += 1
        
        assert left < seq_len
        assert len(frag_ind_sep) == len(frag_emb_sep)
        # Add [SEP] token
        frag_ind_sep.append(tokenizer.sep_token_id)
        frag_emb_sep.append(0)
        arr_ind_sep.append(frag_ind_sep)
        arr_emb_sep.append(frag_emb_sep)

    return arr_ind_sep, arr_emb_sep


# In[78]:


train_cls_ind, train_cls_emb = create_ind_emb_mention_sep(
    arr_ind=train_cls_ind, 
    arr_emb=train_cls_emb_y
)


# In[79]:


dev_cls_ind, dev_cls_emb = create_ind_emb_mention_sep(
    arr_ind=dev_cls_ind, 
    arr_emb=dev_cls_emb_y
)


# In[80]:


## PAD inidices and create attention masks


# In[81]:


def calculate_max_seq(list_list_arr_seq):
    max_len = 0
    for list_arr_seq in list_list_arr_seq:
        for arr_seq in list_arr_seq:
            cur_len = len(arr_seq)
            if cur_len > max_len:
                max_len = cur_len
    return max_len


# In[82]:


print(calculate_max_seq([train_cls_ind, dev_cls_ind]))


# In[83]:


MAX_SEQ_LEN = 256


# In[84]:


def pad_inidices_emb_create_att(list_ind_sep, list_emb_sep, 
                                tokenizer=tokenizer, 
                                max_seq_len=MAX_SEQ_LEN,
                                emb_pad_val=0):
    arr_ind_sep = []
    arr_emb_sep = []
    arr_att_sep = []
    # We pad the indices and embeddings arrays and generate the attention mask
    assert len(list_ind_sep) == len(list_emb_sep)
    for i in range(len(list_ind_sep)):
        cur_ind_sep = list_ind_sep[i]
        cur_emb_sep = list_emb_sep[i]
        assert len(cur_ind_sep) == len(cur_emb_sep)
        cur_len = len(cur_ind_sep)
        cur_pad_len = max_seq_len - cur_len
        # Padding
        arr_ind_sep.append(cur_ind_sep + [tokenizer.pad_token_id] * cur_pad_len)
        arr_emb_sep.append(cur_emb_sep + [emb_pad_val] * cur_pad_len)
        # Attention
        arr_att_sep.append([1] * cur_len + [0] * cur_pad_len)
        
    return np.array(arr_ind_sep), np.array(arr_emb_sep), np.array(arr_att_sep)


# In[85]:


train_cls_ind, train_cls_emb, train_cls_att = pad_inidices_emb_create_att(
    list_ind_sep=train_cls_ind,
    list_emb_sep=train_cls_emb
)


# In[86]:


print(train_cls_code_pre_y.shape, train_cls_code_suf_y.shape, 
      train_cls_ind.shape, train_cls_emb.shape, train_cls_att.shape)


# In[87]:


dev_cls_ind, dev_cls_emb, dev_cls_att = pad_inidices_emb_create_att(
    list_ind_sep=dev_cls_ind,
    list_emb_sep=dev_cls_emb
)


# In[88]:


print(dev_cls_code_pre_y.shape, dev_cls_code_suf_y.shape, 
      dev_cls_ind.shape, dev_cls_emb.shape, dev_cls_att.shape)


# In[89]:


# Train + Dev
train_dev_cls_code_pre_y = np.concatenate((train_cls_code_pre_y, dev_cls_code_pre_y))
train_dev_cls_code_suf_y = np.concatenate((train_cls_code_suf_y, dev_cls_code_suf_y))
train_dev_cls_ind = np.concatenate((train_cls_ind, dev_cls_ind))
train_dev_cls_emb = np.concatenate((train_cls_emb, dev_cls_emb))
train_dev_cls_att = np.concatenate((train_cls_att, dev_cls_att))


# In[90]:


print(train_dev_cls_code_pre_y.shape, train_dev_cls_code_suf_y.shape, 
      train_dev_cls_ind.shape, train_dev_cls_emb.shape, train_dev_cls_att.shape)


# In[91]:


def format_ner_preds(ner_file_name, ner_dir=RES_DIR, doc_list=test_doc_list, arr_frag=test_frag,
                     arr_start_end=test_start_end_frag, arr_word_id=test_word_id, 
                     arr_ind=test_ind, prefix_name="df_test_preds_", suffix_name=".csv"):
    # Load file
    df_preds_ner = pd.read_csv(ner_dir + prefix_name + ner_file_name + ".csv", header=0, sep='\t')
    
    # Obtain CLS-samples to predict
    cls_y_ner, cls_ind_ner, df_cls_ann_ner = create_cls_emb_y_samples(
        df_ann=df_preds_ner, doc_list=doc_list, 
        arr_frag=arr_frag,
        arr_start_end=arr_start_end, arr_word_id=arr_word_id, 
        arr_ind=arr_ind,
        seq_len=SEQ_LEN, empty_samples=EMPTY_SAMPLES, subtask='ner'
    )
    cls_emb_y_ner = np.array([sample[1] for sample in cls_y_ner])
    print(df_preds_ner.shape, df_cls_ann_ner.shape)
    # Add special tokens to separate mentions
    cls_ind_ner, cls_emb_ner = create_ind_emb_mention_sep(
        arr_ind=cls_ind_ner, 
        arr_emb=cls_emb_y_ner
    )
    print(calculate_max_seq([cls_ind_ner]))
    # Add padding
    cls_ind_ner, cls_emb_ner, cls_att_ner = pad_inidices_emb_create_att(
        list_ind_sep=cls_ind_ner,
        list_emb_sep=cls_emb_ner
    )
    print(cls_ind_ner.shape, cls_emb_ner.shape, cls_att_ner.shape)
    return cls_ind_ner, cls_emb_ner, cls_att_ner, df_cls_ann_ner


# In[92]:


dev_cls_ind_ner, dev_cls_emb_ner, dev_cls_att_ner, df_dev_cls_ann_ner = format_ner_preds(
    ner_file_name=df_iob_preds_name,
    doc_list=dev_doc_list,
    arr_frag=dev_frag,
    arr_start_end=dev_start_end_frag,
    arr_word_id=dev_word_id,
    arr_ind=dev_ind,
    prefix_name="df_dev_preds_"
)


# In[93]:


test_cls_ind_ner, test_cls_emb_ner, test_cls_att_ner, df_test_cls_ann_ner = format_ner_preds(
    ner_file_name=df_iob_preds_name
)


# ## Fine-tuning
# 
# Using the corpus of labeled sentences, we fine-tune the model on a multi-label sentence classification task.

# In[94]:


# Set memory growth


# In[95]:


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)


# In[96]:


for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# In[97]:


if model_name.split('_')[0] in ('beto', 'mbert'):
    model = TFBertForTokenClassification_NerAnnEmb.from_pretrained(model_path, from_pt=True)
    
else: # default
    model = TFXLMRobertaForTokenClassification_NerAnnEmb.from_pretrained(model_path, from_pt=True)
    # Mentions start-end additional tokens
    model.resize_token_embeddings(len(tokenizer.get_vocab()))


# In[98]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

code_pre_num_labels = len(code_pre_lab_encoder) - 1 if not EMPTY_SAMPLES else len(code_pre_lab_encoder)
code_suf_num_labels = len(code_suf_lab_encoder) # Some code-suf can be "O"

input_ids = Input(shape=(MAX_SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(MAX_SEQ_LEN,), name='attention_mask', dtype='int64')
ner_ann_ids = Input(shape=(MAX_SEQ_LEN,), name='ner_ann_ids', dtype='int64')

out_cls = model.layers[0](input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          ner_ann_ids=ner_ann_ids)[0][:, 0, :] # take CLS token output representation

# Code-pre
out_cls_code_pre = Dense(units=code_pre_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_cls) # Multi-class classification 
out_cls_code_pre_model = Activation(activation='softmax', name='code_pre_cls_output')(out_cls_code_pre)

# Code-suf
out_cls_code_suf = Dense(units=code_suf_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_cls) # Multi-class classification 
out_cls_code_suf_model = Activation(activation='softmax', name='code_suf_cls_output')(out_cls_code_suf)


model = Model(inputs=[input_ids, attention_mask, ner_ann_ids], outputs=[out_cls_code_pre_model, out_cls_code_suf_model])


# In[99]:


print(model.summary())


# In[100]:


print(model.input)


# In[101]:


print(model.output)


# In[102]:


# GS data
df_dev_gs = format_codiesp_x_gs(dev_gs_path)
df_test_gs = format_codiesp_x_gs(test_gs_path)

valid_codes_D = set(pd.read_csv(codes_d_path, sep='\t', header=None, 
                                  usecols=[0])[0].tolist())
valid_codes_D = set([x.lower() for x in valid_codes_D])


# In[104]:


import tensorflow_addons as tfa
from tensorflow.keras import losses
import time

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'code_pre_cls_output': losses.SparseCategoricalCrossentropy(from_logits=LOGITS),
        'code_suf_cls_output': losses.SparseCategoricalCrossentropy(from_logits=LOGITS)}
loss_weights = {'code_pre_cls_output': 1, 'code_suf_cls_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

start_time = time.time()

history = model.fit(x={'input_ids': train_dev_cls_ind, 
                       'attention_mask': train_dev_cls_att,
                       'ner_ann_ids': train_dev_cls_emb}, 
                    y={'code_pre_cls_output': train_dev_cls_code_pre_y, 
                       'code_suf_cls_output': train_dev_cls_code_suf_y}, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, 
                    verbose=2)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print()


# ## Evaluation

# ### Development

# In[105]:


start_time = time.time()


# In[106]:


y_pred_dev_cls = model.predict({'input_ids': dev_cls_ind_ner, 
                                'attention_mask': dev_cls_att_ner,
                                'ner_ann_ids': dev_cls_emb_ner})


# In[107]:


df_dev_preds = cls_code_norm_preds_brat_format(
    y_pred_cls=y_pred_dev_cls, df_pred_ner=df_dev_cls_ann_ner, 
    code_decoder_list=[code_pre_lab_decoder, code_suf_lab_decoder],
    subtask=subtask_ann,
    code_sep=CODE_SEP,
    codes_pre_o_mask=None,
    codes_pre_suf_mask=None
)


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


y_pred_test_cls = model.predict({'input_ids': test_cls_ind_ner, 
                                 'attention_mask': test_cls_att_ner,
                                'ner_ann_ids': test_cls_emb_ner})


# In[113]:


np.save(file="test_preds_code_pre_" + JOB_NAME + ".npy", arr=y_pred_test_cls[0])


# In[114]:


np.save(file="test_preds_code_suf_" + JOB_NAME + ".npy", arr=y_pred_test_cls[1])


# In[115]:


df_test_preds = cls_code_norm_preds_brat_format(
    y_pred_cls=y_pred_test_cls, df_pred_ner=df_test_cls_ann_ner, 
    code_decoder_list=[code_pre_lab_decoder, code_suf_lab_decoder],
    subtask=subtask_ann,
    code_sep=CODE_SEP,
    codes_pre_o_mask=None,
    codes_pre_suf_mask=None
)


# In[116]:


# Adapt to CodiEsp format
df_test_preds['label_pred'] = TYPE_ANN
df_test_preds['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_test_preds.iterrows()]
df_test_preds = df_test_preds.rename(columns={'code_pred': 'code'})
df_test_preds = df_test_preds[['clinical_case', 'pos_pred', 'label_pred', 'code']]


# In[117]:


print(calculate_codiesp_x_metrics(df_gs=df_test_gs[df_test_gs['label_gs'] == TYPE_ANN], 
                            df_pred=format_codiesp_x_pred_df(df_run=df_test_preds,
                                                             valid_codes=valid_codes_D)))


# In[118]:


end_time = time.time()
print("--- Test evaluation: %s seconds ---" % (end_time - start_time))
print()


# In[119]:


# Save final results DF
df_test_preds.to_csv("df_test_preds_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')


# ### Additional predictions

# #### Ensemble

# In[120]:


for ens_name in ens_model_name_arr:
    test_cls_ind_ner, test_cls_emb_ner, test_cls_att_ner, df_test_cls_ann_ner = format_ner_preds(
        ner_file_name=df_iob_preds_name_ens + ens_name,
        ner_dir=RES_DIR_ENS
    )
    if (hier_iob_exec == 1) and (random_exec == 1):
        df_test_cls_ann_ner.to_csv(RES_DIR_ENS + "df_test_preds_ens_ner_" + ens_name + "_" + TYPE_TASK + "_hier_task_cls_" +                                    model_name + "_ann.csv", index=False, header=True, sep = '\t')
    
    start_time = time.time()
    y_pred_test_cls = model.predict({'input_ids': test_cls_ind_ner, 
                                 'attention_mask': test_cls_att_ner,
                                'ner_ann_ids': test_cls_emb_ner})
    np.save(file="test_preds_code_pre_ens_ner_" + ens_name + "_" + JOB_NAME + ".npy", arr=y_pred_test_cls[0])
    np.save(file="test_preds_code_suf_ens_ner_" + ens_name + "_" + JOB_NAME + ".npy", arr=y_pred_test_cls[1])
    df_test_preds = cls_code_norm_preds_brat_format(
        y_pred_cls=y_pred_test_cls, df_pred_ner=df_test_cls_ann_ner, 
        code_decoder_list=[code_pre_lab_decoder, code_suf_lab_decoder],
        subtask=subtask_ann,
        code_sep=CODE_SEP,
        codes_pre_o_mask=None,
        codes_pre_suf_mask=None
    )
    # Adapt to CodiEsp format
    df_test_preds['label_pred'] = TYPE_ANN
    df_test_preds['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_test_preds.iterrows()]
    df_test_preds = df_test_preds.rename(columns={'code_pred': 'code'})
    df_test_preds = df_test_preds[['clinical_case', 'pos_pred', 'label_pred', 'code']]
    print(calculate_codiesp_x_metrics(df_gs=df_test_gs[df_test_gs['label_gs'] == TYPE_ANN], 
                                      df_pred=format_codiesp_x_pred_df(df_run=df_test_preds,
                                                                       valid_codes=valid_codes_D)))
    end_time = time.time()
    print("--- Ensemble NER " + ens_name + " evaluation: %s seconds ---" % (end_time - start_time))
    print()
    # Save final results DF
    df_test_preds.to_csv("df_test_preds_ens_ner_" + ens_name + "_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')


# #### Multi-task NER

# In[135]:


test_cls_ind_ner, test_cls_emb_ner, test_cls_att_ner, df_test_cls_ann_ner = format_ner_preds(
    ner_file_name=df_iob_preds_name_multi 
)


# In[136]:


start_time = time.time()


# In[137]:


y_pred_test_cls = model.predict({'input_ids': test_cls_ind_ner, 
                                 'attention_mask': test_cls_att_ner,
                                'ner_ann_ids': test_cls_emb_ner})


# In[138]:


np.save(file="test_preds_code_pre_multi_task_ner_" + str(multi_iob_exec) + "_" + JOB_NAME + ".npy", arr=y_pred_test_cls[0])


# In[139]:


np.save(file="test_preds_code_suf_multi_task_ner_" + str(multi_iob_exec) + "_" + JOB_NAME + ".npy", arr=y_pred_test_cls[1])


# In[140]:


df_test_preds = cls_code_norm_preds_brat_format(
    y_pred_cls=y_pred_test_cls, df_pred_ner=df_test_cls_ann_ner, 
    code_decoder_list=[code_pre_lab_decoder, code_suf_lab_decoder],
    subtask=subtask_ann,
    code_sep=CODE_SEP,
    codes_pre_o_mask=None,
    codes_pre_suf_mask=None
)


# In[141]:


# Adapt to CodiEsp format
df_test_preds['label_pred'] = TYPE_ANN
df_test_preds['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_test_preds.iterrows()]
df_test_preds = df_test_preds.rename(columns={'code_pred': 'code'})
df_test_preds = df_test_preds[['clinical_case', 'pos_pred', 'label_pred', 'code']]


# In[142]:


print(calculate_codiesp_x_metrics(df_gs=df_test_gs[df_test_gs['label_gs'] == TYPE_ANN], 
                            df_pred=format_codiesp_x_pred_df(df_run=df_test_preds,
                                                             valid_codes=valid_codes_D)))


# In[143]:


end_time = time.time()
print("--- Multi-task NER evaluation: %s seconds ---" % (end_time - start_time))


# In[144]:


# Save final results DF
df_test_preds.to_csv("df_test_preds_multi_task_ner_" + str(multi_iob_exec) + "_" + JOB_NAME + ".csv", index=False, header=True, sep = '\t')

