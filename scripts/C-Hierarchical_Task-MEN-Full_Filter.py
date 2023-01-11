#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# In[ ]:

# Input args (4): model-name, nº epochs, random seed, random exec

# In[1]:


model_name = "mbert_galen"
epochs = 65
random_seed = 1
random_exec = 1


# 
# <br>
# Estimated hyper-parameters for the remaining models:<br>
# mBERT: epochs=89, random_seed=0<br>
# BETO-Galén: epochs=85, random_seed=1<br>
# BETO: epochs=85, random_seed=2<br>
# XLM-R-Galén: epochs=94, random_seed=2<br>
# XLM-R: epochs=86, random_seed=3<br>
# 

# In[ ]:


import sys
if len(sys.argv) > 1:
    model_name = sys.argv[-5]
    epochs = int(sys.argv[-4])
    random_seed = int(sys.argv[-3])
    random_exec = int(sys.argv[-2])


# In[3]:


hier_iob_exec = random_exec
    
print("Model name:", model_name, 
      "| nº epochs:", epochs,
      "| random seed:", random_seed,
      "| random exec:", random_exec)


# In[ ]:

# In[4]:


root_path = "../"


# In[ ]:

# In[5]:


from transformers import BertTokenizerFast, XLMRobertaTokenizerFast


# All variables that depend on model_name

# In[6]:


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


# In[1]:

# In[7]:


utils_path = root_path + "utils/"
corpus_path = root_path + "datasets/cantemist_v6/"
sub_task_path = "cantemist-norm/"
ss_corpus_path = root_path + "datasets/Cantemist-SSplit-text/"
dev_gs_path = corpus_path + "dev-set2/" + sub_task_path
test_gs_path = corpus_path + "test-set/" + sub_task_path


# In[2]:

# In[8]:


import tensorflow as tf


# Auxiliary components

# In[9]:


sys.path.insert(0, utils_path)
from nlp_utils import *


# In[10]:


print(sys.path)


# Hyper-parameters

# In[11]:


type_tokenizer = "transformers"


# In[12]:


subtask = 'norm'
subtask_ann = subtask + '-iob_code_suf'
text_col = "raw_text"
SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = epochs
LR = 3e-5


# In[13]:


GREEDY = True
IGNORE_VALUE = -100
code_strat = 'o'
ANN_STRATEGY = "word-all"
LOGITS = False


# In[14]:


MENTION_LIMIT_EMB_VALUE = 1


# In[15]:


tf.random.set_seed(random_seed)


# In[16]:


RES_DIR = root_path + "results/Cantemist/final_exec/"


# In[18]:


df_iob_preds_name = "c_hier_task_iob_" + model_name + "_" + str(hier_iob_exec)
EMPTY_SAMPLES = False


# In[19]:


JOB_NAME = "c_hier_task_cls_train_" + model_name + "_" + str(random_exec)
print("\n" + JOB_NAME)


# ## Load text<br>
# <br>
# Firstly, all text files from training and development Cantemist corpora are loaded in different dataframes.<br>
# <br>
# Also, NER-annotations are loaded.

# ### Training corpus

# In[3]:

# In[20]:


train_path = corpus_path + "train-set/" + sub_task_path
train_files = [f for f in os.listdir(train_path) if os.path.isfile(train_path + f) and f.split('.')[-1] == "txt"]
n_train_files = len(train_files)
train_data = load_text_files(train_files, train_path)
df_text_train = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in train_files], 'raw_text': train_data})


# ### Test corpus

# In[5]:

# In[21]:


test_path = corpus_path + "test-set/" + sub_task_path
test_files = [f for f in os.listdir(test_path) if os.path.isfile(test_path + f) and f.split('.')[-1] == 'txt']
test_data = load_text_files(test_files, test_path)
df_text_test = pd.DataFrame({'doc_id': [s.split('.txt')[0] for s in test_files], 'raw_text': test_data})


# ## Process NER annotations<br>
# <br>
# We load and pre-process the NER annotations in BRAT format available for the CodiEsp-X subtask.

# ### Training corpus

# In[6]:

# In[22]:


train_ann_files = [train_path + f for f in os.listdir(train_path) if f.split('.')[-1] == "ann"]


# In[7]:

# In[23]:


df_codes_train_ner = process_brat_norm(train_ann_files).sort_values(["doc_id", "start", "end"])


# In[8]:

# In[24]:


df_codes_train_ner["code_pre"] = df_codes_train_ner["code"].apply(lambda x: x.split('/')[0])
df_codes_train_ner["code_suf"] = df_codes_train_ner["code"].apply(lambda x: '/'.join(x.split('/')[1:]))


# In[9]:

# In[25]:


assert ~df_codes_train_ner[["doc_id", "start", "end"]].duplicated().any()


# In[10]:

# In[26]:


df_codes_train_ner_final = df_codes_train_ner.copy()


# In[11]:

# In[27]:


print(df_codes_train_ner_final.shape[0])


# ## Creation of annotated sequences<br>
# <br>
# We create the corpus used to fine-tune the transformer model on a NER task. In this way, we split the texts into sentences, and convert them into sequences of subtokens. Also, each generated subtoken is assigned a NER label in IOB-2 format.

# In[18]:

# In[28]:


train_dev_codes_pre = sorted(set(df_codes_train_ner_final["code_pre"].values))


# In[19]:

# In[29]:


len(train_dev_codes_pre)


# In[20]:

# In[30]:


train_dev_codes_suf = sorted(set(df_codes_train_ner_final["code_suf"].values))


# In[21]:

# In[31]:


print(len(train_dev_codes_suf))


# In[22]:

# Create IOB-2 and Clinical-Coding label encoders as dict (more computationally efficient)

# In[32]:


iob_lab_encoder = {"B": 0, "I": 1, "O": 2}
iob_lab_decoder = {0: "B", 1: "I", 2: "O"}


# Code-pre

# In[33]:


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

# In[34]:


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


# In[23]:

# In[35]:


print(len(iob_lab_encoder), len(iob_lab_decoder))


# In[24]:

# In[36]:


print(len(code_pre_lab_encoder), len(code_pre_lab_decoder))


# In[25]:

# In[37]:


print(len(code_suf_lab_encoder), len(code_suf_lab_decoder))


# In[26]:

# Text classification (later ignored)

# In[27]:

# In[38]:


train_dev_codes = sorted(set(df_codes_train_ner_final["code"].values))


# In[28]:

# In[39]:


print(len(train_dev_codes))


# In[29]:

# In[40]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[41]:


mlb_encoder = MultiLabelBinarizer()
mlb_encoder.fit([train_dev_codes])


# ### Training corpus

# Only training texts with NER annotations are considered:

# In[30]:

# In[42]:


train_doc_list = sorted(set(df_codes_train_ner_final["doc_id"]))


# In[31]:

# In[43]:


ss_sub_corpus_path = ss_corpus_path + "training/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_train = load_ss_files(ss_files, ss_sub_corpus_path)


# In[32]:

# In[44]:


df_mock = pd.DataFrame({"doc_id": [], "start": [], "end": []})


# In[45]:


train_ind, train_att, train_type, train_y, train_text_y, train_frag, train_start_end_frag,                 train_word_id = ss_create_input_data_ner(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_mock, df_ann_text=df_mock, # ignore text-level
                                    doc_list=train_doc_list, ss_dict=ss_dict_train,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# ### Test corpus<br>
# <br>
# All test texts are considered:

# In[36]:

# In[46]:


test_doc_list = sorted(set(df_text_test["doc_id"]))


# In[37]:

# Sentence-Split data

# In[38]:

# In[47]:


ss_sub_corpus_path = ss_corpus_path + "test-background/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_test = load_ss_files(ss_files, ss_sub_corpus_path)


# In[39]:

# In[48]:


test_ind, test_att, test_type, test_y, test_text_y, test_frag, test_start_end_frag,                 test_word_id = ss_create_input_data_ner(df_text=df_text_test, text_col=text_col, 
                                    df_ann=df_mock, df_ann_text=df_mock, # ignore text-level
                                    doc_list=test_doc_list, ss_dict=ss_dict_test,
                                    tokenizer=tokenizer, 
                                    lab_encoder_list=[iob_lab_encoder, code_pre_lab_encoder, code_suf_lab_encoder], 
                                    text_label_encoder=mlb_encoder, seq_len=SEQ_LEN, ign_value=IGNORE_VALUE, 
                                    strategy=ANN_STRATEGY, greedy=GREEDY, 
                                    subtask=subtask_ann, code_strat=code_strat)


# In[44]:

# Additional Embedding & CLS samples

# In[45]:

# In[49]:


df_codes_train_ner_final = df_codes_train_ner_final.rename(columns={"doc_id": "clinical_case"})
df_codes_train_ner_final['location'] = df_codes_train_ner_final.apply(lambda x: str(x['start']) + ' ' + str(x['end']), 
                                                                      axis=1)


# In[46]:

# In[50]:


train_cls_y, train_cls_ind, df_train_cls_ann = create_cls_emb_y_samples(
    df_ann=df_codes_train_ner_final, doc_list=train_doc_list, arr_frag=train_frag,
    arr_start_end=train_start_end_frag, arr_word_id=train_word_id, arr_ind=train_ind,
    seq_len=SEQ_LEN, empty_samples=EMPTY_SAMPLES, subtask=subtask_ann
)


# In[47]:

# In[51]:


train_cls_code_pre_y = np.array([code_pre_lab_encoder[sample[0][-1][0]] for sample in train_cls_y])
train_cls_code_suf_y = np.array([code_suf_lab_encoder[sample[0][-1][1]] for sample in train_cls_y])
train_cls_emb_y = np.array([sample[1] for sample in train_cls_y])


# In[51]:

# Insert mention-delimiter tokens

# In[52]:

# In[52]:


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


# In[53]:

# In[53]:


train_cls_ind, train_cls_emb = create_ind_emb_mention_sep(
    arr_ind=train_cls_ind, 
    arr_emb=train_cls_emb_y
)


# In[55]:

#  PAD inidices and create attention masks

# In[56]:

# In[54]:


def calculate_max_seq(list_list_arr_seq):
    max_len = 0
    for list_arr_seq in list_list_arr_seq:
        for arr_seq in list_arr_seq:
            cur_len = len(arr_seq)
            if cur_len > max_len:
                max_len = cur_len
    return max_len


# In[57]:

# In[55]:


print(calculate_max_seq([train_cls_ind]))


# In[58]:

# In[56]:


MAX_SEQ_LEN = 256


# In[59]:

# In[57]:


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


# In[60]:

# In[58]:


train_cls_ind, train_cls_emb, train_cls_att = pad_inidices_emb_create_att(
    list_ind_sep=train_cls_ind,
    list_emb_sep=train_cls_emb
)


# In[61]:

# In[59]:


print(train_cls_code_pre_y.shape, train_cls_code_suf_y.shape, 
      train_cls_ind.shape, train_cls_emb.shape, train_cls_att.shape)


# In[66]:

# In[60]:


def format_ner_preds(ner_file_name=df_iob_preds_name, ner_dir=RES_DIR, doc_list=test_doc_list, arr_frag=test_frag,
                     arr_start_end=test_start_end_frag, arr_word_id=test_word_id, 
                     arr_ind=test_ind, prefix_name="df_test_preds_", 
                     suffix_name=".csv", df_preds_ner=None):
    if df_preds_ner is None:
        # Load file
        df_preds_ner = pd.read_csv(ner_dir + prefix_name + ner_file_name + suffix_name, header=0, sep='\t')
    df_preds_ner['location'] = df_preds_ner.apply(lambda x: str(x['start']) + ' ' + str(x['end']), 
                                                      axis=1)
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


# ## Fine-tuning<br>
# <br>
# Using the corpus of labeled sentences, we fine-tune the model on a multi-label sentence classification task.

# In[69]:

# Set memory growth

# In[70]:

# In[61]:


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)


# In[71]:

# In[62]:


for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# In[ ]:

# In[63]:


if model_name.split('_')[0] in ('beto', 'mbert'):
    model = TFBertForTokenClassification_NerAnnEmb.from_pretrained(model_path, from_pt=True)
    
else: # default
    model = TFXLMRobertaForTokenClassification_NerAnnEmb.from_pretrained(model_path, from_pt=True)
    # Mentions start-end additional tokens
    model.resize_token_embeddings(len(tokenizer.get_vocab()))


# In[73]:

# In[64]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform


# In[65]:


code_pre_num_labels = len(code_pre_lab_encoder) - 1 if not EMPTY_SAMPLES else len(code_pre_lab_encoder)
code_suf_num_labels = len(code_suf_lab_encoder) - 1 if not EMPTY_SAMPLES else len(code_suf_lab_encoder)


# In[66]:


input_ids = Input(shape=(MAX_SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(MAX_SEQ_LEN,), name='attention_mask', dtype='int64')
ner_ann_ids = Input(shape=(MAX_SEQ_LEN,), name='ner_ann_ids', dtype='int64')


# In[67]:


out_cls = model.layers[0](input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          ner_ann_ids=ner_ann_ids)[0][:, 0, :] # take CLS token output representation


# Code-pre

# In[68]:


out_cls_code_pre = Dense(units=code_pre_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_cls) # Multi-class classification 
out_cls_code_pre_model = Activation(activation='softmax', name='code_pre_cls_output')(out_cls_code_pre)


# Code-suf

# In[69]:


out_cls_code_suf = Dense(units=code_suf_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_cls) # Multi-class classification 
out_cls_code_suf_model = Activation(activation='softmax', name='code_suf_cls_output')(out_cls_code_suf)


# In[70]:


model = Model(inputs=[input_ids, attention_mask, ner_ann_ids], outputs=[out_cls_code_pre_model, out_cls_code_suf_model])


# In[74]:

# In[71]:


print(model.summary())


# In[75]:

# In[72]:


print(model.input)


# In[76]:

# In[73]:


print(model.output)


# In[77]:

# GS data

# In[74]:


df_test_gs = format_ner_gs(test_gs_path, subtask=subtask)


# In[79]:

# In[75]:


import tensorflow_addons as tfa
from tensorflow.keras import losses
import time


# In[76]:


optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'code_pre_cls_output': losses.SparseCategoricalCrossentropy(from_logits=LOGITS),
        'code_suf_cls_output': losses.SparseCategoricalCrossentropy(from_logits=LOGITS)}
loss_weights = {'code_pre_cls_output': 1, 'code_suf_cls_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)


# In[103]:


start_time = time.time()


# In[ ]:


history = model.fit(x={'input_ids': train_cls_ind, 
                       'attention_mask': train_cls_att,
                       'ner_ann_ids': train_cls_emb}, 
                    y={'code_pre_cls_output': train_cls_code_pre_y, 
                       'code_suf_cls_output': train_cls_code_suf_y}, 
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                    verbose=2)


# In[ ]:


end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print()


# ## Evaluation

# #### Full

# In[78]:


df_test_gs_full = df_test_gs.sort_values(by=["clinical_case", "start_pos_gs", "end_pos_gs"])


# In[79]:


df_test_gs_full_ner = df_test_gs_full[["clinical_case", "start_pos_gs", "end_pos_gs", "span"]].rename( 
    columns={"start_pos_gs": "start", "end_pos_gs": "end", "span": "text"}
)


# In[80]:


df_test_gs_full_ner['clinical_case'] = df_test_gs_full_ner['clinical_case'].apply(
    lambda x: x.split('.')[0]
)


# In[ ]:


# Generate input data


# In[81]:


test_full_cls_ind_ner, test_full_cls_emb_ner, test_full_cls_att_ner, df_test_full_cls_ann_ner = format_ner_preds(
    df_preds_ner=df_test_gs_full_ner
)


# In[ ]:


# Predictions


# In[ ]:


start_time = time.time()


# In[87]:

# In[82]:


y_pred_test_full_cls = model.predict({'input_ids': test_full_cls_ind_ner, 
                                 'attention_mask': test_full_cls_att_ner,
                                'ner_ann_ids': test_full_cls_emb_ner})


# In[88]:

# In[ ]:


np.save(file="test_full_preds_code_pre_" + JOB_NAME + ".npy", arr=y_pred_test_full_cls[0])


# In[89]:

# In[ ]:


np.save(file="test_full_preds_code_suf_" + JOB_NAME + ".npy", arr=y_pred_test_full_cls[1])


# In[90]:

# In[83]:


df_test_full_preds = cls_code_norm_preds_brat_format(
    y_pred_cls=y_pred_test_full_cls, df_pred_ner=df_test_full_cls_ann_ner, 
    code_decoder_list=[code_pre_lab_decoder, code_suf_lab_decoder],
    subtask=subtask_ann,
    codes_pre_o_mask=None,
    codes_pre_suf_mask=None
)


# In[91]:

# In[ ]:


print(round(
    (
        pd.Series(df_test_gs_full["code_gs"].values) == pd.Series(df_test_full_preds["code_pred"].values)
    ).value_counts(normalize=True)[True], 
    4
))


# In[92]:

# In[ ]:


end_time = time.time()
print("--- Full test evaluation: %s seconds ---" % (end_time - start_time))
print()


# In[93]:

# Save final results DF

# In[ ]:


df_test_full_preds.to_csv("df_test_full_preds_" + JOB_NAME + ".csv", index=True, header=True, sep = '\t')


# #### Filtering

# In[89]:


train_dev_mentions = sorted(set([x.lower() for x in set(
    df_codes_train_ner_final["text_ref"].values
)]))


# In[90]:


df_test_gs_filt = df_test_gs[df_test_gs.span.apply(
    lambda x: x.lower() not in train_dev_mentions
)].sort_values(by=["clinical_case", "start_pos_gs", "end_pos_gs"])


# In[91]:


df_test_gs_filt_ner = df_test_gs_filt[["clinical_case", "start_pos_gs", "end_pos_gs", "span"]].rename( 
    columns={"start_pos_gs": "start", "end_pos_gs": "end", "span": "text"}
)


# In[92]:


df_test_gs_filt_ner['clinical_case'] = df_test_gs_filt_ner['clinical_case'].apply(
    lambda x: x.split('.')[0]
)


# In[ ]:


# Generate input data


# In[93]:


test_filt_cls_ind_ner, test_filt_cls_emb_ner, test_filt_cls_att_ner, df_test_filt_cls_ann_ner = format_ner_preds(
    df_preds_ner=df_test_gs_filt_ner
)


# In[ ]:


# Predictions


# In[ ]:


start_time = time.time()


# In[87]:

# In[94]:


y_pred_test_filt_cls = model.predict({'input_ids': test_filt_cls_ind_ner, 
                                 'attention_mask': test_filt_cls_att_ner,
                                'ner_ann_ids': test_filt_cls_emb_ner})


# In[88]:

# In[ ]:


np.save(file="test_filt_preds_code_pre_" + JOB_NAME + ".npy", arr=y_pred_test_filt_cls[0])


# In[89]:

# In[ ]:


np.save(file="test_filt_preds_code_suf_" + JOB_NAME + ".npy", arr=y_pred_test_filt_cls[1])


# In[90]:

# In[95]:


df_test_filt_preds = cls_code_norm_preds_brat_format(
    y_pred_cls=y_pred_test_filt_cls, df_pred_ner=df_test_filt_cls_ann_ner, 
    code_decoder_list=[code_pre_lab_decoder, code_suf_lab_decoder],
    subtask=subtask_ann,
    codes_pre_o_mask=None,
    codes_pre_suf_mask=None
)


# In[91]:

# In[ ]:


print(round(
    (
        pd.Series(df_test_gs_filt["code_gs"].values) == pd.Series(df_test_filt_preds["code_pred"].values)
    ).value_counts(normalize=True)[True], 
    4
))


# In[92]:

# In[ ]:


end_time = time.time()
print("--- Filtering test evaluation: %s seconds ---" % (end_time - start_time))
print()


# In[93]:

# Save final results DF

# In[ ]:


df_test_filt_preds.to_csv("df_test_filt_preds_" + JOB_NAME + ".csv", index=True, header=True, sep = '\t')

