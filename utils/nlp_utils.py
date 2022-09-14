### Module containing auxiliary functions and classes for clinical coding NLP using Transformers


## Load text

import os

def load_text_files(file_names, path):
    """
    It loads the text contained in a set of files into a returned list of strings.
    Code adapted from https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(path + f, "r") as file:
            output.append(file.read())
            
    return output


def load_ss_files(file_names, path):
    """
    It loads the start-end pair of each split sentence from a set of files (start + \t + end line-format expected) into a 
    returned dictionary, where keys are file names and values a list of tuples containing the start-end pairs of the 
    split sentences.
    """
    output = dict()
    for f in file_names:
        with open(path + f, "r") as file:
            f_key = f.split('.')[0]
            output[f_key] = []
            for sent in file:
                output[f_key].append(tuple(map(int, sent.strip().split('\t'))))
            
    return output


import numpy as np
import pandas as pd

def process_labels_norm(df_ann):
    """
    Primarly dessign to process CodiEsp-X annotations.
    """
    
    df_res = []
    for i in range(df_ann.shape[0]):
        ann_i = df_ann.iloc[i].values
        # Separate discontinuous locations and split each location into start and end offset
        ann_loc_i = ann_i[4]
        for loc in ann_loc_i.split(';'):
            split_loc = loc.split(' ')
            df_res.append(np.concatenate((ann_i[:4], [int(split_loc[0]), int(split_loc[1])])))

    return pd.DataFrame(np.array(df_res), 
                        columns=list(df_ann.columns[:-1]) + ["start", "end"]).drop_duplicates()


def process_labels_norm_prueba(df_ann):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    
    df_res = []
    for i in range(df_ann.shape[0]):
        ann_i = df_ann.iloc[i].values
        # Separate discontinuous locations and split each location into start and end offset
        ann_loc_i = ann_i[4]
        arr_loc = ann_loc_i.split(';')
        ann_disc = False if len(arr_loc) == 1 else True
        for loc in arr_loc:
            split_loc = loc.split(' ')
            df_res.append(np.concatenate((ann_i, [int(split_loc[0]), int(split_loc[1]), ann_disc])))

    return pd.DataFrame(np.array(df_res), 
                        columns=list(df_ann.columns) + ["start", "end", "disc"])


from math import ceil

def process_brat_norm(brat_files):
    """
    Primarly dessign to process Cantemist-Norm annotations.
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    """
    
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            i = 0
            for line in ann_file:
                i += 1
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                if i % 2 > 0:
                    # BRAT annotation
                    assert line_split[0] == "T" + str(ceil(i/2))
                    text_ref = line_split[2]
                    location = ' '.join(line_split[1].split(' ')[1:]).split(';')
                else:
                    # Code assignment
                    assert line_split[0] == "#" + str(ceil(i/2))
                    code = line_split[2]
                    # Discontinuous annotations are split into a sequence of continuous annotations
                    for loc in location:
                        split_loc = loc.split(' ')
                        df_res.append([doc_name, code, text_ref, int(split_loc[0]), int(split_loc[1])])

    return pd.DataFrame(df_res, 
columns=["doc_id", "code", "text_ref", "start", "end"])


def process_brat_ner(brat_files):
    """
    Primarly dessign to process Cantemist-NER annotations.
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    """
    
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            for line in ann_file:
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                text_ref = line_split[2]
                location = ' '.join(line_split[1].split(' ')[1:]).split(';')
                # Discontinuous annotations are split into a sequence of continuous annotations
                for loc in location:
                    split_loc = loc.split(' ')
                    df_res.append([doc_name, text_ref, int(split_loc[0]), int(split_loc[1])])

    return pd.DataFrame(df_res, 
columns=["doc_id", "text_ref", "start", "end"])


def process_de_ident_ner(brat_files):
    """
    Primarly dessign to process de-identification annotations from Galén corpus.
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    """
    
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.ann')[0]
            for line in ann_file:
                line_split = line.strip().split('\t')
                if line_split[0][0] == "T":
                    assert len(line_split) == 3
                    text_ref = line_split[2]
                    ann_type = line_split[1].split(' ')[0]
                    location = ' '.join(line_split[1].split(' ')[1:])
                    df_res.append([doc_name, text_ref, ann_type, location])

    return pd.DataFrame(df_res, columns=["doc_id", "text_ref", "type", "location"])



## Whitespace-punctuation tokenization (same as BERT pre-tokenization)
# The next code is adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

import unicodedata

def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

def is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

def is_space(ch):
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'

def is_control(ch):
    """
    Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py#L64
    """
    return unicodedata.category(ch).startswith("C")


def word_start_end(text, start_i=0, cased=True):
    """
    Our aim is to produce both a list of strings containing the text of each word and a list of pairs containing the start and
    end char positions of each word.
    
    start_i: the start position of the first character in the text.
    
    Code adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101
    """
    
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    # Store the start positions of each considered character (ch) in start_arr, 
    # such that sum([len(word) for word in spaced.strip().split()]) = len(start_arr)
    start_arr = [] 
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            start_arr.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not(ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            start_arr.append(start_i)
        # If it is a control char we skip it but take its offset into account
        start_i += 1
    
    assert sum([len(word) for word in spaced.strip().split()]) == len(start_arr)
    
    text_arr, start_end_arr = [], []
    i = 0
    for word in spaced.strip().split():
        text_arr.append(word)
        j = i + len(word)
        start_end_arr.append((start_arr[i], start_arr[j - 1] + 1))
        i = j
        
    return text_arr, start_end_arr



## NER-annotations

def start_end_tokenize(text, tokenizer, start_pos=0):
    """
    Our aim is to produce both a list of sub-tokens and a list of tuples containing the start and
    end char positions of each sub-token.
    """
    # here guille now: FastText
    type_tokenizer = str(type(tokenizer))
    if 'transformers' in type_tokenizer:
        start_end_arr = []
        token_text = tokenizer(text, add_special_tokens=False)
        for i in range(len(token_text['input_ids'])):
            chr_span = token_text.token_to_chars(i)
            start_end_arr.append((chr_span.start + start_pos, chr_span.end + start_pos))

        return tokenizer.convert_ids_to_tokens(token_text['input_ids']), start_end_arr
    
    elif 'fasttext' in type_tokenizer:
        return [text], [(start_pos, start_pos + len(text))]


# Tokenization analysis

def check_overlap_ner(df_ann, doc_list):
    """
    This function returns the named entities in a single document with overlapping
    spans.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            start_i = df_doc.iloc[i]['start']
            end_i = df_doc.iloc[i]['end']
            for j in range(i + 1, len_doc):
                start_j = df_doc.iloc[j]['start']
                end_j = df_doc.iloc[j]['end']
                if start_i < end_j and start_j < end_i:
                    res.append((doc, start_i, end_i, start_j, end_j, (start_i >= start_j and end_i <= end_j) or \
                                                                     (start_j >= start_i and end_j <= end_i)))
                
    return pd.DataFrame(res, columns=["doc_id", "start_1", "end_1", "start_2", "end_2", "contained"])



def eliminate_overlap(df_ann):
    """
    For each pair of existing overlapping annotations in a document, the longer is eliminated.
    Then, the existence of overlapping annotations is re-evaluated.
    """
    df_res = df_ann.copy()
    for doc in sorted(set(df_ann['doc_id'])):
        doc_over = check_overlap_ner(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 >= len_2:
                elim_start = aux_row['start_1']
                elim_end = aux_row['end_1']
            else:
                elim_start = aux_row['start_2']
                elim_end = aux_row['end_2']
            # Eliminate longer overlapping annotation
            df_res = df_res[(df_res['doc_id'] != aux_row['doc_id']) | (df_res['start'] != elim_start) | (df_res['end'] != elim_end)]
            doc_over = check_overlap_ner(df_ann=df_res, doc_list=[doc])
        
    return df_res


def check_overlap_ner_prueba(df_ann, doc_list):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            ann_i = df_doc.iloc[i]
            start_i = ann_i['start']
            end_i = ann_i['end']
            for j in range(i + 1, len_doc):
                ann_j = df_doc.iloc[j]
                start_j = ann_j['start']
                end_j = ann_j['end']
                if start_i < end_j and start_j < end_i:
                    res.append((doc, start_i, end_i, ann_i['type'], ann_i['location'], 
                                start_j, end_j, ann_j['type'], ann_j['location'],  
                                (start_i >= start_j and end_i <= end_j) or (start_j >= start_i and end_j <= end_i)))
                
    return pd.DataFrame(res, columns=["doc_id", "start_1", "end_1", "type_1", "location_1", 
                                      "start_2", "end_2", "type_2", "location_2", "contained"])


def eliminate_overlap_prueba(df_ann):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    df_res = df_ann.copy()
    for doc in sorted(set(df_ann['doc_id'])):
        doc_over = check_overlap_ner_prueba(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 > len_2:
                elim = 1
                elim_start = aux_row['start_1']
                elim_end = aux_row['end_1']
            else:
                elim = 5
                elim_start = aux_row['start_2']
                elim_end = aux_row['end_2']
            # Eliminate longer overlapping annotation
            df_res = df_res[(df_res['doc_id'] != aux_row['doc_id']) | (df_res['start'] != aux_row[elim]) | \
                            (df_res['end'] != aux_row[elim+1]) | (df_res['type'] != aux_row[elim+2]) | \
                            (df_res['location'] != aux_row[elim+3])]
            doc_over = check_overlap_ner_prueba(df_ann=df_res, doc_list=[doc])
        
    return df_res


def check_overlap_ner_code(df_ann, doc_list):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            ann_i = df_doc.iloc[i]
            start_i = ann_i['start']
            end_i = ann_i['end']
            code_i = ann_i['code']
            for j in range(i + 1, len_doc):
                ann_j = df_doc.iloc[j]
                start_j = ann_j['start']
                end_j = ann_j['end']
                code_j = ann_j['code']
                if start_i < end_j and start_j < end_i and code_i != code_j:
                    res.append((doc, start_i, end_i, code_i, ann_i['location'], 
                                start_j, end_j, code_j, ann_j['location'],  
                                (start_i >= start_j and end_i <= end_j) or (start_j >= start_i and end_j <= end_i)))
                
    return pd.DataFrame(res, columns=["doc_id", "start_1", "end_1", "code_1", "location_1", 
                                      "start_2", "end_2", "code_2", "location_2", "contained"])


def eliminate_overlap_code(df_ann):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    df_res = df_ann.copy()
    for doc in sorted(set(df_ann['doc_id'])):
        doc_over = check_overlap_ner_code(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 > len_2:
                elim = 1
            else:
                elim = 5
            # Eliminate longer overlapping annotation
            df_res = df_res[(df_res['doc_id'] != aux_row['doc_id']) | (df_res['start'] != aux_row[elim]) | \
                            (df_res['end'] != aux_row[elim+1]) | (df_res['code'] != aux_row[elim+2]) | \
                            (df_res['location'] != aux_row[elim+3])]
            doc_over = check_overlap_ner_code(df_ann=df_res, doc_list=[doc])
        
    return df_res


def check_overlap_ner_distinct(df_ann, doc_list):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            ann_i = df_doc.iloc[i]
            start_i = ann_i['start']
            end_i = ann_i['end']
            for j in range(i + 1, len_doc):
                ann_j = df_doc.iloc[j]
                start_j = ann_j['start']
                end_j = ann_j['end']
                if start_i < end_j and start_j < end_i and (start_i != start_j or end_i != end_j): # allow duplicates
                    res.append((doc, start_i, end_i, ann_i['type'], ann_i['location'], 
                                start_j, end_j, ann_j['type'], ann_j['location'],  
                                (start_i >= start_j and end_i <= end_j) or (start_j >= start_i and end_j <= end_i)))
                
    return pd.DataFrame(res, columns=["doc_id", "start_1", "end_1", "type_1", "location_1", 
                                      "start_2", "end_2", "type_2", "location_2", "contained"])


def eliminate_overlap_distinct(df_ann):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    df_res = df_ann.copy()
    for doc in sorted(set(df_ann['doc_id'])):
        doc_over = check_overlap_ner_distinct(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 > len_2:
                elim = 1
                elim_start = aux_row['start_1']
                elim_end = aux_row['end_1']
            else:
                elim = 5
                elim_start = aux_row['start_2']
                elim_end = aux_row['end_2']
            # Eliminate longer overlapping annotation
            df_res = df_res[(df_res['doc_id'] != aux_row['doc_id']) | (df_res['start'] != aux_row[elim]) | \
                            (df_res['end'] != aux_row[elim+1]) | (df_res['type'] != aux_row[elim+2]) | \
                            (df_res['location'] != aux_row[elim+3])]
            doc_over = check_overlap_ner_distinct(df_ann=df_res, doc_list=[doc])
        
    return df_res


def check_overlap_ner_diff(df_ann, doc_list):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            ann_i = df_doc.iloc[i]
            start_i = ann_i['start']
            end_i = ann_i['end']
            for j in range(i + 1, len_doc):
                ann_j = df_doc.iloc[j]
                start_j = ann_j['start']
                end_j = ann_j['end']
                if (start_i < end_j) and (start_j < end_i):
                    assert ann_i['type'] != ann_j['type']
                    #if (ann_i['location'] != ann_j['location']) or (ann_i['disc'] != ann_j['disc']): # allow duplicates
                    if ann_i['location'] != ann_j['location']: # allow duplicates
                        res.append((doc, start_i, end_i, ann_i['type'], ann_i['location'], 
                                    start_j, end_j, ann_j['type'], ann_j['location'],  
                                    (start_i >= start_j and end_i <= end_j) or (start_j >= start_i and end_j <= end_i)))
                
    return pd.DataFrame(res, columns=["doc_id", "start_1", "end_1", "type_1", "location_1", 
                                      "start_2", "end_2", "type_2", "location_2", "contained"])


def eliminate_overlap_diff(df_ann):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    df_res = df_ann.copy()
    for doc in sorted(set(df_ann['doc_id'])):
        doc_over = check_overlap_ner_diff(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 > len_2:
                elim = 1
                elim_start = aux_row['start_1']
                elim_end = aux_row['end_1']
            else:
                elim = 5
                elim_start = aux_row['start_2']
                elim_end = aux_row['end_2']
            # Eliminate longer overlapping annotation
            df_res = df_res[(df_res['doc_id'] != aux_row['doc_id']) | (df_res['start'] != aux_row[elim]) | \
                            (df_res['end'] != aux_row[elim+1]) | (df_res['type'] != aux_row[elim+2]) | \
                            (df_res['location'] != aux_row[elim+3])]
            doc_over = check_overlap_ner_diff(df_ann=df_res, doc_list=[doc])
        
    return df_res


def restore_disc_ann(df_ann, df_ann_final):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    res = []
    for index, row in df_ann.iterrows():
        disc_loc = row['location']
        disc_loc_split = disc_loc.split(';')
        assert len(disc_loc_split) > 1
        contain_bool = True
        for loc in disc_loc_split:
            ann_pos = loc.split(' ')
            start, end = int(ann_pos[0]), int(ann_pos[1])
            contain_shape = df_ann_final[(df_ann_final['doc_id'] == row['doc_id']) & (df_ann_final['type'] == row['type']) & \
                         # None of the next cols are part of the unique key, so they are redundant
                         #(df_ann_final['code'] == row['code']) & (df_ann_final['word'] == row['word']) & \
                         (df_ann_final['start'] == start) & (df_ann_final['end'] == end) & \
                         (df_ann_final['location'] == disc_loc)].shape[0]
            if contain_shape == 0:
                contain_bool = False
                break
            else:
                assert contain_shape == 1
        # Add discontinuous annotation
        if contain_bool:
            res.append(row)

    return pd.DataFrame(res)


def check_ann_left_right_direction(row):
    """
    Preserve discontinuous annotations with non-overlapping fragments.
    """
    left_right = True
    raw_loc = row['location']
    arr_loc = raw_loc.split(';')
    len_arr_loc = len(arr_loc)
    if len_arr_loc > 1:
        i = 1
        while (left_right and (i < len_arr_loc)):
            start_left = int(arr_loc[i-1].split(' ')[0])
            start_right = int(arr_loc[i].split(' ')[0])
            left_right = (start_left < start_right)
            i += 1
    
    return left_right


def check_ann_span_sent(df_ann, ss_dict):
    mult_sent, one_sent, no_sent = [], [], []
    for doc in sorted(set(df_ann['doc_id'])):
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        doc_ss = ss_dict[doc]
        # Check if there is any annotation not fully contained in a single sentence
        for index, row in doc_ann.iterrows():
            ann_s = row['start']
            ann_e = row['end']
            i = 0
            cont = False
            while ((i < len(doc_ss)) and (not cont)):
                ss_s = doc_ss[i][0]
                ss_e = doc_ss[i][1]
                if ann_s >= ss_s and ann_s < ss_e:
                    cont = True
                    if ann_e > ss_e:
                        mult_sent.append(row)
                    else:
                        one_sent.append(row)
                i += 1
            if not cont:
                no_sent.append(row)

    return pd.DataFrame(mult_sent), pd.DataFrame(one_sent), pd.DataFrame(no_sent)


def tokenize_ner(df_text, text_col, df_ann, doc_list, tokenizer):
    """
    For sanity check purposes only.
    """
    start_res, end_res = [], []
    for doc in doc_list:
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        # Tokenize text
        doc_ss_token, doc_ss_start_end = start_end_tokenize(text=doc_text, tokenizer=tokenizer, start_pos=0)
        doc_ss_start_end_arr = np.array(doc_ss_start_end)
        for index, row in doc_ann.iterrows():
            # Annotation start char position
            tok_start = np.where(doc_ss_start_end_arr[:, 0] <= row['start'])[0][-1]
            if doc_ss_start_end_arr[tok_start, 0] != row['start']:
                start_res.append((doc, row['start'], row['end'], doc_ss_token, doc_ss_start_end))
            # Annotation end char position
            tok_end = np.where(doc_ss_start_end_arr[:, 1] >= row['end'])[0][0]
            if doc_ss_start_end_arr[tok_end, 1] != row['end']:
                end_res.append((doc, row['start'], row['end'], doc_ss_token, doc_ss_start_end))
    
    return pd.DataFrame(start_res, columns=['doc_id', 'start', 'end', 'tok', 'start_end']), pd.DataFrame(end_res, columns=['doc_id', 'start', 'end', 'tok', 'start_end'])


# Creation of a NER corpus

def ner_iob2_annotate(arr_start_end, df_ann, subtask='ner'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following either IOB-2 NER format 
    (subtask='ner') or IOB-Code NER-Norm format (subtask='norm'), using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: ner, norm
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        #assert labels[tok_start] == "O" # no overlapping annotations are expected
        # Because the presence of two ann in a single word, e.g. "pT3N2Mx" ann in Cantemist dev-set2 cc_onco1427
        if labels[tok_start] != "O": 
            print(labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)
        
        if subtask == 'ner': labels[tok_start] = "B"
        elif subtask == 'norm': labels[tok_start] = "B" + "-" + row['code']
        else: raise Exception('Subtask not implemented!')
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert labels[i] == "O" # no overlapping annotations are expected
                if subtask == 'ner': labels[i] = "I"
                elif subtask == 'norm': labels[i] = "I" + "-" + row['code']
    
    return [labels]


def norm_mention_iob2_annotate(arr_start_end, df_ann):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following either IOB-2 NER format 
    (subtask='ner') or IOB-Code NER-Norm format (subtask='norm'), using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        #assert labels[tok_start] == "O" # no overlapping annotations are expected
        if labels[tok_start] != "O": # Because of the "pT3N2Mx" annotation (two ann in a single word) in dev-set2 cc_onco1427
            print(labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)
        
        labels[tok_start] = "B" + "-" + row['code']
        
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert labels[i] == "O" # no overlapping annotations are expected
                labels[i] = "I"
    
    return [labels]


def norm_iob2_code_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_code'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2, Code] NER-Norm format, 
    using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_code, norm-iob_code-crf
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    iob_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    if subtask.split('-')[-1] != "crf": code_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        #assert labels[tok_start] == "O" # no overlapping annotations are expected
        if iob_labels[tok_start] != "O": # Because of the "pT3N2Mx" annotation (two ann in a single word) in dev-set2 cc_onco1427
            print(iob_labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)
        
        iob_labels[tok_start] = "B"
        if subtask.split('-')[-1] == "crf": iob_labels[tok_start] += ('-' + row["code"])
        else: code_labels[tok_start] = row["code"]
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_labels[i] = "I"
                if subtask.split('-')[-1] == "crf": iob_labels[i] += ('-' + row["code"])
                else: code_labels[i] = row["code"]
                
    if subtask.split('-')[-1] == "crf":
        return [iob_labels]
    
    else:
        return [iob_labels, code_labels]


def norm_iob2_code_suffix_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_code_suf'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2, Code_pre, Code_suf] 
    NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_code_suf, norm-iob_code_suf-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    iob_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_pre_labels = [default_code_value] * len(arr_start_end)
    code_suf_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        #assert labels[tok_start] == "O" # no overlapping annotations are expected
        if iob_labels[tok_start] != "O": # Because of the "pT3N2Mx" annotation (two ann in a single word) in dev-set2 cc_onco1427
            print(iob_labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)
        
        iob_labels[tok_start] = "B"
        code_pre_labels[tok_start] = row["code_pre"]
        code_suf_labels[tok_start] = row["code_suf"] if row["code_suf"] is not None else "O"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_labels[i] = "I"
                if subtask.split('-')[-1] == "mention":
                    code_pre_labels[i] = "I"
                    code_suf_labels[i] = "I"
                else:
                    code_pre_labels[i] = row["code_pre"]
                    code_suf_labels[i] = row["code_suf"] if row["code_suf"] is not None else "O"
    
    return [iob_labels, code_pre_labels, code_suf_labels]


def norm_iob2_code_suffix_h_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_code_suf'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2, Code_pre, Code_suf, H-indicator] 
    NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_code_suf-h, norm-iob_code_suf-h-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    iob_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_pre_labels = [default_code_value] * len(arr_start_end)
    code_suf_labels = [default_code_value] * len(arr_start_end)
    code_h_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        #assert labels[tok_start] == "O" # no overlapping annotations are expected
        if iob_labels[tok_start] != "O": # Because of the "pT3N2Mx" annotation (two ann in a single word) in dev-set2 cc_onco1427
            print(iob_labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)
        
        iob_labels[tok_start] = "B"
        code_pre_labels[tok_start] = row["code_pre"]
        code_suf_labels[tok_start] = row["code_suf"]
        code_h_labels[tok_start] = "H" if row["code_h"] else "O"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_labels[i] = "I"
                if subtask.split('-')[-1] == "mention":
                    code_pre_labels[i] = "I"
                    code_suf_labels[i] = "I"
                    code_h_labels[i] = "I"
                else:
                    code_pre_labels[i] = row["code_pre"]
                    code_suf_labels[i] = row["code_suf"]
                    code_h_labels[i] = code_h_labels[tok_start]
    
    return [iob_labels, code_pre_labels, code_suf_labels, code_h_labels]


def norm_iob2_code_disc_d_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_code_disc_d'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    iob_disc_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
        
        iob_disc_labels[tok_start] = "DB"
        code_labels[tok_start] = row["code"]
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_disc_labels[i] = "DI"
                code_labels[i] = row["code"]
        
        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected

            iob_disc_labels[tok_start] = "DI"
            code_labels[tok_start] = row["code"]

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                    iob_disc_labels[i] = "DI"
                    code_labels[i] = row["code"]
    
    return [iob_disc_labels, code_labels]


def norm_iob2_disc_d_annotate(arr_start_end, df_ann):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (disc)] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    """
    
    iob_disc_labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
        iob_disc_labels[tok_start] = "DB"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                iob_disc_labels[i] = "DI"
        
        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
            iob_disc_labels[tok_start] = "DI"

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                    iob_disc_labels[i] = "DI"
    
    return [iob_disc_labels]


def norm_iob2_diag_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    subtask_diag = "norm-iob_code_suf"
    if subtask.split('-')[-1] == 'mention': 
        subtask_diag += '-mention'
    
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['start', 'end', 'code_pre', 'code_suf']].copy()
    
    return norm_iob2_code_suffix_annotate(arr_start_end=arr_start_end, df_ann=df_ann_diag, ign_value=ign_value, subtask=subtask_diag)


def norm_iob2_diag_disc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_disc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['location', 'code_pre', 'code_suf']].copy()
    
    iob_disc_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_pre_labels = [default_code_value] * len(arr_start_end)
    code_suf_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann_diag.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
        
        iob_disc_labels[tok_start] = "B"
        code_pre_labels[tok_start] = row["code_pre"]
        code_suf_labels[tok_start] = row["code_suf"] if row["code_suf"] is not None else "O"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_disc_labels[i] = "I"
                if subtask.split('-')[-1] == "mention":
                    code_pre_labels[i] = "I"
                    code_suf_labels[i] = "I"
                else:
                    code_pre_labels[i] = row["code_pre"]
                    code_suf_labels[i] = row["code_suf"] if row["code_suf"] is not None else "O"
        
        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected

            iob_disc_labels[tok_start] = "I"
            code_pre_labels[tok_start] = row["code_pre"]
            code_suf_labels[tok_start] = row["code_suf"] if row["code_suf"] is not None else "O"

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                    iob_disc_labels[i] = "I"
                    if subtask.split('-')[-1] == "mention":
                        code_pre_labels[i] = "I"
                        code_suf_labels[i] = "I"
                    else:
                        code_pre_labels[i] = row["code_pre"]
                        code_suf_labels[i] = row["code_suf"] if row["code_suf"] is not None else "O"
    
    return [iob_disc_labels, code_pre_labels, code_suf_labels]


def norm_iob2_disc_annotate(arr_start_end, df_ann, subtask='norm-iob_disc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (disc)] 
    NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_disc
    """
    
    iob_disc_labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
        iob_disc_labels[tok_start] = "B"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                iob_disc_labels[i] = "I"
        
        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
            iob_disc_labels[tok_start] = "I"

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                    iob_disc_labels[i] = "I"
    
    return [iob_disc_labels]


def norm_iob2_cont_disc_annotate(arr_start_end, df_ann):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels = ner_iob2_annotate(arr_start_end=arr_start_end, df_ann=df_ann_cont, subtask="ner")
    
     # Discontinuous
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels = norm_iob2_disc_annotate(arr_start_end=arr_start_end, df_ann=df_ann_disc)
            
    return [iob_cont_labels[0], iob_disc_labels[0]]
    


def norm_iob2_diag_single_disc_d_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_single_disc_d'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['location', 'code']].copy()
    
    return norm_iob2_code_disc_d_annotate(arr_start_end, df_ann_diag, ign_value=ign_value, 
                                          subtask='norm-iob_code_disc_d')


def norm_iob2_diag_cont_disc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_cont_disc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    subtask_cont = "norm-iob_diag"
    if subtask.split('-')[-1] == 'mention': 
        subtask_cont += '-mention'
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels, code_cont_pre_labels, code_cont_suf_labels = norm_iob2_diag_annotate(arr_start_end=arr_start_end, 
                                                                                          df_ann=df_ann_cont, 
                                                                                          ign_value=ign_value, 
                                                                                          subtask=subtask_cont)
    # Discontinuous
    subtask_disc = "norm-iob_diag_disc"
    if subtask.split('-')[-1] == 'mention': 
        subtask_disc += '-mention'
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels, code_disc_pre_labels, code_disc_suf_labels = norm_iob2_diag_disc_annotate(arr_start_end=arr_start_end, 
                                                                                               df_ann=df_ann_disc, 
                                                                                               ign_value=ign_value, 
                                                                                               subtask=subtask_disc)
    ## Merge code labels
    assert len(code_cont_pre_labels) == len(code_cont_suf_labels) == len(code_disc_pre_labels) == len(code_disc_suf_labels)
    def_code_value = "O" if ign_value is None else ign_value
    
    # Code-pre
    code_pre_labels = []
    for code_c_pre, code_d_pre in zip(code_cont_pre_labels, code_disc_pre_labels):
        assert (code_c_pre == def_code_value) or (code_d_pre == def_code_value)
        # Code-pre
        if code_c_pre != def_code_value:
            # Continuous ann
            code_pre_labels.append(code_c_pre)
            
        elif code_d_pre != def_code_value:
            # Discontinuous ann
            code_pre_labels.append(code_d_pre)
            
        else:
            # No ann
            code_pre_labels.append(def_code_value)
        
    # Code-suf
    code_suf_labels = []
    for code_c_suf, code_d_suf in zip(code_cont_suf_labels, code_disc_suf_labels):
        assert (code_c_suf == def_code_value) or (code_d_suf == def_code_value)
        # Code-pre
        if code_c_suf != def_code_value:
            # Continuous ann
            code_suf_labels.append(code_c_suf)
            
        elif code_d_suf != def_code_value:
            # Discontinuous ann
            code_suf_labels.append(code_d_suf)
            
        else:
            # No ann
            code_suf_labels.append(def_code_value)
            
    return [iob_cont_labels, iob_disc_labels, code_pre_labels, code_suf_labels]
    


def norm_iob2_diag_cont_disc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_cont_disc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    subtask_cont = "norm-iob_diag"
    if subtask.split('-')[-1] == 'mention': 
        subtask_cont += '-mention'
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels, code_cont_pre_labels, code_cont_suf_labels = norm_iob2_diag_annotate(arr_start_end=arr_start_end, 
                                                                                          df_ann=df_ann_cont, 
                                                                                          ign_value=ign_value, 
                                                                                          subtask=subtask_cont)
    # Discontinuous
    subtask_disc = "norm-iob_diag_disc"
    if subtask.split('-')[-1] == 'mention': 
        subtask_disc += '-mention'
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels, code_disc_pre_labels, code_disc_suf_labels = norm_iob2_diag_disc_annotate(arr_start_end=arr_start_end, 
                                                                                               df_ann=df_ann_disc, 
                                                                                               ign_value=ign_value, 
                                                                                               subtask=subtask_disc)
    ## Merge code labels
    assert len(code_cont_pre_labels) == len(code_cont_suf_labels) == len(code_disc_pre_labels) == len(code_disc_suf_labels)
    def_code_value = "O" if ign_value is None else ign_value
    
    # Code-pre
    code_pre_labels = []
    for code_c_pre, code_d_pre in zip(code_cont_pre_labels, code_disc_pre_labels):
        assert (code_c_pre == def_code_value) or (code_d_pre == def_code_value)
        # Code-pre
        if code_c_pre != def_code_value:
            # Continuous ann
            code_pre_labels.append(code_c_pre)
            
        elif code_d_pre != def_code_value:
            # Discontinuous ann
            code_pre_labels.append(code_d_pre)
            
        else:
            # No ann
            code_pre_labels.append(def_code_value)
        
    # Code-suf
    code_suf_labels = []
    for code_c_suf, code_d_suf in zip(code_cont_suf_labels, code_disc_suf_labels):
        assert (code_c_suf == def_code_value) or (code_d_suf == def_code_value)
        # Code-pre
        if code_c_suf != def_code_value:
            # Continuous ann
            code_suf_labels.append(code_c_suf)
            
        elif code_d_suf != def_code_value:
            # Discontinuous ann
            code_suf_labels.append(code_d_suf)
            
        else:
            # No ann
            code_suf_labels.append(def_code_value)
            
    return [iob_cont_labels, iob_disc_labels, code_pre_labels, code_suf_labels]
    


def norm_iob2_disc_code_suffix_annotate(arr_start_end, df_ann, ign_value=-100):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (disc), 
    Code-pre, Code-suf] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    """
    
    iob_disc_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_pre_labels = [default_code_value] * len(arr_start_end)
    code_suf_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
        iob_disc_labels[tok_start] = "B"
        code_pre_labels[tok_start] = row["code_pre"]
        code_suf_labels[tok_start] = row["code_suf"] if row["code_suf"] is not None else "O"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                iob_disc_labels[i] = "I"
                code_pre_labels[i] = row["code_pre"]
                code_suf_labels[i] = row["code_suf"] if row["code_suf"] is not None else "O"
        
        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
            iob_disc_labels[tok_start] = "I"
            code_pre_labels[tok_start] = row["code_pre"]
            code_suf_labels[tok_start] = row["code_suf"] if row["code_suf"] is not None else "O"

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                    iob_disc_labels[i] = "I"
                    code_pre_labels[i] = row["code_pre"]
                    code_suf_labels[i] = row["code_suf"] if row["code_suf"] is not None else "O"
    
    return [iob_disc_labels, code_pre_labels, code_suf_labels]


def norm_iob2_cont_disc_code_suffix_annotate(arr_start_end, df_ann, ign_value=-100):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels, code_cont_pre_labels, code_cont_suf_labels = norm_iob2_code_suffix_annotate(
        arr_start_end=arr_start_end, 
        df_ann=df_ann_cont, 
        ign_value=ign_value
    )
    
    # Discontinuous
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels, code_disc_pre_labels, code_disc_suf_labels = norm_iob2_disc_code_suffix_annotate(
        arr_start_end=arr_start_end, 
        df_ann=df_ann_disc, 
        ign_value=ign_value
    )
    
    ## Merge labels
    assert len(iob_cont_labels) == len(iob_disc_labels) == len(code_cont_pre_labels) == len(code_cont_suf_labels) == \
           len(code_disc_pre_labels) == len(code_disc_suf_labels)
    def_code_value = "O" if ign_value is None else ign_value
    
    # Code-pre
    code_pre_labels = []
    for code_c_pre, code_d_pre in zip(code_cont_pre_labels, code_disc_pre_labels):
        if code_c_pre == code_d_pre:
            # Either default_code_value or the same code
            code_pre_labels.append(code_c_pre)
        
        else:
            assert (code_c_pre == def_code_value) or (code_d_pre == def_code_value)
        
            if code_c_pre != def_code_value:
                # Continuous ann
                code_pre_labels.append(code_c_pre)
            
            else:
                # Discontinuous ann
                code_pre_labels.append(code_d_pre)
        
    # Code-suf
    code_suf_labels = []
    for code_c_suf, code_d_suf in zip(code_cont_suf_labels, code_disc_suf_labels):
        if code_c_suf == code_d_suf:
            # Either default_code_value or the same code
            code_suf_labels.append(code_c_suf)
        
        else:
            assert (code_c_suf == def_code_value) or (code_d_suf == def_code_value)
        
            if code_c_suf != def_code_value:
                # Continuous ann
                code_suf_labels.append(code_c_suf)
            
            else:
                # Discontinuous ann
                code_suf_labels.append(code_d_suf)
            
    return [iob_cont_labels, iob_disc_labels, code_pre_labels, code_suf_labels]
    


def norm_iob2_diag_single_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_single'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag_proc, norm-iob_diag_proc-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    subtask_diag = "norm-iob_code"
    if subtask.split('-')[-1] == 'crf': 
        subtask_diag += '-crf'
    
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['start', 'end', 'code']].copy()
    
    return norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_diag, ign_value=ign_value, subtask=subtask_diag)


def norm_iob2_diag_single_cont_disc_d_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_single_cont_disc_d'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    subtask_cont = "norm-iob_diag_single"
    if subtask.split('-')[-1] == 'crf': 
        subtask_cont += '-crf'
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels, code_cont_labels = norm_iob2_diag_single_annotate(arr_start_end=arr_start_end, 
                                                                                          df_ann=df_ann_cont, 
                                                                                          ign_value=ign_value, 
                                                                                          subtask=subtask_cont)
    # Discontinuous
    subtask_disc = "norm-iob_diag_single_disc_d"
    if subtask.split('-')[-1] == 'crf': 
        subtask_disc += '-crf'
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels, code_disc_labels = norm_iob2_diag_single_disc_d_annotate(arr_start_end=arr_start_end, 
                                                                                               df_ann=df_ann_disc, 
                                                                                               ign_value=ign_value, 
                                                                                               subtask=subtask_disc)
    ## Merge labels
    assert len(iob_cont_labels) == len(iob_disc_labels) == len(code_cont_labels) == len(code_disc_labels)
    def_code_value = "O" if ign_value is None else ign_value
    
    # IOB-2
    iob_labels = []
    for iob_c, iob_d in zip(iob_cont_labels, iob_disc_labels):
        assert (iob_c == def_code_value) or (iob_d == def_code_value)
        
        if iob_c != def_code_value:
            # Continuous ann
            iob_labels.append(iob_c)
            
        elif iob_d != def_code_value:
            # Discontinuous ann
            iob_labels.append(iob_d)
            
        else:
            # No ann
            iob_labels.append(def_code_value)
        
    # Code
    code_labels = []
    for code_c, code_d in zip(code_cont_labels, code_disc_labels):
        assert (code_c == def_code_value) or (code_d == def_code_value)
        
        if code_c != def_code_value:
            # Continuous ann
            code_labels.append(code_c)
            
        elif code_d != def_code_value:
            # Discontinuous ann
            code_labels.append(code_d)
            
        else:
            # No ann
            code_labels.append(def_code_value)
        
            
    return [iob_labels, code_labels]
    


def norm_iob2_cont_disc_d_annotate(arr_start_end, df_ann):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (cont-disc)] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    """
    
    # Continuous
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels = ner_iob2_annotate(arr_start_end=arr_start_end, df_ann=df_ann_cont, subtask="ner")[0]
    
    # Discontinuous
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels = norm_iob2_disc_d_annotate(arr_start_end=arr_start_end, df_ann=df_ann_disc)[0]
    
    ## Merge labels
    assert len(iob_cont_labels) == len(iob_disc_labels)
    def_code_value = "O"
    
    # IOB-2
    iob_labels = []
    for iob_c, iob_d in zip(iob_cont_labels, iob_disc_labels):
        assert (iob_c == def_code_value) or (iob_d == def_code_value)
        
        if iob_c != def_code_value:
            # Continuous ann
            iob_labels.append(iob_c)
            
        elif iob_d != def_code_value:
            # Discontinuous ann
            iob_labels.append(iob_d)
            
        else:
            # No ann
            iob_labels.append(def_code_value)        
            
    return [iob_labels]
    


def norm_iob2_proc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_proc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag_proc, norm-iob_diag_proc-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    subtask_proc = "norm-iob_code"
    if subtask.split('-')[-1] == 'crf': 
        subtask_proc += '-crf'
    
    df_ann_proc = df_ann[df_ann['type'] == 'PROCEDIMIENTO'][['start', 'end', 'code']].copy()
    
    return norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_proc, ign_value=ign_value, subtask=subtask_proc)


def norm_iob2_proc_disc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_proc_disc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    df_ann_proc = df_ann[df_ann['type'] == 'PROCEDIMIENTO'][['location', 'code_pre']].copy()
    df_ann_proc.rename(columns={'code_pre': 'code'}, inplace=True)
    
    iob_disc_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann_proc.iterrows():
        ann_loc_split = row['location'].split(';')
        ## First fragment
        loc_start = int(ann_loc_split[0].split(' ')[0])
        loc_end = int(ann_loc_split[0].split(' ')[1])
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected
        
        iob_disc_labels[tok_start] = "B"
        code_labels[tok_start] = row["code"]
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_disc_labels[i] = "I"
                if subtask.split('-')[-1] == "mention":
                    code_labels[i] = "I"
                else:
                    code_labels[i] = row["code"]
        
        ## Subsequent fragments
        ann_loc_len = len(ann_loc_split)
        for ann_i in range(1, ann_loc_len):
            loc_start = int(ann_loc_split[ann_i].split(' ')[0])
            loc_end = int(ann_loc_split[ann_i].split(' ')[1])
            # First subtoken/word of annotation
            tok_start = np.where(arr_start_end[:, 0] <= loc_start)[0][-1] # last subtoken/word <= annotation start
            # Last subtoken/word of annotation
            tok_end = np.where(arr_start_end[:, 1] >= loc_end)[0][0] # first subtoken/word >= annotation end
            assert tok_start <= tok_end
            # Annotate first subtoken/word
            assert iob_disc_labels[tok_start] == "O" # no overlapping annotations are expected

            iob_disc_labels[tok_start] = "I"
            code_labels[tok_start] = row["code"]

            if tok_start < tok_end:
                # Annotation spanning multiple subtokens/words
                for i in range(tok_start + 1, tok_end + 1):
                    assert iob_disc_labels[i] == "O" # no overlapping annotations are expected
                    iob_disc_labels[i] = "I"
                    if subtask.split('-')[-1] == "mention":
                        code_labels[i] = "I"
                    else:
                        code_labels[i] = row["code"]
    
    return [iob_disc_labels, code_labels]


def norm_iob2_proc_cont_disc_c_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_proc_cont_disc_c'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    subtask_cont = "norm-iob_proc"
    if subtask.split('-')[-1] == 'mention': 
        subtask_cont += '-mention'
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels, code_cont_labels = norm_iob2_proc_annotate(arr_start_end=arr_start_end, df_ann=df_ann_cont, 
                                                                ign_value=ign_value, subtask=subtask_cont)
    # Discontinuous
    subtask_disc = "norm-iob_proc_disc"
    if subtask.split('-')[-1] == 'mention': 
        subtask_disc += '-mention'
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels, code_disc_labels = norm_iob2_proc_disc_annotate(arr_start_end=arr_start_end, df_ann=df_ann_disc, 
                                                                     ign_value=ign_value, subtask=subtask_disc)
    ## Merge labels
    assert len(iob_cont_labels) == len(iob_disc_labels) == len(code_cont_labels) == len(code_disc_labels)
    def_code_value = "O" if ign_value is None else ign_value
    
    # IOB-2
    iob_labels = []
    for iob_c, iob_d in zip(iob_cont_labels, iob_disc_labels):
        assert (iob_c == def_code_value) or (iob_d == def_code_value)
        
        if iob_c != def_code_value:
            # Continuous ann
            iob_labels.append(iob_c)
            
        elif iob_d != def_code_value:
            # Discontinuous ann
            iob_labels.append(iob_d)
            
        else:
            # No ann
            iob_labels.append(def_code_value)
        
    # Code
    code_labels = []
    for code_c, code_d in zip(code_cont_labels, code_disc_labels):
        assert (code_c == def_code_value) or (code_d == def_code_value)
        
        if code_c != def_code_value:
            # Continuous ann
            code_labels.append(code_c)
            
        elif code_d != def_code_value:
            # Discontinuous ann
            code_labels.append(code_d)
            
        else:
            # No ann
            code_labels.append(def_code_value)
        
    return [iob_labels, code_labels]
    

def norm_iob2_proc_code_pre_suffix_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_proc_code_pre_suf'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2, Code_pre, Code_suf] 
    NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_code_suf, norm-iob_code_suf-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    iob_labels = ["O"] * len(arr_start_end)
    default_code_value = "O" if ign_value is None else ign_value
    code_pre_labels = [default_code_value] * len(arr_start_end)
    code_int_labels = [default_code_value] * len(arr_start_end)
    code_suf_labels = [default_code_value] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First subtoken/word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last subtoken/word <= annotation start
        # Last subtoken/word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first subtoken/word >= annotation end
        assert tok_start <= tok_end
        # Annotate first subtoken/word
        assert iob_labels[tok_start] == "O" # no overlapping annotations are expected
        
        iob_labels[tok_start] = "B"
        code_pre_labels[tok_start] = row["code_pre"]
        code_int_labels[tok_start] = row["code_int"]
        code_suf_labels[tok_start] = row["code_suf"] if row["code_suf"] is not None else "O"
            
        if tok_start < tok_end:
            # Annotation spanning multiple subtokens/words
            for i in range(tok_start + 1, tok_end + 1):
                assert iob_labels[i] == "O" # no overlapping annotations are expected
                # here guille now: adapt to "mention-first" coding strategy (only assing code label to the first word of the mention)
                iob_labels[i] = "I"
                if subtask.split('-')[-1] == "mention":
                    code_pre_labels[i] = "I"
                    code_int_labels[i] = "I"
                    code_suf_labels[i] = "I"
                else:
                    code_pre_labels[i] = row["code_pre"]
                    code_int_labels[i] = row["code_int"]
                    code_suf_labels[i] = row["code_suf"] if row["code_suf"] is not None else "O"
    
    return [iob_labels, code_pre_labels, code_int_labels, code_suf_labels]


def norm_iob2_proc_disc_d_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_proc_disc_d'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    df_ann_proc = df_ann[df_ann['type'] == 'PROCEDIMIENTO'][['location', 'code']].copy()
    
    return norm_iob2_code_disc_d_annotate(arr_start_end, df_ann_proc, ign_value=ign_value, 
                                          subtask='norm-iob_code_disc_d')


def norm_iob2_proc_cont_disc_d_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_proc_cont_disc_d'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Continuous
    subtask_cont = "norm-iob_proc"
    if subtask.split('-')[-1] == 'crf': 
        subtask_cont += '-crf'
    df_ann_cont = df_ann[~df_ann['disc']].copy()
    iob_cont_labels, code_cont_labels = norm_iob2_proc_annotate(arr_start_end=arr_start_end, 
                                                                                          df_ann=df_ann_cont, 
                                                                                          ign_value=ign_value, 
                                                                                          subtask=subtask_cont)
    # Discontinuous
    subtask_disc = "norm-iob_proc_disc_d"
    if subtask.split('-')[-1] == 'crf': 
        subtask_disc += '-crf'
    df_ann_disc = df_ann[df_ann['disc']].copy()
    iob_disc_labels, code_disc_labels = norm_iob2_proc_disc_d_annotate(arr_start_end=arr_start_end, 
                                                                                               df_ann=df_ann_disc, 
                                                                                               ign_value=ign_value, 
                                                                                               subtask=subtask_disc)
    ## Merge labels
    assert len(iob_cont_labels) == len(iob_disc_labels) == len(code_cont_labels) == len(code_disc_labels)
    def_code_value = "O" if ign_value is None else ign_value
    
    # IOB-2
    iob_labels = []
    for iob_c, iob_d in zip(iob_cont_labels, iob_disc_labels):
        assert (iob_c == def_code_value) or (iob_d == def_code_value)
        
        if iob_c != def_code_value:
            # Continuous ann
            iob_labels.append(iob_c)
            
        elif iob_d != def_code_value:
            # Discontinuous ann
            iob_labels.append(iob_d)
            
        else:
            # No ann
            iob_labels.append(def_code_value)
        
    # Code
    code_labels = []
    for code_c, code_d in zip(code_cont_labels, code_disc_labels):
        assert (code_c == def_code_value) or (code_d == def_code_value)
        
        if code_c != def_code_value:
            # Continuous ann
            code_labels.append(code_c)
            
        elif code_d != def_code_value:
            # Discontinuous ann
            code_labels.append(code_d)
            
        else:
            # No ann
            code_labels.append(def_code_value)
        
            
    return [iob_labels, code_labels]
    


def norm_iob2_diag_proc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_proc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag_proc, norm-iob_diag_proc-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Diagnosis
    subtask_diag = "norm-iob_code_suf"
    if subtask.split('-')[-1] == 'mention': 
        subtask_diag += '-mention'
    
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['start', 'end', 'code_pre', 'code_suf']].copy()
    
    diag_labels = norm_iob2_code_suffix_annotate(arr_start_end=arr_start_end, df_ann=df_ann_diag, ign_value=ign_value, 
                                                  subtask=subtask_diag)
    # Procedure
    subtask_proc = "norm-iob_code"
    if subtask.split('-')[-1] == 'mention': 
        subtask_proc += '-mention'
    
    df_ann_proc = df_ann[df_ann['type'] == 'PROCEDIMIENTO'][['start', 'end', 'code_pre']].copy()
    df_ann_proc.rename(columns={'code_pre': 'code'}, inplace=True)
    
    proc_labels = norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_proc, ign_value=ign_value, subtask=subtask_proc)
    
    return diag_labels + proc_labels


def norm_iob2_diag_single_proc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_single_proc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag_proc, norm-iob_diag_proc-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    subtask_code = "norm-iob_code"
    if subtask.split('-')[-1] == 'mention': 
        subtask_code += '-mention'
    
    # Diagnosis
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['start', 'end', 'code']].copy()
    diag_labels = norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_diag, ign_value=ign_value, subtask=subtask_code)
    
    # Procedure
    df_ann_proc = df_ann[df_ann['type'] == 'PROCEDIMIENTO'][['start', 'end', 'code']].copy()
    proc_labels = norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_proc, ign_value=ign_value, subtask=subtask_code)
    
    return diag_labels + proc_labels


def norm_iob2_diag_single_only_proc_annotate(arr_start_end, df_ann, ign_value=-100, subtask='norm-iob_diag_single_only_proc'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag_proc, norm-iob_diag_proc-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    subtask_code = "norm-iob_code"
    if subtask.split('-')[-1] == 'mention': 
        subtask_code += '-mention'
    
    # Diagnosis
    df_ann_diag = df_ann[df_ann['type'] == 'DIAGNOSTICO'][['start', 'end', 'code']].copy()
    diag_iob_labels, diag_code_labels = norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_diag, 
                                                                    ign_value=ign_value, subtask=subtask_code)
    
    # Procedure
    df_ann_proc = df_ann[df_ann['type'] == 'PROCEDIMIENTO'][['start', 'end', 'code']].copy()
    proc_iob_labels, proc_code_labels = norm_iob2_code_annotate(arr_start_end=arr_start_end, df_ann=df_ann_proc, 
                                                                    ign_value=ign_value, subtask=subtask_code)
    
    # Merge IOB-2
    assert len(diag_iob_labels) == len(proc_iob_labels)
    def_code_value = "O"
    iob_labels = []
    for iob_d, iob_p in zip(diag_iob_labels, proc_iob_labels):
        assert (iob_d == def_code_value) or (iob_p == def_code_value) or (iob_d == iob_p)
        
        if iob_d != def_code_value:
            # Continuous ann
            iob_labels.append(iob_d)
            
        elif iob_p != def_code_value:
            # Discontinuous ann
            iob_labels.append(iob_p)
            
        else:
            # No ann
            iob_labels.append(def_code_value)
    
    # Add "O" labels to codes if 'ign'
    if ign_value is not None:
        def_code_value = ign_value
        diag_code_labels_final, proc_code_labels_final = [], []
        for code_d, code_p in zip(diag_code_labels, proc_code_labels):
            diag_code_labels_final.append(code_d)
            proc_code_labels_final.append(code_p)
            
            if (code_d != def_code_value) and (code_p == def_code_value):
                proc_code_labels_final[-1] = "O"

            elif (code_d == def_code_value) and (code_p != def_code_value):
                diag_code_labels_final[-1] = "O"
    else:
        diag_code_labels_final = diag_code_labels
        proc_code_labels_final = proc_code_labels
        
    
    return [iob_labels, diag_code_labels_final, proc_code_labels_final]


def norm_iob2_diag_single_only_proc_cont_disc_d_annotate(arr_start_end, df_ann, ign_value=-100, 
                                                  subtask='norm-iob_diag_single_only_proc_cont_disc_d'):
    """
    Annotate a sequence of subtokens/words (given their start-end char positions) following [IOB2 (Diag), Diag_code_pre, 
    Diag_code_suf, IOB2 (Proc), Proc_code] NER-Norm format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Implemented subtasks: norm-iob_diag, norm-iob_diag-mention (using "I")
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    # Diagnosis
    subtask_diag = "norm-iob_diag_single_cont_disc_d"
    if subtask.split('-')[-1] == 'crf': 
        subtask_diag += '-crf'
    iob_diag_labels, code_diag_labels = norm_iob2_diag_single_cont_disc_d_annotate(arr_start_end=arr_start_end, 
                                                                                          df_ann=df_ann, 
                                                                                          ign_value=ign_value, 
                                                                                          subtask=subtask_diag)
    # Procedure
    subtask_proc = "norm-iob_proc_cont_disc_d"
    if subtask.split('-')[-1] == 'crf': 
        subtask_proc += '-crf'
    iob_proc_labels, code_proc_labels = norm_iob2_proc_cont_disc_d_annotate(arr_start_end=arr_start_end, 
                                                                                          df_ann=df_ann, 
                                                                                          ign_value=ign_value, 
                                                                                          subtask=subtask_proc)
    ## Merge labels
    assert len(iob_diag_labels) == len(iob_proc_labels) == len(code_diag_labels) == len(code_proc_labels)
    def_code_value = "O"
    
    # IOB-2
    iob_labels = []
    for iob_d, iob_p in zip(iob_diag_labels, iob_proc_labels):
        assert (iob_d == def_code_value) or (iob_p == def_code_value) or (iob_d == iob_p)
        
        if iob_d != def_code_value:
            # Continuous ann
            iob_labels.append(iob_d)
            
        elif iob_p != def_code_value:
            # Discontinuous ann
            iob_labels.append(iob_p)
            
        else:
            # No ann
            iob_labels.append(def_code_value)
        
            
    return [iob_labels, code_diag_labels, code_proc_labels]
    


def convert_word_token(word_text, word_start_end, word_labels, tokenizer, ign_value, strategy, word_pos):
    """
    Given a list of words, the function converts them to a list of subtokens.
    Implemented strategies: word-all, word-first, word-first-x.
    """
    res_sub_token, res_start_end, res_word_id = [], [], []
    # here guille now: probar multiple-labels
    # Multiple labels
    res_labels = [[] for lab_i in range(len(word_labels))]
    for i in range(len(word_text)):
        w_text = word_text[i]
        w_start_end = word_start_end[i]
        sub_token, _ = start_end_tokenize(text=w_text, tokenizer=tokenizer, start_pos=w_start_end[0])
        tok_start_end = [w_start_end] * len(sub_token) # using the word start-end pair as the start-end position of the subtokens
        tok_word_id = [i + word_pos] * len(sub_token)
        res_sub_token.extend(sub_token)
        res_start_end.extend(tok_start_end)
        res_word_id.extend(tok_word_id)
        # Multiple labels
        for lab_i in range(len(word_labels)):
            w_label = word_labels[lab_i][i]
            if strategy.split('-')[1] == "all":
                res_labels[lab_i].extend([w_label] * len(sub_token))
            else:
                subtk_value = "X" if strategy.split('-')[-1] == "x" else ign_value
                res_labels[lab_i].extend([w_label] + [subtk_value] * (len(sub_token) - 1))
        
    return res_sub_token, res_start_end, res_labels, res_word_id


def start_end_tokenize_ner(text, max_seq_len, tokenizer, start_pos, df_ann, ign_value, strategy="word-all", cased=True, word_pos=0, 
                           subtask='ner', code_strat='ign'):
    """
    Given an input text, it returns a list of lists containing the adjacent sequences of subtokens.
    return: list of lists, shape [n_sequences, n_subtokens] (out_sub_token, out_start_end, out_word_id)
            list of lists of lists, shape [n_outputs, n_sequences, n_subtokens] (out_labels)
    """
    
    out_sub_token, out_start_end, out_labels, out_word_id = [], [], [], []
    if strategy.split('-')[0] == "word":
        # Apply whitespace and punctuation pre-tokenization to extract the words from the input text
        word_text, word_chr_start_end = word_start_end(text=text, start_i=start_pos, cased=cased) 
        assert len(word_text) == len(word_chr_start_end)
        if len(subtask.split('-')) == 1:
            # Obtain IOB-2/IOB-Code labels at word-level
            word_labels = ner_iob2_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, subtask=subtask)
        elif subtask == "norm-mention":
            word_labels = norm_mention_iob2_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann)
        elif subtask.split('-')[1] == "iob_code":
            # Multiple labels. here guille now: adapt to other strategies
            word_labels = norm_iob2_code_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif '-'.join(subtask.split('-')[1:3]) == "iob_code_suf-h":
            word_labels = norm_iob2_code_suffix_h_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_code_suf":
            word_labels = norm_iob2_code_suffix_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_cont_disc_code_suf":
            word_labels = norm_iob2_cont_disc_code_suffix_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None)
        elif subtask.split('-')[1] == "iob_diag":
            word_labels = norm_iob2_diag_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_disc":
            word_labels = norm_iob2_diag_disc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_cont_disc":
            word_labels = norm_iob2_diag_cont_disc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_cont_disc_c":
            word_labels = norm_iob2_diag_cont_disc_c_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_cont_disc":
            word_labels = norm_iob2_cont_disc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann)
        elif subtask.split('-')[1] == "iob_cont_disc_d":
            word_labels = norm_iob2_cont_disc_d_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann)
        elif subtask.split('-')[1] == "iob_diag_single":
            word_labels = norm_iob2_diag_single_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_single_disc_d":
            word_labels = norm_iob2_diag_single_disc_d_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_single_cont_disc_d":
            word_labels = norm_iob2_diag_single_cont_disc_d_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_proc":
            word_labels = norm_iob2_proc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_proc_disc":
            word_labels = norm_iob2_proc_disc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_proc_disc_d":
            word_labels = norm_iob2_proc_disc_d_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_proc_cont_disc_c":
            word_labels = norm_iob2_proc_cont_disc_c_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_proc_cont_disc_d":
            word_labels = norm_iob2_proc_cont_disc_d_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_proc_code_pre_suf":
            word_labels = norm_iob2_proc_code_pre_suffix_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_proc":
            word_labels = norm_iob2_diag_proc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_single_proc":
            word_labels = norm_iob2_diag_single_proc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_single_only_proc":
            word_labels = norm_iob2_diag_single_only_proc_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        elif subtask.split('-')[1] == "iob_diag_single_only_proc_cont_disc_d":
            word_labels = norm_iob2_diag_single_only_proc_cont_disc_d_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann, 
                                                  ign_value = ign_value if code_strat == 'ign' else None, subtask=subtask)
        for lab_i in range(len(word_labels)):
            assert len(word_labels[lab_i]) == len(word_text)
        
        # Convert word-level arrays to subtoken-level
        sub_token, start_end, labels, word_id = convert_word_token(word_text=word_text, word_start_end=word_chr_start_end, 
                                            word_labels=word_labels, tokenizer=tokenizer, ign_value=ign_value, strategy=strategy, 
                                            word_pos=word_pos)
    else:
        raise Exception('Strategy not implemented!')
        
    assert len(sub_token) == len(start_end) == len(word_id)
    # Multiple labels
    for lab_i in range(len(labels)):
        out_labels.append([])
        assert len(labels[lab_i]) == len(sub_token)
    
    # Re-split large sub-tokens sequences
    for i in range(0, len(sub_token), max_seq_len):
        out_sub_token.append(sub_token[i:i+max_seq_len])
        out_start_end.append(start_end[i:i+max_seq_len])
        out_word_id.append(word_id[i:i+max_seq_len])
        # Multiple labels
        for lab_i in range(len(labels)):
            out_labels[lab_i].append(labels[lab_i][i:i+max_seq_len])
    
    return out_sub_token, out_start_end, out_labels, out_word_id  
        

def ss_start_end_tokenize_ner(ss_start_end, max_seq_len, text, tokenizer, df_ann, ign_value, strategy="word-all", 
                              cased=True, subtask='ner', code_strat='ign'):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of 
                  the split sentences from the input document text.
    text: document text.
    
    return: 4 lists of lists, the first for the sub-tokens from the re-split sentences, the second for the 
            start-end char positions pairs of the sub-tokens from the re-split sentences, the third for
            the IOB-2/IOB-Code labels associated to the sub-tokens from the re-split sentences, and the forth for the 
            word id of each sub-token.
    """
    out_sub_token, out_start_end, out_labels, out_word_id = [], [], [], []
    n_ss_words = 0
    for ss_start, ss_end in ss_start_end:
        ss_text = text[ss_start:ss_end]
        # annotations spanning multiple adjacent sentences are not considered
        ss_ann = df_ann[(df_ann['start'] >= ss_start) & (df_ann['end'] <= ss_end)]
        ss_sub_token, ss_start_end, ss_labels, ss_word_id = start_end_tokenize_ner(text=ss_text, max_seq_len=max_seq_len,
                            tokenizer=tokenizer, start_pos=ss_start, df_ann=ss_ann, ign_value=ign_value, 
                            strategy=strategy, cased=cased, word_pos=n_ss_words, subtask=subtask, code_strat=code_strat)
        out_sub_token.extend(ss_sub_token)
        out_start_end.extend(ss_start_end)
        out_word_id.extend(ss_word_id)
        # Multiple labels: here guille now
        if len(out_labels) == 0: # first iteration (dirty, as the number of output tensors is not previously defined)
            out_labels = [[] for lab_i in range(len(ss_labels))]
        for lab_i in range(len(ss_labels)):
            out_labels[lab_i].extend(ss_labels[lab_i])
        
        # We update the number of words contained in the document so far
        n_ss_words = ss_word_id[-1][-1] + 1
    
    return out_sub_token, out_start_end, out_labels, out_word_id


def ss_fragment_greedy_ner(ss_token, ss_start_end, ss_labels, ss_word_id, max_seq_len):
    """
    Same as ss_fragment_greedy but also including a labels and word-id arrays
    """
    frag_token, frag_start_end, frag_word_id = [[]], [[]], [[]]
    # Multiple labels
    frag_labels = [[[]] for lab_i in range(len(ss_labels))]
    
    i = 0
    while i < len(ss_token):
        assert len(ss_token[i]) <= max_seq_len
        if len(frag_token[-1]) + len(ss_token[i]) > max_seq_len:
            # Fragment is full, so create a new empty fragment
            frag_token.append([])
            frag_start_end.append([])
            frag_word_id.append([])
            # Multiple labels
            for lab_i in range(len(ss_labels)):
                frag_labels[lab_i].append([])
            
        frag_token[-1].extend(ss_token[i])
        frag_start_end[-1].extend(ss_start_end[i])
        frag_word_id[-1].extend(ss_word_id[i])
        # Multiple labels
        for lab_i in range(len(ss_labels)):
            frag_labels[lab_i][-1].extend(ss_labels[lab_i][i])
        
        i += 1
          
    return frag_token, frag_start_end, frag_labels, frag_word_id


def format_token_ner(token_list, label_list, tokenizer, seq_len, lab_encoder_list, ign_value, fasttext_strat):
    """
    Given a list of sub-tokens and their assigned NER-labels, as well as a tokenizer, it returns their corresponding lists of 
    indices, attention masks, tokens types and transformed labels. Padding is added as appropriate.
    """
    # here guille now: FastText
    type_tokenizer = str(type(tokenizer))
    if 'transformers' in type_tokenizer:    
        token_ids = tokenizer.convert_tokens_to_ids(token_list)
        # Add [CLS] and [SEP] tokens (single sequence)
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)

        # Generate attention mask
        token_len = len(token_ids)
        attention_mask = [1] * token_len

        # Generate token types
        token_type = [0] * token_len
        
        # Add special tokens labels
        # Multiple labels
        token_labels = []
        for lab_i in range(len(label_list)):
            token_labels.append([ign_value] + [lab_encoder_list[lab_i][label] if label != ign_value else label \
                                               for label in label_list[lab_i]] + [ign_value])
            assert len(token_labels[lab_i]) == token_len
        
        # Padding
        pad_len = seq_len - token_len
        token_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type += [0] * pad_len
        
    
    elif 'fasttext' in type_tokenizer:
        # Add special tokens labels
        # Multiple labels
        token_labels = []
        for lab_i in range(len(label_list)):
            token_labels.append([lab_encoder_list[lab_i][label] if label != ign_value else label \
                                               for label in label_list[lab_i]])
        
        # Implement differently according to the fine-tuning/freezed strategy    
        if fasttext_strat == "ft":
            # Fine-tuning strategy (with zero-padding)
            word_id_offset = 2 # 0 = pad, -1 -> 1 = unk token, 0 -> 2 = first known token, etc.
            token_ids = [tokenizer.get_word_id(word) + word_id_offset for word in token_list]
            token_len = len(token_ids)
            attention_mask, token_type = [], []

            # Padding
            pad_len = seq_len - token_len
            token_ids += [0] * pad_len # zero-padding
            
        elif fasttext_strat == "freeze":
            # Freezed embeddings strategy (with np.zeros padding)
            token_ids = [tokenizer.get_word_vector(word) for word in token_list] # shape: (n_sub_tok_i, dim)
            token_len = len(token_ids)
            attention_mask, token_type = [], []

            # Padding
            pad_len = seq_len - token_len
            token_ids += [np.zeros(tokenizer.get_dimension())] * pad_len # zero-padding, final shape: (seq_len, dim)
            
    
    # Multiple labels
    for lab_i in range(len(label_list)):
        token_labels[lab_i].extend([ign_value] * pad_len)

    return token_ids, attention_mask, token_type, token_labels


from copy import deepcopy

def ss_create_input_data_ner(df_text, text_col, df_ann, df_ann_text, doc_list, ss_dict, tokenizer, lab_encoder_list, text_label_encoder, seq_len, ign_value, 
                             strategy="word-all", greedy=False, cased=True, subtask='ner', code_strat='ign', 
                             fasttext_strat="ft"):
    """
    This function generates the data needed to fine-tune a transformer model on a multi-class token classification task, 
    such as Cantemist-NER subtask, following the IOB-2 annotation format.
    """
    
    indices, attention_mask, token_type, labels, text_labels, n_fragments, start_end_offsets, word_ids = [], [], [], [], [], [], [], []
    # here guille now: FastText
    sub_tok_max_seq_len = seq_len
    type_tokenizer = str(type(tokenizer))
    if 'transformers' in type_tokenizer: sub_tok_max_seq_len -= 2
    for doc in doc_list:
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Text classification
        doc_ann_text = df_ann_text[df_ann_text["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        ## Generate annotated subtokens sequences
        if ss_dict is not None:
            # Perform sentence split (SS) on doc text
            doc_ss = ss_dict[doc] # SS start-end pairs of the doc text
            doc_ss_token, doc_ss_start_end, doc_ss_label, doc_ss_word_id = ss_start_end_tokenize_ner(ss_start_end=doc_ss, 
                                        max_seq_len=sub_tok_max_seq_len, text=doc_text, 
                                        tokenizer=tokenizer, df_ann=doc_ann, ign_value=ign_value, strategy=strategy, cased=cased, 
                                        subtask=subtask, code_strat=code_strat)
            assert len(doc_ss_token) == len(doc_ss_start_end) == len(doc_ss_word_id)
            # Multiple labels
            for lab_i in range(len(doc_ss_label)):
                assert len(doc_ss_label[lab_i]) == len(doc_ss_token)
                
            if greedy:
                # Split the list of sub-tokens sentences into sequences comprising multiple sentences
                frag_token, frag_start_end, frag_label, frag_word_id = ss_fragment_greedy_ner(ss_token=doc_ss_token, 
                                ss_start_end=doc_ss_start_end, ss_labels=doc_ss_label, ss_word_id=doc_ss_word_id, 
                                max_seq_len=sub_tok_max_seq_len)
            else: 
                frag_token = deepcopy(doc_ss_token)
                frag_start_end = deepcopy(doc_ss_start_end)
                frag_label = deepcopy(doc_ss_label)
                frag_word_id = deepcopy(doc_ss_word_id)
        else:
            # Generate annotated sequences using text-stream strategy (without considering SS)
            frag_token, frag_start_end, frag_label, frag_word_id = start_end_tokenize_ner(text=doc_text, max_seq_len=sub_tok_max_seq_len,
                            tokenizer=tokenizer, start_pos=0, df_ann=doc_ann, ign_value=ign_value, 
                            strategy=strategy, cased=cased, word_pos=0, subtask=subtask, code_strat=code_strat)
            
        assert len(frag_token) == len(frag_start_end) == len(frag_word_id)
        # Multiple labels
        for lab_i in range(len(frag_label)):
            assert len(frag_label[lab_i]) == len(frag_token)
        # Store the start-end char positions of all the sequences
        start_end_offsets.extend(frag_start_end)
        # Store the sub-tokens word ids of all the sequences
        word_ids.extend(frag_word_id)
        # Store the number of sequences of each doc text
        n_fragments.append(len(frag_token))
        ## Subtokens sequences formatting
        # Multiple labels
        if len(labels) == 0: 
            labels = [[] for lab_i in range(len(frag_label))] # first iteration (dirty, as the number of output tensors is not previously defined)
        for seq_i in range(len(frag_token)):
            f_token = frag_token[seq_i]
            f_start_end = frag_start_end[seq_i]
            f_word_id = frag_word_id[seq_i]
            # Multiple labels
            f_label = []
            for lab_i in range(len(frag_label)):
                f_label.append(frag_label[lab_i][seq_i])
            # sequence length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) == len(f_word_id) <= sub_tok_max_seq_len
            # Multiple labels
            for lab_i in range(len(f_label)):
                assert len(f_label[lab_i]) == len(f_token)
            f_id, f_att, f_type, f_label = format_token_ner(token_list=f_token, label_list=f_label, 
                                                            tokenizer=tokenizer, seq_len=seq_len, 
                                                            lab_encoder_list=lab_encoder_list, ign_value=ign_value, 
                                                            fasttext_strat=fasttext_strat)
            
            # Text classification
            text_frag_labels = []
            # start-end char positions of the whole fragment, i.e. the start position of the first
            # sub-token and the end position of the last sub-token
            frag_start, frag_end = f_start_end[0][0], f_start_end[-1][1]
            for j in range(doc_ann_text.shape[0]):
                doc_ann_cur = doc_ann_text.iloc[j] # current annotation
                # Add the annotations whose text references are contained within the fragment
                if doc_ann_cur['start'] < frag_end and doc_ann_cur['end'] > frag_start:
                    text_frag_labels.append(doc_ann_cur['code'])
            text_labels.append(text_frag_labels)
            
            indices.append(f_id)
            attention_mask.append(f_att)
            token_type.append(f_type)
            # Multiple labels
            for lab_i in range(len(f_label)):
                labels[lab_i].append(f_label[lab_i])
            
    return np.array(indices), np.array(attention_mask), np.array(token_type), np.array(labels), \
           text_label_encoder.transform(text_labels), np.array(n_fragments), start_end_offsets, word_ids



## NER performance evaluation

from sklearn.preprocessing import normalize

def word_seq_preds(tok_seq_word_id, tok_seq_preds, tok_seq_start_end, strategy):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-sum", "word-all-crf", "word-first-crf", 
                            "word-first-x-crf".
    """

    # Convert subtoken-level predictions to word-level predictions
    arr_word_seq_start_end = []
    # Multiple labels
    arr_word_seq_preds = [[] for lab_i in range(len(tok_seq_preds))]
    left = 0
    while left < len(tok_seq_word_id):
        cur_word_id = tok_seq_word_id[left]
        right = left + 1
        while right < len(tok_seq_word_id):
            if tok_seq_word_id[right] != cur_word_id:
                break
            right += 1
        # cur_word_id spans from left to right - 1 subtoken positions
        assert len(set(tok_seq_start_end[left:right])) == 1 # start-end pos of the subtokens correspond to the word start-end pos
        arr_word_seq_start_end.append(tok_seq_start_end[left])
        
        # Multiple labels
        for lab_i in range(len(tok_seq_preds)):
            if strategy.split('-')[-1] == "max":
                # max of predictions made in all subtokens of the word 
                arr_word_seq_preds[lab_i].append(np.max(tok_seq_preds[lab_i][left:right], axis=0))

            elif strategy.split('-')[-1] == "prod":
                # product of predictions made in all subtokens of the word 
                arr_word_seq_preds[lab_i].append(np.prod(tok_seq_preds[lab_i][left:right], axis=0))

            elif strategy.split('-')[-1] == "sum":
                # sum of predictions made in all subtokens of the word 
                arr_word_seq_preds[lab_i].append(np.sum(tok_seq_preds[lab_i][left:right], axis=0))

            elif strategy.split('-')[-1] == "sum_norm":
                # sum of predictions made in all subtokens of the word                 
                arr_word_seq_preds[lab_i].append(normalize(
                    np.sum(tok_seq_preds[lab_i][left:right], axis=0).reshape(1, -1), 
                    norm='l1', 
                    axis=1
                )[0])

            elif '-'.join(strategy.split('-')[-2:]) == "all-crf":
                # label obtaining the relative majority from the predictions made in all subtokens of the word
                # (labels are assumed to be int)
                arr_word_seq_preds[lab_i].append(np.argmax(np.bincount(tok_seq_preds[lab_i][left:right])))

            elif '-'.join(strategy.split('-')[-2:]) == "first-crf":
                # label predicted on the first subtoken of the word
                arr_word_seq_preds[lab_i].append(tok_seq_preds[lab_i][cur_word_id]) # CRF only predicts the first subtoken of each word
            elif strategy.split('-')[1] == "first": # word-first, word-first-x-crf
                # predictions made on the first subtoken of the word
                arr_word_seq_preds[lab_i].append(tok_seq_preds[lab_i][left]) 

            else:
                raise Exception('Word strategy not implemented!')

        left = right
    
    assert cur_word_id == tok_seq_word_id[-1]
    
    return arr_word_seq_preds, arr_word_seq_start_end


def ner_iob2_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, subtask='ner', strategy='word-first'):
    """
    seq_preds: it is assumed to be a list containing a single list (single label, either IOB or IOB-Code), 
    e.g. NER: [[("B"), ("O")]]; NORM: [[("B-8000/3", 0.87), ("O", 0.6)]] (non-CRF), [[("B-8000/3"), ("O")]] (CRF)
    
    subtask: ner, norm, norm-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0].split('-')[0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0].split('-')[0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1], 
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})
            
            if subtask.split('-')[0] == 'norm':
                if subtask == 'norm':
                    if strategy.split('-')[-1] != "crf":
                        # Extract probabilities of the labels predicted within the annotation (from left to right - 1 pos)
                        ann_lab_prob = np.array([pair[1] for pair in seq_preds[0][left:right]])
                        # Select the label with the maximum probability
                        max_lab = left + np.argmax(ann_lab_prob)
                        code_pred = seq_preds[0][max_lab][0].split('-')[1]
                    else:
                        # Extract codes predicted in the annotation (from left to right - 1 pos)
                        ann_codes = [pred[0].split('-')[1] for pred in seq_preds[0][left:right]]
                        # Select the most frequently predicted code within the annotation
                        codes_uniq, codes_freq = np.unique(ann_codes, return_counts=True)
                        code_pred = codes_uniq[np.argmax(codes_freq)]
                        
                elif subtask == 'norm-mention':
                    code_pred = seq_preds[0][left][0].split('-')[1]
                    
                # Add NORM annotation
                res[-1]['code_pred'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res


def mention_seq_preds(seq_preds, mention_strat='max'):
    """
    Given the word-level coding-predictions (labels probabilities) made in a detected mention, the function returns a 
    single coding-prediction (labels probabilities) made for the whole mention.
    
    seq_preds: shape n_words (in the mention) x n_labels (1 for CRF)
    
    mention_strat: first, max, prod, all-crf.
    """
    
    res = None
    if mention_strat == "first":
        # Select the coding-prediction made for the first word of the mention
        res = seq_preds[0]
    elif mention_strat == "max":
        res = np.max(seq_preds, axis=0)
    elif mention_strat == "prod":
        res = np.prod(seq_preds, axis=0)
    elif mention_strat == "sum":
        res = np.sum(seq_preds, axis=0)
    elif mention_strat == "all-crf":
        # Select the most frequently predicted coding-label within the mention
        res = np.argmax(np.bincount(seq_preds)) # predicted labels are assumed to be int
    else:
        raise Exception('Mention strategy not implemented!')
    
    return res


def norm_iob2_code_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, code_lab_decoder_list, 
                                     strategy='word-first', subtask='norm-iob_code', mention_strat='max'):
    """
    seq_preds: it is assumed to be a list containing two lists (double label, IOB + Code), e.g. [[("B"), ("O")], 
    [('8000/3', 0.87), ('8756/3H', 0.2)]] (non-CRF), [[("B"), ("O")], [('8000/3'), ('8756/3H')]] (CRF)
    
    subtask: norm-iob_code, norm_iob_code-mention
    
    code_lab_decoder_list: [code_lab_decoder]
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1], 
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})
            # Coding-predictions made on the detected mention (from left to right - 1 pos)
            mention_code_preds = seq_preds[1][left:right]
            if '-'.join(subtask.split('-')[1:]) == 'iob_code':
                if strategy.split('-')[-1] != "crf":
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))]

                else:
                    code_pred = code_lab_decoder_list[0][mention_seq_preds(seq_preds=mention_code_preds, mention_strat='all-crf')]
                    
            elif '-'.join(subtask.split('-')[1:]) == 'iob_code-mention':
                label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                if strategy.split('-')[-1] != "crf":
                    label_pred = np.argmax(label_pred)
                code_pred = code_lab_decoder_list[0][label_pred]
            
            # Add NORM annotation
            res[-1]['code_pred'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res


def norm_iob2_code_mask_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, code_lab_decoder_list, 
                                     codes_o_mask, mention_strat='max'):
    """
    IT ONLY WORKS FOR code_strat='o', != 'crf'
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1], 
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})
            # Coding-predictions made on the detected mention (from left to right - 1 pos)
            mention_code_preds = seq_preds[1][left:right]
            code_pred = code_lab_decoder_list[0][np.argmax(np.multiply(codes_o_mask, 
                            mention_seq_preds(seq_preds=mention_code_preds, mention_strat=mention_strat)))]
            
            # Add NORM annotation
            res[-1]['code_pred'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res


def norm_iob2_code_suffix_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, code_lab_decoder_list, 
                                            strategy='word-first', subtask='norm-iob_code_suf', mention_strat='max',
                                            code_sep='/'):
    """
    seq_preds: it is assumed to be a list containing 3/4 lists (IOB + Code-pre + Code-suf + (optional) H-indicator), 
    e.g. [[("B"), ("O")], [('8000', 0.87), ('8756', 0.2)], [('/3', 0.6), ('/3H', 0.3)]] (non-CRF), 
    [[("B"), ("O")], [('8000'), ('8756')], [('/3'), ('/3H')]] (CRF)
    
    code_lab_decoder_list: [code_pre_lab_decoder, code_suf_lab_decoder, (optional) code_h_lab_decoder]
    
    subtask: norm-iob_code_suf, norm-iob_code_suf-mention, norm-iob_code_suf-h, norm-iob_code_suf-h-mention
    """
    
    n_output = len(seq_preds)
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0], 
                        'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1]})
            
            # Extract predicted NORM-code
            code_pred = []
            for lab_i in range(1, n_output):
                mention_code_preds = seq_preds[lab_i][left:right]
                if subtask.split('-')[-1] != 'mention':
                    if strategy.split('-')[-1] != "crf":
                        # here guille now: not adapted to word-first
                        code_pred.append(code_lab_decoder_list[lab_i-1][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))])
                        
                    else:
                        code_pred.append(code_lab_decoder_list[lab_i-1][mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat='all-crf')])
                
                else:
                    label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                    if strategy.split('-')[-1] != "crf":
                        label_pred = np.argmax(label_pred)
                    code_pred.append(code_lab_decoder_list[lab_i-1][label_pred])
                    
            if code_pred[-1] == "O": # no suffix predicted
                code_pred = code_pred[:-1]
                
            # Join code-prefix and code-suffix
            code_pred = code_sep.join(code_pred)       
            
            # Add NORM annotation
            res[-1]['code_pred'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res


def norm_iob2_code_suffix_mask_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, code_lab_decoder_list, 
                                            codes_pre_suf_mask, codes_pre_o_mask,
                                            mention_strat='max', code_sep='/'):
    """
    IT ONLY WORKS FOR code_strat='o', != 'crf'
    """
    
    n_output = len(seq_preds)
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0], 
                        'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1]})
            
            # Extract predicted NORM-code
            # HERE GUILLE NOW: mask impossible codes
            # Also mask "O" in both pre and suf labels (since an ann has been detected)
            code_pred = []
            # Extract mention-probs
            mention_code_pre_preds = mention_seq_preds(seq_preds=seq_preds[1][left:right], 
                                                       mention_strat=mention_strat)
            mention_code_suf_preds = mention_seq_preds(seq_preds=seq_preds[2][left:right], 
                                                       mention_strat=mention_strat)
            # Mask "O" pre-label
            mention_code_pre_preds = np.multiply(codes_pre_o_mask, mention_code_pre_preds)
            label_pre_pred = np.argmax(mention_code_pre_preds)
            # Mask impossible suf-labels
            mention_code_suf_preds = np.multiply(codes_pre_suf_mask[label_pre_pred], mention_code_suf_preds) # be careful with label_pre_pred (in case it is "O")
            label_suf_pred = np.argmax(mention_code_suf_preds)
            # Append code-pre
            code_pred.append(code_lab_decoder_list[0][label_pre_pred])
            # Append code-suf
            code_pred.append(code_lab_decoder_list[1][label_suf_pred])
            
            if code_pred[-1] == "O": # no suffix predicted
                code_pred = code_pred[:-1]
            
            # Join code-prefix and code-suffix
            code_pred = code_sep.join(code_pred)       
            
            # Add NORM annotation
            res[-1]['code_pred'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res


def norm_code_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col):
    """
    seq_preds: it is assumed to be a list containing a single list (single label, either IOB or IOB-Code), 
    e.g. NER: [[("B"), ("O")]]; NORM: [[("B-8000/3", 0.87), ("O", 0.6)]] (non-CRF), [[("B-8000/3"), ("O")]] (CRF)
    
    subtask: ner, norm, norm-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] != "O":
            code_pred = seq_preds[0][left][0]
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != code_pred:
                    break
                right += 1
            # Add NORM annotation
            res.append({'clinical_case': doc_id, 'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1], 
                        'code_pred': code_pred, 'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})
                
            left = right # next sub-token different from code_pred, or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res

                                      
def norm_iob2_diag_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    
    ## Diagnosis
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                right += 1
                
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                        'label_pred': 'DIAGNOSTICO'})
            
            # Extract predicted NORM-code
            code_pred = []
            for lab_i in range(1, 3):
                mention_code_preds = seq_preds[lab_i][left:right]
                if subtask.split('-')[-1] != 'mention':
                    if strategy.split('-')[-1] != "crf":
                        # here guille now: not adapted to word-first
                        code_pred.append(code_lab_decoder_list[lab_i-1][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))])
                
                else:
                    label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                    if strategy.split('-')[-1] != "crf":
                        label_pred = np.argmax(label_pred)
                    code_pred.append(code_lab_decoder_list[lab_i-1][label_pred])
                    
            if code_pred[-1] == "O": # no code-suffix predicted
                code_pred = code_pred[:-1]
                
            # Join code-prefix and code-suffix
            code_pred = '.'.join(code_pred)       
            
            # Add NORM annotation
            res[-1]['code'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    return res


def norm_iob2_diag_disc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_disc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    
    ## Diagnosis
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            ann_iter_pos = list(range(left, right))
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "B"):
                if seq_preds[0][left][0] == "I":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    ann_iter_pos += list(range(left, right))
                    
                    left = right
                    
                else:
                    left += 1
            
            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'pos_pred': ann_start_pos + ' ' + ann_end_pos, 'label_pred': 'DIAGNOSTICO'})
            
            # Extract predicted NORM-code
            code_pred = []
            for lab_i in range(1, 3):
                mention_code_preds = [seq_preds[lab_i][pos] for pos in ann_iter_pos]
                if subtask.split('-')[-1] != 'mention':
                    if strategy.split('-')[-1] != "crf":
                        # here guille now: not adapted to word-first
                        code_pred.append(code_lab_decoder_list[lab_i-1][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))])
                
                else:
                    label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                    if strategy.split('-')[-1] != "crf":
                        label_pred = np.argmax(label_pred)
                    code_pred.append(code_lab_decoder_list[lab_i-1][label_pred])
                    
            if code_pred[-1] == "O": # no code-suffix predicted
                code_pred = code_pred[:-1]
                
            # Join code-prefix and code-suffix
            code_pred = '.'.join(code_pred)       
            
            # Add NORM annotation
            res[-1]['code'] = code_pred
            
        else:
            left += 1
    
    return res


def norm_iob2_disc_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 1 list (IOB (disc))
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "B"):
                if seq_preds[0][left][0] == "I":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    
                    left = right
                    
                else:
                    left += 1
            
            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'start': int(ann_start_pos), 'end': int(ann_end_pos), 
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0],
                        'location': ann_pos})
            
        else:
            left += 1
    
    return res


def norm_iob2_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 1 list (IOB (disc))
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "DB":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "DI"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "DB"):
                if seq_preds[0][left][0] == "DI":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "DI"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    
                    left = right
                    
                else:
                    left += 1
            
            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'start': int(ann_start_pos), 'end': int(ann_end_pos), 
                        'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0]})
            
        else:
            left += 1
    
    return res


def norm_iob2_code_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_code_disc_d', mention_strat='max',
                                          code_type='DIAGNOSTICO'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "DB":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "DI"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            ann_iter_pos = list(range(left, right))
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "DB"):
                if seq_preds[0][left][0] == "DI":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "DI"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    ann_iter_pos += list(range(left, right))
                    
                    left = right
                    
                else:
                    left += 1
            
            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'pos_pred': ann_start_pos + ' ' + ann_end_pos, 'label_pred': code_type})
            
            # Extract predicted NORM-code            
            mention_code_preds = [seq_preds[1][pos] for pos in ann_iter_pos]
            if subtask.split('-')[-1] != 'mention':
                if strategy.split('-')[-1] != "crf":
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))]
                    
            # Add NORM annotation
            res[-1]['code'] = code_pred
            
        else:
            left += 1
    
    return res


def norm_iob2_diag_single_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_disc_d', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    return norm_iob2_code_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat,
                                          code_type='DIAGNOSTICO')


def norm_iob2_code_only_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_code_only_disc_d', mention_strat='max',
                                          code_type='DIAGNOSTICO'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "DB":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "DI"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            ann_iter_pos = list(range(left, right))
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "DB"):
                if seq_preds[0][left][0] == "DI":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "DI"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    ann_iter_pos += list(range(left, right))
                    
                    left = right
                    
                else:
                    left += 1
            
            # Extract predicted NORM-code            
            mention_code_preds = [seq_preds[1][pos] for pos in ann_iter_pos]
            if subtask.split('-')[-1] != 'mention':
                if strategy.split('-')[-1] != "crf":
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))]
                    
            # Add annotation
            if code_pred != 'O':
                # Add annotation
                ann_start_pos = ann_pos.split(' ')[0]
                ann_end_pos = ann_pos.split(' ')[-1]
                res.append({'clinical_case': doc_id, 'pos_pred': ann_start_pos + ' ' + ann_end_pos, 'label_pred': code_type,
                            'code': code_pred})
            
        else:
            left += 1
    
    return res


def norm_iob2_diag_single_only_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_only_disc_d', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    return norm_iob2_code_only_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat,
                                          code_type='DIAGNOSTICO')


def norm_iob2_diag_cont_disc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_cont_disc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Continuous
    seq_preds_cont = [seq_preds[0], seq_preds[2], seq_preds[3]]
    labels_cont = norm_iob2_diag_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_cont, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Discontinuous
    seq_preds_disc = [seq_preds[1], seq_preds[2], seq_preds[3]]
    labels_disc = norm_iob2_diag_disc_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_disc, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_cont + labels_disc


def norm_iob2_cont_disc_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 2 lists (IOB (cont) + IOB (disc))
    """
    
    # Continuous
    seq_preds_cont = [seq_preds[0]]
    labels_cont = ner_iob2_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_cont, seq_start_end=seq_start_end, 
            df_text=df_text, text_col=text_col, subtask='ner')
    for ann_dic in labels_cont:
        ann_dic['location'] = str(ann_dic['start']) + ' ' + str(ann_dic['end'])
    
    # Discontinuous
    seq_preds_disc = [seq_preds[1]]
    labels_disc = norm_iob2_disc_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_disc, seq_start_end=seq_start_end,
            df_text=df_text, text_col=text_col)
    
    return labels_cont + labels_disc


def norm_iob2_disc_code_suffix_mask_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, 
                                code_lab_decoder_list, codes_pre_suf_mask, codes_pre_o_mask,
                                mention_strat='max', code_sep='.'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 1 list (IOB (disc))
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            ann_iter_pos = list(range(left, right))
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "B"):
                if seq_preds[0][left][0] == "I":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    ann_iter_pos += list(range(left, right))
                    
                    left = right
                    
                else:
                    left += 1
            
            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0],
                        'start': int(ann_start_pos), 'end': int(ann_end_pos), 
                        'location': ann_pos})
            
            # Extract predicted NORM-code
            code_pred = []
            # Extract mention-probs
            mention_code_pre_preds = mention_seq_preds(seq_preds=[seq_preds[1][pos] for pos in ann_iter_pos], 
                                                       mention_strat=mention_strat)
            mention_code_suf_preds = mention_seq_preds(seq_preds=[seq_preds[2][pos] for pos in ann_iter_pos], 
                                                       mention_strat=mention_strat)
            # Mask "O" pre-label
            mention_code_pre_preds = np.multiply(codes_pre_o_mask, mention_code_pre_preds)
            label_pre_pred = np.argmax(mention_code_pre_preds)
            # Mask impossible suf-labels
            mention_code_suf_preds = np.multiply(codes_pre_suf_mask[label_pre_pred], mention_code_suf_preds) # be careful with label_pre_pred (in case it is "O")
            label_suf_pred = np.argmax(mention_code_suf_preds)
            # Append code-pre
            code_pred.append(code_lab_decoder_list[0][label_pre_pred])
            # Append code-suf
            code_pred.append(code_lab_decoder_list[1][label_suf_pred])
            
            if code_pred[-1] == "O": # no suffix predicted
                code_pred = code_pred[:-1]
            
            # Join code-prefix and code-suffix
            code_pred = code_sep.join(code_pred)       
            
            # Add NORM annotation
            res[-1]['code_pred'] = code_pred
            
        else:
            left += 1
    
    return res


def norm_iob2_cont_disc_code_suffix_mask_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col, 
                                            code_lab_decoder_list, codes_pre_suf_mask, codes_pre_o_mask,
                                            mention_strat='max', code_sep='.'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 4 lists (IOB (cont) + IOB (disc) + Code-pre + Code-suf)
    
    code_lab_decoder_list: [code_pre_lab_decoder, code_suf_lab_decoder]
    """
    
    # Continuous
    seq_preds_cont = [seq_preds[0], seq_preds[2], seq_preds[3]]
    labels_cont = norm_iob2_code_suffix_mask_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_cont, 
                                          seq_start_end=seq_start_end, 
                                          df_text=df_text, text_col=text_col, code_lab_decoder_list=code_lab_decoder_list, 
                                          codes_pre_suf_mask=codes_pre_suf_mask, codes_pre_o_mask=codes_pre_o_mask, 
                                          mention_strat=mention_strat, code_sep=code_sep)
    for ann_dic in labels_cont:
        ann_dic['location'] = str(ann_dic['start']) + ' ' + str(ann_dic['end'])
    # here guille priority now
    # Discontinuous
    seq_preds_disc = [seq_preds[1], seq_preds[2], seq_preds[3]]
    labels_disc = norm_iob2_disc_code_suffix_mask_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_disc, 
                                          seq_start_end=seq_start_end, 
                                          df_text=df_text, text_col=text_col, code_lab_decoder_list=code_lab_decoder_list, 
                                          codes_pre_suf_mask=codes_pre_suf_mask, codes_pre_o_mask=codes_pre_o_mask, 
                                          mention_strat=mention_strat, code_sep=code_sep)
    
    return labels_cont + labels_disc


def norm_iob2_cont_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, df_text, text_col):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 1 list (IOB (cont-disc))
    """
    
    # Continuous
    labels_cont = ner_iob2_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
            df_text=df_text, text_col=text_col, subtask='ner')
    
    # Discontinuous
    labels_disc = norm_iob2_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end,
            df_text=df_text, text_col=text_col)
    
    return labels_cont + labels_disc


def norm_iob2_diag_single_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0].split('-')[0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0].split('-')[0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                        'label_pred': 'DIAGNOSTICO'})
            
            # Extract predicted NORM-code
            code_pred = None
            if strategy.split('-')[-1] != "crf":
                mention_code_preds = seq_preds[1][left:right]
                # here guille now: not adapted to word-first
                code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                mention_strat=mention_strat))]
            else:
                # Extract codes predicted in the annotation (from left to right - 1 pos)
                ann_codes = [pred[0].split('-')[1] for pred in seq_preds[0][left:right]]
                # Select the most frequently predicted code within the annotation
                codes_uniq, codes_freq = np.unique(ann_codes, return_counts=True)
                code_pred = codes_uniq[np.argmax(codes_freq)]
                    
            # Add NORM annotation
            res[-1]['code'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    
    return res


def norm_iob2_diag_single_only_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_only', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            
            # Extract predicted NORM-code
            code_pred = None
            mention_code_preds = seq_preds[1][left:right]
            if subtask.split('-')[-1] != 'mention':
                if strategy.split('-')[-1] != "crf":
                    # here guille now: not adapted to word-first
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                    mention_strat=mention_strat))]
                
            else:
                label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                if strategy.split('-')[-1] != "crf":
                    label_pred = np.argmax(label_pred)
                code_pred = code_lab_decoder_list[0][label_pred]
            
            if code_pred != 'O':
                # Add annotation
                res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                            'label_pred': 'DIAGNOSTICO', 'code': code_pred})
                    
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    
    return res


def norm_iob2_diag_single_cont_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_cont_disc_d', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Continuous
    labels_cont = norm_iob2_diag_single_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Discontinuous
    labels_disc = norm_iob2_diag_single_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_cont + labels_disc


def norm_iob2_diag_single_only_cont_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_only_cont_disc_d', 
                                          mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Continuous
    labels_cont = norm_iob2_diag_single_only_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Discontinuous
    labels_disc = norm_iob2_diag_single_only_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_cont + labels_disc


def norm_iob2_proc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0].split('-')[0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0].split('-')[0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                        'label_pred': 'PROCEDIMIENTO'})
            
            # Extract predicted NORM-code
            code_pred = None
            if strategy.split('-')[-1] != "crf":
                mention_code_preds = seq_preds[1][left:right]
                # here guille now: not adapted to word-first
                code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                mention_strat=mention_strat))]
            else:
                # Extract codes predicted in the annotation (from left to right - 1 pos)
                ann_codes = [pred[0].split('-')[1] for pred in seq_preds[0][left:right]]
                # Select the most frequently predicted code within the annotation
                codes_uniq, codes_freq = np.unique(ann_codes, return_counts=True)
                code_pred = codes_uniq[np.argmax(codes_freq)]
                    
            # Add NORM annotation
            res[-1]['code'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    
    return res


def norm_iob2_proc_only_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_only', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            
            # Extract predicted NORM-code
            code_pred = None
            mention_code_preds = seq_preds[1][left:right]
            if subtask.split('-')[-1] != 'mention':
                if strategy.split('-')[-1] != "crf":
                    # here guille now: not adapted to word-first
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                    mention_strat=mention_strat))]
                
            else:
                label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                if strategy.split('-')[-1] != "crf":
                    label_pred = np.argmax(label_pred)
                code_pred = code_lab_decoder_list[0][label_pred]
            
            if code_pred != 'O':
                # Add annotation
                res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                            'label_pred': 'PROCEDIMIENTO', 'code': code_pred})
                    
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    
    return res


def norm_iob2_proc_disc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_disc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            ## First fragment
            right = left + 1
            while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                right += 1
            
            # Save pos
            ann_pos = str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
            ann_iter_pos = list(range(left, right))
            
            left = right
            while (left < len(seq_preds[0])) and (seq_preds[0][left][0] != "B"):
                if seq_preds[0][left][0] == "I":
                    ## Subsequent fragment
                    right = left + 1
                    while (right < len(seq_preds[0])) and (seq_preds[0][right][0] == "I"):
                        right += 1
                    
                    # Save pos
                    ann_pos += ';' + str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right-1][1])
                    ann_iter_pos += list(range(left, right))
                    
                    left = right
                    
                else:
                    left += 1
            
            # Add NER annotation
            # For discontinuous annotations, just keep the first and last offset
            ann_start_pos = ann_pos.split(' ')[0]
            ann_end_pos = ann_pos.split(' ')[-1]
            res.append({'clinical_case': doc_id, 'pos_pred': ann_start_pos + ' ' + ann_end_pos, 'label_pred': 'PROCEDIMIENTO'})
            
            # Extract predicted NORM-code
            code_pred = None
            mention_code_preds = [seq_preds[1][pos] for pos in ann_iter_pos]
            if subtask.split('-')[-1] != 'mention':
                if strategy.split('-')[-1] != "crf":
                    # here guille now: not adapted to word-first
                    code_pred = code_lab_decoder_list[0][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                    mention_strat=mention_strat))]

            else:
                label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                if strategy.split('-')[-1] != "crf":
                    label_pred = np.argmax(label_pred)
                code_pred = code_lab_decoder_list[0][label_pred]
                    
            # Add NORM annotation
            res[-1]['code'] = code_pred
            
        else:
            left += 1
    
    return res


def norm_iob2_proc_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_disc_d', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    return norm_iob2_code_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat,
                                          code_type='PROCEDIMIENTO')


def norm_iob2_proc_only_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_only_disc_d', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    return norm_iob2_code_only_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat,
                                          code_type='PROCEDIMIENTO')


def norm_iob2_proc_cont_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_cont_disc_d', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Continuous
    labels_cont = norm_iob2_proc_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Discontinuous
    labels_disc = norm_iob2_proc_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_cont + labels_disc


def norm_iob2_proc_only_cont_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_only_cont_disc_d', 
                                          mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Continuous
    labels_cont = norm_iob2_proc_only_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Discontinuous
    labels_disc = norm_iob2_proc_only_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=code_lab_decoder_list, 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_cont + labels_disc


def norm_iob2_proc_code_pre_suffix_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_proc_code_pre_suf', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                        'label_pred': 'PROCEDIMIENTO'})
            
            # Extract predicted NORM-code
            code_pred = []
            for lab_i in range(1, 4):
                mention_code_preds = seq_preds[lab_i][left:right]
                if subtask.split('-')[-1] != 'mention':
                    if strategy.split('-')[-1] != "crf":
                        # here guille now: not adapted to word-first
                        code_pred.append(code_lab_decoder_list[lab_i-1][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))])
                
                else:
                    label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                    if strategy.split('-')[-1] != "crf":
                        label_pred = np.argmax(label_pred)
                    code_pred.append(code_lab_decoder_list[lab_i-1][label_pred])
                    
            if code_pred[-1] == "O": # no code-suffix predicted
                code_pred = code_pred[:-1]
                
            # Join code-prefix and code-suffix
            code_pred = ''.join(code_pred)
                    
            # Add NORM annotation
            res[-1]['code'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    
    return res


def norm_iob2_diag_proc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_proc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder, proc_code_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    res = []
    
    ## Diagnosis
    left = 0
    while left < len(seq_preds[0]):
        if seq_preds[0][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[0]):
                if seq_preds[0][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                        'label_pred': 'DIAGNOSTICO'})
            
            # Extract predicted NORM-code
            code_pred = []
            for lab_i in range(1, 3):
                mention_code_preds = seq_preds[lab_i][left:right]
                if subtask.split('-')[-1] != 'mention':
                    if strategy.split('-')[-1] != "crf":
                        # here guille now: not adapted to word-first
                        code_pred.append(code_lab_decoder_list[lab_i-1][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                        mention_strat=mention_strat))])
                
                else:
                    label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                    if strategy.split('-')[-1] != "crf":
                        label_pred = np.argmax(label_pred)
                    code_pred.append(code_lab_decoder_list[lab_i-1][label_pred])
                    
            if code_pred[-1] == "O": # no code-suffix predicted
                code_pred = code_pred[:-1]
                
            # Join code-prefix and code-suffix
            code_pred = '.'.join(code_pred)       
            
            # Add NORM annotation
            res[-1]['code'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    ## Procedure
    left = 0
    while left < len(seq_preds[3]):
        if seq_preds[3][left][0] == "B":
            right = left + 1
            while right < len(seq_preds[3]):
                if seq_preds[3][right][0] != "I":
                    break
                right += 1
            # Add NER annotation
            res.append({'clinical_case': doc_id, 'pos_pred': str(seq_start_end[left][0]) + ' ' + str(seq_start_end[right - 1][1]), 
                        'label_pred': 'PROCEDIMIENTO'})
            
            # Extract predicted NORM-code
            code_pred = None
            mention_code_preds = seq_preds[4][left:right]
            if subtask.split('-')[-1] != 'mention':
                if strategy.split('-')[-1] != "crf":
                    # here guille now: not adapted to word-first
                    code_pred = code_lab_decoder_list[2][np.argmax(mention_seq_preds(seq_preds=mention_code_preds, 
                                                                    mention_strat=mention_strat))]
                
            else:
                label_pred = mention_seq_preds(seq_preds=mention_code_preds, mention_strat='first')
                if strategy.split('-')[-1] != "crf":
                    label_pred = np.argmax(label_pred)
                code_pred = code_lab_decoder_list[2][label_pred]
                    
            # Add NORM annotation
            res[-1]['code'] = code_pred
                
            left = right # next sub-token different from "I", or len(seq_preds[0]) (out of bounds)
        else:
            left += 1
    
    
    return res


def norm_iob2_diag_single_proc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_proc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Diagnosis
    seq_preds_diag = [seq_preds[0], seq_preds[1]]
    labels_diag = norm_iob2_diag_single_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_diag, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=[code_lab_decoder_list[0]], 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Procedures
    seq_preds_proc = [seq_preds[2], seq_preds[3]]
    labels_proc = norm_iob2_proc_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_proc, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=[code_lab_decoder_list[1]], 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_diag + labels_proc


def norm_iob2_diag_single_only_proc_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                          strategy='word-first', subtask='norm-iob_diag_single_only_proc', mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Diagnosis
    seq_preds_diag = [seq_preds[0], seq_preds[1]]
    labels_diag = norm_iob2_diag_single_only_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_diag, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=[code_lab_decoder_list[0]], 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Procedures
    seq_preds_proc = [seq_preds[0], seq_preds[2]]
    labels_proc = norm_iob2_proc_only_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_proc, seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=[code_lab_decoder_list[1]], 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_diag + labels_proc


def norm_iob2_diag_single_only_proc_cont_disc_d_extract_seq_preds(doc_id, seq_preds, seq_start_end, code_lab_decoder_list, 
                                        strategy='word-first', subtask='norm-iob_diag_single_only_proc_cont_disc_d', 
                                        mention_strat='max'):
    """
    NOT ADAPTED TO CRF (NEITHER TO -H INDICATOR).
    seq_preds: it is assumed to be a list containing 5 lists (IOB (diag) + Diag-code-pre + Diag-code-suf + IOB (proc) + Proc-Code)
    
    code_lab_decoder_list: [diag_code_pre_lab_decoder, diag_code_suf_lab_decoder]
    
    subtask: norm-iob_diag_proc, norm-iob_code_suf-mention
    """
    
    # Diagnosis
    seq_preds_diag = [seq_preds[0], seq_preds[1]]
    labels_diag = norm_iob2_diag_single_only_cont_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_diag, 
                                          seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=[code_lab_decoder_list[0]], 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    # Procedures
    seq_preds_proc = [seq_preds[0], seq_preds[2]]
    labels_proc = norm_iob2_proc_only_cont_disc_d_extract_seq_preds(doc_id=doc_id, seq_preds=seq_preds_proc, 
                                          seq_start_end=seq_start_end, 
                                          code_lab_decoder_list=[code_lab_decoder_list[1]], 
                                          strategy=strategy, subtask=subtask, mention_strat=mention_strat)
    
    return labels_diag + labels_proc


def seq_ner_preds_brat_format(doc_list, fragments, arr_start_end, arr_word_id, arr_preds, strategy="word-first", 
                              crf_mask_seq_len=None,
                              type_tokenizer='transformers'):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-all-crf", "word-first-crf", "word-first-x-crf".
    """
    
    arr_doc_seq_start_end = []
    # Multiple labels
    arr_doc_seq_preds = [[] for lab_i in range(len(arr_preds))]
    i = 0
    for d in range(len(doc_list)):
        n_frag = fragments[d]
        # Extract subtoken-level arrays for each document (joining adjacent fragments)
        doc_tok_start_end = [ss_pair for frag in arr_start_end[i:i+n_frag] for ss_pair in frag]
        doc_tok_word_id = [w_id for frag in arr_word_id[i:i+n_frag] for w_id in frag]
        assert len(doc_tok_start_end) == len(doc_tok_word_id)
        
        if strategy.split('-')[0] == "word":
            # Extract subtoken-level predictions, ignoring special tokens (CLS, SEQ, PAD)
            # (CLS, SEP only for transformers, not for fasttext)
            # Multiple labels
            doc_tok_preds = []
            # here guille now: FastText
            inf = sup = 1
            if type_tokenizer == 'fasttext':
                inf = sup = 0
            for lab_i in range(len(arr_preds)):
                if strategy.split('-')[-1] != "crf":
                    # doc_tok_preds[lab_i] shape: n_tok (per doc) x n_labels (3 for NER, 2*n_codes + 1 for NER-Norm)
                    doc_tok_preds.append(np.array([preds for j in range(i, i+n_frag) \
                        for preds in arr_preds[lab_i][j][inf:len(arr_start_end[j])+sup]]))
                    assert doc_tok_preds[-1].shape[0] == len(doc_tok_start_end)
                else:
                    # arr_preds does not contain predictions made on "ignored" tokens 
                    # (either special tokens or secondary tokens for "word-first" strategy),
                    # but it contains predictions for right-padding-CRF tokens.
                    # crf_mask_seq_len is expected not to be None, indicating the
                    # number of "not right-padded" tokens in each fragment;
                    # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
                    # for all outputs (this may be changed when implementing "mention-first" approach).
                    # doc_tok_preds shape: n_tok (per doc) x 1 (int label)
                    doc_tok_preds.append(np.array([preds for j in range(i, i+n_frag) \
                        for preds in arr_preds[lab_i][j][:crf_mask_seq_len[j]]]))
                    if strategy.split('-')[-2] == "all":
                        assert doc_tok_preds[-1].shape[0] == len(doc_tok_start_end)
                    elif strategy.split('-')[-2] == "first":
                        assert doc_tok_preds[-1].shape[0] == (doc_tok_word_id[-1] + 1)
                    elif '-'.join(strategy.split('-')[-3:-1]) == "first-x":
                        assert doc_tok_preds[-1].shape[0] == len(doc_tok_start_end)
                    else:
                        raise Exception('Strategy not implemented!')
        
            # Convert subtoken-level arrays to word-level
            doc_word_seq_preds, doc_word_seq_start_end = word_seq_preds(tok_seq_word_id=doc_tok_word_id, 
                                        tok_seq_preds=doc_tok_preds, tok_seq_start_end=doc_tok_start_end, strategy=strategy)
            assert len(doc_word_seq_start_end) == (doc_tok_word_id[-1] + 1)
            
            # Multiple labels
            for lab_i in range(len(arr_preds)):
                assert len(doc_word_seq_preds[lab_i]) == len(doc_word_seq_start_end)
                arr_doc_seq_preds[lab_i].append(doc_word_seq_preds[lab_i]) # final shape: n_doc x n_words (per doc) x [n_labels (e.g. 3 for NER) or 1 (CRF)]
            arr_doc_seq_start_end.append(doc_word_seq_start_end) # final shape: n_doc x n_words (per doc) x 2 (start-end pair)
        
        else:
            raise Exception('Strategy not implemented!')

        i += n_frag

    return arr_doc_seq_preds, arr_doc_seq_start_end


def extract_seq_preds(arr_doc_seq_preds, arr_doc_seq_start_end, doc_list, lab_decoder_list, df_text, text_col, 
                      subtask='ner', strategy="word-all", mention_strat='max', code_sep='/',
                      codes_pre_suf_mask=None, codes_pre_o_mask=None):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-all-crf", "word-first-crf".
    """
    
    n_output = len(arr_doc_seq_preds)
    ann_res = []
    for d in range(len(doc_list)):
        doc = doc_list[d]
        # Multiple labels
        doc_seq_preds = []
        for lab_i in range(n_output):
            doc_seq_preds.append(arr_doc_seq_preds[lab_i][d]) # shape: n_words (per doc) x [n_labels (e.g. 3 for NER) or 1 (CRF)]
        doc_seq_start_end = arr_doc_seq_start_end[d] # shape: n_words (per doc) x 2 (start-end pair)
        # here guille now (probar): depending on the subtask
        doc_seq_lab = []
        if len(subtask.split('-')) == 1 or subtask.split('-')[1] == 'mention': # ner, norm, norm-mention
            # Single label (custom)
            if subtask == "ner":
                # IOB label
                if strategy.split('-')[-1] != "crf":
                    doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                else:
                    doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])

            elif subtask.split('-')[0] == "norm":
                # IOB-Code label
                doc_seq_lab.append([])
                for i in range(len(doc_seq_preds[0])):
                    if strategy.split('-')[-1] != "crf":
                        max_j = np.argmax(doc_seq_preds[0][i])
                        # append both the predicted label and its probability
                        doc_seq_lab[0].append((lab_decoder_list[0][max_j], doc_seq_preds[0][i][max_j])) # final shape: n_words (per doc) x 2
                    else:
                        doc_seq_lab[0].append((lab_decoder_list[0][doc_seq_preds[0][i]],)) # final shape: n_words (per doc) x 1
                        
            else:
                raise Exception("Subtask not implemented!")
            
            ann_res.extend(ner_iob2_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end, 
                                            df_text=df_text, text_col=text_col, subtask=subtask, strategy=strategy))
        
        elif subtask.split('-')[1] == "iob_code":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code predictions
            # Mention strategy
            doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels (1 for CRF)
            ann_res.extend(norm_iob2_code_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            df_text=df_text, text_col=text_col, code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_code_mask":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code predictions
            # Mention strategy
            doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels (1 for CRF)
            ann_res.extend(norm_iob2_code_mask_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            df_text=df_text, text_col=text_col, code_lab_decoder_list=lab_decoder_list[1:],
                                            codes_o_mask=codes_pre_o_mask,
                                            mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_code_suf":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(1, n_output):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_code_suffix_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            df_text=df_text, text_col=text_col, code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat, 
                                            code_sep=code_sep))
        
        elif subtask.split('-')[1] == "iob_code_suf_mask":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(1, n_output):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_code_suffix_mask_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            df_text=df_text, text_col=text_col, code_lab_decoder_list=lab_decoder_list[1:],
                                            codes_pre_suf_mask=codes_pre_suf_mask, codes_pre_o_mask=codes_pre_o_mask,
                                            mention_strat=mention_strat, code_sep=code_sep))
        
        # here guille now: avoid IOB
        elif subtask.split('-')[1] == "code":
            # Code label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_code_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            df_text=df_text, text_col=text_col))
            
        elif subtask.split('-')[1] == "iob_diag":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(1, 3):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ann_res.extend(norm_iob2_diag_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_disc" or subtask.split('-')[1] == "iob_diag_cont_disc_c":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(1, 3):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ann_res.extend(norm_iob2_diag_disc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_cont_disc":
            ## Diagnosis
            # IOB label (continuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # IOB label (discontinuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[1], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[1]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(2, 4):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ann_res.extend(norm_iob2_diag_cont_disc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_cont_disc":
            ## Diagnosis
            # IOB label (continuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # IOB label (discontinuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[1], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[1]])
                
            ann_res.extend(norm_iob2_cont_disc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end, 
                df_text=df_text, text_col=text_col))
            
        elif subtask.split('-')[1] == "iob_cont_disc_code_suf_mask":
            # IOB label (continuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # IOB label (discontinuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[1], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[1]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(2, 4):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ann_res.extend(norm_iob2_cont_disc_code_suffix_mask_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end, df_text=df_text, text_col=text_col,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            codes_pre_suf_mask=codes_pre_suf_mask, 
                                            codes_pre_o_mask=codes_pre_o_mask,
                                            mention_strat=mention_strat, code_sep=code_sep))
            
        elif subtask.split('-')[1] == "iob_cont_disc_d":
            ## Diagnosis
            # IOB label (continuous-discontinuous)
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                
            ann_res.extend(norm_iob2_cont_disc_d_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end, 
                df_text=df_text, text_col=text_col))
            
        elif subtask.split('-')[1] == "iob_diag_single":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                # Code and (optionally) h-indicator predictions
                doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_iob2_diag_single_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_single_disc_d":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                # Code and (optionally) h-indicator predictions
                doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_iob2_diag_single_disc_d_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_single_cont_disc_d":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                # Code and (optionally) h-indicator predictions
                doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_iob2_diag_single_cont_disc_d_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_proc":
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                # Code and (optionally) h-indicator predictions
                doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_iob2_proc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_proc_disc" or subtask.split('-')[1] == "iob_proc_cont_disc_c":
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_proc_disc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_proc_disc_d":
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                # Code and (optionally) h-indicator predictions
                doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_iob2_proc_disc_d_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_proc_cont_disc_d":
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
                # Code and (optionally) h-indicator predictions
                doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
                        
            ann_res.extend(norm_iob2_proc_cont_disc_d_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_proc_code_pre_suf":
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code-prefix, code-intermediate and code-suffix
            for lab_i in range(1, 4):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ann_res.extend(norm_iob2_proc_code_pre_suffix_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_proc":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code-prefix and code-suffix and (optionally) h-indicator predictions
            for lab_i in range(1, 3):
                # Mention strategy
                doc_seq_lab.append(doc_seq_preds[lab_i]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[3], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[3]])
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[4]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_diag_proc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_single_proc":
            ## Diagnosis
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ## Procedure
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[2], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[2]])
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[3]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_diag_single_proc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_single_only_proc":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            ## Diagnosis
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ## Procedure
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[2]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_diag_single_only_proc_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        elif subtask.split('-')[1] == "iob_diag_single_only_proc_cont_disc_d":
            # IOB label
            if strategy.split('-')[-1] != "crf":
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in np.argmax(doc_seq_preds[0], axis=1)])
            else:
                doc_seq_lab.append([(lab_decoder_list[0][pred],) for pred in doc_seq_preds[0]])
            ## Diagnosis
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[1]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                
            ## Procedure
            # Code and (optionally) h-indicator predictions
            doc_seq_lab.append(doc_seq_preds[2]) # final shape: n_words (per doc) x n_labels (1 for CRF)
                        
            ann_res.extend(norm_iob2_diag_single_only_proc_cont_disc_d_extract_seq_preds(doc_id=doc, seq_preds=doc_seq_lab, 
                                            seq_start_end=doc_seq_start_end,
                                            code_lab_decoder_list=lab_decoder_list[1:],
                                            strategy=strategy, subtask=subtask, mention_strat=mention_strat))
            
        else:
            raise Exception("Subtask not implemented!")
    
    return ann_res


def ner_preds_brat_format(doc_list, fragments, preds, start_end, word_id, lab_decoder_list, df_text, text_col, 
                          strategy="word-all", subtask="ner", crf_mask_seq_len=None, mention_strat='max',
                          type_tokenizer='transformers', code_sep='/',
                          codes_pre_suf_mask=None, codes_pre_o_mask=None):
    """
    Implemented strategies: "word-first", "word-max", "word-prod", "word-all-crf", "word-first-crf".
    """
    # Post-process the subtoken predictions for each document, obtaining word-level predictions
    arr_doc_seq_preds, arr_doc_seq_start_end = seq_ner_preds_brat_format(doc_list=doc_list, fragments=fragments, 
                            arr_start_end=start_end, arr_word_id=word_id, arr_preds=preds, strategy=strategy,
                            crf_mask_seq_len=crf_mask_seq_len,
                            type_tokenizer=type_tokenizer)
    
    # Extract the predicted mentions from the word-level predictions
    ann_res = extract_seq_preds(arr_doc_seq_preds=arr_doc_seq_preds, arr_doc_seq_start_end=arr_doc_seq_start_end, 
                                doc_list=doc_list, lab_decoder_list=lab_decoder_list, df_text=df_text, 
                                text_col=text_col, subtask=subtask, strategy=strategy, mention_strat=mention_strat,
                                code_sep=code_sep,
                                codes_pre_suf_mask=codes_pre_suf_mask, codes_pre_o_mask=codes_pre_o_mask)
            
    return pd.DataFrame(ann_res)


def ens_ner_preds_brat_format(doc_list, ens_doc_word_preds, ens_doc_word_start_end, lab_decoder_list, df_text, text_col, ens_eval_strat='max',
                              strategy="word-all", subtask="ner", mention_strat='max', norm_words=False):
    """
    Implemented strategies: "max", "prod", "sum".
    NOT ADAPTED TO CRF.
    """
    
    n_output = len(ens_doc_word_preds[0]) # the shapes of predictions from all models in the ensemble are assumed to be the same
    # Sanity check: same word start-end arrays obatined from different models
    doc_word_start_end = ens_doc_word_start_end[0]
    for i in range(len(doc_list)):
        aux_san_arr = np.array(doc_word_start_end[i])
        for j in range(1, len(ens_doc_word_start_end)):
            comp_arr = np.array(ens_doc_word_start_end[j][i])
            assert np.array_equal(aux_san_arr, comp_arr)
    
    # Merge predictions made by all models
    arr_doc_preds = [] # final shape: n_out x n_docs x n_words (per_doc) x n_labels
    for lab_i in range(n_output):
        arr_doc_preds.append([])
        for d in range(len(doc_list)):
            if norm_words:
                arr_ens_word_preds = np.array([normalize(np.array(word_preds[lab_i][d]), norm='l1', axis=1) for word_preds in ens_doc_word_preds]) 
            else:
                arr_ens_word_preds = np.array([word_preds[lab_i][d] for word_preds in ens_doc_word_preds]) 
            # shape: n_ens x n_words (per doc) x n_labels (e.g. 3 for NER) 
            if ens_eval_strat == "max":
                arr_word_preds = np.max(arr_ens_word_preds, axis=0)
            elif ens_eval_strat == "prod":
                arr_word_preds = np.prod(arr_ens_word_preds, axis=0)
            elif ens_eval_strat == "sum":
                arr_word_preds = np.sum(arr_ens_word_preds, axis=0)
            else:
                raise Exception('Ensemble evaluation strategy not implemented!')
                
            arr_doc_preds[lab_i].append(arr_word_preds)

    # Extract the annotations from the merged predictions
    ann_res = extract_seq_preds(arr_doc_seq_preds=arr_doc_preds, arr_doc_seq_start_end=doc_word_start_end, doc_list=doc_list, 
                      lab_decoder_list=lab_decoder_list, df_text=df_text, text_col=text_col, 
                      subtask=subtask, strategy=strategy, mention_strat=mention_strat)
    
    return pd.DataFrame(ann_res)


import shutil

def write_ner_ann(df_pred_ann, out_path, ann_label="MORFOLOGIA_NEOPLASIA", subtask='ner'):
    """
    df_pred_ann: pd.DataFrame with the format defined in extract_seq_preds_iob2/extract_seq_preds_iob2_code function.
    
    Write a set of NER-annotations from different documents in BRAT format.
    """
    
    # Create a new output directory
    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.mkdir(out_path)
    
    for doc in sorted(set(df_pred_ann['clinical_case'])):
        doc_pred_ann = df_pred_ann[df_pred_ann['clinical_case'] == doc]
        with open(out_path + doc + ".ann", "w") as out_file:
            i = 1
            for index, row in doc_pred_ann.iterrows():
                start = row['start']
                end = row['end']
                out_file.write("T" + str(i) + "\t" + ann_label + " " + str(start) + " " + 
                               str(end) + "\t" + row['text'][start:end].replace("\n", " ") + "\n")
                if subtask == 'norm':
                    code = row['code_pred']
                    out_file.write("#" + str(i) + "\t" + "AnnotatorNotes " + "T" + str(i) + "\t" + code + "\n")
                
                i += 1
    return


# Cantemist-NER/NORM

import ann_parsing

def format_ner_gs(gs_path, subtask='ner'):
    """
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_ner_norm.py
    """
    
    if subtask=='norm':
        gs = ann_parsing.main(gs_path, ['MORFOLOGIA_NEOPLASIA'], with_notes=True)
        
        if gs.shape[0] == 0:
            raise Exception('There are not parsed Gold Standard annotations')
        
        gs.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 'code_gs',
                      'start_pos_gs', 'end_pos_gs']
        
    elif subtask=='ner':
        gs = ann_parsing.main(gs_path, ['MORFOLOGIA_NEOPLASIA'], with_notes=False)
        
        if gs.shape[0] == 0:
            raise Exception('There are not parsed Gold Standard annotations')
        
        gs.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 
                      'start_pos_gs', 'end_pos_gs']
        
    else:
        raise Exception('Error! Subtask name not properly set up')
    
    return gs


def format_ner_pred(gs_path, pred_path, subtask='ner'):
    """
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_ner_norm.py
    """
    
    # Get ANN files in Gold Standard
    ann_list_gs = list(filter(lambda x: x[-4:] == '.ann', os.listdir(gs_path)))
    
    if subtask=='norm':
        pred = ann_parsing.main(pred_path, ['MORFOLOGIA_NEOPLASIA','MORFOLOGIA-NEOPLASIA'], with_notes=True)
        
        if pred.shape[0] == 0:
            raise Exception('There are not parsed predicted annotations')
        
        pred.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 'code_pred',
                      'start_pos_pred', 'end_pos_pred']
    elif subtask=='ner':
        pred = ann_parsing.main(pred_path, ['MORFOLOGIA_NEOPLASIA','MORFOLOGIA-NEOPLASIA'], with_notes=False)
        
        if pred.shape[0] == 0:
            raise Exception('There are not parsed predicted annotations')
        
        pred.columns = ['clinical_case', 'mark', 'label', 'offset', 'span',
                      'start_pos_pred', 'end_pos_pred']
    else:
        raise Exception('Error! Subtask name not properly set up')

    # Remove predictions for files not in Gold Standard
    pred_gs_subset = pred.loc[pred['clinical_case'].isin(ann_list_gs),:]
    
    return pred_gs_subset


def format_ner_pred_df(gs_path, df_preds, subtask):
    """
    df_preds: same format as returned by ner_preds_brat_format function.
    
    return: pd.DataFrame with the columns expected in calculate_ner_metrics function.
    """
    
    # Get ANN files in Gold Standard
    ann_list_gs = list(filter(lambda x: x[-4:] == '.ann', os.listdir(gs_path)))
    
    df_preds_res = df_preds.copy()
    
    # Add .ann suffix
    df_preds_res['clinical_case'] = df_preds_res['clinical_case'].apply(lambda x: x + '.ann')
    
    # Remove predictions for files not in Gold Standard
    df_pred_gs_subset = df_preds_res.loc[df_preds_res['clinical_case'].isin(ann_list_gs),:]
    
    df_pred_gs_subset['offset'] = df_pred_gs_subset.apply(lambda x: str(x['start']) + ' ' + str(x['end']), axis=1)
    
    # here guille now (probar): depending on the subtask
    if subtask == 'ner':
        return df_pred_gs_subset[['clinical_case', 'offset', 'start', 'end']]
    elif subtask == 'norm':
        return df_pred_gs_subset[['clinical_case', 'offset', 'start', 'end', 'code_pred']]
    else:
        raise Exception('Error! Subtask name not properly set up')


import warnings
def calculate_ner_metrics(gs, pred, subtask='ner'):
    """
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_ner_norm.py
    """
    
    # Predicted Positives:
    Pred_Pos = pred.drop_duplicates(subset=['clinical_case', "offset"]).shape[0]
    
    # Gold Standard Positives:
    GS_Pos = gs.drop_duplicates(subset=['clinical_case', "offset"]).shape[0]
    
    # Eliminate predictions not in GS (prediction needs to be in same clinical
    # case and to have the exact same offset to be considered valid!!!!)
    df_sel = pd.merge(pred, gs, 
                      how="right",
                      on=["clinical_case", "offset"])
    
    if subtask=='norm':
        # Check if codes are equal
        df_sel["is_valid"] = \
            df_sel.apply(lambda x: (x["code_gs"] == x["code_pred"]), axis=1)
    elif subtask=='ner':
        is_valid = df_sel.apply(lambda x: x.isnull().any()==False, axis=1)
        df_sel = df_sel.assign(is_valid=is_valid.values)
    else:
        raise Exception('Error! Subtask name not properly set up')
    

    # There are two annotations with two valid codes. Any of the two codes is considered as valid
    if subtask=='norm':
        df_sel = several_codes_one_annot(df_sel)
        
    # True Positives:
    TP = df_sel[df_sel["is_valid"] == True].shape[0]
    
    # Calculate Final Metrics:
    P = TP / Pred_Pos
    R = TP / GS_Pos
    if (P+R) == 0:
        F1 = 0
    else:
        F1 = (2 * P * R) / (P + R)
    
    
    if (any([F1, P, R]) > 1):
        warnings.warn('Metric greater than 1! You have encountered an undetected bug, please, contact antonio.miranda@bsc.es!')
                                            
    return round(P, 4), round(R, 4), round(F1, 4)


def several_codes_one_annot(df_sel):
    
    # If any of the two valid codes is predicted, give both as good
    if any(df_sel.loc[(df_sel['clinical_case']=='cc_onco838.ann') & 
                  (df_sel['offset'] == '2509 2534')]['is_valid']):
        df_sel.loc[(df_sel['clinical_case']=='cc_onco838.ann') &
                   (df_sel['offset'] == '2509 2534'),'is_valid'] = True
            
    if any(df_sel.loc[(df_sel['clinical_case']=='cc_onco1057.ann') & 
                      (df_sel['offset'] == '2791 2831')]['is_valid']):
        df_sel.loc[(df_sel['clinical_case']=='cc_onco1057.ann') &
                   (df_sel['offset'] == '2791 2831'),'is_valid'] = True
        
    # Remove one of the entries where there are two valid codes
    df_sel.drop(df_sel.loc[(df_sel['clinical_case']=='cc_onco838.ann') &
                    (df_sel['offset'] == '2509 2534') & 
                    (df_sel['code_gs']=='8441/0')].index, inplace=True)
    
    df_sel.drop(df_sel.loc[(df_sel['clinical_case']=='cc_onco1057.ann') &
            (df_sel['offset'] == '2791 2831') & 
            (df_sel['code_gs']=='8803/3')].index, inplace=True)
        
    return df_sel


# CodiEsp-X

def format_codiesp_x_gs(filepath, gs_headers=["clinical_case","label_gs", "code", "ref", "pos_gs"]):
    '''
    Code copied from: https://github.com/TeMU-BSC/codiesp-evaluation-script/blob/master/codiespX_evaluation.py#L17
    
    DESCRIPTION: Load Gold Standard table
    
    INPUT: 
        filepath: str
            route to TSV file with Gold Standard.
    
    OUTPUT: 
        gs_data: pandas dataframe
            with columns:['clinical_case','label_gs','code','ref','pos_gs','start_pos_gs','end_pos_gs']
    '''
    # Check GS format:
    check = pd.read_csv(filepath, sep='\t', header = None, nrows=1)
    if check.shape[1] != 5:
        raise ImportError('The GS file does not have 5 columns. Then, it was not imported')
    
    gs_data = pd.read_csv(filepath, sep="\t", names=gs_headers)
    gs_data.code = gs_data.code.str.lower()
    
    gs_data['start_pos_gs'], gs_data['aux_end_gs'] = gs_data['pos_gs'].str.split(' ', 1).str
    
    # In case there are discontinuous annotations, just keep the first and 
    # last offset and consider everything in between as part of the reference.
    gs_data["end_pos_gs"] = gs_data['aux_end_gs'].apply(lambda x: x.split(' ')[-1]) 
    gs_data = gs_data.drop(["aux_end_gs"], axis=1)
    
    gs_data['start_pos_gs'] = gs_data['start_pos_gs'].astype("int")
    gs_data['end_pos_gs'] = gs_data['end_pos_gs'].astype("int")
    
    return gs_data


def format_codiesp_x_pred(filepath, valid_codes, 
             run_headers=["clinical_case","pos_pred","label_pred", "code"]):
    '''
    Code copied from: https://github.com/TeMU-BSC/codiesp-evaluation-script/blob/master/codiespX_evaluation.py#L49
    
    DESCRIPTION: Load Predictions table
        
    INPUT: 
        filepath: str
            route to TSV file with Predictions.
        valid_codes: set
            set of valid codes of this subtask
    
    OUTPUT: 
        run_data: pandas dataframe
            with columns:[clinical_case, label_pred, code, start_pos_pred, end_pos_pred]
    '''
    # Check predictions format
    check = pd.read_csv(filepath, sep='\t', header = None, nrows=1)
    if check.shape[1] != 4:
        raise ImportError('The predictions file does not have 4 columns. Then, it was not imported')
        
    run_data = pd.read_csv(filepath, sep="\t", names=run_headers)
    run_data.code = run_data.code.str.lower()
    
    # Check predictions types
    if all(run_data.dtypes == pd.Series({'clinical_case': object,
                                         'pos_pred': object,
                                         'label_pred': object,
                                         'code': object})) == False:
        warnings.warn('The predictions file has wrong types')
        
    # Check if predictions file is empty
    if run_data.shape[0] == 0:
        is_empty = 1
        warnings.warn('The predictions file is empty')
    else:
        is_empty = 0
        
    # Remove codes predicted but not in list of valid codes
    run_data = run_data[run_data['code'].isin(valid_codes)]
    if (run_data.shape[0] == 0) & (is_empty == 0):
        warnings.warn('None of the predicted codes are considered valid codes')
        
    # Split position into starting and end positions
    run_data['start_pos_pred'], run_data['end_pos_pred'] = run_data['pos_pred'].str.split(' ', 1).str
    run_data['start_pos_pred'] = run_data['start_pos_pred'].astype("int")
    run_data['end_pos_pred'] = run_data['end_pos_pred'].astype("int")
    run_data = run_data.drop("pos_pred", axis=1)
    
    return run_data


def format_codiesp_x_pred_df(df_run, valid_codes):
    '''
    Code adapted from: https://github.com/TeMU-BSC/codiesp-evaluation-script/blob/master/codiespX_evaluation.py#L49
    
    DESCRIPTION: Load Predictions dataframe
        
    INPUT: 
        df_pred: pandas dataframe
            table with Predictions with columns: ["clinical_case", "pos_pred", "label_pred", "code"]
            (format returned by norm_iob2_diag_proc_extract_seq_preds function)
        valid_codes: set
            set of valid codes of this subtask
    
    OUTPUT: 
        run_data: pandas dataframe
            with columns:[clinical_case, label_pred, code, start_pos_pred, end_pos_pred]
    '''
    df_pred = df_run.copy()
    df_pred.code = df_pred.code.str.lower()
    
    # Check predictions types
    if all(df_pred.dtypes == pd.Series({'clinical_case': object,
                                         'pos_pred': object,
                                         'label_pred': object,
                                         'code': object})) == False:
        warnings.warn('The predictions file has wrong types')
        
    # Check if predictions file is empty
    if df_pred.shape[0] == 0:
        is_empty = 1
        warnings.warn('The predictions file is empty')
    else:
        is_empty = 0
        
    # Remove codes predicted but not in list of valid codes
    df_pred = df_pred[df_pred['code'].isin(valid_codes)]
    if (df_pred.shape[0] == 0) & (is_empty == 0):
        warnings.warn('None of the predicted codes are considered valid codes')
        
    # Split position into starting and end positions
    df_pred['start_pos_pred'], df_pred['end_pos_pred'] = df_pred['pos_pred'].str.split(' ', 1).str
    df_pred['start_pos_pred'] = df_pred['start_pos_pred'].astype("int")
    df_pred['end_pos_pred'] = df_pred['end_pos_pred'].astype("int")
    df_pred = df_pred.drop("pos_pred", axis=1)
    
    return df_pred


def calculate_codiesp_x_metrics(df_gs, df_pred, tol = 10):
    '''
    Code adapted from: https://github.com/TeMU-BSC/codiesp-evaluation-script/blob/master/codiespX_evaluation.py#L122
    
    DESCRIPTION: Calculate task X metrics:
    
    In case a code has several references, just acknowledging one is enough.
    In case of discontinuous references, the reference is considered to 
    start at the start position of the first part of the reference and to 
    end at the final position of the last part of the reference.
    
    INPUT: 
        df_gs: pandas dataframe
            with the Gold Standard. Columns are those output by the function read_gs.
        df_pred: pandas dataframe
            with the predictions. Columns are those output by the function read_run.
    
    OUTPUT: 
        P: float
            Micro-average precision
        R: float
            Micro-average recall
        F1: float
            Micro-average F-score
    '''
    
    # Predicted Positives:
    Pred_Pos = df_pred.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    
    # Gold Standard Positives:
    GS_Pos = df_gs.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    
    # Eliminate predictions not in GS
    df_sel = pd.merge(df_pred, df_gs, 
                      how="right",
                      on=["clinical_case", "code"])
    
    # Check if GS reference is inside predicted interval
    df_sel["start_space"] = (df_sel["start_pos_gs"] - df_sel["start_pos_pred"])
    df_sel["end_space"] = (df_sel["end_pos_pred"] - df_sel["end_pos_gs"])
    df_sel["is_valid"] = df_sel.apply(lambda x: ((x["start_space"] <= tol) & 
                                                 (x["start_space"] >= 0) &
                                                 (x["end_space"] <= tol) &
                                                 (x["end_space"] >= 0)), axis=1)
    
    # Remove duplicates that appear in case there are codes with several references in GS
    # In case just one of the references is predicted, mark the code as True
    df_final = df_sel.sort_values(by="is_valid",
                                  ascending=True).drop_duplicates(
                                      subset=["clinical_case", "code"],
                                      keep="last")

    # True Positives:
    TP = df_final[df_final["is_valid"] == True].shape[0]
    
    # Calculate Final Metrics:
    P = TP / Pred_Pos
    R = TP / GS_Pos
    if (P+R) == 0:
        F1 = 0
        warnings.warn('Global F1 score automatically set to zero to avoid division by zero')
        return P, R, F1
    F1 = (2 * P * R) / (P + R)
                                            
    return round(P, 4), round(R, 4), round(F1, 4)


def calculate_codiesp_ner_metrics(df_gs, df_pred):
    '''
    Evaluate NER performance on the CodiEsp-X task, computing P, R and F1 metrics.
    '''
    
    # Rename columns
    df_gs_class = df_gs.copy()
    df_gs_class = df_gs_class.rename(columns={'start_pos_gs': 'start', 'end_pos_gs': 'end'})
    
    df_pred_class = df_pred.copy()
    df_pred_class = df_pred_class.rename(columns={'start_pos_pred': 'start', 'end_pos_pred': 'end'})
    
    # Predicted Positives:
    df_pred_class = df_pred_class.drop_duplicates(subset=["clinical_case", "start", "end"])
    Pred_Pos = df_pred_class.shape[0]

    # Gold Standard Positives:
    df_gs_class = df_gs_class.drop_duplicates(subset=["clinical_case", "start", "end"])
    GS_Pos = df_gs_class.shape[0]

    # Eliminate predictions not in GS (prediction needs to be in same clinical
    # case and to have the exact same offset to be considered valid!!!!)
    df_sel = pd.merge(df_pred_class, df_gs_class, 
                      how="right",
                      on=["clinical_case", "start", "end"])

    # Check if offsets are equal
    is_valid = df_sel.apply(lambda x: x.isnull().any()==False, axis=1)
    df_sel = df_sel.assign(is_valid=is_valid.values)
    

    # True Positives:
    TP = df_sel[df_sel["is_valid"] == True].shape[0]

    # Calculate Final Metrics:
    P = TP / Pred_Pos if Pred_Pos > 0 else 0
    R = TP / GS_Pos
    if (P+R) == 0:
        F1 = 0
    else:
        F1 = (2 * P * R) / (P + R)
                                            
    return round(P, 4), round(R, 4), round(F1, 4)


# Galén de-identification
def calculate_anon_metrics(gs, pred):
    """
    gs: df with columns "clinical_case, start, end, code_gs"
    pred: df with columns "clinical_case, start, end, code_pred"
    """
    
    res_dict = {}
    for code_class in sorted(set(gs["code_gs"])):
        gs_class = gs[gs["code_gs"] == code_class].copy()
        pred_class = pred[pred["code_pred"] == code_class].copy()
    
        # Predicted Positives:
        Pred_Pos = pred_class.drop_duplicates(subset=["clinical_case", "start", "end"]).shape[0]

        # Gold Standard Positives:
        GS_Pos = gs_class.drop_duplicates(subset=["clinical_case", "start", "end"]).shape[0]

        # Eliminate predictions not in GS (prediction needs to be in same clinical
        # case and to have the exact same offset to be considered valid!!!!)
        df_sel = pd.merge(pred_class, gs_class, 
                          how="right",
                          on=["clinical_case", "start", "end"])

        # Check if codes are equal
        df_sel["is_valid"] = \
                df_sel.apply(lambda x: (x["code_gs"] == x["code_pred"]), axis=1)

        # True Positives:
        TP = df_sel[df_sel["is_valid"] == True].shape[0]

        # Calculate Final Metrics:
        P = TP / Pred_Pos if Pred_Pos > 0 else 0
        R = TP / GS_Pos
        if (P+R) == 0:
            F1 = 0
        else:
            F1 = (2 * P * R) / (P + R)

        if (any([F1, P, R]) > 1):
            warnings.warn('Metric greater than 1! You have encountered an undetected bug, please, contact antonio.miranda@bsc.es!')
        
        # Save results
        res_dict[code_class] = {'P': P, 'R': R, 'F1': F1}
    
    res_df = pd.DataFrame(res_dict).transpose()
    
    # Also return macro-averaged metrics                                        
    return res_df, round(np.mean(res_df['P']), 3), round(np.mean(res_df['R']), 3), round(np.mean(res_df['F1']), 3)



## Text classification evaluation

def max_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list):
    """
    Convert fragment-level to doc-level predictions, usin max criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(y_frag_pred[i_frag:i_frag+n_frag].max(axis=0))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list)


def prob_codiesp_prediction_format(y_pred, label_encoder_classes, doc_list):
    """
    Given a matrix of predicted probabilities (m_docs x n_codes), for each document, this procedure stores all the
    codes sorted according to their probability values in descending order. Finally, predictions are saved in a dataframe
    defined following CodiEsp submission format (see https://temu.bsc.es/codiesp/index.php/2020/02/06/submission/).
    """
    
    # Sanity check
    assert y_pred.shape[0] == len(doc_list)
    
    pred_doc, pred_code, pred_rank = [], [], []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        # Codes are sorted according to their probability values in descending order
        codes_sort = [label_encoder_classes[j] for j in np.argsort(pred)[::-1]]
        pred_code += codes_sort
        pred_doc += [doc_list[i]]*len(codes_sort)
        # For compatibility with format_predictions function
        pred_rank += list(range(1, len(codes_sort)+1))
            
    # Save predictions in CodiEsp submission format
    return pd.DataFrame({"doc_id": pred_doc, "code": pred_code, "rank": pred_rank})


from trectools import TrecQrel, TrecRun, TrecEval

def compute_map(valid_codes, pred, gs_out_path=None, pred_out_path=None):
    """
    Custom function to compute MAP evaluation metric. 
    Code adapted from https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py
    """
    
    # Input args default values
    if gs_out_path is None: gs_out_path = './intermediate_gs_file.txt'
    if pred_out_path is None: pred_out_path = './intermediate_pred_file.txt'
    
    ###### 2. Format predictions as TrecRun format: ######
    format_predictions(pred, pred_out_path, valid_codes)
    
    
    ###### 3. Calculate MAP ######
    # Load GS from qrel file
    qrels = TrecQrel(gs_out_path)

    # Load pred from run file
    run = TrecRun(pred_out_path)

    # Calculate MAP
    te = TrecEval(run, qrels)
    MAP = te.get_map(trec_eval=False) # With this option False, rank order is taken from the given document order
    
    ###### 4. Return results ######
    return MAP


# Code adapted from: https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py
    
def format_predictions(pred, output_path, valid_codes, 
                       system_name = 'xx', pred_names = ['query','docid', 'rank']):
    '''
    DESCRIPTION: Add extra columns to Predictions table to match 
    trectools library standards.
        
    INPUT: 
        pred: pd.DataFrame
                Predictions.
        output_path: str
            route to TSV where intermediate file is stored
        valid_codes: set
            set of valid codes of this subtask

    OUTPUT: 
        stores TSV files with columns  with columns ['query', "q0", 'docid', 'rank', 'score', 'system']
    
    Note: Dataframe headers chosen to match library standards.
          More informative INPUT headers would be: 
          ["clinical case","code"]

    https://github.com/joaopalotti/trectools#file-formats
    '''
    # Rename columns
    pred.columns = pred_names
    
    # Not needed to: Check if predictions are empty, as all codes sorted by prob, prob-thr etc., are returned
    
    # Add columns needed for the library to properly import the dataframe
    pred['q0'] = 'Q0'
    pred['score'] = float(10) 
    pred['system'] = system_name 
    
    # Reorder and rename columns
    pred = pred[['query', "q0", 'docid', 'rank', 'score', 'system']]
    
    # Lowercase codes
    pred["docid"] = pred["docid"].str.lower()
    
    # Not needed to: Remove codes predicted twice in the same clinical case
    
    # Not needed to: Remove codes predicted but not in list of valid codes
    
    # Write dataframe to Run file
    pred.to_csv(output_path, index=False, header=None, sep = '\t')


# Code copied from: https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py

def format_gs(filepath, output_path=None, gs_names = ['qid', 'docno']):
    '''
    DESCRIPTION: Load Gold Standard table.
    
    INPUT: 
        filepath: str
            route to TSV file with Gold Standard.
        output_path: str
            route to TSV where intermediate file is stored    
    
    OUTPUT: 
        stores TSV files with columns ["query", "q0", "docid", "rel"].
    
    Note: Dataframe headers chosen to match library standards. 
          More informative headers for the INPUT would be: 
          ["clinical case","label","code","relevance"]
    
    # https://github.com/joaopalotti/trectools#file-formats
    '''
    in_header = 0 if 'cantemist' in filepath else None # Guille, since only the GS file from the Cantemist corpus has a header
    
    # Input args default values
    if output_path is None: output_path = './intermediate_gs_file.txt' 
    
    # Check GS format:
    check = pd.read_csv(filepath, sep='\t', header = in_header, nrows=1)
        
    if check.shape[1] != 2:
        raise ImportError('The GS file does not have 2 columns. Then, it was not imported')
    
    # Import GS
    gs = pd.read_csv(filepath, sep='\t', header = in_header, names = gs_names)  
        
    # Preprocessing
    gs["q0"] = str(0) # column with all zeros (q0) # Columnn needed for the library to properly import the dataframe
    gs["rel"] = str(1) # column indicating the relevance of the code (in GS, all codes are relevant)
    if 'code_pre' not in filepath:
        gs.docno = gs.docno.str.lower() # Lowercase codes
    gs = gs[['qid', 'q0', 'docno', 'rel']]
    
    # Remove codes predicted twice in the same clinical case 
    # (they are present in GS because one code may have several references)
    gs = gs.drop_duplicates(subset=['qid','docno'],  
                            keep='first')  # Keep first of the predictions

    # Write dataframe to Qrel file
    gs.to_csv(output_path, index=False, header=None, sep=' ')



## NER loss and callbacks

import tensorflow as tf

class TokenClassificationLoss(tf.keras.losses.Loss):
    """
    Code adapted from https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFTokenClassificationLoss
    """
    
    def __init__(self, from_logits=True, ignore_val=-100, 
                 reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        self.reduction = reduction
        super(TokenClassificationLoss, self).__init__(**kwargs)
        
    
    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits, reduction=self.reduction
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val
        reduced_preds = tf.boolean_mask(tf.reshape(y_pred, (-1, y_pred.shape[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
        
        return loss_fn(labels, reduced_preds)


class TokenClassificationLossSampleWeight(tf.keras.losses.Loss):
    """
    Code adapted from https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFTokenClassificationLoss
    """
    
    def __init__(self, weak_label, weak_weight_value=1, strong_weight_value=2, from_logits=True, ignore_val=-100, **kwargs):
        self.weak_label = weak_label
        self.weak_weight_value = weak_weight_value
        self.strong_weight_value = strong_weight_value
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        super(TokenClassificationLossSampleWeight, self).__init__(**kwargs)
        
    
    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits,
            reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val
        reduced_preds = tf.boolean_mask(tf.reshape(y_pred, (-1, y_pred.shape[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
        
        # sample weight
        sample_weight = tf.where(labels == self.weak_label, x=self.weak_weight_value, y=self.strong_weight_value)
        loss_value = loss_fn(labels, reduced_preds)
        
        return loss_value * tf.cast(sample_weight, loss_value.dtype)


class SequenceClassificationLoss(tf.keras.losses.Loss):
    
    def __init__(self, from_logits=True, ignore_val=-100, **kwargs):
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        super(SequenceClassificationLoss, self).__init__(**kwargs)
        
    
    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val # reshape to avoid incompatible y_true shape of (None, 1)
        reduced_preds = tf.boolean_mask(y_pred, active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
        
        return loss_fn(y_true=labels, y_pred=reduced_preds)


class EarlyNER(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, train_gs_path, val_gs_path, patience=10, strategy="word-all", subtask="ner", 
                 logits=True, n_output=1, df_val_gs_ner=None, val_gs_path_ner=None, mention_strat='max',
                 type_tokenizer='transformers', codes_pre_suf_mask=None, codes_pre_o_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.train_gs_path = train_gs_path
        self.val_gs_path = val_gs_path
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.logits = logits
        self.n_output = n_output
        self.df_val_gs_ner = df_val_gs_ner
        self.val_gs_path_ner = val_gs_path_ner
        self.mention_strat = mention_strat
        self.type_tokenizer = type_tokenizer
        self.codes_pre_suf_mask = codes_pre_suf_mask
        self.codes_pre_o_mask = codes_pre_o_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
            
        df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train, 
                                              start_end=self.train_start_end, word_id=self.train_word_id, 
                                              lab_decoder_list=self.label_decoder_list, 
                                              df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                              subtask=self.subtask, crf_mask_seq_len=train_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer,
                                            codes_pre_suf_mask=self.codes_pre_suf_mask, 
                                            codes_pre_o_mask=self.codes_pre_o_mask)
        if df_pred_train.shape[0] == 0:
            p_train = r_train = f1_train = 0
        else:
            p_train, r_train, f1_train = calculate_ner_metrics(gs=self.df_train_gs, 
                                                           pred=format_ner_pred_df(gs_path=self.train_gs_path, df_preds=df_pred_train, 
                                                                                   subtask=self.subtask.split('-')[0]),
                                                           subtask=self.subtask.split('-')[0])
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer,
                                            codes_pre_suf_mask=self.codes_pre_suf_mask, 
                                            codes_pre_o_mask=self.codes_pre_o_mask)
        if df_pred_val.shape[0] == 0:
            p_val = r_val = f1_val = 0
        else:
            p_val, r_val, f1_val = calculate_ner_metrics(gs=self.df_val_gs, 
                                                     pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_pred_val, 
                                                                             subtask=self.subtask.split('-')[0]), 
                                                     subtask=self.subtask.split('-')[0])
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        if self.df_val_gs_ner is not None:
            df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer)
            p_val_ner, r_val_ner, f1_val_ner = calculate_ner_metrics(gs=self.df_val_gs, 
                                                         pred=format_ner_pred_df(gs_path=self.val_gs_path_ner, df_preds=df_pred_val, 
                                                                                 subtask='ner'), 
                                                         subtask='ner')
            logs['p_val_ner'] = p_val_ner
            logs['r_val_ner'] = r_val_ner
            logs['f1_val_ner'] = f1_val_ner
        
            print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s' % 
              (str(p_train),str(r_train),str(f1_train),
               str(p_val),str(r_val),str(f1_val),str(p_val_ner),str(r_val_ner),str(f1_val_ner)),end=100*' '+'\n')
        else:
            print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s' % 
              (str(p_train),str(r_train),str(f1_train),
               str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


class EarlyNER_IOB_Code(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, train_gs_path, val_gs_path, iob_train_preds, iob_val_preds, patience=10, 
                 strategy="word-all", subtask="ner", 
                 logits=True, n_output=1, mention_strat='max',
                 type_tokenizer='transformers', codes_pre_suf_mask=None, codes_pre_o_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.train_gs_path = train_gs_path
        self.val_gs_path = val_gs_path
        self.iob_train_preds = iob_train_preds
        self.iob_val_preds = iob_val_preds
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.logits = logits
        self.n_output = n_output
        self.mention_strat = mention_strat
        self.type_tokenizer = type_tokenizer
        self.codes_pre_suf_mask = codes_pre_suf_mask
        self.codes_pre_o_mask = codes_pre_o_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
            
        # Add IOB preds
        y_pred_train = [self.iob_train_preds] + y_pred_train
        
        df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train, 
                                              start_end=self.train_start_end, word_id=self.train_word_id, 
                                              lab_decoder_list=self.label_decoder_list, 
                                              df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                              subtask=self.subtask, crf_mask_seq_len=train_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer,
                                            codes_pre_suf_mask=self.codes_pre_suf_mask, 
                                            codes_pre_o_mask=self.codes_pre_o_mask)
        if df_pred_train.shape[0] == 0:
            p_train = r_train = f1_train = 0
        else:
            p_train, r_train, f1_train = calculate_ner_metrics(gs=self.df_train_gs, 
                                                           pred=format_ner_pred_df(gs_path=self.train_gs_path, df_preds=df_pred_train, 
                                                                                   subtask=self.subtask.split('-')[0]),
                                                           subtask=self.subtask.split('-')[0])
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        # Add IOB preds
        y_pred_val = [self.iob_val_preds] + y_pred_val
        
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer,
                                            codes_pre_suf_mask=self.codes_pre_suf_mask, 
                                            codes_pre_o_mask=self.codes_pre_o_mask)
        if df_pred_val.shape[0] == 0:
            p_val = r_val = f1_val = 0
        else:
            p_val, r_val, f1_val = calculate_ner_metrics(gs=self.df_val_gs, 
                                                     pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_pred_val, 
                                                                             subtask=self.subtask.split('-')[0]), 
                                                     subtask=self.subtask.split('-')[0])
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s' % 
          (str(p_train),str(r_train),str(f1_train),
           str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



def create_cls_emb_y_samples(df_ann, doc_list, arr_frag, arr_start_end, arr_word_id, arr_ind, 
                             seq_len, empty_samples=False, subtask="norm-iob_code", 
                             only_mention=False):
    """
    IT PROCESSES BOTH CONTINUOUS AND DISCONTINUOUS ANNs.
    
    If empty_samples is True, we also add an empty sample for each fragment not containing any mention.
    
    return: list [((i_frag, label), emb_arr)] of shape n_frag, where: 
    *i_frag* (int) as the index of the fragment; 
    *label* (str) in ["O", code_1, ...code_n], or (code_pre, code_suf) (depending on subtask);
    *emb_arr* is a binary matrix (0, 1) of shape seq_len (with 1 value only for 
    tokens of a mention, and with 0 value also for the [CLS], [SEP] and [PAD] tokens).
    
    We also return the indices of the corresponding fragments, to be compatible with the returned embeddings,
    as well as a DataFrame containing the subset of the annotations that could be transformed into
    cls-samples, i.e. "completely" contained in a single fragment.
    """
    
    # Sanity check
    assert len(doc_list) == len(arr_frag)
    assert len(arr_start_end) == len(arr_word_id) == len(arr_ind)

    # Sort annotation
    df_ann = df_ann.sort_values(by=["clinical_case", "start", "end"])
    
    name_subtask = subtask
    if len(subtask.split('-')) > 1:
        name_subtask = subtask.split('-')[1]

    arr_cls_y = []
    arr_cls_ind = []
    arr_cls_ann = [] # Annotations completely contained in a single fragment
    for doc_i in range(len(doc_list)):
        # Calculate the start & end frag_i of doc_i
        start_frag_i = sum(arr_frag[:doc_i])
        end_frag_i = start_frag_i + arr_frag[doc_i] # end + 1
        # Extract annotations of doc_i
        doc_ann = df_ann[df_ann["clinical_case"] == doc_list[doc_i]]
        for index, row in doc_ann.iterrows():
            # Find the fragment containing the current ann
            found = False
            frag_i = start_frag_i
            while (frag_i < end_frag_i) and (found == False):
                # Check if frag_i "completely" contains current ann
                arr_start_end_frag_i = np.array(arr_start_end[frag_i]) # shape: [n_tok_frag_i, 2]
                tok_start = np.where(arr_start_end_frag_i[:, 0] <= row['start'])[0]
                tok_end = np.where(arr_start_end_frag_i[:, 1] >= row['end'])[0]
                if (len(tok_start) > 0) and (len(tok_end) > 0):
                    # The whole ann is contained within the fragment
                    found = True
                    arr_word_id_frag_i = arr_word_id[frag_i] # list shape: [n_tok_frag_i, 1]
                    assert len(arr_word_id_frag_i) == arr_start_end_frag_i.shape[0]
                    if not only_mention:
                        cls_emb = np.array([0] * seq_len) # create initial embeddings
                    else:
                        cls_ind = []
                    # Iterate over ann segments
                    for segment in row['location'].split(';'):
                        ann_seg_pos = segment.split(' ')
                        ann_seg_start = int(ann_seg_pos[0])
                        ann_seg_end = int(ann_seg_pos[1])
                        seg_tok_start = np.where(arr_start_end_frag_i[:, 0] <= ann_seg_start)[0]
                        seg_tok_end = np.where(arr_start_end_frag_i[:, 1] >= ann_seg_end)[0]
                        assert len(seg_tok_start) > 0 and len(seg_tok_end) > 0
                
                        seg_tok_i_start = seg_tok_start[-1] # position index of the last sub-token of the first word of the ann
                        seg_tok_i_end = seg_tok_end[0] # position index of the first sub-token of the last word of the ann
                        # Extract ids of the first and last word of the ann-seg  
                        word_id_start = arr_word_id_frag_i[seg_tok_i_start]
                        word_id_end = arr_word_id_frag_i[seg_tok_i_end]
                        # Update CLS embedding, with 1s in the positions of the sub-tokens from the words
                        # inside the ann, and 0s in the remaining positions
                        seg_cls_emb = [1 if (w_id >= word_id_start and w_id <= word_id_end) else 0 for w_id in arr_word_id_frag_i]
                        # Add 0s for [CLS], [SEP] and (potentially) [PAD] tokens
                        seg_cls_emb = [0] + seg_cls_emb + [0] * (seq_len - len(arr_word_id_frag_i) - 1)
                        if not only_mention:
                            cls_emb += np.array(seg_cls_emb)
                        else:
                            assert len(seg_cls_emb) == len(arr_ind[frag_i]) == seq_len
                            for pos in range(seq_len):
                                if seg_cls_emb[pos] == 1:
                                    cls_ind.append(arr_ind[frag_i][pos])
                    
                    if name_subtask == "iob_code":
                        label = row['code']
                    elif name_subtask == "iob_code_suf":
                        label = (row['code_pre'], row['code_suf'])
                    else:
                        label = 'O'
                    if not only_mention:
                        # Check no overlapping ann-segments
                        assert cls_emb.min() == 0 and cls_emb.max() == 1
                        arr_cls_y.append(((frag_i, label), cls_emb))
                        # Append indices of the corresponding fragment
                        arr_cls_ind.append(arr_ind[frag_i])
                    else:
                        arr_cls_y.append((frag_i, label))
                        arr_cls_ind.append(cls_ind)
                        
                    # Append ann
                    arr_cls_ann.append(row)

                frag_i += 1
    
    if not only_mention:
        if empty_samples:
            # Add empty cls samples & ind for fragments not containing ann
            all_frag_i = set(range(0, len(arr_start_end)))
            ann_frag_i = set([sample[0][0] for sample in arr_cls_y])
            empty_frag_i = all_frag_i - ann_frag_i
            label = 'O'
            if name_subtask == "iob_code_suf":
                label = ('O', 'O')
            for frag_i in sorted(empty_frag_i):
                cls_y = np.array([0] * seq_len)
                arr_cls_y.append(((frag_i, label), cls_y))
                arr_cls_ind.append(arr_ind[frag_i])
        
        arr_cls_ind = np.array(arr_cls_ind)
    
    return arr_cls_y, arr_cls_ind, pd.DataFrame(arr_cls_ann)



def cls_code_extract_preds(y_pred_cls, df_pred_ner, code_decoder,
                           codes_o_mask=None):
    # y_pred_cls shape: n_samples x n_codes
    df_pred_norm = []
    
    if codes_o_mask is not None:
        y_pred_cls = np.multiply(codes_o_mask, y_pred_cls)
    
    y_pred_cls_code = np.argmax(y_pred_cls, axis=1)
    i = 0
    for index, row in df_pred_ner.iterrows():
        row['code_pred'] = code_decoder[y_pred_cls_code[i]]
        df_pred_norm.append(row)
        i += 1
    
    return pd.DataFrame(df_pred_norm)



def cls_code_suffix_extract_preds(y_pred_cls, df_pred_ner, 
                                  code_decoder_list,
                                  code_sep='/',
                                  codes_pre_o_mask=None,
                                  codes_pre_suf_mask=None):
    # y_pred_cls shape: 2 x n_samples x n_codes (depending on pre or suf)
    
    # In future versions, this could be re-implemented
    # to be included in the previous function
    
    df_pred_norm = []
    
    y_pred_cls_pre = y_pred_cls[0]
    y_pred_cls_suf = y_pred_cls[1]
    
    if codes_pre_o_mask is not None:
        y_pred_cls_pre = np.multiply(codes_pre_o_mask, y_pred_cls_pre)
    
    y_pred_cls_pre_code = np.argmax(y_pred_cls_pre, axis=1) # shape: n_samples
    i = 0
    for index, row in df_pred_ner.iterrows():
        label_pre = y_pred_cls_pre_code[i]
        code_pre = code_decoder_list[0][label_pre]
        frag_i_pred_cls_suf = y_pred_cls_suf[i]
        if codes_pre_suf_mask is not None:
            frag_i_pred_cls_suf = np.multiply(
                codes_pre_suf_mask[label_pre],
                frag_i_pred_cls_suf
            )
        label_suf = np.argmax(frag_i_pred_cls_suf)
        code_suf = code_decoder_list[1][label_suf]
        code_pred = code_pre
        if code_suf != 'O':
            code_pred += code_sep + code_suf
        row['code_pred'] = code_pred
        df_pred_norm.append(row)
        i += 1
    
    return pd.DataFrame(df_pred_norm)



def cls_code_norm_preds_brat_format(y_pred_cls, df_pred_ner, 
                                    code_decoder_list,
                                    subtask='norm-iob_code',
                                    code_sep='/',
                                    codes_pre_o_mask=None,
                                    codes_pre_suf_mask=None):
    
    # codes_o_mask shape: n_codes + 1 ('O label')
    # codes_pre_suf shape: n_codes_pre x n_codes_suf (+ 1 if 'O' label)
    
    df_pred_norm = None
    
    name_subtask = subtask.split('-')[1]
    
    if name_subtask == "iob_code":
        df_pred_norm = cls_code_extract_preds(
            y_pred_cls=y_pred_cls, df_pred_ner=df_pred_ner, 
            code_decoder=code_decoder_list[0],
            codes_o_mask=codes_pre_o_mask
        )
        
    elif name_subtask == "iob_code_suf":
        df_pred_norm = cls_code_suffix_extract_preds(
            y_pred_cls=y_pred_cls, df_pred_ner=df_pred_ner, 
            code_decoder_list=code_decoder_list,
            code_sep=code_sep,
            codes_pre_o_mask=codes_pre_o_mask,
            codes_pre_suf_mask=codes_pre_suf_mask
        )
    
    else:
        raise Exception("Subtask not implemented!")
    
    return df_pred_norm
    


class EarlyNER_IOB_Code_CLS(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, df_pred_ner_train, 
                 df_pred_ner_val, label_decoder_list, 
                 df_train_gs, df_val_gs, train_gs_path, val_gs_path, 
                 n_output=1, patience=10, subtask="norm-iob_code", 
                 codes_pre_o_mask=None,
                 codes_pre_suf_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.df_pred_ner_train = df_pred_ner_train
        self.df_pred_ner_val = df_pred_ner_val
        self.label_decoder_list = label_decoder_list
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.train_gs_path = train_gs_path
        self.val_gs_path = val_gs_path
        self.n_output = n_output
        self.patience = patience
        self.subtask = subtask
        self.codes_pre_o_mask = codes_pre_o_mask
        self.codes_pre_suf_mask = codes_pre_suf_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data    
        y_pred_train_cls = self.model.predict(self.X_train)
        
        if self.n_output > 1:
            y_pred_train_cls = y_pred_train_cls[-1]
        
        df_pred_train = cls_code_norm_preds_brat_format(
            y_pred_cls=y_pred_train_cls, 
            df_pred_ner=self.df_pred_ner_train, 
            code_decoder_list=self.label_decoder_list,
            codes_pre_o_mask=self.codes_pre_o_mask,
            codes_pre_suf_mask=self.codes_pre_suf_mask,
            subtask=self.subtask
        )
        
        if df_pred_train.shape[0] == 0:
            p_train = r_train = f1_train = 0
        else:
            p_train, r_train, f1_train = calculate_ner_metrics(gs=self.df_train_gs, 
                                                           pred=format_ner_pred_df(gs_path=self.train_gs_path, df_preds=df_pred_train, 
                                                                                   subtask=self.subtask.split('-')[0]),
                                                           subtask=self.subtask.split('-')[0])
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        ### Val data
        y_pred_val_cls = self.model.predict(self.X_val)
        
        if self.n_output > 1:
            y_pred_val_cls = y_pred_val_cls[-1]
        
        df_pred_val = cls_code_norm_preds_brat_format(
            y_pred_cls=y_pred_val_cls, 
            df_pred_ner=self.df_pred_ner_val, 
            code_decoder_list=self.label_decoder_list,
            codes_pre_o_mask=self.codes_pre_o_mask,
            codes_pre_suf_mask=self.codes_pre_suf_mask,
            subtask=self.subtask
        )
        
        if df_pred_val.shape[0] == 0:
            p_val = r_val = f1_val = 0
        else:
            p_val, r_val, f1_val = calculate_ner_metrics(gs=self.df_val_gs, 
                                                     pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_pred_val, 
                                                                             subtask=self.subtask.split('-')[0]), 
                                                     subtask=self.subtask.split('-')[0])
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s' % 
          (str(p_train),str(r_train),str(f1_train),
           str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



class EarlyNER_IOB_Code_CLS_Val(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_val, 
                 df_pred_ner_val, label_decoder_list, 
                 df_val_gs, val_gs_path, 
                 patience=10, subtask="norm-iob_code", 
                 codes_pre_o_mask=None,
                 codes_pre_suf_mask=None):
        self.X_val = x_val
        self.df_pred_ner_val = df_pred_ner_val
        self.label_decoder_list = label_decoder_list
        self.df_val_gs = df_val_gs
        self.val_gs_path = val_gs_path
        self.patience = patience
        self.subtask = subtask
        self.codes_pre_o_mask = codes_pre_o_mask
        self.codes_pre_suf_mask = codes_pre_suf_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Val data
        y_pred_val_cls = self.model.predict(self.X_val)
        
        df_pred_val = cls_code_norm_preds_brat_format(
            y_pred_cls=y_pred_val_cls, 
            df_pred_ner=self.df_pred_ner_val, 
            code_decoder_list=self.label_decoder_list,
            codes_pre_o_mask=self.codes_pre_o_mask,
            codes_pre_suf_mask=self.codes_pre_suf_mask,
            subtask=self.subtask
        )
        
        if df_pred_val.shape[0] == 0:
            p_val = r_val = f1_val = 0
        else:
            p_val, r_val, f1_val = calculate_ner_metrics(gs=self.df_val_gs, 
                                                     pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_pred_val, 
                                                                             subtask=self.subtask.split('-')[0]), 
                                                     subtask=self.subtask.split('-')[0])
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rval_p: %s - val_r: %s - val_f1: %s' % 
          (str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

        

class EarlyNER_IOB_Code_CLS_Multi_Task(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col,
                 label_decoder_list, train_doc_list, val_doc_list, train_start_end, val_start_end,
                 train_word_id, val_word_id, df_train_gs, df_val_gs, train_gs_path, val_gs_path, 
                 patience=10, strategy="word-prod", subtask="norm-iob_code", 
                 type_tokenizer="transformers", seq_len=128, empty_samples=False, 
                 n_cls_output=1, codes_pre_o_mask=None, codes_pre_suf_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.train_gs_path = train_gs_path
        self.val_gs_path = val_gs_path
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.type_tokenizer = type_tokenizer
        self.seq_len = seq_len
        self.empty_samples = empty_samples
        self.n_cls_output = n_cls_output
        self.codes_pre_o_mask = codes_pre_o_mask
        self.codes_pre_suf_mask = codes_pre_suf_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data    
        y_pred_train_iob = self.model.predict(self.X_train)[0]
        
        # Obtain formatted NER predictions
        df_train_preds_iob = ner_preds_brat_format(
            doc_list=self.train_doc_list, fragments=self.frag_train, preds=[y_pred_train_iob], 
            start_end=self.train_start_end, word_id=self.train_word_id, 
            lab_decoder_list=[self.label_decoder_list[0]], 
            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
            subtask='ner', type_tokenizer=self.type_tokenizer
        )
        
        if df_train_preds_iob.shape[0] == 0:
            p_ner_train = r_ner_train = f1_ner_train = p_train = r_train = f1_train = 0
        
        else:
            p_ner_train, r_ner_train, f1_ner_train = calculate_ner_metrics(gs=self.df_train_gs, 
                        pred=format_ner_pred_df(gs_path=self.train_gs_path, df_preds=df_train_preds_iob, 
                                                subtask='ner'),
                        subtask='ner')
            
            # Obtain CLS-samples to predict
            train_cls_y_iob, train_cls_ind_iob, df_train_cls_ann_iob = create_cls_emb_y_samples(
                df_ann=df_train_preds_iob, doc_list=self.train_doc_list, 
                arr_frag=self.frag_train,
                arr_start_end=self.train_start_end, arr_word_id=self.train_word_id, 
                arr_ind=self.X_train['input_ids'],
                seq_len=self.seq_len, empty_samples=self.empty_samples, subtask='ner'
            )
            train_cls_emb_y_iob = np.array([sample[1] for sample in train_cls_y_iob])
            train_cls_att_iob = np.array([self.X_train['attention_mask'][sample[0][0]] for sample in train_cls_y_iob])
            
            # CLS predictions
            y_pred_train_cls = self.model.predict({'input_ids': train_cls_ind_iob, 
                                            'attention_mask': train_cls_att_iob,
                                            'ner_ann_ids': train_cls_emb_y_iob})[1:]
            
            if self.n_cls_output == 1:
                y_pred_train_cls = y_pred_train_cls[0]
            
            df_train_preds_cls = cls_code_norm_preds_brat_format(
                y_pred_cls=y_pred_train_cls, df_pred_ner=df_train_cls_ann_iob, 
                code_decoder_list=self.label_decoder_list[1:],
                subtask=self.subtask,
                codes_pre_o_mask=self.codes_pre_o_mask
            )
            
            p_train, r_train, f1_train = calculate_ner_metrics(gs=self.df_train_gs, 
                                    pred=format_ner_pred_df(gs_path=self.train_gs_path, df_preds=df_train_preds_cls, 
                                                            subtask=self.subtask.split('-')[0]),
                                    subtask=self.subtask.split('-')[0])
            
        logs['p_ner'] = p_ner_train
        logs['r_ner'] = r_ner_train
        logs['f1_ner'] = f1_ner_train
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        
        ### Val data    
        y_pred_val_iob = self.model.predict(self.X_val)[0]
        
        # Obtain formatted NER predictions
        df_val_preds_iob = ner_preds_brat_format(
            doc_list=self.val_doc_list, fragments=self.frag_val, preds=[y_pred_val_iob], 
            start_end=self.val_start_end, word_id=self.val_word_id, 
            lab_decoder_list=[self.label_decoder_list[0]], 
            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
            subtask='ner', type_tokenizer=self.type_tokenizer
        )
        
        if df_val_preds_iob.shape[0] == 0:
            p_ner_val = r_ner_val = f1_ner_val = p_val = r_val = f1_val = 0
        
        else:
            p_ner_val, r_ner_val, f1_ner_val = calculate_ner_metrics(gs=self.df_val_gs, 
                        pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_val_preds_iob, 
                                                subtask='ner'),
                        subtask='ner')
            
            # Obtain CLS-samples to predict
            val_cls_y_iob, val_cls_ind_iob, df_val_cls_ann_iob = create_cls_emb_y_samples(
                df_ann=df_val_preds_iob, doc_list=self.val_doc_list, 
                arr_frag=self.frag_val,
                arr_start_end=self.val_start_end, arr_word_id=self.val_word_id, 
                arr_ind=self.X_val['input_ids'],
                seq_len=self.seq_len, empty_samples=self.empty_samples, subtask='ner'
            )
            val_cls_emb_y_iob = np.array([sample[1] for sample in val_cls_y_iob])
            val_cls_att_iob = np.array([self.X_val['attention_mask'][sample[0][0]] for sample in val_cls_y_iob])
            
            # CLS predictions
            y_pred_val_cls = self.model.predict({'input_ids': val_cls_ind_iob, 
                                            'attention_mask': val_cls_att_iob,
                                            'ner_ann_ids': val_cls_emb_y_iob})[1:]
            
            if self.n_cls_output == 1:
                y_pred_val_cls = y_pred_val_cls[0]
            
            df_val_preds_cls = cls_code_norm_preds_brat_format(
                y_pred_cls=y_pred_val_cls, df_pred_ner=df_val_cls_ann_iob, 
                code_decoder_list=self.label_decoder_list[1:],
                subtask=self.subtask,
                codes_pre_o_mask=self.codes_pre_o_mask
            )
            
            p_val, r_val, f1_val = calculate_ner_metrics(gs=self.df_val_gs, 
                                    pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_val_preds_cls, 
                                                            subtask=self.subtask.split('-')[0]),
                                    subtask=self.subtask.split('-')[0])
            
        logs['p_ner_val'] = p_ner_val
        logs['r_ner_val'] = r_ner_val
        logs['f1_ner_val'] = f1_ner_val
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rp_ner: %s - r_ner: %s - f1_ner: %s / p: %s - r: %s - f1: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s / val_p: %s - val_r: %s - val_f1: %s' % 
          (str(p_ner_train),str(r_ner_train),str(f1_ner_train),str(p_train),str(r_train),str(f1_train),
           str(p_ner_val),str(r_ner_val),str(f1_ner_val),str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



class EarlyCodiEspX_IOB_Disc_Code_CLS_Multi_Task(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col,
                 label_decoder_list, train_doc_list, val_doc_list, train_start_end, val_start_end,
                 train_word_id, val_word_id, df_train_gs, df_val_gs, valid_codes,
                 patience=10, strategy="word-prod", subtask="norm-iob_code_suf", 
                 type_tokenizer="transformers", seq_len=128, empty_samples=False, 
                 n_cls_output=2, type_ann='DIAGNOSTICO', code_sep='.',
                 codes_pre_o_mask=None, codes_pre_suf_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.type_tokenizer = type_tokenizer
        self.seq_len = seq_len
        self.empty_samples = empty_samples
        self.n_cls_output = n_cls_output
        self.type_ann = type_ann
        self.code_sep = code_sep
        self.codes_pre_o_mask = codes_pre_o_mask
        self.codes_pre_suf_mask = codes_pre_suf_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data    
        y_pred_train_iob = self.model.predict(self.X_train)[:2]
        
        # Obtain formatted NER predictions
        df_train_preds_iob = ner_preds_brat_format(
            doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train_iob, 
            start_end=self.train_start_end, word_id=self.train_word_id, 
            lab_decoder_list=[self.label_decoder_list[0]], 
            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
            subtask='norm-iob_cont_disc', type_tokenizer=self.type_tokenizer
        )
        
        if df_train_preds_iob.shape[0] == 0:
            p_ner_train = r_ner_train = f1_ner_train = p_train = r_train = f1_train = 0
        
        else:
            # Adapt to CodiEsp format
            df_train_preds_iob_eval = df_train_preds_iob.copy()
            df_train_preds_iob_eval['label_pred'] = self.type_ann
            df_train_preds_iob_eval['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_train_preds_iob_eval.iterrows()]
            df_train_preds_iob_eval['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
            df_train_preds_iob_eval = df_train_preds_iob_eval[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NER metrics
            p_ner_train, r_ner_train, f1_ner_train = calculate_codiesp_ner_metrics(
                df_gs=self.df_train_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_train_preds_iob_eval, valid_codes=self.valid_codes)
            )
            
            # Obtain CLS-samples to predict
            train_cls_y_iob, train_cls_ind_iob, df_train_cls_ann_iob = create_cls_emb_y_samples(
                df_ann=df_train_preds_iob, doc_list=self.train_doc_list, 
                arr_frag=self.frag_train,
                arr_start_end=self.train_start_end, arr_word_id=self.train_word_id, 
                arr_ind=self.X_train['input_ids'],
                seq_len=self.seq_len, empty_samples=self.empty_samples, subtask='ner'
            )
            train_cls_emb_y_iob = np.array([sample[1] for sample in train_cls_y_iob])
            train_cls_att_iob = np.array([self.X_train['attention_mask'][sample[0][0]] for sample in train_cls_y_iob])
            
            # CLS predictions
            y_pred_train_cls = self.model.predict({'input_ids': train_cls_ind_iob, 
                                            'attention_mask': train_cls_att_iob,
                                            'ner_ann_ids': train_cls_emb_y_iob})[2:]
            
            if self.n_cls_output == 1:
                y_pred_train_cls = y_pred_train_cls[0]
            
            df_train_preds_cls = cls_code_norm_preds_brat_format(
                y_pred_cls=y_pred_train_cls, df_pred_ner=df_train_cls_ann_iob, 
                code_decoder_list=self.label_decoder_list[1:],
                subtask=self.subtask,
                code_sep=self.code_sep,
                codes_pre_o_mask=self.codes_pre_o_mask,
                codes_pre_suf_mask=self.codes_pre_suf_mask
            )
            
            # Adapt to CodiEsp format
            df_train_preds_cls['label_pred'] = self.type_ann
            df_train_preds_cls['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_train_preds_cls.iterrows()]
            df_train_preds_cls = df_train_preds_cls.rename(columns={'code_pred': 'code'})
            df_train_preds_cls = df_train_preds_cls[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            
            p_train, r_train, f1_train = calculate_codiesp_x_metrics(
                df_gs=self.df_train_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_train_preds_cls, valid_codes=self.valid_codes)
            )
            
        logs['p_ner'] = p_ner_train
        logs['r_ner'] = r_ner_train
        logs['f1_ner'] = f1_ner_train
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        
        ### Val data    
        y_pred_val_iob = self.model.predict(self.X_val)[:2]
        
        # Obtain formatted NER predictions
        df_val_preds_iob = ner_preds_brat_format(
            doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val_iob, 
            start_end=self.val_start_end, word_id=self.val_word_id, 
            lab_decoder_list=[self.label_decoder_list[0]], 
            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
            subtask='norm-iob_cont_disc', type_tokenizer=self.type_tokenizer
        )
        
        if df_val_preds_iob.shape[0] == 0:
            p_ner_val = r_ner_val = f1_ner_val = p_val = r_val = f1_val = 0
        
        else:
            # Adapt to CodiEsp format
            df_val_preds_iob_eval = df_val_preds_iob.copy()
            df_val_preds_iob_eval['label_pred'] = self.type_ann
            df_val_preds_iob_eval['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_val_preds_iob_eval.iterrows()]
            df_val_preds_iob_eval['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
            df_val_preds_iob_eval = df_val_preds_iob_eval[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NER metrics
            p_ner_val, r_ner_val, f1_ner_val = calculate_codiesp_ner_metrics(
                df_gs=self.df_val_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_val_preds_iob_eval, valid_codes=self.valid_codes)
            )
            
            # Obtain CLS-samples to predict
            val_cls_y_iob, val_cls_ind_iob, df_val_cls_ann_iob = create_cls_emb_y_samples(
                df_ann=df_val_preds_iob, doc_list=self.val_doc_list, 
                arr_frag=self.frag_val,
                arr_start_end=self.val_start_end, arr_word_id=self.val_word_id, 
                arr_ind=self.X_val['input_ids'],
                seq_len=self.seq_len, empty_samples=self.empty_samples, subtask='ner'
            )
            val_cls_emb_y_iob = np.array([sample[1] for sample in val_cls_y_iob])
            val_cls_att_iob = np.array([self.X_val['attention_mask'][sample[0][0]] for sample in val_cls_y_iob])
            
            # CLS predictions
            y_pred_val_cls = self.model.predict({'input_ids': val_cls_ind_iob, 
                                            'attention_mask': val_cls_att_iob,
                                            'ner_ann_ids': val_cls_emb_y_iob})[2:]
            
            if self.n_cls_output == 1:
                y_pred_val_cls = y_pred_val_cls[0]
            
            df_val_preds_cls = cls_code_norm_preds_brat_format(
                y_pred_cls=y_pred_val_cls, df_pred_ner=df_val_cls_ann_iob, 
                code_decoder_list=self.label_decoder_list[1:],
                subtask=self.subtask,
                code_sep=self.code_sep,
                codes_pre_o_mask=self.codes_pre_o_mask,
                codes_pre_suf_mask=self.codes_pre_suf_mask
            )
            
            # Adapt to CodiEsp format
            df_val_preds_cls['label_pred'] = self.type_ann
            df_val_preds_cls['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_val_preds_cls.iterrows()]
            df_val_preds_cls = df_val_preds_cls.rename(columns={'code_pred': 'code'})
            df_val_preds_cls = df_val_preds_cls[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            
            p_val, r_val, f1_val = calculate_codiesp_x_metrics(
                df_gs=self.df_val_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_val_preds_cls, valid_codes=self.valid_codes)
            )
            
        logs['p_ner_val'] = p_ner_val
        logs['r_ner_val'] = r_ner_val
        logs['f1_ner_val'] = f1_ner_val
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rp_ner: %s - r_ner: %s - f1_ner: %s / p: %s - r: %s - f1: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s / val_p: %s - val_r: %s - val_f1: %s' % 
          (str(p_ner_train),str(r_ner_train),str(f1_ner_train),str(p_train),str(r_train),str(f1_train),
           str(p_ner_val),str(r_ner_val),str(f1_ner_val),str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



class EarlyNormOnly(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Validation P, R, F1 Norm values are reported at the end of each epoch.
    """
    
    def __init__(self, x_val, frag_val, text_val, text_col, label_decoder_list, 
                 val_doc_list, val_start_end, val_word_id, 
                 df_val_gs, val_gs_path, iob_val_preds, patience=10, strategy="word-all", subtask="ner", 
                 logits=True, n_output=1, df_val_gs_ner=None, val_gs_path_ner=None, mention_strat='max'):
        self.X_val = x_val
        self.frag_val = frag_val
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.val_doc_list = val_doc_list
        self.val_start_end = val_start_end
        self.val_word_id = val_word_id
        self.df_val_gs = df_val_gs
        self.val_gs_path = val_gs_path
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.logits = logits
        self.n_output = n_output
        self.df_val_gs_ner = df_val_gs_ner
        self.val_gs_path_ner = val_gs_path_ner
        self.mention_strat = mention_strat
        self.iob_val_preds = iob_val_preds
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
        
        y_pred_val = [self.iob_val_preds] + y_pred_val
            
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat)
        p_val, r_val, f1_val = calculate_ner_metrics(gs=self.df_val_gs, 
                                                     pred=format_ner_pred_df(gs_path=self.val_gs_path, df_preds=df_pred_val, 
                                                                             subtask=self.subtask.split('-')[0]), 
                                                     subtask=self.subtask.split('-')[0])
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        if self.df_val_gs_ner is not None:
            df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat)
            p_val_ner, r_val_ner, f1_val_ner = calculate_ner_metrics(gs=self.df_val_gs, 
                                                         pred=format_ner_pred_df(gs_path=self.val_gs_path_ner, df_preds=df_pred_val, 
                                                                                 subtask='ner'), 
                                                         subtask='ner')
            logs['p_val_ner'] = p_val_ner
            logs['r_val_ner'] = r_val_ner
            logs['f1_val_ner'] = f1_val_ner
        
            print('\rval_p: %s - val_r: %s - val_f1: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s' % 
              (str(p_val),str(r_val),str(f1_val),str(p_val_ner),str(r_val_ner),str(f1_val_ner)),end=100*' '+'\n')
        else:
            print('\rval_p: %s - val_r: %s - val_f1: %s' % 
              (str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

        
        
class EarlyCodiEspX(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, valid_codes, patience=10, strategy="word-all", subtask="ner", 
                 logits=True, n_output=1, eval_ner=False, mention_strat='max', code_sep='/',
                 type_tokenizer='transformers', type_ann='DIAGNOSTICO', 
                 codes_pre_suf_mask=None, codes_pre_o_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.logits = logits
        self.n_output = n_output
        self.eval_ner = eval_ner
        self.mention_strat = mention_strat
        self.code_sep = code_sep
        self.type_tokenizer = type_tokenizer
        self.type_ann = type_ann
        self.codes_pre_suf_mask = codes_pre_suf_mask
        self.codes_pre_o_mask = codes_pre_o_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
            
        df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train, 
                                              start_end=self.train_start_end, word_id=self.train_word_id, 
                                              lab_decoder_list=self.label_decoder_list, 
                                              df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                              subtask=self.subtask, crf_mask_seq_len=train_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer,
                                            code_sep=self.code_sep,
                                            codes_pre_suf_mask=self.codes_pre_suf_mask, 
                                            codes_pre_o_mask=self.codes_pre_o_mask)
        if df_pred_train.shape[0] == 0:
            p_train = r_train = f1_train = 0
        else:
            # Adapt to CodiEsp format
            df_pred_train['label_pred'] = self.type_ann
            df_pred_train['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train.iterrows()]
            df_pred_train = df_pred_train.rename(columns={'code_pred': 'code'})
            df_pred_train = df_pred_train[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NORM metrics
            p_train, r_train, f1_train = calculate_codiesp_x_metrics(
                df_gs=self.df_train_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_train, valid_codes=self.valid_codes)
            )
            
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat, 
                                            type_tokenizer=self.type_tokenizer,
                                            code_sep=self.code_sep,
                                            codes_pre_suf_mask=self.codes_pre_suf_mask, 
                                            codes_pre_o_mask=self.codes_pre_o_mask)
        if df_pred_val.shape[0] == 0:
            p_val = r_val = f1_val = 0
        else:
            # Adapt to CodiEsp format
            df_pred_val['label_pred'] = self.type_ann
            df_pred_val['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val.iterrows()]
            df_pred_val = df_pred_val.rename(columns={'code_pred': 'code'})
            df_pred_val = df_pred_val[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NORM metrics
            p_val, r_val, f1_val = calculate_codiesp_x_metrics(
                df_gs=self.df_val_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_val, valid_codes=self.valid_codes)
            )
            
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        
        if self.eval_ner:
            # Train
            df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train, 
                                            start_end=self.train_start_end, word_id=self.train_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=train_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
            if df_pred_train.shape[0] == 0:
                p_train_ner = r_train_ner = f1_train_ner = 0
            else:
                # Adapt to CodiEsp format
                df_pred_train['label_pred'] = self.type_ann
                df_pred_train['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train.iterrows()]
                df_pred_train['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
                df_pred_train = df_pred_train[['clinical_case', 'pos_pred', 'label_pred', 'code']]
                # Compute NER metrics
                p_train_ner, r_train_ner, f1_train_ner = calculate_codiesp_ner_metrics(
                    df_gs=self.df_train_gs[self.df_train_gs['label_gs'] == self.type_ann], 
                    df_pred=format_codiesp_x_pred_df(df_run=df_pred_train, valid_codes=self.valid_codes)
                )
            
            logs['p_ner'] = p_train_ner
            logs['r_ner'] = r_train_ner
            logs['f1_ner'] = f1_train_ner
            
            # Val
            df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=val_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
            
            if df_pred_val.shape[0] == 0:
                p_val_ner = r_val_ner = f1_val_ner = 0
            else:
                # Adapt to CodiEsp format
                df_pred_val['label_pred'] = self.type_ann
                df_pred_val['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val.iterrows()]
                df_pred_val['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
                df_pred_val = df_pred_val[['clinical_case', 'pos_pred', 'label_pred', 'code']]
                # Compute NER metrics
                p_val_ner, r_val_ner, f1_val_ner = calculate_codiesp_ner_metrics(
                    df_gs=self.df_val_gs[self.df_val_gs['label_gs'] == self.type_ann], 
                    df_pred=format_codiesp_x_pred_df(df_run=df_pred_val, valid_codes=self.valid_codes)
                )
            
            logs['p_val_ner'] = p_val_ner
            logs['r_val_ner'] = r_val_ner
            logs['f1_val_ner'] = f1_val_ner
        
            print('\rp: %s - r: %s - f1: %s | p_ner: %s - r_ner: %s - f1_ner: %s | val_p: %s - val_r: %s - val_f1: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s' % 
              (str(p_train),str(r_train),str(f1_train),
               str(p_train_ner),str(r_train_ner),str(f1_train_ner),
               str(p_val),str(r_val),str(f1_val),
               str(p_val_ner),str(r_val_ner),str(f1_val_ner)),end=100*' '+'\n')
            
        else:
            print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s' % 
              (str(p_train),str(r_train),str(f1_train),
               str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val >= self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

        
        
class EarlyCodiEspX_IOB(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, valid_codes, patience=10, strategy="word-all", 
                 logits=True, n_output=1, subtask='ner', type_tokenizer='transformers', type_ann='DIAGNOSTICO'):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.patience = patience
        self.strategy = strategy
        self.logits = logits
        self.n_output = n_output
        self.subtask = subtask
        self.type_tokenizer = type_tokenizer
        self.type_ann = type_ann
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
            
        df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train, 
                                            start_end=self.train_start_end, word_id=self.train_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=train_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
        if df_pred_train.shape[0] == 0:
            p_train_ner = r_train_ner = f1_train_ner = 0
        else:
            # Adapt to CodiEsp format
            df_pred_train['label_pred'] = self.type_ann
            df_pred_train['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train.iterrows()]
            df_pred_train['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
            df_pred_train = df_pred_train[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NER metrics
            p_train_ner, r_train_ner, f1_train_ner = calculate_codiesp_ner_metrics(
                df_gs=self.df_train_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_train, valid_codes=self.valid_codes)
            )

        logs['p_ner'] = p_train_ner
        logs['r_ner'] = r_train_ner
        logs['f1_ner'] = f1_train_ner
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
            
        if df_pred_val.shape[0] == 0:
            p_val_ner = r_val_ner = f1_val_ner = 0
        else:
            # Adapt to CodiEsp format
            df_pred_val['label_pred'] = self.type_ann
            df_pred_val['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val.iterrows()]
            df_pred_val['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
            df_pred_val = df_pred_val[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NER metrics
            p_val_ner, r_val_ner, f1_val_ner = calculate_codiesp_ner_metrics(
                df_gs=self.df_val_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_val, valid_codes=self.valid_codes)
            )

        logs['p_val_ner'] = p_val_ner
        logs['r_val_ner'] = r_val_ner
        logs['f1_val_ner'] = f1_val_ner

        print('\rp_ner: %s - r_ner: %s - f1_ner: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s' % 
          (str(p_train_ner),str(r_train_ner),str(f1_train_ner),
           str(p_val_ner),str(r_val_ner),str(f1_val_ner)),end=100*' '+'\n')
        
        # Early-stopping
        if (f1_val_ner >= self.best):
            self.best = f1_val_ner
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



class EarlyCodiEspX_IOB_Code_CLS(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, df_pred_ner_train, 
                 df_pred_ner_val, label_decoder_list, 
                 df_train_gs, df_val_gs,
                 valid_codes, patience=10, subtask="norm-iob_code", 
                 type_ann='DIAGNOSTICO',
                 code_sep='.',
                 codes_pre_o_mask=None,
                 codes_pre_suf_mask=None):
        self.X_train = x_train
        self.X_val = x_val
        self.df_pred_ner_train = df_pred_ner_train
        self.df_pred_ner_val = df_pred_ner_val
        self.label_decoder_list = label_decoder_list
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.patience = patience
        self.subtask = subtask
        self.type_ann = type_ann
        self.code_sep = code_sep
        self.codes_pre_o_mask = codes_pre_o_mask
        self.codes_pre_suf_mask = codes_pre_suf_mask
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data    
        y_pred_train_cls = self.model.predict(self.X_train)
        
        df_pred_train = cls_code_norm_preds_brat_format(
            y_pred_cls=y_pred_train_cls, 
            df_pred_ner=self.df_pred_ner_train, 
            code_decoder_list=self.label_decoder_list,
            code_sep=self.code_sep,
            codes_pre_o_mask=self.codes_pre_o_mask,
            codes_pre_suf_mask=self.codes_pre_suf_mask,
            subtask=self.subtask
        )
        
        if df_pred_train.shape[0] == 0:
            p_train = r_train = f1_train = 0
        else:
            # Adapt to CodiEsp format
            df_pred_train['label_pred'] = self.type_ann
            df_pred_train['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train.iterrows()]
            df_pred_train = df_pred_train.rename(columns={'code_pred': 'code'})
            df_pred_train = df_pred_train[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            
            p_train, r_train, f1_train = calculate_codiesp_x_metrics(
                df_gs=self.df_train_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_train, valid_codes=self.valid_codes)
            )
            
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        ### Val data
        y_pred_val_cls = self.model.predict(self.X_val)
        
        df_pred_val = cls_code_norm_preds_brat_format(
            y_pred_cls=y_pred_val_cls, 
            df_pred_ner=self.df_pred_ner_val, 
            code_decoder_list=self.label_decoder_list,
            code_sep=self.code_sep,
            codes_pre_o_mask=self.codes_pre_o_mask,
            codes_pre_suf_mask=self.codes_pre_suf_mask,
            subtask=self.subtask
        )
        
        if df_pred_val.shape[0] == 0:
            p_val = r_val = f1_val = 0
        else:
            # Adapt to CodiEsp format
            df_pred_val['label_pred'] = self.type_ann
            df_pred_val['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val.iterrows()]
            df_pred_val = df_pred_val.rename(columns={'code_pred': 'code'})
            df_pred_val = df_pred_val[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            
            p_val, r_val, f1_val = calculate_codiesp_x_metrics(
                df_gs=self.df_val_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_val, valid_codes=self.valid_codes)
            )
            
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s' % 
          (str(p_train),str(r_train),str(f1_train),
           str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val >= self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

        
        
class EarlyCodiEspX_IOB_Multi(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, valid_codes, patience=10, strategy="word-all", 
                 logits=True, n_output=1, type_tokenizer='transformers', type_ann='DIAGNOSTICO'):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.patience = patience
        self.strategy = strategy
        self.logits = logits
        self.n_output = n_output
        self.type_tokenizer = type_tokenizer
        self.type_ann = type_ann
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
        
        ## Diagnosis    
        df_pred_train_diag = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train[:1], 
                                            start_end=self.train_start_end, word_id=self.train_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=train_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
        # Adapt to CodiEsp format
        df_pred_train_diag['label_pred'] = self.type_ann
        df_pred_train_diag['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train_diag.iterrows()]
        df_pred_train_diag['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
        df_pred_train_diag = df_pred_train_diag[['clinical_case', 'pos_pred', 'label_pred', 'code']]
        
        ## Procedure
        df_pred_train_proc = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train[1:], 
                                            start_end=self.train_start_end, word_id=self.train_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=train_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
        # Adapt to CodiEsp format
        df_pred_train_proc['label_pred'] = self.type_ann
        df_pred_train_proc['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train_proc.iterrows()]
        df_pred_train_proc['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
        df_pred_train_proc = df_pred_train_proc[['clinical_case', 'pos_pred', 'label_pred', 'code']]
        
        df_pred_train = pd.concat([df_pred_train_diag, df_pred_train_proc])
        p_train_ner, r_train_ner, f1_train_ner = calculate_codiesp_ner_metrics(
            df_gs=self.df_train_gs, 
            df_pred=format_codiesp_x_pred_df(df_run=df_pred_train, valid_codes=self.valid_codes)
        )

        logs['p_ner'] = p_train_ner
        logs['r_ner'] = r_train_ner
        logs['f1_ner'] = f1_train_ner
        
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        ## Diagnosis    
        df_pred_val_diag = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val[:1], 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=val_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
        # Adapt to CodiEsp format
        df_pred_val_diag['label_pred'] = self.type_ann
        df_pred_val_diag['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val_diag.iterrows()]
        df_pred_val_diag['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
        df_pred_val_diag = df_pred_val_diag[['clinical_case', 'pos_pred', 'label_pred', 'code']]
        
        ## Procedure    
        df_pred_val_proc = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val[1:], 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask='ner', crf_mask_seq_len=val_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
        # Adapt to CodiEsp format
        df_pred_val_proc['label_pred'] = self.type_ann
        df_pred_val_proc['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val_proc.iterrows()]
        df_pred_val_proc['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
        df_pred_val_proc = df_pred_val_proc[['clinical_case', 'pos_pred', 'label_pred', 'code']]
        
        df_pred_val = pd.concat([df_pred_val_diag, df_pred_val_proc])
        p_val_ner, r_val_ner, f1_val_ner = calculate_codiesp_ner_metrics(
            df_gs=self.df_val_gs, 
            df_pred=format_codiesp_x_pred_df(df_run=df_pred_val, valid_codes=self.valid_codes)
        )

        logs['p_val_ner'] = p_val_ner
        logs['r_val_ner'] = r_val_ner
        logs['f1_val_ner'] = f1_val_ner

        print('\rp_ner: %s - r_ner: %s - f1_ner: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s' % 
          (str(p_train_ner),str(r_train_ner),str(f1_train_ner),
           str(p_val_ner),str(r_val_ner),str(f1_val_ner)),end=100*' '+'\n')
        
        # Early-stopping
        if (f1_val_ner >= self.best):
            self.best = f1_val_ner
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


        
class EarlyCodiEspX_IOB_Multi_Weight(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring NER F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, text_train, text_val, text_col, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, valid_codes, patience=10, strategy="word-all", 
                 logits=True, n_output=2, i_output=0, subtask='ner', type_tokenizer='transformers', 
                 type_ann='DIAGNOSTICO'):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.text_train = text_train
        self.text_val = text_val
        self.text_col = text_col
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.patience = patience
        self.strategy = strategy
        self.logits = logits
        self.n_output = n_output
        self.i_output = i_output # 0 indicates diagnostic, 1 procedure
        self.subtask = subtask
        self.type_tokenizer = type_tokenizer
        self.type_ann = type_ann
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
            
        df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, 
                                            preds=[y_pred_train[self.i_output]], 
                                            start_end=self.train_start_end, word_id=self.train_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_train, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=train_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
        if df_pred_train.shape[0] == 0:
            p_train_ner = r_train_ner = f1_train_ner = 0
        else:
            # Adapt to CodiEsp format
            df_pred_train['label_pred'] = self.type_ann
            df_pred_train['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_train.iterrows()]
            df_pred_train['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
            df_pred_train = df_pred_train[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NER metrics
            p_train_ner, r_train_ner, f1_train_ner = calculate_codiesp_ner_metrics(
                df_gs=self.df_train_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_train, valid_codes=self.valid_codes)
            )

        logs['p_ner'] = p_train_ner
        logs['r_ner'] = r_train_ner
        logs['f1_ner'] = f1_train_ner
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, 
                                            preds=[y_pred_val[self.i_output]], 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=self.text_val, text_col=self.text_col, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            type_tokenizer=self.type_tokenizer)
            
        if df_pred_val.shape[0] == 0:
            p_val_ner = r_val_ner = f1_val_ner = 0
        else:
            # Adapt to CodiEsp format
            df_pred_val['label_pred'] = self.type_ann
            df_pred_val['pos_pred'] = [str(row['start']) + ' ' + str(row['end']) for index, row in df_pred_val.iterrows()]
            df_pred_val['code'] = 'n23' if self.type_ann == 'DIAGNOSTICO' else 'bn20' # example of valid code
            df_pred_val = df_pred_val[['clinical_case', 'pos_pred', 'label_pred', 'code']]
            # Compute NER metrics
            p_val_ner, r_val_ner, f1_val_ner = calculate_codiesp_ner_metrics(
                df_gs=self.df_val_gs, 
                df_pred=format_codiesp_x_pred_df(df_run=df_pred_val, valid_codes=self.valid_codes)
            )

        logs['p_val_ner'] = p_val_ner
        logs['r_val_ner'] = r_val_ner
        logs['f1_val_ner'] = f1_val_ner

        print('\rp_ner: %s - r_ner: %s - f1_ner: %s | val_p_ner: %s - val_r_ner: %s - val_f1_ner: %s' % 
          (str(p_train_ner),str(r_train_ner),str(f1_train_ner),
           str(p_val_ner),str(r_val_ner),str(f1_val_ner)),end=100*' '+'\n')
        
        # Early-stopping
        if (f1_val_ner >= self.best):
            self.best = f1_val_ner
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



class EarlyAnon(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring F1 metric on validation dataset.
    Both training and validation P, R, F1 values are reported at the end of each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, label_decoder_list, 
                 train_doc_list, val_doc_list, train_start_end, val_start_end, train_word_id, val_word_id, 
                 df_train_gs, df_val_gs, patience=10, strategy="word-all", subtask="ner", 
                 logits=True, n_output=1, mention_strat='max'):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.label_decoder_list = label_decoder_list
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_start_end = train_start_end
        self.val_start_end = val_start_end
        self.train_word_id = train_word_id
        self.val_word_id = val_word_id
        self.df_train_gs = df_train_gs
        self.df_val_gs = df_val_gs
        self.patience = patience
        self.strategy = strategy
        self.subtask = subtask
        self.logits = logits
        self.n_output = n_output
        self.mention_strat = mention_strat
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_train = [y_pred_train]
        if self.strategy.split('-')[-1] != "crf":
            train_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_train[lab_i] = tf.nn.softmax(logits=y_pred_train[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            train_crf_mask_seq_len = y_pred_train[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_train[lab_i] = y_pred_train[lab_i][0]
        
        # Columns: clinical case, start, end, code_pred
        df_pred_train = ner_preds_brat_format(doc_list=self.train_doc_list, fragments=self.frag_train, preds=y_pred_train, 
                                              start_end=self.train_start_end, word_id=self.train_word_id, 
                                              lab_decoder_list=self.label_decoder_list, 
                                              df_text=None, text_col=None, strategy=self.strategy, 
                                              subtask=self.subtask, crf_mask_seq_len=train_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat)
        
        # To avoid errors: in the first epochs, the predicted codes are not valid, so ignore those predictions
        if df_pred_train.shape[0] == 0:
            print("Corrupted train predictions!")
            p_train, r_train, f1_train = .0, .0, .0
        else:
            _, p_train, r_train, f1_train = calculate_anon_metrics(gs=self.df_train_gs, pred=df_pred_train)
        logs['p'] = p_train
        logs['r'] = r_train
        logs['f1'] = f1_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1: # single output tensor
            y_pred_val = [y_pred_val]
        if self.strategy.split('-')[-1] != "crf":
            val_crf_mask_seq_len = None
            if self.logits:
                # Multiple labels
                for lab_i in range(self.n_output):
                    y_pred_val[lab_i] = tf.nn.softmax(logits=y_pred_val[lab_i], axis=-1).numpy()
        else:
            # when using multiple output tensors, crf_mask_seq_len is assumed to be the same
            # for all outputs (this may be changed when implementing "mention-first" approach)
            val_crf_mask_seq_len = y_pred_val[0][-2] # the crf_mask_seq_len of the first output tensor
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = y_pred_val[lab_i][0]
            
        df_pred_val = ner_preds_brat_format(doc_list=self.val_doc_list, fragments=self.frag_val, preds=y_pred_val, 
                                            start_end=self.val_start_end, word_id=self.val_word_id, 
                                            lab_decoder_list=self.label_decoder_list, 
                                            df_text=None, text_col=None, strategy=self.strategy, 
                                            subtask=self.subtask, crf_mask_seq_len=val_crf_mask_seq_len, 
                                            mention_strat=self.mention_strat)
        
        # To avoid errors: in the first epochs, the predicted codes are not valid, so ignore those predictions
        if df_pred_val.shape[0] == 0:
            print("Corrupted val predictions!")
            p_val, r_val, f1_val = .0, .0, .0
        else:
            _, p_val, r_val, f1_val = calculate_anon_metrics(gs=self.df_val_gs, pred=df_pred_val)
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val
        
        print('\rp: %s - r: %s - f1: %s | val_p: %s - val_r: %s - val_f1: %s' % 
              (str(p_train),str(r_train),str(f1_train),
               str(p_val),str(r_val),str(f1_val)),end=100*' '+'\n')
            
        
        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


# Text classification
class EarlyMAP(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring MAP metric on both training and validation datasets.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, train_doc_list, val_doc_list, 
                 patience=10, label_encoder_cls=None, train_gs_file_text=None, 
                 val_gs_file_text=None, train_pred_file_text=None, val_pred_file_text=None, 
                 valid_codes=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.patience = patience
        self.label_encoder_cls = label_encoder_cls
        self.train_gs_file_text = train_gs_file_text
        self.val_gs_file_text = val_gs_file_text
        self.train_pred_file_text = train_pred_file_text
        self.val_pred_file_text = val_pred_file_text
        self.valid_codes = valid_codes
        
        
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None
    
    
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        
        df_text_pred_train = max_fragment_prediction(y_frag_pred=y_pred_train, n_fragments=self.frag_train, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.train_doc_list)
        map_train = round(compute_map(valid_codes=self.valid_codes, pred=df_text_pred_train, gs_out_path=self.train_gs_file_text, 
                                      pred_out_path=self.train_pred_file_text), 3)
        
        logs['map'] = map_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        
        df_text_pred_val = max_fragment_prediction(y_frag_pred=y_pred_val, n_fragments=self.frag_val, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.val_doc_list)
        map_val = round(compute_map(valid_codes=self.valid_codes, pred=df_text_pred_val, gs_out_path=self.val_gs_file_text, 
                                    pred_out_path=self.val_pred_file_text), 3)
        
        logs['val_map'] = map_val
            
            
        print('\rmap: %s | val_map: %s' % 
            (str(map_train),str(map_val)),end=100*' '+'\n')
        
        
        # Early-stopping
        if (map_val > self.best):
            self.best = map_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

        
class MAP_Horov(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring MAP metric on both training and validation datasets.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, train_doc_list, val_doc_list, 
                 label_encoder_cls=None, train_gs_file_text=None, 
                 val_gs_file_text=None, train_pred_file_text=None, val_pred_file_text=None, 
                 valid_codes=None):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.label_encoder_cls = label_encoder_cls
        self.train_gs_file_text = train_gs_file_text
        self.val_gs_file_text = val_gs_file_text
        self.train_pred_file_text = train_pred_file_text
        self.val_pred_file_text = val_pred_file_text
        self.valid_codes = valid_codes
        
        
    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## P, R, F1
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        
        df_text_pred_train = max_fragment_prediction(y_frag_pred=y_pred_train, n_fragments=self.frag_train, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.train_doc_list)
        map_train = round(compute_map(valid_codes=self.valid_codes, pred=df_text_pred_train, gs_out_path=self.train_gs_file_text, 
                                      pred_out_path=self.train_pred_file_text), 3)
        
        logs['map'] = map_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        
        df_text_pred_val = max_fragment_prediction(y_frag_pred=y_pred_val, n_fragments=self.frag_val, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.val_doc_list)
        map_val = round(compute_map(valid_codes=self.valid_codes, pred=df_text_pred_val, gs_out_path=self.val_gs_file_text, 
                                    pred_out_path=self.val_pred_file_text), 3)
        
        logs['val_map'] = map_val
            
            
        print('\rmap: %s | val_map: %s' % 
            (str(map_train),str(map_val)),end=100*' '+'\n')

        

## NER-ANN Embeddings

NER_ANN_VOCAB_SIZE = 2

# Custom BERT model

from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_tf_utils import shape_list, get_initializer

class TFBertEmbeddings_NerAnnEmb(tf.keras.layers.Layer):
    """Construct the embeddings from word, position, token_type and ner_ann embeddings.""" # Guille

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.ner_ann_vocab_size = NER_ANN_VOCAB_SIZE # Guille
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.embeddings_sum = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        
        with tf.name_scope("ner_ann_embeddings"): # Guille
            self.ner_ann_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.ner_ann_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        ner_ann_ids: tf.Tensor = None, # Guille
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        
        if ner_ann_ids is None: # Guille
            ner_ann_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        ner_ann_embeds = tf.gather(params=self.ner_ann_embeddings, indices=ner_ann_ids) # Guille
        final_embeddings = self.embeddings_sum(inputs=[inputs_embeds, position_embeds, token_type_embeds, 
                                                       ner_ann_embeds]) # Guille
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


from transformers.modeling_tf_utils import input_processing, keras_serializable, TFModelInputType
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling
from transformers.models.bert.modeling_tf_bert import TFBertEncoder, TFBertPooler
from typing import Optional, Tuple, Union

@keras_serializable
class TFBertMainLayer_NerAnnEmb(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFBertEmbeddings_NerAnnEmb(config, name="embeddings") # Guille
        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        ner_ann_ids: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            ner_ann_ids=ner_ann_ids, # Guille
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(dims=input_shape, value=0)
        
        if inputs["ner_ann_ids"] is None: # Guille
            inputs["ner_ann_ids"] = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            token_type_ids=inputs["token_type_ids"],
            ner_ann_ids=inputs["ner_ann_ids"], # Guille
            inputs_embeds=inputs["inputs_embeds"],
            training=inputs["training"],
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(inputs["attention_mask"], (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not inputs["return_dict"]:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


from transformers import TFBertPreTrainedModel
from transformers.modeling_tf_utils import TFTokenClassificationLoss
from transformers.modeling_tf_outputs import TFTokenClassifierOutput

class TFBertForTokenClassification_NerAnnEmb(TFBertPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer_NerAnnEmb(config, add_pooling_layer=False, name="bert") # Guille
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
    
    # Guille
    """
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    """
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        ner_ann_ids: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            ner_ann_ids=ner_ann_ids, # Guille
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            ner_ann_ids=inputs["ner_ann_ids"], # Guille
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(inputs=sequence_output, training=inputs["training"])
        logits = self.classifier(inputs=sequence_output)
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTokenClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)



# Custom XLM-R model

class TFRobertaEmbeddings_NerAnnEmb(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 1
        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.ner_ann_vocab_size = NER_ANN_VOCAB_SIZE # Guille
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.embeddings_sum = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("ner_ann_embeddings"): # Guille
            self.ner_ann_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.ner_ann_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = tf.math.cumsum(mask, axis=1) * mask

        return incremental_indices + self.padding_idx

    def call(self, input_ids=None, position_ids=None, token_type_ids=None, ner_ann_ids=None, # Guille
             inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if ner_ann_ids is None: # Guille
            ner_ann_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids=input_ids)
            else:
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )
                position_ids = tf.tile(input=position_ids, multiples=(input_shape[0], 1))

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        ner_ann_embeds = tf.gather(params=self.ner_ann_embeddings, indices=ner_ann_ids) # Guille
        final_embeddings = self.embeddings_sum(inputs=[inputs_embeds, position_embeds, token_type_embeds, 
                                                       ner_ann_embeds]) # Guille
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_tf_roberta import TFRobertaEncoder, TFRobertaPooler

@keras_serializable
class TFRobertaMainLayer_NerAnnEmb(tf.keras.layers.Layer):
    config_class = RobertaConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.encoder = TFRobertaEncoder(config, name="encoder")
        self.pooler = TFRobertaPooler(config, name="pooler") if add_pooling_layer else None
        # The embeddings must be the last declaration in order to follow the weights order
        self.embeddings = TFRobertaEmbeddings_NerAnnEmb(config, name="embeddings") # Guille

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer.get_input_embeddings
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer.set_input_embeddings
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer.call
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        ner_ann_ids: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            ner_ann_ids=ner_ann_ids, # Guille
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(tensor=inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(tensor=inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(dims=input_shape, value=0)
        
        if inputs["ner_ann_ids"] is None: # Guille
            inputs["ner_ann_ids"] = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            token_type_ids=inputs["token_type_ids"],
            ner_ann_ids=inputs["ner_ann_ids"], # Guille
            inputs_embeds=inputs["inputs_embeds"],
            training=inputs["training"],
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(inputs["attention_mask"], (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not inputs["return_dict"]:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


from transformers.models.roberta.modeling_tf_roberta import TFRobertaPreTrainedModel

class TFRobertaForTokenClassification_NerAnnEmb(TFRobertaPreTrainedModel, TFTokenClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer_NerAnnEmb(config, add_pooling_layer=False, name="roberta") # Guille
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
    
    # Guille
    """
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    """
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        ner_ann_ids=None, # Guille
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            ner_ann_ids=ner_ann_ids, # Guille
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.roberta(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            ner_ann_ids=inputs["ner_ann_ids"], # Guille
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=inputs["training"])
        logits = self.classifier(sequence_output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForTokenClassification.serving_output
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTokenClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig

class TFXLMRobertaForTokenClassification_NerAnnEmb(TFRobertaForTokenClassification_NerAnnEmb):
    """
    This class overrides :class:`~transformers.TFRobertaForTokenClassification_NerAnnEmb`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """ # Guille

    config_class = XLMRobertaConfig

    

## "Mentions"-attention at the final layer

from transformers.models.bert.modeling_tf_bert import TFBertLayer, TFBertEmbeddings
from transformers.modeling_tf_outputs import TFBaseModelOutput

class TFBertEncoder_Mention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TFBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        mention_attention_mask: tf.Tensor, # Guille
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # First 11 layers (Guille)
        for i, layer_module in enumerate(self.layer[:len(self.layer) - 1]): # Guille
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Last 12th layer (Guille)
        i = len(self.layer) - 1
        layer_module = self.layer[i]
        
        layer_outputs = layer_module(
            hidden_states=hidden_states,
            attention_mask=mention_attention_mask, # Guille
            head_mask=head_mask[i],
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


@keras_serializable
class TFBertMainLayer_Mention(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFBertEmbeddings(config, name="embeddings")
        self.encoder = TFBertEncoder_Mention(config, name="encoder") # Guille
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        mention_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mention_attention_mask=mention_attention_mask, # Guille
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        if inputs["mention_attention_mask"] is None: # Guille
            inputs["mention_attention_mask"] = tf.fill(dims=input_shape, value=1)

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            token_type_ids=inputs["token_type_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            training=inputs["training"],
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(inputs["attention_mask"], (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        
        # Mention attention mask (Guille)
        extended_mention_attention_mask = tf.reshape(inputs["mention_attention_mask"], 
                                                     (input_shape[0], 1, 1, input_shape[1]))
        extended_mention_attention_mask = tf.cast(extended_mention_attention_mask, dtype=embedding_output.dtype)
        extended_mention_attention_mask = tf.multiply(tf.subtract(one_cst, extended_mention_attention_mask), 
                                                      ten_thousand_cst)
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            mention_attention_mask=extended_mention_attention_mask,
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not inputs["return_dict"]:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFBertForClassification_Mention(TFBertPreTrainedModel):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer_Mention(config, add_pooling_layer=False, name="bert") # Guille


    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        mention_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mention_attention_mask=mention_attention_mask, # Guille
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mention_attention_mask=inputs["mention_attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        
        return outputs



## "Mentions"-embedding at the final layer

NER_ANN_NORM = False

class TFNerAnnEmb(tf.keras.layers.Layer):
    """Construct the embeddings for ner_ann embeddings."""

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.ner_ann_vocab_size = NER_ANN_VOCAB_SIZE
        self.norm = NER_ANN_NORM
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.embeddings_sum = tf.keras.layers.Add()
        if self.norm:
            self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
            self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("ner_ann_embeddings"):
            self.ner_ann_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.ner_ann_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def call(
        self,
        ner_ann_ids: tf.Tensor, # Guille
        inputs_embeds: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        
        ner_ann_embeds = tf.gather(params=self.ner_ann_embeddings, indices=ner_ann_ids)
        final_embeddings = self.embeddings_sum(inputs=[inputs_embeds, ner_ann_embeds])
        if self.norm:
            final_embeddings = self.LayerNorm(inputs=final_embeddings)
            final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
    

class TFBertEncoder_MentionEmb(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        
        self.layer = [TFBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        self.emb_layer = TFNerAnnEmb(config=config) # Guile

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        ner_ann_ids: tf.Tensor, # Guille
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # First 11 layers (Guille)
        for i, layer_module in enumerate(self.layer[:len(self.layer) - 1]): # Guille
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Last 12th layer (Guille)
        i = len(self.layer) - 1
        layer_module = self.layer[i]
        
        hidden_emb_states = self.emb_layer(
            ner_ann_ids=ner_ann_ids,
            inputs_embeds=hidden_states
        )
        
        layer_outputs = layer_module(
            hidden_states=hidden_emb_states, # Guille
            attention_mask=attention_mask, # Guille
            head_mask=head_mask[i],
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


@keras_serializable
class TFBertMainLayer_MentionEmb(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFBertEmbeddings(config, name="embeddings")
        self.encoder = TFBertEncoder_MentionEmb(config, name="encoder") # Guille
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        ner_ann_ids: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            ner_ann_ids=ner_ann_ids, # Guille
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        if inputs["ner_ann_ids"] is None: # Guille
            inputs["ner_ann_ids"] = tf.fill(dims=input_shape, value=0)

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            token_type_ids=inputs["token_type_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            training=inputs["training"],
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(inputs["attention_mask"], (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            ner_ann_ids=inputs["ner_ann_ids"], # Guille
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not inputs["return_dict"]:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFBertForClassification_MentionEmb(TFBertPreTrainedModel):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"mlm___cls",
        r"nsp___cls",
        r"cls.predictions",
        r"cls.seq_relationship",
    ]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer_MentionEmb(config=config,
                                               add_pooling_layer=False, 
                                               name="bert") # Guille


    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        ner_ann_ids: Optional[Union[np.ndarray, tf.Tensor]] = None, # Guille
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            ner_ann_ids=ner_ann_ids, # Guille
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            ner_ann_ids=inputs["ner_ann_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        
        return outputs