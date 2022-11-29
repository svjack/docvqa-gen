#### py39_cp_cp
from zh_mt5_model import *
from en_t2t_model import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import spacy
import pandas as pd
import numpy as np

import re
from tqdm import tqdm
from copy import deepcopy
import pathlib

import json
import pickle as pkl

from tqdm import tqdm
from easynmt import EasyNMT


#path = "svjack/prompt-extend-chinese"
### https://huggingface.co/svjack/squad_gen_qst_zh_v0
#path = "model/squad_gen_qst_zh_v0"
path = "svjack/squad_gen_qst_zh_v0"
#### upload to huggingface
asker_zh = T5_B(path,
    device = "cpu")

zh_nlp = spacy.load("zh_core_web_sm")
en_nlp = spacy.load("en_core_web_sm")

trans_model = EasyNMT('opus-mt')

def detect_language(text):
    assert type(text) == type("")
    # detect_list.append(trans_model.language_detection_fasttext(prompt))
    lang = trans_model.language_detection_fasttext(text)
    lang = lang.lower().strip()
    if "zh" not in lang and "en" not in lang:
        lang = "others"
    if "zh" in lang:
        lang = "zh"
    if "en" in lang:
        lang = "en"
    assert lang in ["en", "zh", "others"]
    return lang

def drop_duplicates_by_col(df, on_col = "aug_sparql_query"):
    assert hasattr(df, "size")
    assert on_col in df.columns.tolist()
    req = []
    set_ = set([])
    for i, r in df.iterrows():
        if r[on_col] not in set_:
            set_.add(r[on_col])
            req.append(r)
    return pd.DataFrame(req)

def sent_with_ents(sent, en_nlp):
    assert type(sent) == type("")
    doc = en_nlp(sent)
    return (sent, pd.Series(doc.ents).map(
    lambda span: (span.text, span.label_)
).values.tolist())

def gen_ask_by_span_zh(asker ,sent, span):
    if type(span) == type(""):
        span = [span]
    if not span:
        return []
    sent = sent.replace("|", "")
    span = list(map(lambda x: x.replace("|", ""), span))
    x = list(map(lambda x: "{}|{}".format(sent, x), span))
    return list(map(
        lambda y: asker.predict(y)
        , x))

#### list return
def gen_ask_by_span(asker, sent, span, lang):
    assert lang in ["en", "zh"]
    if lang == "zh":
        return gen_ask_by_span_zh(asker ,sent, span)
    else:
        return gen_ask_by_span_en(t2t, sent, span)


def filter_ent_cate(ent_list, maintain_cate_list = [
    "DATE", "FAC", "GPE", "LOC", "PERSON"
]):
    if not ent_list:
        return []
    return list(filter(lambda t2: t2[1] in maintain_cate_list, ent_list))

def batch_as_list(a, batch_size = int(100000)):
    req = []
    for ele in a:
        if not req:
            req.append([])
        if len(req[-1]) < batch_size:
            req[-1].append(ele)
        else:
            req.append([])
            req[-1].append(ele)
    return req

def gen_qst_to_df(paragraph,
                nlp = zh_nlp,
                asker = asker_zh,
                nlp_input = None,
                maintain_cate_list = [
"DATE", "FAC", "GPE", "LOC", "PERSON"
    ], limit_ents_size = 10, batch_size = 4
                      ):
    if limit_ents_size is None:
        limit_ents_size = 10000
    assert type(paragraph) == type("")
    lang = detect_language(paragraph)
    if lang != "zh":
        lang = "en"
    nlp = en_nlp if lang == "en" else zh_nlp

    if nlp_input is None:
        _, entity_list = sent_with_ents(paragraph, nlp)
    else:
        _, entity_list = deepcopy(nlp_input)
    if maintain_cate_list:
        entity_list = filter_ent_cate(entity_list, maintain_cate_list = maintain_cate_list)
    entity_list = entity_list[:limit_ents_size]
    if not entity_list:
        return None
    l = batch_as_list(entity_list, batch_size)
    for ele in tqdm(l):
        ents = list(map(lambda x: x[0], ele))
        ent_cates = list(map(lambda x: x[1], ele))
        #questions = gen_ask_by_span_zh(asker, paragraph, ents)
        questions = gen_ask_by_span(asker, paragraph, ents, lang)
        assert len(ele) == len(ent_cates) == len(questions)
        #return [ele, ent_cates, questions, ans]
        batch_l = list(map(pd.Series, [ents, ent_cates, questions]))
        batch_df = pd.concat(batch_l, axis = 1)
        batch_df.columns = ["entity", "entity_cate", "question",]
        yield batch_df

'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import spacy
import pandas as pd
import numpy as np

from haystack.nodes import FARMReader

from simplet5 import SimpleT5
from haystack.utils import SquadData
import re
import seaborn as sns
from tqdm import tqdm
from fuzzywuzzy import fuzz
from copy import deepcopy
import pathlib

import json
#import jionlp as jio

import pickle as pkl

import sqlite_utils
from tqdm import tqdm

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pkl.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


asker_1 = SimpleT5()
asker_1.load_model(
    "mt5",
    "/Users/svjack/temp/haystack_on_wiki/model/squad_gen_qst/simplet5-epoch-0-train-loss-1.8256-val-loss-2.0754/",
    use_gpu = False
)


zh_nlp = spacy.load("zh_core_web_sm")

def drop_duplicates_by_col(df, on_col = "aug_sparql_query"):
    assert hasattr(df, "size")
    assert on_col in df.columns.tolist()
    req = []
    set_ = set([])
    for i, r in df.iterrows():
        if r[on_col] not in set_:
            set_.add(r[on_col])
            req.append(r)
    return pd.DataFrame(req)

def sent_with_ents(sent, en_nlp):
    assert type(sent) == type("")
    doc = en_nlp(sent)
    return (sent, pd.Series(doc.ents).map(
    lambda span: (span.text, span.label_)
).values.tolist())

def gen_ask_by_span(asker ,sent, span):
    if type(span) == type(""):
        span = [span]
    if not span:
        return []
    sent = sent.replace("|", "")
    span = list(map(lambda x: x.replace("|", ""), span))
    x = list(map(lambda x: "{}|{}".format(sent, x), span))
    return list(map(
        lambda y: asker.predict(y)[0]
        , x))

def ask_paragraph(reader, question, paragraphs_list):
    if type(paragraphs_list) == type(""):
        paragraphs_list = [paragraphs_list]
    prd = reader.predict_on_texts(
    question = question,
    texts = paragraphs_list
)
    query = prd["query"]
    no_ans_gap = prd["no_ans_gap"]
    answers = list(map(lambda x: x.to_dict() ,prd["answers"]))
    return {
        "query": query,
        "no_ans_gap": no_ans_gap,
        "answers": answers
    }

def filter_ent_cate(ent_list, maintain_cate_list = [
    "DATE", "FAC", "GPE", "LOC", "PERSON"
]):
    if not ent_list:
        return []
    return list(filter(lambda t2: t2[1] in maintain_cate_list, ent_list))

def batch_as_list(a, batch_size = int(100000)):
    req = []
    for ele in a:
        if not req:
            req.append([])
        if len(req[-1]) < batch_size:
            req[-1].append(ele)
        else:
            req.append([])
            req[-1].append(ele)
    return req

def gen_qst_to_df(paragraph,
                nlp = zh_nlp,
                asker = asker_1,
                nlp_input = None,
                maintain_cate_list = [
"DATE", "FAC", "GPE", "LOC", "PERSON"
    ], limit_ents_size = 10, batch_size = 4
                      ):
    if limit_ents_size is None:
        limit_ents_size = 10000
    assert type(paragraph) == type("")
    if nlp_input is None:
        _, entity_list = sent_with_ents(paragraph, nlp)
    else:
        _, entity_list = deepcopy(nlp_input)
    if maintain_cate_list:
        entity_list = filter_ent_cate(entity_list, maintain_cate_list = maintain_cate_list)
    entity_list = entity_list[:limit_ents_size]
    if not entity_list:
        return None
    l = batch_as_list(entity_list, batch_size)
    for ele in tqdm(l):
        ents = list(map(lambda x: x[0], ele))
        ent_cates = list(map(lambda x: x[1], ele))
        questions = gen_ask_by_span(asker, paragraph, ents)
        assert len(ele) == len(ent_cates) == len(questions)
        #return [ele, ent_cates, questions, ans]
        batch_l = list(map(pd.Series, [ents, ent_cates, questions]))
        batch_df = pd.concat(batch_l, axis = 1)
        batch_df.columns = ["entity", "entity_cate", "question",]
        yield batch_df

def ask_para_in_to_df(paragraph,
                      nlp = zh_nlp,
                      asker = asker_1,
                      reader = squad_reader,
                      nlp_input = None,
                      maintain_cate_list = [
    "DATE", "FAC", "GPE", "LOC", "PERSON"
], limit_ents_size = 10, batch_size = 4):
    if limit_ents_size is None:
        limit_ents_size = 10000
    assert type(paragraph) == type("")
    if nlp_input is None:
        _, entity_list = sent_with_ents(paragraph, nlp)
    else:
        _, entity_list = deepcopy(nlp_input)
    if maintain_cate_list:
        entity_list = filter_ent_cate(entity_list, maintain_cate_list = maintain_cate_list)
    entity_list = entity_list[:limit_ents_size]
    if not entity_list:
        return None
    l = batch_as_list(entity_list, batch_size)
    for ele in tqdm(l):
        ents = list(map(lambda x: x[0], ele))
        ent_cates = list(map(lambda x: x[1], ele))
        questions = gen_ask_by_span(asker, paragraph, ents)
        assert len(ele) == len(ent_cates) == len(questions)
        ans = []
        for qst in questions:
            a = ask_paragraph(reader ,qst, [paragraph])
            ans.append(a)
        assert len(ele) == len(ent_cates) == len(questions) == len(ans)
        #return [ele, ent_cates, questions, ans]
        batch_l = list(map(pd.Series, [ents, ent_cates, questions, ans]))
        batch_df = pd.concat(batch_l, axis = 1)
        batch_df.columns = ["entity", "entity_cate", "question", "answer"]
        yield batch_df

def ask_para_in_to_loop_to_final(paragraph,
                                 nlp_input = None,
                                 limit_ents_size = 100,
                                ea_sim_threshold = 50,
                                 maintain_cate_list = [
    "DATE", "FAC", "GPE", "LOC", "PERSON"
]
                                ):
    ask_gen = ask_para_in_to_df(paragraph, nlp_input = nlp_input, limit_ents_size = limit_ents_size,
                               maintain_cate_list = maintain_cate_list
                               )
    req = []
    for ele in ask_gen:
        req.append(ele)
    if not req:
        return None
    req_df = pd.concat(req, axis = 0)
    req_df["no_ans_gap"] = req_df["answer"].map(lambda x: x["no_ans_gap"])
    req_df["ans"] = req_df["answer"].map(lambda x: x["answers"]).map(
    lambda x: list(map(lambda y: (y["answer"] ,y["score"]), x))
)
    req_df = req_df.explode("ans").dropna()
    req_df["answer_text"] = req_df["ans"].map(lambda x: x[0])
    req_df["score"] = req_df["ans"].map(lambda x: x[1])
    req_df["entity_answer_sim"] = req_df.apply(lambda x: fuzz.ratio(x["entity"], x["answer_text"]), axis = 1)
    req_df = req_df[
    req_df["entity_answer_sim"] >= ea_sim_threshold
]
    if req_df.size == 0:
        return None
    req_df = req_df.sort_values(by = "score", ascending = False)[["entity", "entity_cate", "question", "no_ans_gap", "answer_text", "entity_answer_sim"]]
    req_df = drop_duplicates_by_col(req_df, on_col="question")
    return req_df

def retrieve_sent_split(sent,
                       stops_split_pattern = "|".join(map(lambda x: r"\{}".format(x),
                                                                 "。？"))
                       ):
    if not sent.strip():
        return []

    split_list = re.split(stops_split_pattern, sent)
    return split_list

def text_to_sent_list(text, sent_length_upper_bnd = 128, sent_length_lower_bnd = 8):
    sent_list = list(filter(lambda x: len(x) >= sent_length_lower_bnd and len(x) <= sent_length_upper_bnd,
    retrieve_sent_split(text)))
    if not sent_list:
        return []
    return sent_list

def length_join(l, length_threshold = 420):
    assert type(l) == type([])
    l = list(filter(lambda x: len(x) <= length_threshold, l))
    if len(l) <= 1:
        return []
    req = []
    now_str = ""
    for ele in l:
        if len(now_str) <= length_threshold:
            now_str = now_str + " " + ele
        else:
            req.append(now_str)
            now_str = ele
    req.append(now_str)
    #print(req)
    #req = list(filter(lambda x: len(x) >= length_threshold and len(x) <= length_threshold + flow_size, req))
    req_ = []
    for ele in req:
        if ele not in req_:
            req_.append(ele)
    return req_
'''
