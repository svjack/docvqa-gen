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
