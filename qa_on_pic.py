import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from qa_on_image import *
from qa_on_context import *

from paddlenlp import Taskflow
zh_docprompt = Taskflow("document_intelligence", lang="ch")
en_docprompt = Taskflow("document_intelligence", lang="en")

import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
import json
from tqdm import tqdm

# Chinese image
table_engine = PPStructure(recovery=True)

def zh_length_ratio(x):
    return sum(map(len ,re.findall(u"[\u4e00-\u9fa5]+", x))) / len(x) if len(x) > 0 else 0

def generate_qa_on_context(img_path, table_engine,
text_length_threshold = 50, prob_threshold = 0.7):
    #img = _decode_image(r["l"])[:, :, :3]
    img_array = read_img_to_3d_array(img_path)
    result = table_engine(img_array)
    #result = list(map(lambda d: dict(map(lambda t2: (t2[0], t2[1].tolist()) if hasattr(t2[1], "tolist") else t2, d.items())),result))
    result = list(map(lambda d: dict(
        filter(lambda x: x is not None ,map(lambda t2: None if hasattr(t2[1], "tolist") else t2, d.items()))
    ),result))
    result_df = pd.DataFrame(result)
    #text_df = result_df[result_df["type"] == "text"]
    text_df = result_df
    if not text_df.size:
        return None
    text_series = text_df["res"].map(lambda x: "".join(map(lambda y: y["text"],
    filter(lambda z: type(z) == type({}) and "confidence" in z and z["confidence"] >= prob_threshold ,x)
    )))
    if not text_series.size:
        return None
    text_series = text_series[text_series.map(lambda x: len(x) >= text_length_threshold)]
    if not text_series.size:
        return None

    lang = pd.Series([" ".join(text_series.values.tolist())]).map(detect_language).value_counts().index.tolist()[0]
    if lang != "zh":
        lang = "en"
    if lang == "zh":
        text_series = text_series[text_series.map(lambda x: zh_length_ratio(x) >= 0.6)]

    if not text_series.size:
        return None
    req = []
    for text in tqdm(text_series.values.tolist()):
        output = gen_qst_to_df(text)
        for ele in output:
            if hasattr(ele, "size"):
                req.append(ele)
    if not req:
        return None
    req_df = pd.concat(req, axis = 0)
    req_df = req_df[["question", "entity"]].rename(columns = {
        "entity": "answer"
    })
    req = []
    q_set = set([])
    for i, r in req_df.iterrows():
        if r["question"] not in q_set:
            q_set.add(r["question"])
            req.append(r)
    req_df = pd.DataFrame(req)
    req_df["source"] = ["context"] * len(req_df)
    return req_df

def run_qa_by_document_intelligence(img_path ,qa_df, prob_threshold = 0.7,
                                   drop_extract = True):
    qa_df = qa_df[["question", "answer", "source"]]
    lang = pd.Series([" ".join(qa_df["question"].values.tolist())]).map(detect_language).value_counts().index.tolist()[0]
    if lang != "zh":
        lang = "en"
    print("lang :" ,lang)
    if lang == "en":
        docprompt = en_docprompt
    else:
        docprompt = zh_docprompt

    pd_qa_df = pd.DataFrame(docprompt(
[{"doc": img_path, "prompt":qa_df["question"].values.tolist()}]
))
    pd_qa_df = pd_qa_df.explode("result")
    pd_qa_df = pd.concat([pd_qa_df.reset_index().iloc[:, 1:] ,pd.json_normalize(pd_qa_df["result"]).reset_index().iloc[:, 1:]],
         axis = 1)[["prompt", "value", "prob"]].rename(
    columns = {
        "prompt": "question"
    }
)
    pd_qa_cat_df = pd.merge(pd_qa_df, qa_df, on = "question")[["question", "value", "answer", "prob", "source"]].rename(
    columns = {
        "value": "pd_intelli_answer",
        "answer": "pd_extract_answer"
    }
)
    pd_qa_cat_df = pd_qa_cat_df.sort_values(by = "prob", ascending = False)
    if drop_extract:
        del pd_qa_cat_df["pd_extract_answer"]
    pd_qa_cat_df = pd_qa_cat_df[
    pd_qa_cat_df["prob"] >= prob_threshold
    ]
    return pd_qa_cat_df

def generate_qa(img_path, ppstructure_bin_path, table_engine, cate_limit_size = 5,
    prob_threshold = 0.2,
):
    if type(cate_limit_size) != type(0):
        cate_limit_size = 32
    image_qa_df = generate_qa_on_image(img_path, ppstructure_bin_path)
    context_qa_df = generate_qa_on_context(img_path, table_engine,)
    req = []
    if hasattr(image_qa_df, "size"):
        req.append(image_qa_df.head(cate_limit_size))
    if hasattr(context_qa_df, "size"):
        req.append(context_qa_df.head(cate_limit_size))
    if not req:
        return None
    question_answer_df = pd.concat(req, axis = 0)
    req = []
    q_set = set([])
    for i, r in question_answer_df.iterrows():
        if r["question"] not in q_set:
            q_set.add(r["question"])
            req.append(r)
    question_answer_df = pd.DataFrame(req)
    qa_df = run_qa_by_document_intelligence(img_path ,question_answer_df, prob_threshold = prob_threshold)
    return qa_df

'''
img_path = "imgs/f1b8b765-6b2a-11ed-b3dd-b360cf1d86b5.png"
img_path = "imgs/f8baba44-6b2a-11ed-b3dd-b360cf1d86b5.png"
Image.open(img_path)
qa_df = generate_qa(img_path, ppstructure_bin_path, table_engine)
'''


'''
### text2text install

### eng t2t
!sudo apt-get install libopenblas-dev -y
!sudo apt-get install libomp-dev -y
!pip install text2text
!pip install faiss-cpu

import text2text as t2t
t2t.Handler(["I like HuiLin Chen's song. [SEP] HuiLin Chen"], src_lang='en').question()

!ls -lah qg_model.bin

pip install easynmt

### paddle install
pip install paddlepaddle
pip install paddlenlp
pip install paddleocr

git clone https://github.com/PaddlePaddle/PaddleOCR
cd PaddleOCR/ppstructure
mkdir inference
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar && tar -xf ser_vi_layoutxlm_xfund_infer.tar
cd ..

### donut install
git clone https://github.com/clovaai/donut
cd donut
python setup.py install
pip install -U datasets

pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

pip install Polygon3
pip install sconf
#pip install lanms-neo

git clone https://github.com/gen-ko/lanms-neo
cd lanms-neo
python setup.py install

###
'''
