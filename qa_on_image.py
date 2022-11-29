import sys
import os
import pandas as pd
import numpy as np
import shutil

import pandas as pd
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import re

from easynmt import EasyNMT
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

#ppstructure_bin_path = "/Users/svjack/opt/anaconda3/envs/py39_cp/bin/python"
ppstructure_bin_path = sys.executable
assert os.path.exists(ppstructure_bin_path)

#donut_proj_path = "/Users/svjack/temp/donut-master/"
donut_proj_path = "donut"
sys.path.insert(0 ,donut_proj_path)

from train import *
en_model_path = "model/en_rev_result_overfit_v0"
zh_model_path = "model/zh_rev_result_overfit"
assert os.path.exists(en_model_path)
assert os.path.exists(zh_model_path)

from PIL import Image

from donut import DonutModel
import torch

import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
import json
from tqdm import tqdm

def demo_process_vqa(input_img, question, lang):
    #global pretrained_model, task_prompt, task_name
    global zh_pretrained_model, en_pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    user_prompt = task_prompt.replace("{user_input}", question)
    if lang == "en":
        output = en_pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]
    else:
        output = zh_pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]
    return output

task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
en_pretrained_model = DonutModel.from_pretrained(en_model_path)
zh_pretrained_model = DonutModel.from_pretrained(zh_model_path)

if torch.cuda.is_available():
    en_pretrained_model.half()
    device = torch.device("cuda")
    en_pretrained_model.to(device)

if torch.cuda.is_available():
    zh_pretrained_model.half()
    device = torch.device("cuda")
    zh_pretrained_model.to(device)

en_pretrained_model.eval()
zh_pretrained_model.eval()
print("have load !")

def read_img_to_3d_array(img_path):
    img = Image.open(img_path)
    img_array = np.asarray(img)
    if len(img_array.shape) == 3:
        #img_array = img_array[:, :, :3]
        pass
    else:
        assert len(img_array.shape) == 2
        h, w = img_array.shape
        img_array = img_array.reshape([h, w, 1])
        img_array = np.concatenate([img_array] * 3, axis = -1)
    assert len(img_array.shape) == 3
    img_array = img_array[:, :, :3]
    return img_array

def add_question_to_answer_df(img_path ,answer_df, lang,
unique_answer = True, unique_question = True):
    assert os.path.exists(img_path)
    assert hasattr(answer_df, "size")
    img_array = read_img_to_3d_array(img_path)

    req = []
    an_set = set([])
    for i, r in tqdm(answer_df.iterrows()):
        d = r.to_dict()
        answer = d["transcription"]
        if unique_answer:
            if answer not in an_set:
                an_set.add(answer)
            else:
                continue
        qa_d = demo_process_vqa(img_array, answer, lang)
        pred_question = qa_d["answer"]
        d["question"] = pred_question
        req.append(d)
    df = pd.DataFrame(req).rename(columns = {"transcription": "answer"})
    if unique_question:
        req = []
        q_set = set([])
        for i, r in df.iterrows():
            if r["question"] not in q_set:
                q_set.add(r["question"])
                req.append(r)
        df = pd.DataFrame(req)
    if not df.size:
        return None
    #req_df = pd.DataFrame(req)
    df = df[["question", "answer"]]
    df["source"] = ["image"] * len(df)
    return df

def generate_qa_on_image(img_path, ppstructure_bin_path):
    cmd_format = '''{} predict_system.py \
      --kie_algorithm=LayoutXLM \
      --ser_model_dir=./inference/ser_vi_layoutxlm_xfund_infer \
      --image_dir={} \
      --ser_dict_path=../ppocr/utils/dict/kie_dict/xfund_class_list.txt \
      --vis_font_path=../doc/fonts/simfang.ttf \
      --ocr_order_method="tb-yx" \
      --mode=kie
    '''
    cmd_format = '''{} ppstructure/predict_system.py \
      --kie_algorithm=LayoutXLM \
      --ser_model_dir=ppstructure/inference/ser_vi_layoutxlm_xfund_infer \
      --image_dir={} \
      --ser_dict_path=ppocr/utils/dict/kie_dict/xfund_class_list.txt \
      --vis_font_path=doc/fonts/simfang.ttf \
      --ocr_order_method="tb-yx" \
      --mode=kie
    '''

    output_path = "output"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    cmd = cmd_format.format(ppstructure_bin_path, img_path)
    os.system(cmd)

    output_file_path = os.path.join(output_path, "kie/{}/res_0_kie.txt".format(img_path.split("/")[-1].split(".")[0]))
    assert os.path.exists(output_file_path)

    res_df = pd.read_csv(output_file_path, sep = "\t", header = None)
    res_df = pd.DataFrame(pd.DataFrame(json.loads(res_df.iloc[0, 1]))["ocr_info"].values.tolist())
    if res_df.size == 0:
        return None
    all_ans_token_list = res_df["transcription"].drop_duplicates().values.tolist()
    lang = pd.Series([" ".join(all_ans_token_list)]).map(detect_language).value_counts().index.tolist()[0]
    if lang != "zh":
        lang = "en"

    answer_df = res_df[res_df["pred"] == "ANSWER"]
    if lang == "zh":
        answer_df = answer_df[answer_df["transcription"].map(lambda x: len(x) <= 10)]
    else:
        answer_df = answer_df[answer_df["transcription"].map(lambda x: len(x) <= 30)]
    if answer_df.size == 0:
        return None
    assert answer_df.size > 0
    answer_df["transcription"] = answer_df["transcription"].map(lambda x: x.split("：")[-1].split(":")[-1] if not re.findall(r"[0-9][:：][0-9]", x) else x)
    print("lang: ", lang)
    question_answer_df = add_question_to_answer_df(img_path, answer_df, lang)
    question_answer_df = question_answer_df[["question", "answer", "source"]]
    return question_answer_df
