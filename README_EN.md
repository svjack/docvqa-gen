<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">docvqa-gen</h3>

  <p align="center">
   		Question Answering dataset generator of Document Visual in English and Chinese
    <br />
  </p>
</p>

[中文介绍](README.md)

### Brief introduction
Nowadays, many question answering framework provide a function about make customize dataset generation to give a support of training model in the data dimension. For example, [primeqa](https://github.com/primeqa/primeqa) provide many toolkits about every kinds of question generation which its Multilingual Question Generation: Supports generation of questions for effective domain adaptation over tables and multilingual text. And [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/question_generation/unimo-text) also implement this function in Chinese NLP domain. Tracing the source, they may get inspiration from some frameworks like [text2text](https://github.com/artitw/text2text). And i also draw this idea from text2text in the early of 2022.4<br/>
As [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/question_generation/unimo-text) have release this function in not long ago, i think its time to release a demo project about promote this function to [OCR field](https://en.wikipedia.org/wiki/Optical_character_recognition), And give a interface about generate questions on image. This is the data composition of [DocVQA task](https://www.docvqa.org)

The core is to train a question generator on images, if someone provide a answer, this generator will give the question. This need is met by a OCR free framework named [donut](https://github.com/clovaai/donut), which in its DocVQA variation, it can perform question answer on image. In some situation, the performance of this model can not overcome some framework with the help of OCR.

Because, a all in one construction makes the generator error can not be well deal with in a isolated manner in different blocks. This may make the conclusion too relay on the training dataset, and bring challenges to image augmentation tasks in training this kinds for model.

But this does not affect the convenience to use [donut](https://github.com/clovaai/donut) to train a question generator. Because the decoder in donut use [hyunwoongko/asian-bart-ecjk](https://huggingface.co/hyunwoongko/asian-bart-ecjk) which makes this framework works with languages include English Chinese Japanese Korean, And the developers of this excellent model are mainly from South Korea. This makes a opportunity to deal with the question generation in ecjk domain.

### Minimize Installtion
If someone only want to use the trained [donut](https://github.com/clovaai/donut) model to generate questions in a image. I have upload a early stopped version of them to huggingface hub in English and Chinese domains respectively. You can download them from [svjack/question_generator_by_zh_on_pic](https://huggingface.co/svjack/question_generator_by_zh_on_pic) and [svjack/question_generator_by_en_on_pic](https://huggingface.co/svjack/question_generator_by_en_on_pic) and simply install  [donut](https://github.com/clovaai/donut) by
```bash
pip install torch 
pip install transformers==4.11.3
pip install opencv-python==4.6.0.66
pip install donut-python
```
This will help you use them. (you can use git-lfs to download them and use DonutModel.from_pretrained(en_model_path) to init weights as [donut](https://github.com/clovaai/donut) do)

### One step forward
In NLP domain, if give you a paragraph, how can you get some questions and their answers from this paragraph, The whole target can be divided into some blocks.<br/>

Below is the self train and build NLP Example deploy in HuggingFace Space.<br/>
<b>[Question generate 🍩 demonstration](https://huggingface.co/spaces/svjack/Question-Generator)</b>

#### Question generate on Paragraph Results illustration
<table><caption></caption>
<tbody>
<tr>
<td>Image</td>
<td><img src="imgs/en_nlp_input.png" alt="Girl in a jacket" width="450" height="150"></td>
<td><img src="imgs/zh_nlp_input.png" alt="Girl in a jacket" width="450" height="150"></td>
</tr>
<tr>
<td>Question Answer From Above pics</td>
<td><img src="imgs/en_nlp_output.png" alt="Girl in a jacket" width="450" height="450"></td>
<td><img src="imgs/zh_nlp_output.png" alt="Girl in a jacket" width="450" height="450"></td>
</tr>
</tbody>
</table>

Firstly, determine what kind of questions answers people mainly interested. One common used NLP task answer this problem --- NER, in many topics, people mainly care about Named-entity, so many datasets are constructed centered with NE. So you can extract Named-entity as answers. If the answer come from text, then a native NER is required.<br/>
<b>When it comes to image</b>, This demand met by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s [kie module](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppstructure/kie) in [PPStructure](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppstructure)

Secondly, use a generator generate question about this answer.<br/>
<b>When it comes to image</b>, This have done with the help of [donut](https://github.com/clovaai/donut), But because the variation of dataset, donut question generator may not works when the image construct by some long paragraph. This make the demand about, extract the long paragraph by OCR, and generate question on the paragraph recognize by OCR. In English domain, [text2text](https://github.com/artitw/text2text) deal with it by its Handler, and I release my [svjack/squad_gen_qst_zh_v0](https://huggingface.co/svjack/squad_gen_qst_zh_v0) works for Chinese.

Thirdly, Valid the question generate by a squad style model.<br/>
<b>When it comes to image</b>, valid questions generated by [donut generator](https://huggingface.co/svjack/question_generator_by_en_on_pic) and [text generator](https://github.com/artitw/text2text). This require a model that support perform question answer on Document images write by English and Chinese. This met by [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)'s [Document Intelligence](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout)

After the above discussion, Task of generate questions from paragraph have been promoted to generate questions from document images. And [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) give the support about this improve from [donut](https://github.com/clovaai/donut) --- a only generator <b>"One Step Forward"</b>, with its compatibility in English and Chinese. And the dataset generation of DocVQA has done in the
theoretical point of view.

### Fully Installtion
If someone want to use the function of "One step forward" to generate questions and answers from Document image, other than [donut](https://github.com/clovaai/donut) you should also install model to perform NER (here i choose [spacy](https://github.com/explosion/spaCy)) [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and download some require models to local. For simplify the install process, in the [notebook.ipynb](notebook.ipynb), it integrated the every Installtion steps and run demo in a tiny jupyter notebook. You can run this notebook in any well telecommunication notebook server (as Kaggle Colab or your local jupyter server)<br/>
Or you can install python packages by requiremnets.txt by 
```bash
pip install -r requirements.txt
```
and run apt-get install to the packages in packages.txt download [svjack/question_generator_by_zh_on_pic](https://huggingface.co/svjack/question_generator_by_zh_on_pic) and [svjack/question_generator_by_en_on_pic](https://huggingface.co/svjack/question_generator_by_en_on_pic).


### Example deploy in Huggingface Space (without Validation in above Third step)

<!--
<b>[Question generate on Document Image 🍩 demonstration](https://huggingface.co/spaces/svjack/Question-Generator-On-Documnet-Image)</b>
-->

#### Question generate on Document Image Results illustration
<table><caption></caption>
<tbody>
<tr>
<td>Image</td>
<td><img src="imgs/en_img.png" alt="Girl in a jacket" width="450" height="450"></td>
<td><img src="imgs/zh_img.png" alt="Girl in a jacket" width="450" height="450"></td>
</tr>
<tr>
<td>Question Answer From Above pics</td>
<td><img src="imgs/en_output.png" alt="Girl in a jacket" width="450" height="450"></td>
<td><img src="imgs/zh_output.png" alt="Girl in a jacket" width="450" height="450"></td>
</tr>
</tbody>
</table>

## HuggingFace demo

### model demo
|Name |HuggingFace model link| HuggingFace space link | Language |
|---------|--------|-------|-------|
| Question Generator from English Document Image 🦅| https://huggingface.co/svjack/question_generator_by_en_on_pic | https://huggingface.co/spaces/svjack/Question-Generator-on-English-Doc | English |
| Question Generator from Chinese Document Image 🐰| https://huggingface.co/svjack/question_generator_by_zh_on_pic | https://huggingface.co/spaces/svjack/Question-Generator-on-Chinese-Doc | Chinese |

### Mannully Examples
Because the paths are fixed in the project. All functions to check the examples should be run in the root dir of the project.
And i suggest to run [notebook.ipynb](notebook.ipynb) to finish the Installtion before check the examples.
check [imgs](imgs) content, before use them.

#### question generation by set answer manually
* 1
```python
from qa_on_image import *
img_path = "imgs/en_img.png"
input_img = read_img_to_3d_array(img_path)
demo_process_vqa(input_img, "605-7227", "en")
```
This will give the output:
```json
{'question': '605-7227', 'answer': 'What is the Phone #?'}
```
* 2
```python
from qa_on_image import *
img_path = "imgs/zh_img.png"
input_img = read_img_to_3d_array(img_path)
demo_process_vqa(input_img, "零钱通", "zh")
```
This will give the output:
```json
{'question': '零钱通', 'answer': '支付方式是什么?'}
```

#### One step forward (generate questions from Document image) example
* 1
```python
from qa_on_pic import *
img_path = "imgs/en_img.png"
qa_df = generate_qa(img_path, ppstructure_bin_path, table_engine)
qa_df.values.tolist()
```
This will give the output:
```json
[['What is the supplier name?', 'Coyne Beahm Shouse, Inc', 1.0, 'image'],
 ['What is the Address?', '6522 Bryan Bouievard', 1.0, 'image'],
 ['What is the Phone #?', '(336) 605-7227', 0.99, 'image'],
 ['What is the Effective Date?', '9/7/2005', 0.97, 'image'],
 ['What was the other name of the other Supplier Name?',
  'Coyne Beahm Shouse, Inc',
  0.97,
  'context'],
 ['What must be included in the contract?',
  'signed Bid Waiver',
  0.91,
  'context'],
 ['What is the number of the street number in the city?', '1', 0.8, 'context'],
 ['What is the job assignment?', '9/7/2005', 0.59, 'image']]
```

* 2
```python
from qa_on_pic import *
img_path = "imgs/zh_img.png"
qa_df = generate_qa(img_path, ppstructure_bin_path, table_engine)
qa_df.values.tolist()
```
This will give the output:
```json
[['账单的商户全称是什么?', '云城区小木船酒行', 1.0, 'image'],
 ['当前状态是什么?', '支付成功', 1.0, 'image'],
 ['支付方式是什么?', '零钱通', 0.99, 'image'],
 ['账单的支付金额是多少?', '2980.00', 0.85, 'image']]
```

* 3
```python
from qa_on_pic import *
img_path = "imgs/en_context.png"
qa_df = generate_qa(img_path, ppstructure_bin_path, table_engine)
qa_df.values.tolist()
```
This will give the output:
```json
[['When must the payment terms be completed?',
  'prior to September 1, 1994',
  1.0,
  'context'],
 ['How often are Safeway" Hot Shoe" awards provided?',
  'annual',
  0.88,
  'context'],
 ['Who received $ 200 in Safeway Gift Certificates?',
  'Winner',
  0.82,
  'context']]
```

* 4
```python
from qa_on_pic import *
img_path = "imgs/zh_context_1.png"
qa_df = generate_qa(img_path, ppstructure_bin_path, table_engine)
qa_df.values.tolist()
```
This will give the output:
```json
[['公司何时发布《发行股份及支付现金购买资产并募集资金汇合备', '2018年12月27日', 1.0, 'context'],
 ['中金黄金在购买其持有的矿业', '90%股权', 0.96, 'context'],
 ['右下角的数字是什么?', '2020', 0.88, 'image'],
 ['该图中的人民币是指什么?', '当前价', 0.82, 'image'],
 ['谁为其持有的矿业股份 并获得其持有的矿业股份?', '中金黄金', 0.55, 'context']]
```

## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/docvqa-gen](https://github.com/svjack/docvqa-gen)<br/>

HuggingFace Space Link:<br/>
<!--
[https://huggingface.co/spaces/svjack/Question-Generator-On-Documnet-Image](https://huggingface.co/spaces/svjack/Question-Generator-On-Documnet-Image)<br/>
-->
[https://huggingface.co/spaces/svjack/Question-Generator](https://huggingface.co/spaces/svjack/Question-Generator)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)

* [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)
* [diffusers](https://github.com/huggingface/diffusers)
* [diffusiondb](https://github.com/poloclub/diffusiondb)
* [Taiyi-Stable-Diffusion-1B-Chinese-v0.1](IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1)
* [prompt-extend](https://github.com/daspartho/prompt-extend)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [Stable-Diffusion-Pokemon](https://github.com/svjack/Stable-Diffusion-Pokemon)
* [svjack](https://huggingface.co/svjack)
-->
* [text2text](https://github.com/artitw/text2text)
* [donut](https://github.com/clovaai/donut)
* [primeqa](https://github.com/primeqa/primeqa)
* [DocVQA task](https://www.docvqa.org)
* [spacy](https://github.com/explosion/spaCy)
* [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)  
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [svjack](https://huggingface.co/svjack)
